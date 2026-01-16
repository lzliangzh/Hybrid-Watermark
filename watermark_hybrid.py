import torch
import numpy as np
from scipy.stats import norm, truncnorm
from scipy.special import betainc
from functools import reduce
from utils.rid_utils import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

# 假设 util.py 中的 fft, ifft, ring_mask, make_Fourier_ringid_pattern 已在全局作用域可用
# 或者你可以通过 import 引入

class HybridWatermarker:
    def __init__(self, 
                 device,
                 # Gaussian Shading 参数
                 gs_ch_factor=1, gs_hw_factor=1, 
                 # RingID 参数
                 ring_radius=RADIUS, ring_radius_cutoff=RADIUS_CUTOFF,
                 # 通用参数
                 fpr_target=1e-6, 
                 user_number=1,
                 user_id=0,
                 # 设为 True 以打印acc/distance、可视化水印情况
                 debug: bool = False,
                 use_chacha: bool = False):
        
        self.device = device
        self.debug = debug
        self.use_chacha = use_chacha
        
        # --- 1. Gaussian Shading 配置 (针对 HETER_WATERMARK_CHANNEL) ---
        self.gs_ch = gs_ch_factor
        self.gs_hw = gs_hw_factor
        self.gs_channels = [0,3]
        self.latent_shape = (4, 64, 64)
        
        self.gs_mark_length = (len(self.gs_channels) * 64 * 64) // (self.gs_ch * self.gs_hw * self.gs_hw) # GS 的有效位长度 (在指定的通道内)
        self.gs_threshold_vote = 1 if self.gs_hw == 1 and self.gs_ch == 1 else self.gs_ch * self.gs_hw * self.gs_hw // 2

        
        # --- 2. RingID 配置 ---
        self.ring_radius = ring_radius
        self.ring_radius_cutoff = ring_radius_cutoff
        self.ring_mark_length = ring_radius - ring_radius_cutoff
        
        self.ring_value_range = 32
        self.ring_channels = [1,2] 
        
        self.ring_mask_single = torch.tensor(
            ring_mask(
                size=64,
                r_out=self.ring_radius, 
                r_in=self.ring_radius_cutoff)
            ).to(self.device)
        self.ring_masks = torch.stack([self.ring_mask_single for _ in self.ring_channels]).to(self.device)
        

        
        # --- 3. 统计与阈值设置 (Detection/Traceability) ---
        self.user_number = user_number
        self.tau_gs = self._calculate_bit_threshold(self.gs_mark_length, fpr_target, 1)
        self.tau_gs_traceable = self._calculate_bit_threshold(self.gs_mark_length, fpr_target, user_number)
        self.ring_threshold_dist = 30.8 # 判定 L1 距离的阈值, 29.6 <=> fpr=1e-6, 30.8 <=> fpr=1%. 需引入函数计算
        # self.tau_ring = self._calculate_bit_threshold(self.ring_mark_length * len(self.ring_channels), fpr_target, user_number)
        
        # 统计计数器
        self.tp_detection_count = 0
        self.tp_traceability_count = 0

        # 密钥信息
        self.gs_key = None
        self.gs_nonce = None
        self.gs_msg = None
        self.ring_message = None
        self.ring_key_values = None # RingID 的模式数值
        
        # 预生成 RingID 频域 Patterns (user_nums 个, 供后续随机抽取)
        self.ring_candidate_patterns, self.ring_user_ids = \
            self._generate_candidate_database(self.user_number)
        self.ring_current_user_id = user_id
        self.output_latents = None

    def _stream_key_encrypt(self, bits: np.ndarray):
        if not self.use_chacha:
            return bits
        self.gs_key = get_random_bytes(32)
        self.gs_nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.gs_key, nonce=self.gs_nonce)
        m_byte = cipher.encrypt(np.packbits(bits).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))[: bits.size]
        return m_bit

    def _stream_key_decrypt(self, bits: np.ndarray):
        if not self.use_chacha:
            return bits
        cipher = ChaCha20.new(key=self.gs_key, nonce=self.gs_nonce)
        m_byte = cipher.decrypt(np.packbits(bits).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))[: bits.size]
        return m_bit

    def _derive_ring_key_nonce(self, user_id: int):
        gen = torch.Generator(device="cpu").manual_seed(int(user_id))
        key = torch.randint(0, 256, (32,), generator=gen, dtype=torch.uint8).numpy().tobytes()
        nonce = torch.randint(0, 256, (12,), generator=gen, dtype=torch.uint8).numpy().tobytes()
        return key, nonce

    def _ring_encrypt_message(self, user_id: int, message: torch.Tensor):
        if not self.use_chacha:
            return message
        bits = message.flatten().detach().cpu().numpy().astype(np.uint8)
        key, nonce = self._derive_ring_key_nonce(user_id)
        cipher = ChaCha20.new(key=key, nonce=nonce)
        m_byte = cipher.encrypt(np.packbits(bits).tobytes())
        m_bit = np.unpackbits(np.frombuffer(m_byte, dtype=np.uint8))[: bits.size]
        return torch.from_numpy(m_bit).to(self.device).reshape(message.shape)

    def _calculate_bit_threshold(self, length, fpr, user_num):
        # 简单的阈值计算逻辑，用于判定检测成功
        for i in range(length):
            p_val = betainc(i + 1, length - i, 0.5) * user_num
            if p_val <= fpr:
                return i / length
        return 0.8 # 默认

    def _trunc_sampling(self, message):
        """Gaussian Shading 的截断正态采样"""
        m_flat = message.flatten()
        m_bin = (m_flat > 0.5).astype(np.int64)
        z = np.zeros_like(m_bin, dtype=np.float32)
        ppf = [norm.ppf(0.0), norm.ppf(0.5), norm.ppf(1.0)]
        for i, val in enumerate(m_bin):
            z[i] = truncnorm.rvs(ppf[int(val)], ppf[int(val) + 1])
        return torch.from_numpy(z).reshape(len(self.gs_channels), 64, 64).to(self.device).half()


    def visualize_hybrid_latents(self, output_latents, save_path="hybrid_debug.png"):
        """
        output_latents: [1, 4, 64, 64]
        """
        # 转换到 CPU
        latents = output_latents[0].detach().cpu()
        
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        
        # --- 1. 可视化通道 0, 3 (频域 FFT) ---
        for i, ch in enumerate(self.ring_channels):
            # 执行 FFT
            # 假设 fft 函数返回的是复数张量
            ch_fft = fft(output_latents[:, ch:ch+1])[0, 0].real.detach().cpu().numpy()
            ch_fft = np.nan_to_num(ch_fft, nan=0.0, posinf=0.0, neginf=0.0)
            
            im = axes[i].imshow(ch_fft, cmap='RdBu_r', vmin=-64, vmax=64)
            axes[i].set_title(f"Channel {ch} FFT (Real Part)")
            fig.colorbar(im, ax=axes[i])

        # --- 2. 可视化通道 1, 2 (空域 Spatial) ---
        for i, ch in enumerate(self.gs_channels):
            # 直接显示空域数值
            ch_spatial = latents[ch].numpy()
            ch_spatial = np.nan_to_num(ch_spatial, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Gaussian Shading 的值通常在 [-2, 2] 左右（截断正态分布）
            im = axes[i+2].imshow(ch_spatial, cmap='viridis')
            axes[i+2].set_title(f"Channel {ch} Spatial (GS)")
            fig.colorbar(im, ax=axes[i+2])

        plt.suptitle("Hybrid Watermark: RingID (Freq) vs Gaussian Shading (Spatial)")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        plt.close()

    def _generate_candidate_database(self, num_users):
        """
        生成多个用户的候选水印库
        """
        candidate_patterns = []
        user_ids = []
        
        for user_id in tqdm(range(num_users)):
            # 1. 为每个用户生成唯一的 Ring Message (bit 序列)
            # 注意：这里需要固定随机种子以保证之后能重新生成同样的 key
            torch.manual_seed(user_id) 
            user_ring_message = torch.randint(0, 2, (self.ring_mark_length, len(self.ring_channels))).to(self.device)
            user_ring_message = self._ring_encrypt_message(user_id, user_ring_message)
            
            # 2. 映射为数值
            user_key_values = torch.where(user_ring_message == 1, self.ring_value_range, -self.ring_value_range)
                
            # 3. 生成该用户对应的 Pattern
            # 这里的 base_latents 可以是全 0，因为 Pattern 主要是叠加量
            pattern = make_Fourier_ringid_pattern(
                device=self.device,
                key_value_combination=user_key_values,
                no_watermark_latents=torch.zeros(1, *self.latent_shape).to(self.device), 
                radius=self.ring_radius,
                radius_cutoff=self.ring_radius_cutoff,
                ring_watermark_channel=self.ring_channels,
                heter_watermark_channel=[],
            )
            
            candidate_patterns.append(pattern)
            user_ids.append(user_id)
            
        # Spatial Shift
        # 注: 没有 Spatial Shift, rotate 75 的 detection 和 identify 都非常差.
        for pattern in candidate_patterns:
            # 这里使用了 fftshift 对反变换后的空间域信号进行了平移
            pattern[:, self.ring_channels, ...] = fft(
                torch.fft.fftshift(ifft(pattern[:, self.ring_channels, ...]), dim = (-1, -2))
            )
            
        return candidate_patterns, user_ids
            
    def create_watermark_and_return_w(self, base_latents, user_id):
        """
        混合注入：
        1. 在指定通道注入 Gaussian Shading (空间域变换)
        2. 在指定通道注入 RingID (频域变换)
        """
        # self.stats['total'] += 1
        output_latents = base_latents.clone()
        self.ring_current_user_id = user_id
        
        # --- 准备 Gaussian Shading (空间域) ---
        # 生成随机 Key 和 Message
        self.gs_msg = torch.randint(0, 2, (len(self.gs_channels) // self.gs_ch, 64 // self.gs_hw, 64 // self.gs_hw), device=self.device)
        # 扩展消息并与 Key 异或
        sd = self.gs_msg.repeat(1, self.gs_ch, self.gs_hw, self.gs_hw)
        if self.use_chacha:
            m_bits = self._stream_key_encrypt(sd.flatten().cpu().numpy())
            m = m_bits.reshape(len(self.gs_channels), 64, 64)
        else:
            self.gs_key = torch.randint(0, 2, (len(self.gs_channels), 64, 64), device=self.device)
            m = ((sd + self.gs_key) % 2).cpu().numpy()
        gs_w = self._trunc_sampling(m)
        
        # 替换指定通道
        for i, ch in enumerate(self.gs_channels):
            output_latents[:, ch] = gs_w[i]
        
        # --- 使用 generate_Fourier_watermark_latents 注入 Ring Pattern [user_id] ---

        output_latents = generate_Fourier_watermark_latents(
            device=self.device,
            radius=self.ring_radius,
            radius_cutoff=self.ring_radius_cutoff,
            watermark_region_mask=self.ring_masks,
            watermark_channel=self.ring_channels,
            original_latents=output_latents,
            watermark_pattern=self.ring_candidate_patterns[self.ring_current_user_id]
        )
        
        self.output_latents = output_latents
        if self.debug:
            self.visualize_hybrid_latents(output_latents, "after_injection.png")
        
        return output_latents
    
    def visualize_watermark_debug(self, latents, save_path="debug_watermark.png"):
        """
        在RingID水印的第1个通道上，可视化 RingID 的三个核心组件：Pattern, Mask, 以及提取的 FFT。
        """
        # 1. 获取频域数据
        latents_fft = fft(latents) # [1, 4, 64, 64]
        init_latents_fft = fft(self.output_latents)
        # 2. 准备绘图数据 (选取第一个水印通道进行展示，例如 ch=1)
        ch = self.ring_channels[0]
        
        # 我们看实部 (real)，因为 RingID 的 Pattern 主要体现在实部振幅上
        pattern_viz = init_latents_fft[0, ch].real.cpu().numpy()
        mask_viz = self.ring_masks[0].cpu().numpy()
        fft_viz = latents_fft[0, ch].real.detach().cpu().numpy()
        
        # 对 FFT 结果进行对数缩放，方便观察高频细节（可选）
        fft_log = np.log1p(np.abs(fft_viz))

        # 3. 开始绘图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # A. 预设的 Pattern (应该是清晰的 $\pm 32$ 构成的圆环)
        im0 = axes[0].imshow(pattern_viz, cmap='RdBu_r')
        axes[0].set_title(f"Target Pattern (Ch {ch})")
        fig.colorbar(im0, ax=axes[0])

        # B. 掩码区域 (应该在半径 3-14 之间)
        im1 = axes[1].imshow(mask_viz, cmap='gray')
        axes[1].set_title("Watermark Mask Area")

        # C. 实际提取的频谱 (应该能隐约看到 Pattern 的影子)
        im2 = axes[2].imshow(fft_viz, cmap='RdBu_r', vmin=-64, vmax=64)
        axes[2].set_title("Extracted FFT (Real)")
        fig.colorbar(im2, ax=axes[2])

        # D. 叠加显示 (将 Mask 覆盖在 FFT 上)
        overlay = fft_viz * mask_viz
        im3 = axes[3].imshow(overlay, cmap='RdBu_r', vmin=-64, vmax=64)
        axes[3].set_title("Extracted FFT in Mask")
        fig.colorbar(im3, ax=axes[3])

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Debug image saved to {save_path}")
        plt.close()
        
    def _diffusion_inverse(self, watermark_sd):
        """
        将提取到的 sd (64x64) 还原为 watermark (8x8)
        watermark_sd: [C, 64, 64] (当前 C=1)
        """
        # 1. 计算步长：64 / 8 = 8
        hw_stride = 64 // self.gs_hw  # 如果 gs_hw=8，步长就是 8
        
        # 2. 空间维度拆分并堆叠，准备投票
        # 先拆高 (H)，再拆宽 (W)
        # split 后变成多个 [C, 8, 64] -> cat 变成 [C*8, 8, 64]
        h_split = torch.cat(torch.split(watermark_sd, hw_stride, dim=1), dim=0)
        # 再对 W 拆：[C*8, 8, 64] -> [C*8*8, 8, 8]
        w_split = torch.cat(torch.split(h_split, hw_stride, dim=2), dim=0)
        
        # 3. 执行投票
        # w_split 现在包含了所有的重复块。对第一个维度求和。
        # 阈值判定：如果超过一半的副本是 1，则判定为 1
        vote = torch.sum(w_split, dim=0)
        
        # 这里的阈值 self.gs_threshold_vote 应该是 (64/8 * 64/8) / 2 = 32
        res = torch.zeros_like(vote)
        res[vote > self.gs_threshold_vote] = 1
        return res
    

    def _eval_gs(self, latents):
        """评估 Gaussian Shading 通道"""
        # 提取指定通道并二值化
        extracted_ch = latents[0, self.gs_channels]
        reversed_m = (extracted_ch > 0).int()
        # 解密
        if self.use_chacha:
            reversed_bits = self._stream_key_decrypt(reversed_m.flatten().cpu().numpy())
            reversed_sd = torch.from_numpy(reversed_bits).to(self.device).reshape(len(self.gs_channels), 64, 64)
        else:
            reversed_sd = (reversed_m + self.gs_key) % 2
        channel_accs = []
        for i in range(len(self.gs_channels)):
            reversed_watermark = self._diffusion_inverse(reversed_sd[i:i+1])
            acc = (reversed_watermark == self.gs_msg[i]).float().mean().item()
            channel_accs.append(acc)
        
        avg_acc = np.mean(channel_accs)
        is_detected = avg_acc >= self.tau_gs
        is_traceable = avg_acc >= self.tau_gs_traceable
        return is_detected, is_traceable, acc

    def _eval_ring(self, latents):
        """评估 RingID 通道"""
        # 转到频域
        latents_fft = fft(latents)
        distances = []
        for pattern in self.ring_candidate_patterns:
            d = get_distance(
                pattern, 
                latents_fft, 
                self.ring_masks, 
                channel=self.ring_channels,
                p=1,
                mode="complex"
            )
            distances.append(d / len(self.ring_channels))
        
        best_match_idx = np.argmin(distances)
        min_dist = distances[best_match_idx]
        if self.debug:
            print(self.ring_current_user_id, best_match_idx, min_dist, min(distances))
        is_detected = min_dist < self.ring_threshold_dist
        is_traceable = is_detected and (best_match_idx == self.ring_current_user_id)
        
        return is_detected, is_traceable, -min_dist
    
    @torch.no_grad()
    def eval_watermark(self, reversed_latents):
        """
        1. 计算 GS 的正确率
        2. 计算 RingID 的正确率
        3. 只要有一个达到 tau / threshold，detection 成功
        4. traceability 逻辑根据具体业务定义（此处采用混合检出即追溯）
        """
        is_gs_detected, is_gs_traceable, gs_score = self._eval_gs(reversed_latents)
        is_ring_detected, is_ring_traceable, ring_score = self._eval_ring(reversed_latents)
        
        if self.debug:
            self.visualize_watermark_debug(reversed_latents, "debug_ch1.png")
            print("Gaussian Shading Acc: ", gs_score)
            print("RingID L1 Distance: ", ring_score)
            print("GS & RID Detected:", is_gs_detected, is_ring_detected)
            print("GS & RID Traceable:", is_gs_traceable, is_ring_traceable)
        
        if is_gs_detected or is_ring_detected:
            self.tp_detection_count += 1

        if (is_gs_detected or is_ring_detected) and (is_gs_traceable or is_ring_traceable):
            self.tp_traceability_count += 1 

        # （供 save_metrics 使用）
        return np.mean([gs_score, ring_score])

    def get_tpr(self):
        # 返回 detection 和 traceability 的累计计数
        return self.tp_detection_count, self.tp_traceability_count