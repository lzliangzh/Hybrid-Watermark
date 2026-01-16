import argparse
import copy
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPProcessor
from stable_diffusion.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from utils.optim_utils import *
from utils.io_utils import *
from utils.image_utils import *
from watermark_hybrid import *





def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 固定随机性
    g = torch.Generator(device=device).manual_seed(args.gen_seed)
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            dtype=torch.float16,
            # revision='fp16',
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    #reference model for CLIP Score
    if args.reference_model is not None:
        ref_model = CLIPModel.from_pretrained(args.reference_model).to(device)
        ref_clip_processor = CLIPProcessor.from_pretrained(args.reference_model)

    # dataset
    dataset, prompt_key = get_dataset(args)

    # # class for watermark
    # if args.chacha:
    #     watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number,
    #                                         generator=g)
    # else:
    #     #a simple implement,
    #     watermark = Gaussian_Shading(args.channel_copy, args.hw_copy, args.fpr, args.user_number)

    
    watermark = HybridWatermarker(
        device=device,
        gs_ch_factor=args.channel_copy,
        gs_hw_factor=args.hw_copy,
        ring_radius=RADIUS,
        ring_radius_cutoff=RADIUS_CUTOFF,
        fpr_target=args.fpr,
        user_number=args.user_number,
        debug=True,
        use_chacha=args.chacha
    )
    os.makedirs(args.output_path, exist_ok=True)

    # assume at the detection time, the original prompt is unknown
    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    #acc
    acc = []
    #CLIP Scores
    clip_scores = []

    #test
    for i in tqdm(range(args.num)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]

        #generate with watermark
        set_random_seed(seed)
        base_latent = pipe.get_random_latents()
        random_user_id = int(torch.randint(0, args.user_number, (1,)).item())
        init_latents_w = watermark.create_watermark_and_return_w(base_latent, random_user_id)
        
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=1,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            generator=g
        )
        
        image_w = outputs_w.images[0]
        image_w.save(f"{args.output_path}/{i}_w.png")
        
        
        # # No Watermark
        # random_gaussian_latents = torch.randn(
        #     (1, 4, args.image_length // 8, args.image_length // 8),
        #     generator=g,
        #     device=device, 
        #     dtype=torch.float16
        # ).to()
        # 
        # outputs = pipe(
        #     current_prompt,
        #     num_images_per_prompt=1,
        #     guidance_scale=args.guidance_scale,
        #     num_inference_steps=args.num_inference_steps,
        #     height=args.image_length,
        #     width=args.image_length,
        #     latents=random_gaussian_latents,
        #     generator=g
        # )
        
        # image = outputs.images[0]
        # image.save(f"{args.output_path}/{i}.png")

        # distortion
        image_w_distortion = image_distortion(image_w, seed, args)

        image_w_distortion.save(f"{args.output_path}/{i}_distorted.png")
        
        # reverse img
        image_w_distortion = transform_img(image_w_distortion).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(image_w_distortion, sample=False)
        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.num_inversion_steps,
        )

        #acc metric
        acc_metric = watermark.eval_watermark(reversed_latents_w)
        print(acc_metric)
        acc.append(acc_metric)

        #CLIP Score
        if args.reference_model is not None:
            socre = measure_similarity([image_w], current_prompt, ref_model,
                                              ref_clip_processor, device)
            clip_socre = socre[0].item()
        else:
            clip_socre = 0
        clip_scores.append(clip_socre)

    #tpr metric
    tpr_detection, tpr_traceability = watermark.get_tpr()
    print(watermark.get_tpr())
    
    #save metrics
    save_metrics(args, tpr_detection, tpr_traceability, acc, clip_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=20, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=128, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='manojb/stable-diffusion-2-1-base')

    # for image distortion
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rotate', default=None, type=float)


    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
