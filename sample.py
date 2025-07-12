from diffusers_helper.hf_login import login

import os

import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
# from diffusers_helper.thread_utils import AsyncStream, async_run # 移除 Gradio 相关的线程工具
# from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html # 移除 Gradio UI 工具
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
import oss2

bucket = oss2.Bucket(
    oss2.Auth(*tuple(open('oss_aksk.txt').read().strip().split('\n'))),
	'oss-cn-hangzhou.aliyuncs.com', 'metac-video-pitcturebook'
)


parser = argparse.ArgumentParser()
# 添加执行所需的参数
parser.add_argument("--input_image", type=str, required=True, help="Path to the input image.")
parser.add_argument("--prompt", type=str, required=True, help="Positive prompt.")
parser.add_argument("--n_prompt", type=str, default="", help="Negative prompt.")
parser.add_argument("--seed", type=int, default=31337, help="Random seed.")
parser.add_argument("--total_second_length", type=float, default=5.0, help="Total video length in seconds.")
parser.add_argument("--steps", type=int, default=25, help="Number of diffusion steps.")
parser.add_argument("--gs", type=float, default=10.0, help="Distilled CFG Scale.")
parser.add_argument("--mp4_crf", type=int, default=16, help="MP4 CRF value (lower means better quality).")
parser.add_argument("--output_dir", type=str, default='./outputs/', help="Directory to save the output video.")
parser.add_argument("--gpu_memory_preservation", type=float, default=6.0, help="GPU memory to preserve during inference (GB).")
parser.add_argument("--use_teacache", action='store_true', help="Enable TeaCache optimization.")
# 保留或移除 Gradio 特定的参数，根据需要
# parser.add_argument('--share', action='store_true')
# parser.add_argument("--server", type=str, default='0.0.0.0')
# parser.add_argument("--port", type=int, required=False)
# parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

print(args)

# --- 模型加载和设置代码保持不变 ---
os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

# stream = AsyncStream() # 移除

outputs_folder = args.output_dir # 使用参数指定的输出目录
os.makedirs(outputs_folder, exist_ok=True)


@torch.no_grad()
def generate_video(input_image_path, prompt, n_prompt, seed, total_second_length, steps, gs, gpu_memory_preservation, use_teacache, mp4_crf):
    # --- 从 worker 函数改编的核心逻辑 ---
    # 固定或移除 worker 中依赖 Gradio 的参数
    latent_window_size = 9 # 固定值
    cfg = 1.0 # 固定值
    rs = 0.0 # 固定值

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    final_output_filename = None

    # stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...')))) # 移除 UI 更新
    print('Starting ...')

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

        # Text encoding
        # stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...')))) # 移除 UI 更新
        print('Text encoding ...')

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1: # 保持逻辑，即使 cfg 固定
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # Processing input image
        # stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Image processing ...')))) # 移除 UI 更新
        print('Image processing ...')

        # 从文件加载图像
        input_image = Image.open(input_image_path).convert('RGB')
        input_image = np.array(input_image)

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=640)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)

        # 保存处理后的输入图像（可选）
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}_input.png'))

        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        # stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...')))) # 移除 UI 更新
        print('VAE encoding ...')

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        start_latent = vae_encode(input_image_pt, vae)

        # CLIP Vision
        # stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...')))) # 移除 UI 更新
        print('CLIP Vision encoding ...')

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        # stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...')))) # 移除 UI 更新
        print('Start sampling ...')

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size

            # if stream.input_queue.top() == 'end': # 移除 Gradio 停止逻辑
            #     stream.output_queue.push(('end', None))
            #     return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not high_vram:
                unload_complete_models()
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                # 移除预览生成和 UI 更新
                # preview = d['denoised']
                # preview = vae_decode_fake(preview)
                # preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                # preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                # if stream.input_queue.top() == 'end': # 移除 Gradio 停止逻辑
                #     stream.output_queue.push(('end', None))
                #     raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = f'Sampling {current_step}/{steps}'
                # desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds (FPS-30). The video is being extended now ...'
                # stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint)))) # 移除 UI 更新
                print(f"  Step: {current_step}/{steps} ({percentage}%)") # 打印进度到控制台
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg, # 使用固定的 cfg
                distilled_guidance_scale=gs, # 使用参数 gs
                guidance_rescale=rs, # 使用固定的 rs
                num_inference_steps=steps, # 使用参数 steps
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            # 保存中间或最终视频
            output_filename = os.path.join(outputs_folder, f'{job_id}.mp4') # 使用固定的最终文件名
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
            final_output_filename = output_filename # 更新最终文件名

            print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')
            print(f'Saved intermediate video to {output_filename}')

            # stream.output_queue.push(('file', output_filename)) # 移除 UI 更新

            if is_last_section:
                break
    except Exception as e: # 使用更通用的异常处理
        traceback.print_exc()
        print(f"Error during generation: {e}")
    finally: # 确保模型在结束或出错时卸载
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    # stream.output_queue.push(('end', None)) # 移除 UI 更新
    print('Generation finished.')
    return final_output_filename # 返回最终文件路径

# --- 移除 Gradio UI 定义和启动代码 ---
# def process(...): ...
# def end_process(...): ...
# block = gr.Blocks(...)
# with block: ...
# block.launch(...)

if __name__ == "__main__":
    if not os.path.exists(args.input_image):
        print(f"Error: Input image not found at {args.input_image}")
        exit(1)

    output_video_path = generate_video(
        input_image_path=args.input_image,
        prompt=args.prompt,
        n_prompt=args.n_prompt,
        seed=args.seed,
        total_second_length=args.total_second_length,
        steps=args.steps,
        gs=args.gs,
        gpu_memory_preservation=args.gpu_memory_preservation,
        use_teacache=args.use_teacache,
        mp4_crf=args.mp4_crf
    )

    if output_video_path:
        print(f"Successfully generated video: {output_video_path}")
    else:
        print("Video generation failed.")
