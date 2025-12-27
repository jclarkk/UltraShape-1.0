import os
import sys
import argparse
import torch
import gc
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import gradio as gr
import trimesh

# Add project root to path
sys.path.append(os.getcwd())

from ultrashape.rembg import BackgroundRemover
from ultrashape.utils.misc import instantiate_from_config
from ultrashape.surface_loaders import SharpEdgeSurfaceLoader
from ultrashape.utils import voxelize_from_point
from ultrashape.pipelines import UltraShapePipeline 

# Global variables to cache the model
MODEL_CACHE = {}

def get_pipeline_cached(config_path, ckpt_path, device='cuda', low_vram=False):
    # Check if we have a valid cached pipeline for this checkpoint
    if "pipeline" in MODEL_CACHE and MODEL_CACHE.get("ckpt_path") == ckpt_path:
        print("Using cached pipeline...")
        return MODEL_CACHE["pipeline"], MODEL_CACHE["config"]

    # Clear old cache if it exists (e.g. different checkpoint)
    if MODEL_CACHE:
        print("Clearing old model cache...")
        MODEL_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading config from {config_path}...")
    config = OmegaConf.load(config_path)
    
    print("Instantiating VAE...")
    vae = instantiate_from_config(config.model.params.vae_config)
    
    print("Instantiating DiT...")
    dit = instantiate_from_config(config.model.params.dit_cfg)
    
    print("Instantiating Conditioner...")
    conditioner = instantiate_from_config(config.model.params.conditioner_config)
    
    print("Instantiating Scheduler & Processor...")
    scheduler = instantiate_from_config(config.model.params.scheduler_cfg)
    image_processor = instantiate_from_config(config.model.params.image_processor_cfg)
    
    print(f"Loading weights from {ckpt_path}...")
    weights = torch.load(ckpt_path, map_location='cpu')
    
    vae.load_state_dict(weights['vae'], strict=True)
    dit.load_state_dict(weights['dit'], strict=True)
    conditioner.load_state_dict(weights['conditioner'], strict=True)
    
    vae.eval().to(device)
    dit.eval().to(device)
    conditioner.eval().to(device)
    
    if hasattr(vae, 'enable_flashvdm_decoder'):
        vae.enable_flashvdm_decoder()

    print("Creating Pipeline...")
    pipeline = UltraShapePipeline(
        vae=vae,
        model=dit,
        scheduler=scheduler,
        conditioner=conditioner,
        image_processor=image_processor
    )

    if low_vram:
        pipeline.enable_model_cpu_offload()
    
    MODEL_CACHE["pipeline"] = pipeline
    MODEL_CACHE["config"] = config
    MODEL_CACHE["ckpt_path"] = ckpt_path
    
    return pipeline, config

def predict(
    image_input,
    mesh_input,
    steps,
    scale,
    octree_res,
    chunk_size,
    seed,
    remove_bg,
    ckpt_path,
    low_vram,
    face_count
):
    # Aggressive memory cleanup at start
    gc.collect()
    torch.cuda.empty_cache()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = "configs/infer_dit_refine.yaml"
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
            
        pipeline, config = get_pipeline_cached(config_path, ckpt_path, device, low_vram)

        token_num = config.model.params.vae_config.params.num_latents
        voxel_res = config.model.params.vae_config.params.voxel_query_res
        
        print(f"Initializing Surface Loader (Token Num: {token_num})...")
        loader = SharpEdgeSurfaceLoader(
            num_sharp_points=204800,
            num_uniform_points=204800,
        )

        print(f"Processing inputs...")
        if image_input is None:
            raise ValueError("Image input is required")
        if mesh_input is None:
            raise ValueError("Mesh input is required")

        # Handle image input
        if isinstance(image_input, dict): 
            # In newer gradio versions Image component might return a dict for mask etc, but usually just PIL/numpy
            # if type='pil' it is PIL.Image
            pass
        
        image = image_input.convert("RGBA")
        
        if remove_bg or image.mode != 'RGBA':
            rembg = BackgroundRemover()
            image = rembg(image)
        
        # Handle mesh input - Gradio Model3D returns path to file
        surface = loader(mesh_input, normalize_scale=scale).to(device, dtype=torch.float16)
        pc = surface[:, :, :3] # [B, N, 3]
        
        # Voxelize
        _, voxel_idx = voxelize_from_point(pc, token_num, resolution=voxel_res)
        
        print("Running diffusion process...")
        gen_device = "cpu" if low_vram else device
        generator = torch.Generator(gen_device).manual_seed(int(seed))
        
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            mesh_out_list, _ = pipeline(
                image=image,
                voxel_cond=voxel_idx,
                generator=generator,
                box_v=1.0,
                mc_level=0.0,
                octree_resolution=int(octree_res),
                num_chunks=int(chunk_size),
                num_inference_steps=int(steps),
                target_face_count=int(face_count),
            )
        
        # Save output
        output_dir = "outputs_gradio"
        os.makedirs(output_dir, exist_ok=True)
        base_name = "output"
        save_path = os.path.join(output_dir, f"{base_name}_refined.glb")
        
        mesh_out = mesh_out_list[0]
        mesh_out.export(save_path)
        print(f"Successfully saved to {save_path}")
        
        return save_path

    finally:
        # Aggressive memory cleanup at end
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UltraShape Gradio App")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to split checkpoint (.pt)")
    parser.add_argument("--share", action="store_true", help="Share the gradio app")
    parser.add_argument("--low_vram", action="store_true", help="Optimize for low VRAM usage (not implemented)")
    
    args = parser.parse_args()
    
    # Define Gradio Interface
    with gr.Blocks(title="UltraShape Inference") as demo:
        gr.Markdown("# UltraShape Inference: Mesh & Image Refinement")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Input Image", image_mode="RGBA")
                mesh_input = gr.Model3D(label="Input Coarse Mesh (.glb, .obj)")
                
                with gr.Accordion("Advanced Parameters", open=True):
                    steps = gr.Slider(minimum=1, maximum=200, value=50, step=1, label="Inference Steps")
                    scale = gr.Slider(minimum=0.1, maximum=2.0, value=0.99, label="Mesh Normalization Scale")
                    octree_res = gr.Slider(minimum=64, maximum=2048, value=1024, step=64, label="Octree Resolution")
                    chunk_size = gr.Slider(minimum=512, maximum=10000, value=2048, step=512, label="Chunk Size (Lower if OOM)")
                    face_count = gr.Slider(minimum=10000, maximum=2000000, value=500000, step=10000, label="Target Face Count")
                    seed = gr.Number(value=42, label="Random Seed")
                    remove_bg = gr.Checkbox(label="Remove Background", value=False)
                
                run_btn = gr.Button("Run Inference", variant="primary")
            
            with gr.Column():
                output_model = gr.Model3D(label="Refined Output Mesh")
        
        # We need to pass args.ckpt to the predict function
        # We can use a lambda or partial, but since args is available in scope...
        # Better to pass it explicitly via a state or partial
        
        run_btn.click(
            fn=lambda img, mesh, s, sc, oct, chk, sd, rm: predict(img, mesh, s, sc, oct, chk, sd, rm, args.ckpt, args.low_vram, face_count),
            inputs=[image_input, mesh_input, steps, scale, octree_res, chunk_size, seed, remove_bg],
            outputs=[output_model]
        )
        
    demo.launch(share=args.share, server_name='0.0.0.0', server_port=7860)
