import torch
import numpy as np
import comfy.samplers
from typing import List, Tuple, Dict, Any, Optional, Union

from .autoregressive_transformer import AutoregressiveTransformer
from .rolling_diffusion_decoder import RollingDiffusionDecoder

class AutoregressiveRollingDiffusionSampler:
    """
    ComfyUI custom node that implements an Autoregressive Transformer and Rolling Diffusion-like Decoder
    using Black Forest Lab's Flux-Dev model for accurate text-to-image generation.
    
    This node combines autoregressive generation with rolling diffusion to create high-quality images
    in a manner similar to GPT-4o's image generation approach.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node."""
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "autoregressive_order": (["top-to-bottom", "left-to-right", "spiral-out"], {"default": "top-to-bottom"}),
                "rolling_window_size": ("INT", {"default": 64, "min": 16, "max": 512, "step": 16}),
                "rolling_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8}),
                "attention_persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"
    
    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        autoregressive_order,
        rolling_window_size,
        rolling_overlap,
        attention_persistence
    ):
        """
        Main entry point for the sampler node.
        
        Args:
            model: The model to use for generation
            positive: Positive conditioning (prompt)
            negative: Negative conditioning (negative prompt)
            latent_image: Initial latent representation
            seed: Random seed for reproducibility
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            sampler_name: Name of the base sampler to use
            scheduler: Name of the scheduler to use
            denoise: Strength of denoising
            autoregressive_order: Order for autoregressive generation
            rolling_window_size: Size of the rolling window
            rolling_overlap: Overlap between windows
            attention_persistence: How much attention persists between windows
            
        Returns:
            Generated latent representation
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Get device
        device = model.device
        
        # Initialize components
        autoregressive_transformer = AutoregressiveTransformer(
            order=autoregressive_order,
            attention_persistence=attention_persistence,
            device=device
        )
        
        rolling_diffusion_decoder = RollingDiffusionDecoder(
            window_size=rolling_window_size,
            overlap=rolling_overlap,
            device=device
        )
        
        # Create conditioning dictionary
        conditioning = {
            "positive": positive,
            "negative": negative,
            "cfg": cfg
        }
        
        # Step 1: Generate sequence using autoregressive transformer
        print(f"Starting autoregressive generation with order: {autoregressive_order}")
        transformed_latent = autoregressive_transformer.generate_sequence(
            model=model,
            latent=latent_image,
            conditioning=conditioning,
            num_steps=steps // 2,  # Split steps between the two processes
            callback=lambda progress: print(f"Autoregressive progress: {progress:.2f}")
        )
        
        # Step 2: Process the latent using rolling diffusion
        print(f"Starting rolling diffusion with window size: {rolling_window_size}, overlap: {rolling_overlap}")
        final_latent = rolling_diffusion_decoder.process_latent(
            model=model,
            latent=transformed_latent,
            conditioning=conditioning,
            steps=steps // 2,  # Split steps between the two processes
            callback=lambda progress: print(f"Rolling diffusion progress: {progress:.2f}")
        )
        
        print("Sampling completed")
        return (final_latent,)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AutoregressiveRollingDiffusionSampler": AutoregressiveRollingDiffusionSampler
}

# Node display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoregressiveRollingDiffusionSampler": "Autoregressive Rolling Diffusion Sampler"
}
