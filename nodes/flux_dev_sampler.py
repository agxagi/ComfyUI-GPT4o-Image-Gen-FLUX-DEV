import os
import torch
from typing import Dict, Any, Optional, Union

from .autoregressive_transformer import AutoregressiveTransformer
from .rolling_diffusion_decoder import RollingDiffusionDecoder
from .flux_dev_integration import FluxDevModelIntegration

class FluxDevAutoRegressiveRollingDiffusionSampler:
    """
    ComfyUI custom node that implements an Autoregressive Transformer and Rolling Diffusion-like Decoder
    using Black Forest Lab's Flux-Dev model for accurate text-to-image generation.
    
    This node combines autoregressive generation with rolling diffusion to create high-quality images
    in a manner similar to GPT-4o's image generation approach, specifically using the Flux-Dev model.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define the input parameters for the node."""
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape with mountains and a lake"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, distorted"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "autoregressive_order": (["top-to-bottom", "left-to-right", "spiral-out"], {"default": "top-to-bottom"}),
                "rolling_window_size": ("INT", {"default": 64, "min": 16, "max": 512, "step": 16}),
                "rolling_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8}),
                "attention_persistence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "enable_cpu_offload": (["True", "False"], {"default": "True"}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"
    
    def sample(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        seed: int,
        steps: int,
        cfg: float,
        autoregressive_order: str,
        rolling_window_size: int,
        rolling_overlap: int,
        attention_persistence: float,
        enable_cpu_offload: str
    ):
        """
        Main entry point for the sampler node.
        
        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative text prompt for image generation
            width: Width of the generated image
            height: Height of the generated image
            seed: Random seed for reproducibility
            steps: Number of sampling steps
            cfg: Classifier-free guidance scale
            autoregressive_order: Order for autoregressive generation
            rolling_window_size: Size of the rolling window
            rolling_overlap: Overlap between windows
            attention_persistence: How much attention persists between windows
            enable_cpu_offload: Whether to enable CPU offloading to save VRAM
            
        Returns:
            Generated latent representation and image
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        
        # Get device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Parse boolean from string
        enable_cpu_offload_bool = (enable_cpu_offload == "True")
        
        # Initialize Flux-Dev model integration
        print("Initializing Flux-Dev model integration")
        flux_integration = FluxDevModelIntegration(
            device=device,
            enable_cpu_offload=enable_cpu_offload_bool
        )
        
        # Load the model
        flux_integration.load_model()
        
        # Initialize components
        print("Initializing autoregressive transformer and rolling diffusion decoder")
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
        
        # Calculate latent dimensions (typically 1/8 of the image dimensions for most models)
        latent_height = height // 8
        latent_width = width // 8
        
        # Create initial latent (random noise)
        print(f"Creating initial latent with dimensions: {latent_height}x{latent_width}")
        latent = torch.randn(
            (1, 4, latent_height, latent_width),
            device=device
        )
        
        # Process prompts
        print("Processing prompts")
        positive_conditioning = flux_integration.process_prompt(prompt)
        negative_conditioning = flux_integration.process_prompt(negative_prompt)
        
        # Create conditioning dictionary
        conditioning = {
            "positive": positive_conditioning,
            "negative": negative_conditioning,
            "cfg": cfg
        }
        
        # Step 1: Generate sequence using autoregressive transformer
        print(f"Starting autoregressive generation with order: {autoregressive_order}")
        transformed_latent = autoregressive_transformer.generate_sequence(
            model=flux_integration.model,
            latent=latent,
            conditioning=conditioning,
            num_steps=steps // 2,  # Split steps between the two processes
            callback=lambda progress: print(f"Autoregressive progress: {progress:.2f}")
        )
        
        # Step 2: Process the latent using rolling diffusion
        print(f"Starting rolling diffusion with window size: {rolling_window_size}, overlap: {rolling_overlap}")
        final_latent = rolling_diffusion_decoder.process_latent(
            model=flux_integration.model,
            latent=transformed_latent,
            conditioning=conditioning,
            steps=steps // 2,  # Split steps between the two processes
            callback=lambda progress: print(f"Rolling diffusion progress: {progress:.2f}")
        )
        
        # Convert latent to image
        print("Converting latent to image")
        image = flux_integration.latent_to_image(final_latent)
        
        print("Sampling completed")
        return (final_latent, image)

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxDevAutoRegressiveRollingDiffusionSampler": FluxDevAutoRegressiveRollingDiffusionSampler
}

# Node display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxDevAutoRegressiveRollingDiffusionSampler": "Flux-Dev Autoregressive Rolling Diffusion Sampler"
}
