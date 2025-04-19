import os
import sys
from typing import Dict, Any, List

# Import all node modules
from .nodes.autoregressive_transformer import AutoregressiveTransformer
from .nodes.rolling_diffusion_decoder import RollingDiffusionDecoder
from .nodes.flux_dev_integration import FluxDevModelIntegration
from .nodes.flux_dev_sampler import FluxDevAutoRegressiveRollingDiffusionSampler
from .nodes.autoregressive_rolling_diffusion_sampler import AutoregressiveRollingDiffusionSampler

# Collect all node class mappings
NODE_CLASS_MAPPINGS = {
    "FluxDevAutoRegressiveRollingDiffusionSampler": FluxDevAutoRegressiveRollingDiffusionSampler,
    "AutoregressiveRollingDiffusionSampler": AutoregressiveRollingDiffusionSampler
}

# Collect all node display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxDevAutoRegressiveRollingDiffusionSampler": "Flux-Dev Autoregressive Rolling Diffusion Sampler",
    "AutoregressiveRollingDiffusionSampler": "Autoregressive Rolling Diffusion Sampler"
}

print(f"Initialized Autoregressive Transformer and Rolling Diffusion Sampler custom nodes")
print(f"Available nodes: {', '.join(NODE_CLASS_MAPPINGS.keys())}")
