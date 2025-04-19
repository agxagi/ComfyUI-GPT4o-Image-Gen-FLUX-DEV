# ComfyUI Autoregressive Transformer and Rolling Diffusion Sampler

This package contains a custom ComfyUI node that implements an Autoregressive Transformer and Rolling Diffusion-like Decoder using Black Forest Lab's Flux-Dev model for accurate text-to-image generation, similar to GPT-4o's image generation approach.

## Contents

- `nodes/`: Python modules implementing the custom node components
  - `autoregressive_transformer.py`: Implementation of the autoregressive transformer
  - `rolling_diffusion_decoder.py`: Implementation of the rolling diffusion decoder
  - `flux_dev_integration.py`: Integration with the Flux-Dev model
  - `flux_dev_sampler.py`: Standalone sampler node using Flux-Dev model
  - `autoregressive_rolling_diffusion_sampler.py`: Generic sampler node for use with any model
- `__init__.py`: Package initialization and node registration
- `README.md`: Documentation and usage instructions
- `install.sh`: Installation script
- `test_custom_node.py`: Unit tests for the implementation

## Installation

Run the included installation script:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-autoregressive-rolling-diffusion.git
cd comfyui-autoregressive-rolling-diffusion
bash install.sh
```

## Usage

See README.md for detailed usage instructions and examples.

## License

MIT License
