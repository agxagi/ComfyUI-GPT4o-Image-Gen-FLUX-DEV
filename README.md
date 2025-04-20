# Autoregressive Transformer and Rolling Diffusion Sampler for ComfyUI ( Work in Progress )

A custom ComfyUI node that implements an Autoregressive Transformer and Rolling Diffusion-like Decoder using Black Forest Lab's Flux-Dev model for accurate text-to-image generation, similar to GPT-4o's image generation approach.
![GitBanner](https://github.com/user-attachments/assets/9ed09756-6fdb-4d35-89e7-a821adead9ca)

## Overview

This custom node brings the power of autoregressive generation and rolling diffusion to ComfyUI, enabling high-quality text-to-image generation with the Flux-Dev model. The implementation follows a similar approach to GPT-4o's image generation system, which combines autoregressive transformers with a rolling diffusion-like decoder.

## Features

- **Autoregressive Generation**: Generate images in different orders (top-to-bottom, left-to-right, spiral-out)
- **Rolling Diffusion**: Process the image in overlapping windows for coherent results
- **Flux-Dev Integration**: Leverage Black Forest Lab's Flux-Dev model for high-quality generation
- **Customizable Parameters**: Fine-tune the generation process with various parameters

## Installation

1. Make sure you have ComfyUI installed and working
2. Clone this repository into your ComfyUI custom_nodes directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourusername/comfyui-autoregressive-rolling-diffusion.git
```

3. Install the required dependencies:

```bash
pip install torch diffusers transformers
```

4. Restart ComfyUI

## Usage

After installation, you'll find two new nodes in the ComfyUI interface:

1. **Flux-Dev Autoregressive Rolling Diffusion Sampler**: A standalone node that takes a text prompt and generates an image using the Flux-Dev model
2. **Autoregressive Rolling Diffusion Sampler**: A node that can be integrated with existing ComfyUI workflows, taking a model, conditioning, and latent image as inputs
![Screenshot (885)](https://github.com/user-attachments/assets/8f1146d8-fe0d-4e90-b3d9-5997bb4126b0)

### Flux-Dev Autoregressive Rolling Diffusion Sampler

This node provides a simple interface for text-to-image generation:

- **prompt**: Text description of the image you want to generate
- **negative_prompt**: Text description of what you don't want in the image
- **width**: Width of the generated image (default: 512)
- **height**: Height of the generated image (default: 512)
- **seed**: Random seed for reproducibility
- **steps**: Number of sampling steps (default: 20)
- **cfg**: Classifier-free guidance scale (default: 7.0)
- **autoregressive_order**: Order for autoregressive generation (top-to-bottom, left-to-right, spiral-out)
- **rolling_window_size**: Size of the rolling window (default: 64)
- **rolling_overlap**: Overlap between windows (default: 16)
- **attention_persistence**: How much attention persists between windows (default: 0.5)
- **enable_cpu_offload**: Whether to enable CPU offloading to save VRAM

### Autoregressive Rolling Diffusion Sampler

This node integrates with existing ComfyUI workflows:

- **model**: The model to use for generation
- **positive**: Positive conditioning (prompt)
- **negative**: Negative conditioning (negative prompt)
- **latent_image**: Initial latent representation
- **seed**: Random seed for reproducibility
- **steps**: Number of sampling steps
- **cfg**: Classifier-free guidance scale
- **sampler_name**: Name of the base sampler to use
- **scheduler**: Name of the scheduler to use
- **denoise**: Strength of denoising
- **autoregressive_order**: Order for autoregressive generation
- **rolling_window_size**: Size of the rolling window
- **rolling_overlap**: Overlap between windows
- **attention_persistence**: How much attention persists between windows

## Example Workflows

### Basic Text-to-Image Generation

1. Add a **Flux-Dev Autoregressive Rolling Diffusion Sampler** node to your workflow
2. Enter your prompt and adjust parameters as needed
3. Connect the output to a **VAE Decode** node to convert the latent to an image
4. Connect the **VAE Decode** output to a **Save Image** node

### Advanced Integration

1. Add a **Load Checkpoint** node to load your model
2. Add **CLIP Text Encode** nodes for positive and negative prompts
3. Add an **Empty Latent Image** node to create an initial latent
4. Add an **Autoregressive Rolling Diffusion Sampler** node
5. Connect the model, conditioning, and latent to the sampler
6. Connect the sampler output to a **VAE Decode** node
7. Connect the **VAE Decode** output to a **Save Image** node

## Parameters Explained

### Autoregressive Order

- **top-to-bottom**: Generate the image row by row from top to bottom
- **left-to-right**: Generate the image column by column from left to right
- **spiral-out**: Generate the image starting from the center and spiraling outward

### Rolling Window Size

The size of the window used for processing. Larger windows capture more context but require more memory.

### Rolling Overlap

The amount of overlap between adjacent windows. More overlap results in smoother transitions but increases computation time.

### Attention Persistence

How much attention information persists between generation steps. Higher values maintain more context but may reduce diversity.

## Technical Details

The implementation consists of four main components:

1. **ComfyUI Node Interface**: Handles integration with ComfyUI
2. **Flux-Dev Model Integration**: Manages loading and interfacing with the Flux-Dev model
3. **Autoregressive Transformer**: Implements the autoregressive generation approach
4. **Rolling Diffusion Decoder**: Implements the row-by-row decoding process

The generation process follows these steps:

1. Initialize the model and components
2. Generate tokens in an autoregressive manner using the transformer
3. Process the generated tokens using rolling diffusion
4. Convert the processed latent to an image

## Troubleshooting

### Common Issues

- **Out of Memory**: Try reducing the rolling window size or enabling CPU offloading
- **Slow Generation**: Reduce the number of steps or the rolling overlap
- **Poor Image Quality**: Increase the number of steps or adjust the CFG scale

### Error Messages

- **"CUDA out of memory"**: Your GPU doesn't have enough VRAM. Try enabling CPU offloading or reducing the image size.
- **"Model not found"**: Make sure you have an internet connection for downloading the Flux-Dev model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Blog
https://www.ababiyaworku.com/research/gpt4o-image-gen

## Acknowledgments

- Black Forest Lab for the Flux-Dev model
- OpenAI for the GPT-4o image generation research
- ComfyUI team for the amazing framework
