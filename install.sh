#!/bin/bash

# Installation script for ComfyUI Autoregressive Transformer and Rolling Diffusion Sampler

echo "Installing ComfyUI Autoregressive Transformer and Rolling Diffusion Sampler..."

# Check if running from ComfyUI custom_nodes directory
if [[ ! -d "../ComfyUI" && ! -d "../../ComfyUI" ]]; then
    echo "Warning: This script should be run from within the ComfyUI/custom_nodes directory."
    echo "Installation may not work correctly otherwise."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

# Install dependencies
echo "Installing dependencies..."
pip install torch diffusers transformers

# Create directories if they don't exist
mkdir -p web

echo "Installation complete!"
echo "Please restart ComfyUI to use the new nodes."
echo
echo "Available nodes:"
echo "- Flux-Dev Autoregressive Rolling Diffusion Sampler"
echo "- Autoregressive Rolling Diffusion Sampler"
echo
echo "See README.md for usage instructions."
