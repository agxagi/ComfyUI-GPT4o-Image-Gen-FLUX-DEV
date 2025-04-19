@echo off
echo Installing ComfyUI Autoregressive Transformer and Rolling Diffusion Sampler...

:: Check if running from ComfyUI custom_nodes directory
if not exist "..\ComfyUI" (
    if not exist "..\..\ComfyUI" (
        echo Warning: This script should be run from within the ComfyUI\custom_nodes directory.
        echo Installation may not work correctly otherwise.
        set /p CONTINUE="Continue anyway? (y/n) "
        if /i not "%CONTINUE%"=="y" (
            echo Installation cancelled.
            exit /b 1
        )
    )
)

:: Install dependencies
echo Installing dependencies...
pip install torch diffusers transformers

:: Create directories if they don't exist
if not exist "web" mkdir web

echo Installation complete!
echo Please restart ComfyUI to use the new nodes.
echo.
echo Available nodes:
echo - Flux-Dev Autoregressive Rolling Diffusion Sampler
echo - Autoregressive Rolling Diffusion Sampler
echo.
echo See README.md for usage instructions.

pause
