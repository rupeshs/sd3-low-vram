## Stable Diffusion 3 Low VRAM

I want a quick way to run [SD3 medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium) on my NVIDIA RTX 3060 6GB laptop GPU.
Probably this will work with as 2GB graphics card too.

![SD3 LOW VRAM](https://raw.githubusercontent.com/rupeshs/sd3-low-vram/main/sd3-lowvram.png)
## How to run?
Install Python 3.10 or 3.11
- Download this repo
- Run `install.bat` wait for the installation to  complete
- Run `start.bat` , enjoy!

:exclamation: Model will be downloaded from Hugging Face you need 24GB+ free disk space. 
By default models will be download to system drive (C: drive) you can change it by 
running :

`set HF_HOME=D:/mycache`

## Test Environment

- Windows 10
- System RAM - 16GB
- GPU - NVIDIA RTX 3060 Laptop 
- Python 3.10.10