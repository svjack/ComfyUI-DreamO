```bash
comfy launch -- --listen 0.0.0.0

wget https://huggingface.co/ByteDance/DreamO/resolve/main/dreamo.safetensors

wget https://huggingface.co/ByteDance/DreamO/resolve/main/dreamo_cfg_distill.safetensors

cp dreamo*.safetensors ComfyUI/models/loras

huggingface-cli login

in file ComfyUI-DreamO.nodes.comfy_nodes.py
#login(token=hf_token)
or 
past in DreanO Load Model node
```

# ComfyUI-DreamO

https://github.com/bytedance/DreamO
ComfyUI Warpper

Note:
This only a warpper. if you need full native comfyui impl. you can find other.


open offload can run on low vram, no open offload need 40GB.

Flux model auto download to models/diffusers

lora need to download to models/lora

lora:
https://huggingface.co/ByteDance/DreamO/tree/main



IP reference (subject)

![show](./assets/show_1.png)


style reference 

note: prompt need add "generate a same style image."

![show](./assets/show_2.png)


ID (face) + IP  reference

![show](./assets/show_3.png)


## online run:

https://www.comfyonline.app/machine_manager (DreamO Template)

