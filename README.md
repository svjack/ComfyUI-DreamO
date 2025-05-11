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

### vim run_xiang_card.py
```python
import os
import time
import subprocess
from pathlib import Path

# Configuration
SEED = 661695664686456
IMAGE_PATH = 'xiang_image.jpg'
STYLE_IMAGE_PATH = '31UR1xM3MWL._AC_UF894,1000_QL80_.jpg'
OUTPUT_DIR = 'ComfyUI/output'
PYTHON_PATH = '/environment/miniconda3/bin/python'

def get_latest_output_count():
    """Return the number of PNG files in the output directory"""
    try:
        return len(list(Path(OUTPUT_DIR).glob('*.png')))
    except:
        return 0

def wait_for_new_output(initial_count):
    """Wait until a new PNG file appears in the output directory"""
    timeout = 60  # seconds
    start_time = time.time()

    while time.time() - start_time < timeout:
        current_count = get_latest_output_count()
        if current_count > initial_count:
            time.sleep(1)  # additional 1 second delay
            return True
        time.sleep(0.5)
    return False

def generate_script(seed):
    """Generate the ComfyUI script with the given seed"""
    script_content = f"""from comfy_script.runtime import *
load()
from comfy_script.runtime.nodes import *
with Workflow():
    dreamo_pipe = DreamOLoadModel('', True, 'dreamo.safetensors', 'dreamo_cfg_distill.safetensors', 'None')
    bg_rm_model = BgRmModelLoad()
    face_helper = FaceModelLoad()
    image, _ = LoadImage('{IMAGE_PATH}')
    image2, _ = LoadImage('{STYLE_IMAGE_PATH}')
    image3 = DreamOGenerate(dreamo_pipe, bg_rm_model, face_helper, image,
        'id', 'a man hold a blank wooden sign', 1024, 1024, 30, 10, {seed}, image2, 'ip')
    SaveImage(image3, 'ComfyUI')
"""
    return script_content

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Main generation loop
    while True:
        # Generate script with current seed
        script = generate_script(SEED)

        # Write script to file
        with open('run_dreamo_generation.py', 'w') as f:
            f.write(script)

        # Get current output count before running
        initial_count = get_latest_output_count()

        # Run the script
        print(f"Generating image with seed: {SEED}")
        subprocess.run([PYTHON_PATH, 'run_dreamo_generation.py'])

        # Wait for new output
        if not wait_for_new_output(initial_count):
            print("Timeout waiting for new output. Continuing to next generation.")

        # Increment seed for next generation
        SEED -= 1

if __name__ == "__main__":
    main()

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

