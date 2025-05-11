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

# Score it 
```bash
mkdir Xiang_Card_DreamO_Images

cp ComfyUI/output/*.png Xiang_Card_DreamO_Images

huggingface-cli upload svjack/Xiang_Card_DreamO_Images Xiang_Card_DreamO_Images --repo-type dataset
```

### Colab
```python
!git clone https://huggingface.co/spaces/svjack/Face-Similarity

import os
os.chdir("Face-Similarity")

!pip install -r requirements.txt
!pip install datasets

from app import *

!huggingface-cli login

!huggingface-cli download svjack/Xiang_Card_DreamO_Images --local-dir Xiang_Card_DreamO_Images --repo-type dataset

!ls Xiang_Card_DreamO_Images


from gradio_client import Client, handle_file

client = Client("http://localhost:7860")
result = client.predict(
		image1=handle_file('Xiang_Card_DreamO_Images/ComfyUI_00001_.png'),
		image2=handle_file('xiang_image.jpg'),
		api_name="/predict"
)
print(result)


import os
from gradio_client import Client, handle_file
from datasets import Dataset
import pandas as pd
from datasets import Dataset, Image as HFImage
from PIL import Image
from tqdm import tqdm

# 初始化Gradio客户端
client = Client("http://localhost:7860")

# 遍历文件夹并处理PNG文件
def process_folder(folder_path, reference_image):
    results = []

    # 遍历文件夹中的所有PNG文件[3](@ref)
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)

            try:
                # 调用API计算相似度[1](@ref)
                result = client.predict(
                    image1=handle_file(file_path),
                    image2=handle_file(reference_image),
                    api_name="/predict"
                )

                # 提取相似度分数
                #score = float(result.replace("图片相似度:", "").strip())
                score = float(result[1].replace("🔍 Distance Score:", "").strip())

                # 将结果添加到列表中
                results.append({
                    "image": file_path,
                    "score": score
                })

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")

    return results

# 主处理流程
def main():
    # 设置路径
    folder_path = "Xiang_Card_DreamO_Images"
    reference_image = "xiang_image.jpg"

    # 处理文件夹
    results = process_folder(folder_path, reference_image)

    # 创建DataFrame并按分数降序排序
    df = pd.DataFrame(results)
    df = df.sort_values(by="score", ascending=True)

    # 创建HuggingFace数据集[5,7](@ref)
    dataset = Dataset.from_pandas(df)

    # 将image列转换为图片类型[7,9](@ref)
    def load_image(example):
        example["image"] = Image.open(example["image"])
        return example

    dataset = dataset.map(load_image)
    dataset = dataset.cast_column("image", HFImage())

    # 保存数据集
    dataset.save_to_disk("similarity_dataset")
    print("数据集已创建并保存为 similarity_dataset")

    return dataset

if __name__ == "__main__":
    dataset = main()

from datasets import load_from_disk
load_from_disk("similarity_dataset/").remove_columns(["__index_level_0__"])

load_from_disk("similarity_dataset/").remove_columns(["__index_level_0__"]).push_to_hub("svjack/Xiang_Card_DreamO_Images_Scores")
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

