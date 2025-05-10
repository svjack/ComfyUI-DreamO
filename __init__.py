


# 注册节点
from .nodes.comfy_nodes import DreamOLoadModel, DreamOGenerate, BgRmModelLoad, FaceModelLoad


NODE_CLASS_MAPPINGS = {
    "DreamOLoadModel": DreamOLoadModel,
    "DreamOGenerate": DreamOGenerate,
    "BgRmModelLoad": BgRmModelLoad,
    "FaceModelLoad": FaceModelLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamOLoadModel": "DreamO Load Model",
    "DreamOGenerate": "DreamO Generate",
    "BgRmModelLoad": "BgRmModelLoad",
    "FaceModelLoad": "FaceModelLoad",
} 
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]