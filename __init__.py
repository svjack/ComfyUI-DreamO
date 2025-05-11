


# 注册节点
from .nodes.comfy_nodes import DreamOLoadModel, DreamOGenerate, BgRmModelLoad, FaceModelLoad, DreamOLoadModelFromLocal


NODE_CLASS_MAPPINGS = {
    "DreamOLoadModel": DreamOLoadModel,
    "DreamOGenerate": DreamOGenerate,
    "BgRmModelLoad": BgRmModelLoad,
    "FaceModelLoad": FaceModelLoad,
    "DreamOLoadModelFromLocal": DreamOLoadModelFromLocal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamOLoadModel": "DreamO Load Model",
    "DreamOGenerate": "DreamO Generate",
    "BgRmModelLoad": "BgRmModelLoad",
    "FaceModelLoad": "FaceModelLoad",
    "DreamOLoadModelFromLocal": "DreamO Load Model From Local",
} 
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]