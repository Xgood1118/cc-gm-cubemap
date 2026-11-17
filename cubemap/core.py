# -*- coding: utf-8 -*-
"""立方体贴图接缝与过滤"""

from dataclasses import dataclass


@dataclass
class Cube:
    """六面数据。"""
    faces: dict


def face_uv(direction):
    """方向转面与 UV"""
    raise NotImplementedError("face_uv 尚未实现")


def sample(cube, direction):
    """采样"""
    raise NotImplementedError("sample 尚未实现")


def seam_error(cube, direction):
    """算接缝误差"""
    raise NotImplementedError("seam_error 尚未实现")

