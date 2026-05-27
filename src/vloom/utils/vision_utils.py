import os
import logging
import random
from typing import Dict, Any, List


logger = logging.getLogger(__name__)

def generate_random_bbox(image_size: int = 1000, min_size: int = 50, max_size: int = 300) -> List[int]:
    """生成随机bbox"""
    size = random.randint(min_size, max_size)
    x1 = random.randint(0, image_size - size)
    y1 = random.randint(0, image_size - size)
    x2 = x1 + size
    y2 = y1 + size
    return [x1, y1, x2, y2]


def generate_noisy_bbox(bbox: List[int], image_size: int = 1000, noise_level: float = 0.1) -> List[int]:
    """生成带有噪声的bbox"""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # 随机偏移
    dx = int(w * noise_level * (random.random() - 0.5) * 2)
    dy = int(h * noise_level * (random.random() - 0.5) * 2)
    
    # 随机缩放
    dw = int(w * noise_level * (random.random() - 0.5) * 2)
    dh = int(h * noise_level * (random.random() - 0.5) * 2)
    
    nx1 = max(0, x1 + dx)
    ny1 = max(0, y1 + dy)
    nx2 = min(image_size, x2 + dx + dw)
    ny2 = min(image_size, y2 + dy + dh)
    
    return [nx1, ny1, nx2, ny2]

def normalize_bbox(bbox: List[float], width: int, height: int) -> List[float]:
    """将绝对坐标转换为 0-1 的归一化坐标"""
    x, y, w, h = bbox
    return [x / width, y / height, w / width, h / height]

def yolo_to_qwen_bbox(yolo_bbox: List[float], image_size: int = 1000) -> List[int]:
    """将YOLO格式bbox (归一化中心点+宽高) 转换为Qwen格式 (绝对像素坐标 [x1, y1, x2, y2])
    
    Args:
        yolo_bbox: [class_id, x_center, y_center, width, height] 或 [x_center, y_center, width, height]
        image_size: Qwen 这里的坐标通常归一化到 1000x1000
    """
    # 处理可能包含 class_id 的情况
    if len(yolo_bbox) == 5:
        _, x_center, y_center, width, height = yolo_bbox
    else:
        x_center, y_center, width, height = yolo_bbox
    
    # 转换为目标尺寸的像素坐标
    x_center_px = x_center * image_size
    y_center_px = y_center * image_size
    width_px = width * image_size
    height_px = height * image_size
    
    # 计算左上角和右下角坐标
    x1 = int(x_center_px - width_px / 2)
    y1 = int(y_center_px - height_px / 2)
    x2 = int(x_center_px + width_px / 2)
    y2 = int(y_center_px + height_px / 2)
    
    # 确保坐标在 0-1000 范围内
    return [
        max(0, min(image_size, x1)),
        max(0, min(image_size, y1)),
        max(0, min(image_size, x2)),
        max(0, min(image_size, y2))
    ]

def coco_to_qwen_bbox(coco_bbox: List[float], img_w: int, img_h: int, image_size: int = 1000) -> List[int]:
    """将COCO格式 (x_min, y_min, w, h, 绝对像素) 转换为 Qwen 格式 (归一化到1000的 x1,y1,x2,y2)"""
    x_min, y_min, w, h = coco_bbox
    
    # 先归一化
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    # 再利用现有函数转为 Qwen 格式
    return yolo_to_qwen_bbox([x_center, y_center, w_norm, h_norm], image_size)
