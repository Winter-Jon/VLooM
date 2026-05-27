from typing import List, Dict, Any, Optional, Union, Type
from dataclasses import dataclass, field
from pathlib import Path
import logging
import aiofiles

from .base import BaseDataset, DatasetItem
from ..utils import yolo_to_qwen_bbox
from ..config import DatasetConfig

logger = logging.getLogger(__name__)

_DEFAULT_LABEL_MAP = {0: "Benign", 1: "Malignant"}


class YoloDataset(BaseDataset):
    def __init__(self, name, description, image_dir: Path, labels_dir: Path, label_map: Optional[Dict[int, str]] = None):
        super().__init__(name, description)
        self.input_dir = Path(image_dir)
        self.labels_dir = Path(labels_dir)
        self.label_map = label_map or _DEFAULT_LABEL_MAP

        self.load_index()

    def load_index(self):
        """扫描目录构建文件列表"""
        if not self.input_dir.exists():
            logger.error(f"输入目录不存在: {self.input_dir}")
            return

        self.items = sorted([p.name for p in self.input_dir.iterdir() if p.is_file()])
        logger.info(f"YoloDataset 加载了 {len(self.items)} 张图片")

    def _parse_yolo_line(self, line: str) -> Optional[Dict]:
        parts = line.strip().split()
        if len(parts) != 5:
            return None

        class_id = int(parts[0])
        label_name = self.label_map.get(class_id, f"class_{class_id}")
        yolo_bbox = [float(x) for x in parts[1:]]
        qwen_bbox = yolo_to_qwen_bbox([class_id] + yolo_bbox)

        return {
            "label": label_name,
            "class_id": class_id,
            "yolo_bbox": yolo_bbox,
            "qwen_bbox": qwen_bbox
        }

    async def _read_label_file(self, path: Path) -> Optional[Dict]:
        try:
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                line = await f.readline()
                return self._parse_yolo_line(line)
        except Exception:
            return None

    async def get_item(self, index: int) -> DatasetItem:
        filename = self.items[index]

        img_file = self.input_dir / filename
        stem = img_file.stem

        template_vars = {"img_name": stem}

        label_path = self.labels_dir / f"{stem}.txt"
        if label_path.exists():
            if label_data := await self._read_label_file(label_path):
                template_vars["gt_label"] = label_data['label']
                template_vars["gt_bbox"] = label_data['qwen_bbox']
            else:
                raise ValueError("标签格式错误")
        else:
            raise ValueError(f"无法读取标签文件: {label_path}")

        return DatasetItem(index, stem, str(img_file), template_vars)


@DatasetConfig.register_subclass("yolo")
@dataclass
class YoloConfig(DatasetConfig):
    image_dir: Path = field(default=Path(""), metadata={"help": "图像目录"})
    labels_dir: Optional[Path] = field(default=None, metadata={"help": "YOLO 格式的 txt 标签目录"})
    label_map: Optional[Dict[int, str]] = field(default=None, metadata={"help": "class_id -> label_name 映射，默认 {0: Benign, 1: Malignant}"})

    def dataset_class(self) -> Type["YoloDataset"]:
        return YoloDataset
