
from typing import List, Dict, Any, Union, Optional, Type
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from .base import BaseDataset, DatasetItem
from ..utils import coco_to_qwen_bbox
from ..config import DatasetConfig

logger = logging.getLogger(__name__)

class CocoDataset(BaseDataset):
    def __init__(self, name, description, data_dir: Path, img_subdir: str = "images", json_file: str = "train.json"):
        super().__init__(name, description)
        self.data_dir = Path(data_dir)
        
        self.img_dir = self.data_dir / img_subdir
        self.json_path = self.data_dir / json_file
        
        self.img_info_map: Dict[int, Dict] = {}
        self.img_to_anns: Dict[int, List[Dict]] = {}
        self.categories: Dict[int, str] = {}
        
        self.load_index()

    def load_index(self):
        """加载 COCO 格式 JSON"""
        if not self.json_path.exists():
            raise FileNotFoundError(f"JSON 标注文件不存在: {self.json_path}")

        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.categories = {c['id']: c['name'] for c in data.get('categories', [])}
            self.img_info_map = {img['id']: img for img in data.get('images', [])}
            
            for ann in data.get('annotations', []):
                img_id = ann['image_id']
                if img_id in self.img_info_map:
                    if img_id not in self.img_to_anns:
                        self.img_to_anns[img_id] = []
                    self.img_to_anns[img_id].append(ann)

            self.items = sorted(list(self.img_to_anns.keys()))
            logger.info(f"KidneyDataset 加载了 {len(self.items)} 个有效样本")
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")

    async def get_item(self, index: int) -> DatasetItem:
        img_id = self.items[index]
        img_info = self.img_info_map[img_id]
        anns = self.img_to_anns[img_id]
        
        img_name = img_info['file_name']
        img_path = self.img_dir / img_name
        
        # 取第一个标注 (如需多标注需修改此处逻辑)
        main_ann = anns[0]
        
        w, h = img_info['width'], img_info['height']
        
        # 处理 BBox
        coco_box = main_ann.get('bbox', [])
        qwen_bbox = coco_to_qwen_bbox(coco_box, w, h) if coco_box else []
        
        label_name = self.categories.get(main_ann['category_id'], "Unknown")

        template_vars = {
            "img_name": Path(img_name).stem,
            "label_id": main_ann['category_id'],
            "gt_label": label_name,
            "gt_bbox": qwen_bbox,
            "width": w,
            "height": h
        }

        return DatasetItem(
            idx=index,
            img_name=img_name,
            img_path=str(img_path),
            template_vars=template_vars
        )
        
@DatasetConfig.register_subclass("coco")
@dataclass
class CocoConfig(DatasetConfig):
    data_dir: Path = Path("")
    json_file: str = "train.json"
    img_subdir: str = "images"
    
    def dataset_class(self) -> Type["CocoDataset"]:
        return CocoDataset