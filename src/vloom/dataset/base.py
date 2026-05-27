import os
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass

from ..config import DatasetConfig

logger = logging.getLogger(__name__)

def create_dataset(config: 'DatasetConfig') -> 'BaseDataset':
    """根据配置创建数据集实例"""
    dataset_class = config.dataset_class()
    if isinstance(config, DatasetConfig):
        dataset = dataset_class(**config.__dict__)
    else:
        raise ValueError(f"未知的数据集配置类型: {type(config)}")
    return dataset


@dataclass
class DatasetItem:
    """标准化的数据集返回对象"""
    idx: int
    img_name: str
    img_path: Path
    template_vars: Dict[str, Any]  # 直接用于 Jinja2 渲染的参数
    valid: bool = True

class BaseDataset(ABC):
    """数据集抽象基类"""
    
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.items: list = []  # 存储样本标识符列表

    @abstractmethod
    def load_index(self):
        pass

    @abstractmethod
    async def get_item(self, index: int) -> DatasetItem:
        pass

    def __len__(self):
        return len(self.items)

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Return the configuration schema for this dataset.
        Usually dataset config matches the fields of the dataset subclass.
        """
        return {}
