from typing import Any, List, Dict, Union, Tuple
import logging
import tempfile
import uuid
from pathlib import Path
from PIL import Image, ImageDraw
from .base import BaseTool
from .registry import ToolClassRegistry
import asyncio
import functools

logger = logging.getLogger(__name__)


def _require_pycocotools():
    try:
        from pycocotools.coco import COCO  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "pycocotools is required for COCO-based tools. "
            "Install it with: pip install vloom[coco]"
        )


def draw_bboxes_on_image(img_path: Path, bboxes: List[List[float]], output_dir: Path = None) -> Path:
    """
    Draw bounding boxes on an image and save to a temporary file.
    """
    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    output_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        color = colors[i % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        draw.text((x, y - 10), f"Box {i+1}", fill=color)

    out_name = f"vis_{uuid.uuid4().hex[:8]}.jpg"
    out_path = output_dir / out_name
    img.save(out_path)
    return out_path


@ToolClassRegistry.register_tool("vision_tool")
class VisionTool(BaseTool):
    """
    Vision Tool that returns bounding boxes for an image from COCO annotations.
    Can be configured with GT (train.json) or Vertical Model predictions (predict_all.json).
    """
    name: str = "vision_tool"
    description: str = "Get bounding boxes for objects in an image. Returns bboxes and a visualization."

    def __init__(self, annotation_path: Path, image_dir: Path = None, vis_output_dir: Path = None):
        _require_pycocotools()
        from pycocotools.coco import COCO

        self.annotation_path = Path(annotation_path)
        if not self.annotation_path.exists():
            raise ValueError(f"COCO annotation file not found: {annotation_path}")

        self.image_dir = Path(image_dir) if image_dir else self.annotation_path.parent / "images_final"
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None

        logger.info(f"Loading COCO annotations from {self.annotation_path}...")
        self.coco = COCO(str(self.annotation_path))
        self._filename_to_id = {img['file_name']: img['id'] for img in self.coco.imgs.values()}
        logger.info(f"COCO annotations loaded. {len(self._filename_to_id)} images indexed.")

    async def execute(self, image_id: int = None, image_name: str = None, **kwargs) -> Dict[str, Any]:
        if image_id is None and image_name:
            image_id = self._filename_to_id.get(image_name)
            if image_id is None:
                return {"error": f"Image name '{image_name}' not found in annotations."}

        if image_id is None:
            return {"error": "Must provide either image_id or image_name."}

        try:
            img_id = int(image_id)
        except ValueError:
            return {"error": f"Invalid image_id {image_id}"}

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = [ann['bbox'] for ann in anns]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info['file_name']

        if not img_path.exists():
            return {"bboxes": bboxes, "error": f"Image file not found: {img_path}"}

        loop = asyncio.get_running_loop()
        vis_path = await loop.run_in_executor(None, draw_bboxes_on_image, img_path, bboxes, self.vis_output_dir)

        return {
            "bboxes": bboxes,
            "visualization": str(vis_path)
        }

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_id": {
                            "type": "integer",
                            "description": "The ID of the image to analyze."
                        },
                        "image_name": {
                            "type": "string",
                            "description": "The filename of the image to analyze (alternative to image_id)."
                        }
                    }
                }
            }
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "annotation_path": {"type": "str", "help": "Path to the COCO JSON annotation file.", "required": True},
            "image_dir": {"type": "str", "help": "Path to the image directory.", "required": False},
            "vis_output_dir": {"type": "str", "help": "Path to save visualization images.", "required": False}
        }


@ToolClassRegistry.register_tool("proposal_tool")
class ProposalTool(BaseTool):
    """
    Proposal Tool that returns candidate bounding boxes from a vertical model (predict_all.json).
    Essentially the same as VisionTool but semantically for "proposals" rather than GT.
    """
    name: str = "proposal_tool"
    description: str = "Get candidate bounding boxes from a vertical detection model. Returns bboxes and visualization."

    def __init__(self, annotation_path: Path, image_dir: Path = None, vis_output_dir: Path = None):
        _require_pycocotools()
        from pycocotools.coco import COCO

        self.annotation_path = Path(annotation_path)
        if not self.annotation_path.exists():
            raise ValueError(f"Proposal annotation file not found: {annotation_path}")

        self.image_dir = Path(image_dir) if image_dir else self.annotation_path.parent / "images_final"
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None

        logger.info(f"Loading proposal annotations from {self.annotation_path}...")
        self.coco = COCO(str(self.annotation_path))
        self._filename_to_id = {img['file_name']: img['id'] for img in self.coco.imgs.values()}
        logger.info(f"Proposal annotations loaded. {len(self._filename_to_id)} images indexed.")

    async def execute(self, image_id: int = None, image_name: str = None, **kwargs) -> Dict[str, Any]:
        if image_id is None and image_name:
            image_id = self._filename_to_id.get(image_name)
            if image_id is None:
                return {"error": f"Image name '{image_name}' not found."}

        if image_id is None:
            return {"error": "Must provide either image_id or image_name."}

        try:
            img_id = int(image_id)
        except ValueError:
            return {"error": f"Invalid image_id {image_id}"}

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = [ann['bbox'] for ann in anns]
        categories = [ann.get('category_id') for ann in anns]

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info['file_name']

        vis_path = None
        if img_path.exists():
            loop = asyncio.get_running_loop()
            vis_path = await loop.run_in_executor(None, draw_bboxes_on_image, img_path, bboxes, self.vis_output_dir)

        return {
            "bboxes": bboxes,
            "categories": categories,
            "visualization": str(vis_path) if vis_path else None
        }

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_id": {"type": "integer", "description": "The ID of the image."},
                        "image_name": {"type": "string", "description": "The filename of the image."}
                    }
                }
            }
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "annotation_path": {"type": "str", "help": "Path to the proposal JSON file.", "required": True},
            "image_dir": {"type": "str", "help": "Path to the image directory.", "required": False},
            "vis_output_dir": {"type": "str", "help": "Path to save visualization images.", "required": False}
        }


@ToolClassRegistry.register_tool("visualizer_tool")
class VisualizerTool(BaseTool):
    """
    Visualizer Tool that draws user-provided bounding boxes on an image.
    Used for self-correction tasks where the model provides its own predictions.
    """
    name: str = "visualizer_tool"
    description: str = "Draw bounding boxes on an image. Provide the image name and bboxes to visualize."

    def __init__(self, image_dir: Path, vis_output_dir: Path = None):
        self.image_dir = Path(image_dir)
        self.vis_output_dir = Path(vis_output_dir) if vis_output_dir else None

        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {image_dir}")
        logger.info(f"VisualizerTool initialized. Image dir: {self.image_dir}")

    async def execute(self, image_name: str, bboxes: List[List[float]], **kwargs) -> Dict[str, Any]:
        img_path = self.image_dir / image_name
        if not img_path.exists():
            return {"error": f"Image not found: {img_path}"}

        loop = asyncio.get_running_loop()
        vis_path = await loop.run_in_executor(None, draw_bboxes_on_image, img_path, bboxes, self.vis_output_dir)
        return {"visualization": str(vis_path)}

    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_name": {"type": "string", "description": "The filename of the image."},
                        "bboxes": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                            "description": "List of [x, y, w, h] bounding boxes to draw."
                        }
                    },
                    "required": ["image_name", "bboxes"]
                }
            }
        }

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "image_dir": {"type": "str", "help": "Path to the image directory.", "required": True},
            "vis_output_dir": {"type": "str", "help": "Path to save visualization images.", "required": False}
        }
