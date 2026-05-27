from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import os
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..core.llm_client import LLMClient
from ..config import TaskConfig
from ..memory.conversation import Conversation
from ..utils.lifecycle_logger import LifeCycleLogger

class BaseAgent(ABC):
    """Agent基类，定义标准接口"""
    def __init__(self, config: TaskConfig, template_root: Path, llm_client: LLMClient, log_interval: int = 10):
        self.config = config
        self.template_root = Path(template_root) if template_root else None
        self.llm = llm_client
        self.jinja_env = self._init_jinja(self.template_root)
        self.lifecycle_logger = LifeCycleLogger(dry_run=self.llm.dry_run, log_interval=log_interval)

    def create_conversation(self, system_message: Optional[str] = None) -> Conversation:
        """Create a new conversation instance."""
        # Use a default max_messages from config if available, else 50
        max_msgs = getattr(self.config, 'max_history', 50)
        return Conversation(system_message=system_message, max_messages=max_msgs)

    def _init_jinja(self, root_dir: Optional[Path]) -> Optional[Environment]:
        if not root_dir or not root_dir.exists():
            # 允许为空，运行时可能报错
            return None
        return Environment(
            loader=FileSystemLoader(root_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def render_template(self, template: Union[str, Path], context: Dict[str, Any]) -> str:
        if not self.jinja_env:
             # Fallback for no jinja env, only supports string formatting
             if isinstance(template, Path):
                 raise RuntimeError("Jinja environment not initialized, cannot render file template.")
             try:
                 return template.format(**context)
             except:
                 pass
             raise RuntimeError("Jinja environment not initialized")

        if isinstance(template, Path):
            # It's a file path
            # Jinja loader expects POSIX paths relative to root
            rel_path = template.as_posix()
            template_obj = self.jinja_env.get_template(rel_path)
            return template_obj.render(**context)
        else:
            # It's a raw string template
            template_obj = self.jinja_env.from_string(template)
            return template_obj.render(**context)

    async def build_user_message(self, text: str, image_path: Optional[Path] = None, image_kwargs: Dict = {}) -> List[Dict[str, Any]]:
        """构建标准用户消息
        Args:
            text: 文本提示
            image_path: 图片路径 (可选)
            image_kwargs: 图片处理参数 (min_pixels, max_pixels, etc.)
        """
        content = [{"type": "text", "text": text}]
        
        if image_path:
            # Merge task-level image config with kwargs
            # Priority: kwarg > task_config > default
            task_img_cfg = self.config.image_config or {}
            final_img_kwargs = {**task_img_cfg, **image_kwargs}
            
            # Extract preprocessing args (max_size) vs model args (min_pixels)
            # image_to_data_url only needs max_size
            max_size = final_img_kwargs.get('max_image_size', 1024)
            
            from ..core.image_processor import image_to_data_url
            image_url = await image_to_data_url(image_path, max_size=max_size)
            
            if image_url:
                img_content = self.format_image_content(image_url, **final_img_kwargs)
                content.append(img_content)
            else:
                # Log warning? handled by image_to_data_url logging usually
                pass

        msg = {
            "role": "user",
            "content": content
        }
        
        # Inject model-specific image parameters if present in kwargs
        # This aligns with how sglang/vLLM expects these parameters for Qwen-VL
        for param in ["min_pixels", "max_pixels"]:
            if param in final_img_kwargs:
                msg[param] = final_img_kwargs[param]
                
        return [msg]

    def format_image_content(self, image_url: str, **kwargs) -> Dict[str, Any]:
        """格式化图片内容部分"""
        # Standard OpenAI format
        content = {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                # "detail": kwargs.get("detail", "auto") # if supported
            }
        }
        # Some models use min_pixels/max_pixels inside image_url dict or parallel to it
        # OpenAI standard puts detail inside image_url.
        # Qwen/DeepSeek via vLLM/SGLang might respect extras.
        # We'll leave it simple for now or inject if needed.
        if "detail" in kwargs:
             content["image_url"]["detail"] = kwargs["detail"]
             
        # If specific models need min_pixels inside content, add here.
        # But usually min_pixels/max_pixels are sent in 'extra_body' or top-level params (like in original core.py).
        # We will assume they are passed via generation_kwargs to client.chat call.
        return content

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """
        Return the configuration schema for `agent_kwargs` or other config.
        Returns:
            Dict[str, Dict[str, Any]]: Param name -> {type, help, required}
        """
        return {}

    @abstractmethod
    async def run(self, item: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        """执行Agent任务"""
        pass
