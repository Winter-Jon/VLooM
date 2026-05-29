from functools import cached_property
import os
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
import draccus

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from .dataset import BaseDataset

# =============================================================================
# Note: draccus has native support for pathlib.Path type conversion
# No custom encoder/decoder registration is needed. Simply declare fields
# with Path type and draccus will automatically convert str -> Path when
# parsing YAML and Path -> str when dumping.
# =============================================================================

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Dataset Configurations

@dataclass
class DatasetConfig(draccus.ChoiceRegistry):
    name: str = ""
    description: str = ""
    
    def dataset_class(self) -> Type["BaseDataset"]:
        raise NotImplementedError


# Task Configurations

@dataclass
class TaskConfig(draccus.ChoiceRegistry):
    name: str = ""
    description: str = ""
    agent_type: str = "basic" # Default agent type

    # Prompt templates (dict of name -> relative path or inline template)
    templates: Dict[str, Union[str, Path]] = field(default_factory=dict)
    
    # Backward compatibility alias
    # template_files: Dict[str, str] = field(default_factory=dict) # Optional if we want strictly new config

    # Extra arguments for agent logic (e.g. thresholds, triggers)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Image processing config (min_pixels, max_pixels, etc.)
    image_config: Dict[str, Any] = field(default_factory=dict)

    # Tool configuration
    tools: List[str] = field(default_factory=list, metadata={"help": "List of tool names to enable"})
    tool_config: Dict[str, Any] = field(default_factory=dict, metadata={"help": "Configuration for specific tools"})

    def __post_init__(self):
        # Smart conversion: if string ends with extension, convert to Path. Otherwise keep as str (inline).
        if self.templates:
            new_templates = {}
            for k, v in self.templates.items():
                if isinstance(v, str):
                    # heuristic to detect file path
                    # You can extend this list if needed
                    if v.lower().endswith(('.j2', '.jinja', '.html', '.txt', '.md', '.json')):
                        new_templates[k] = Path(v)
                    else:
                        new_templates[k] = v
                else:
                    new_templates[k] = v
            self.templates = new_templates

        # Validate tools
        if self.tools:
            # Check for duplicates
            if len(self.tools) != len(set(self.tools)):
                raise ValueError(f"Duplicate tools found in task {self.name}: {self.tools}")
            
        # Validate tool_config keys
        if self.tool_config:
            for tool_name in self.tool_config:
                if tool_name not in self.tools:
                    # Warning or Error? Let's just warn via print for now as logger might not be setup
                    print(f"Warning: Tool '{tool_name}' configured in tool_config but not listed in tools list.")

@TaskConfig.register_subclass("default")
@dataclass
class DefaultTaskConfig(TaskConfig):
    """通用任务配置"""
    pass




@dataclass
class ModelConfig:
    """模型与API参数配置"""
    model_name: str = "models/Qwen3-VL-235B-A22B-Thinking"
    base_url: str = "http://127.0.0.1:30000/v1"
    api_key: str = ""
    api_key_envs: List[str] = field(
        default_factory=lambda: ["OPENAI_API_KEY", "YUNWU_API_KEY"],
        metadata={"help": "Environment variables to check when api_key is empty or EMPTY"},
    )
    generation_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "max_tokens": 8192,
        "temperature": 1.0,
        "top_p": 0.9,
        "min_pixels": 65536,
        "max_pixels": 10035200,
        "max_image_size": 1024,
        "timeout": 300,
    })
    dry_run: bool = field(default=False, metadata={"action": "store_true"})
    # Custom strategy class path (e.g., "preprocess.strategy.ToolDryRunStrategy")
    strategy_class: Optional[str] = field(default=None, metadata={"help": "Custom LLM strategy class path"})

@dataclass
class PathConfig:
    output_root: Path = Path("./outputs")
    exp_name: Optional[str] = None

    @cached_property
    def save_dir(self) -> Path:
        # output_root is already Path due to draccus decoder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.output_root / self.exp_name if self.exp_name else self.output_root / timestamp
        target.mkdir(parents=True, exist_ok=True)
        return target
    
    template_dir: Path = Path("templates")


@dataclass
class PipelineConfig(PathConfig):
    imports: List[str] = field(default_factory=list, metadata={"help": "额外导入的模块列表"})
    
    datasets: List[DatasetConfig] = field(default_factory=list, metadata={"help": "数据集配置列表"})
    model: ModelConfig = field(default_factory=ModelConfig, metadata={"help": "模型配置"})

    task_definitions: List[TaskConfig] = field(default_factory=list, metadata={"help": "任务定义列表"})
    auto_task_dir: Optional[Path] = field(default=None, metadata={"help": "自动加载任务模板的目录"})
    run_tasks: List[str] = field(default_factory=list, metadata={"help": "要执行的任务名称列表"})
    
    max_concurrent: int = field(default=64, metadata={"help": "异步并发请求数量"})
    log_level: LogLevel = LogLevel.INFO
    log_interval: int = field(default=100, metadata={"help": "LifeCycle日志输出间隔 (First 5 + every N)"})

    dry_run: bool = field(default=False, metadata={"action": "store_true", "help": "是否开启Dry Run模式 (仅输出Prompt不调用LLM)"})
