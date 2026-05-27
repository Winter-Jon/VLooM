import logging
from typing import Dict, Type, Optional
from typing import Dict, Type, Optional
from .base import BaseAgent
# Agents now register themselves, so we don't import them here to avoid circular dependency
# if agents import AgentRegistry. 
# However, agents need to be imported SOMEWHERE for the decorator to run.
# That is handled by __init__.py
from ..config import TaskConfig, DefaultTaskConfig, PipelineConfig
from ..core.llm_client import LLMClient
from pathlib import Path

logger = logging.getLogger(__name__)

class AgentRegistry:
    _registry: Dict[str, Type[BaseAgent]] = {}

    def __init__(self, template_root: str, llm_client: LLMClient, log_interval: int = 10):
        self.template_root = template_root
        self.llm_client = llm_client
        self.log_interval = log_interval
        self._agents: Dict[str, BaseAgent] = {}
        # _classes is now just a pointer to the class-level registry
        # ensuring all instances share the same registration
        self._classes = AgentRegistry._registry
    
    @classmethod
    def register_agent(cls, name: str):
        """Decorator to register an agent class."""
        def decorator(subclass: Type[BaseAgent]):
            cls._registry[name] = subclass
            logger.info(f"Registered agent '{name}': {subclass.__name__}")
            return subclass
        return decorator

    def register_agent_class(self, type_name: str, cls: Type[BaseAgent]):
        """Manual registration (legacy support)"""
        self._classes[type_name] = cls

    def create_agent(self, config: TaskConfig) -> BaseAgent:
        # Determine agent class. 
        # For now, we rely on TaskConfig having a type? Or infer from name?
        # The existing config.py TaskConfig doesn't have 'type'.
        # We can assume 'default' for now, OR if name is 'reflection' use ReflectiveAgent.
        # Ideally TaskConfig should have 'agent_type'.
        # I'll update TaskConfig later or use a heuristic here.
        
        # Heuristic: if config.name == "reflection" or "reflective" in config.name -> ReflectiveAgent
        # Else BasicAgent.
        
        # Use agent_type from config, default to 'basic'
        agent_type = config.agent_type or "basic"
            
        if not self._classes.get(agent_type):
             from .basic_agent import BasicAgent
             # BasicAgent is typically registered via decorator, but if not imported yet?
             # If we import it, the decorator runs.
        
        cls = self._classes.get(agent_type)
        if not cls:
             # Fallback if still not found, though basic_agent import should register it if it has decorator
             from .basic_agent import BasicAgent
             cls = BasicAgent

        try:
            agent = cls(config, self.template_root, self.llm_client, log_interval=self.log_interval)
            return agent
        except Exception as e:
            logger.error(f"Failed to create agent for {config.name}: {e}")
            raise

    def get_agent(self, task_name: str) -> Optional[BaseAgent]:
        return self._agents.get(task_name)

    def register_task(self, config: TaskConfig):
        agent = self.create_agent(config)
        self._agents[config.name] = agent

    def load_from_config(self, cfg: PipelineConfig):
        """Loads tasks from PipelineConfig (manual definitions + auto-discovery)."""
        # 1. Register Config Tasks
        if cfg.task_definitions:
            for task_cfg in cfg.task_definitions:
                self.register_task(task_cfg)
                
        # 2. Register Auto Tasks
        if cfg.auto_task_dir:
            root_path = Path(self.template_root)
            scan_path = root_path / cfg.auto_task_dir

            if not scan_path.exists():
                logger.warning(f"Auto task directory not found: {scan_path}")
            else:
                auto_task_names = []
                for file_path in scan_path.glob("*.j2"):
                    task_name = file_path.stem
                    # Skip if already registered (e.g. via config or explicit registration)
                    if self.get_agent(task_name):
                        continue
                    
                    try:
                        rel_file_path = file_path.relative_to(root_path).as_posix()
                    except ValueError:
                        # Fallback if file is not relative to template_root (should mostly be true)
                        rel_file_path = file_path.as_posix()

                    auto_config = DefaultTaskConfig(
                        name=task_name,
                        description=f"Auto-discovered: {file_path.name}",
                        templates={"usr": rel_file_path}
                    )

                    self.register_task(auto_config)
                    auto_task_names.append(task_name)
                
                if auto_task_names:
                    logger.info(f"Auto-registered tasks: {auto_task_names}")

        # Check missing tasks
        missing_tasks = [t for t in cfg.run_tasks if not self.get_agent(t)]
        if missing_tasks:
            logger.warning(f"Warning: Tasks not found in registry: {missing_tasks}")

