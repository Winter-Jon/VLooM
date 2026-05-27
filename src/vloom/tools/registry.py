from typing import Dict, Type, Optional, List, Any
import logging
from .base import BaseTool

logger = logging.getLogger(__name__)


class ToolClassRegistry:
    """Class-level registry for tool classes, mirroring AgentRegistry pattern."""
    _registry: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register_tool(cls, name: str):
        """Decorator to register a tool class."""
        def decorator(tool_cls: Type[BaseTool]):
            cls._registry[name] = tool_cls
            logger.info(f"Registered tool class '{name}': {tool_cls.__name__}")
            return tool_cls
        return decorator

    @classmethod
    def get_tool_class(cls, name: str) -> Optional[Type[BaseTool]]:
        return cls._registry.get(name)

    @classmethod
    def list_registered_tools(cls) -> List[str]:
        return list(cls._registry.keys())


class ToolRegistry:
    """Instance-level registry for managing tool instances."""
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool):
        """Register a tool instance."""
        if tool.name in self._tools:
            logger.warning(f"Tool {tool.name} already registered. Overwriting.")
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of tool schemas."""
        return [t.to_schema() for t in self._tools.values()]

    async def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found.")
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return f"Error: {str(e)}"
