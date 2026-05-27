from abc import ABC, abstractmethod
from typing import Any, Dict, Type, Optional

class BaseTool(ABC):
    """
    Abstract base class for tools.
    """
    name: str = "base_tool"
    description: str = "Base description"
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with provided arguments."""
        pass

    def to_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema for the tool (compatible with OpenAI function calling or similar).
        Default implementation returns a simple description. Subclasses should override.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {}, # Define specific parameters in subclasses
                }
            }
        }
