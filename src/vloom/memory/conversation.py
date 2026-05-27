from typing import List, Dict, Any, Optional, Union
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class Conversation:
    """
    Manages conversation history with support for token limits (simple truncation)
    and persistence.
    """
    def __init__(self, system_message: Optional[str] = None, max_messages: int = 50):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        if system_message:
            self.add_message("system", system_message)

    def add_message(self, role: str, content: Any):
        """Add a message to the history."""
        message = {"role": role, "content": content}
        self.messages.append(message)
        self._truncate()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the conversation."""
        return self.messages

    def clear(self):
        """Clear conversation history."""
        self.messages = []

    def _truncate(self):
        """Simple truncation based on message count (keeping system message)."""
        if len(self.messages) > self.max_messages:
            # Preserve system message if it exists at index 0
            if self.messages and self.messages[0]["role"] == "system":
                system_msg = self.messages[0]
                # Keep system_msg + last (max_messages-1) messages
                self.messages = [system_msg] + self.messages[-(self.max_messages-1):]
            else:
                self.messages = self.messages[-self.max_messages:]
            
            logger.debug(f"Conversation truncated. Current length: {len(self.messages)}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation to dictionary."""
        return {
            "messages": self.messages,
            "max_messages": self.max_messages
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """Load conversation from dictionary."""
        conversation = cls(max_messages=data.get("max_messages", 50))
        conversation.messages = data.get("messages", [])
        return conversation

    def save(self, filepath: Path):
        """Save conversation to a file."""
        import aiofiles # Optional, but good practice if available, or just standard open
        # Using standard open for simplicity unless async is required heavily
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Path) -> 'Conversation':
        """Load conversation from a file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
