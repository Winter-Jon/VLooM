import logging
import json
from typing import Any, Dict, List, Optional
from .tool_agent import ToolAgent
from ..tools.vision_tool import VisionTool
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

@AgentRegistry.register_agent("vision_tool") # Register as specific agent type
class VisionToolAgent(ToolAgent):
    """
    Agent specialized for using the Vision Tool.
    Auto-initializes VisionTool from config.
    """
    def __init__(self, config, template_root, llm_client):
        super().__init__(config, template_root, llm_client)
        
        tool_config = getattr(config, 'tool_config', {})
        
        # Explicitly initialize VisionTool
        ann_path = tool_config.get("vision_tool", {}).get("annotation_path")
        if ann_path:
            try:
                v_tool = VisionTool(ann_path)
                self.register_tool(v_tool)
                logger.info("VisionTool registered in VisionToolAgent.")
            except Exception as e:
                logger.error(f"Failed to init VisionTool: {e}")
        else:
            logger.warning("VisionToolAgent initialized but 'annotation_path' missing in tool_config['vision_tool'].")

    # We can override run or other methods if specific logic is needed,
    # but for now reusing ToolAgent's ReAct loop is fine.
