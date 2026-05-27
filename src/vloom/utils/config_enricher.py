
import logging
import yaml
from typing import Dict, Any, List
from ..agents.registry import AgentRegistry
from ..tools.registry import ToolRegistry
# Need a way to get Tool Class from tool name. Registry currently stores instances or mapping.
# We need to peek into the plugin module or registry mapping.
from ..dataset.base import BaseDataset
from ..config import DatasetConfig

logger = logging.getLogger(__name__)

def enrich_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traverse the config (Tasks, Datasets) and inject schema information for missing fields.
    """
    
    # 1. Enrich Datasets
    if "datasets" in config:
        for ds_cfg in config["datasets"]:
             # This part is tricky because we need the Dataset CLASS.
             # In current VLooM, datasets might be hardcoded or via Draccus.
             # If "name" corresponds to a registered dataset, we find it.
             # For now, we skip or assume a generic pattern if a class is available.
             pass

    # 2. Enrich Tasks
    if "task_definitions" in config:
        for task in config["task_definitions"]:
            agent_type = task.get("agent_type")
            if agent_type:
                # Get Agent Class
                agent_cls = AgentRegistry._registry.get(agent_type)
                if agent_cls and hasattr(agent_cls, "get_config_schema"):
                    schema = agent_cls.get_config_schema()
                    if schema:
                        if "agent_kwargs" not in task:
                            task["agent_kwargs"] = {}
                        
                        for param, info in schema.items():
                            if param not in task["agent_kwargs"]:
                                req_str = "REQUIRED" if info.get("required") else "Optional"
                                help_str = info.get("help", "")
                                task["agent_kwargs"][param] = f"<{req_str}: {info.get('type')}> {help_str}"

            # Tools
            tools = task.get("tools", [])
            for tool_name in tools:
                # We need to find the Tool Class. 
                # ToolRegistry is typically instantiated per agent.
                # However, if we assume Tools are registered globally or we can import them.
                # Since we don't have a global tool class registry (only instances inside agents),
                # We might need to inspect imported modules or specific know classes.
                
                # HACK: For now, specific check or requires a GlobalToolRegistry mapping names to Classes.
                # Let's try to look up in our known tools (BaseTool subclasses) if possible.
                # For this demo, we check if it is VisionTool.
                tool_cls = None
                from ..tools.vision_tool import VisionTool
                if tool_name == "vision_tool":
                    tool_cls = VisionTool
                
                if tool_cls and hasattr(tool_cls, "get_config_schema"):
                    schema = tool_cls.get_config_schema()
                    if schema:
                        if "tool_config" not in task:
                            task["tool_config"] = {}
                        if tool_name not in task["tool_config"]:
                            task["tool_config"][tool_name] = {}
                            
                        for param, info in schema.items():
                             if param not in task["tool_config"][tool_name]:
                                req_str = "REQUIRED" if info.get("required") else "Optional"
                                help_str = info.get("help", "")
                                task["tool_config"][tool_name][param] = f"<{req_str}: {info.get('type')}> {help_str}"
                                
    return config

def main_enrich():
    import sys
    import yaml
    
    if len(sys.argv) < 2:
        print("Usage: python -m vloom.utils.config_enricher <partial_config.yaml>")
        return
        
    path = sys.argv[1]
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
 
    enriched = enrich_config(data)
    
    print(yaml.dump(enriched, sort_keys=False))

if __name__ == "__main__":
    main_enrich()
