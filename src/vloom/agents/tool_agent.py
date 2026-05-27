import logging
import json
from typing import Any, Dict, List, Optional
from .basic_agent import BasicAgent
from ..tools.registry import ToolRegistry, ToolClassRegistry
from ..tools.base import BaseTool
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

@AgentRegistry.register_agent("tool")
class ToolAgent(BasicAgent):
    """
    Agent capable of executing tools.
    Expects tool calls in JSON format: {"tool": "name", "args": {...}}
    """
    def __init__(self, config, template_root, llm_client, log_interval=10):
        super().__init__(config, template_root, llm_client, log_interval=log_interval)
        self.tools = ToolRegistry()

        # Load tools based on config
        requested_tools = getattr(config, 'tools', [])
        tool_config_map = getattr(config, 'tool_config', {})

        # Ensure built-in tool modules are imported so @register_tool decorators fire
        try:
            from ..tools import vision_tool  # noqa: F401
        except ImportError:
            pass

        for tool_name in requested_tools:
            tool_cls = ToolClassRegistry.get_tool_class(tool_name)
            if tool_cls:
                kwargs = tool_config_map.get(tool_name, {})
                try:
                    tool_instance = tool_cls(**kwargs)
                    self.register_tool(tool_instance)
                except Exception as e:
                    logger.error(f"Failed to instantiate tool {tool_name} with kwargs {kwargs}: {e}")
                    raise e
            else:
                logger.warning(f"Unknown tool requested: {tool_name}. Skipping auto-instantiation.")
        
    def register_tool(self, tool: BaseTool):
        self.tools.register(tool)

    async def run(self, item: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        # 1. Prepare Prompt
        templates = self.config.templates or self.config.template_files or {}
        context = item.template_vars
        prompt = self._build_prompt(templates, context)

        should_log = self.lifecycle_logger.log_cycle_start(self.config.name, item.img_name)

        # 2. Initialize Conversation
        conv = self.create_conversation()
        
        # Add available tools description to prompt
        tool_schemas = json.dumps(self.tools.list_tools(), indent=2)
        if tool_schemas != "[]":
            prompt = f"Available Tools:\n{tool_schemas}\n\n{prompt}"

        initial_msgs = await self.build_user_message(
            text=prompt,
            image_path=item.img_path,
            image_kwargs=kwargs
        )
        for msg in initial_msgs:
            conv.add_message(msg["role"], msg["content"])

        max_turns = 5
        current_turn = 0
        final_response = ""
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        tool_history = []  # Track tool calls for output

        while current_turn < max_turns:
            current_turn += 1
            logger.info(f"ToolAgent Turn {current_turn}")

            # Chat
            try:
                self.lifecycle_logger.log_question(conv.get_messages(), should_log=should_log)
                
                response = await self.llm.chat(
                    model=model_name,
                    messages=conv.get_messages(),
                    **kwargs
                )
            except Exception as e:
                logger.error(f"LLM error: {e}")
                break

            content = response.choices[0].message.content.strip()
            final_response = content
            
            # Track usage
            if response.usage:
                total_usage["prompt_tokens"] += response.usage.prompt_tokens
                total_usage["completion_tokens"] += response.usage.completion_tokens
                total_usage["total_tokens"] += response.usage.total_tokens

            conv.add_message("assistant", content)
            
            self.lifecycle_logger.log_response(model_name, content, should_log=should_log)

            # Check for tool call
            tool_call = self._parse_tool_call(content)
            
            if tool_call:
                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("args", {})
                logger.info(f"Detected tool call: {tool_name}({tool_args})")
                
                self.lifecycle_logger.log_tool_use(tool_name, tool_args, should_log=should_log)
                result = await self.tools.execute_tool(tool_name, **tool_args)
                self.lifecycle_logger.log_tool_result(result, should_log=should_log)
                
                # Track for output
                tool_history.append({"tool": tool_name, "args": tool_args, "result": result})
                
                # Build feedback message (potentially multimodal)
                feedback_content = self._build_tool_feedback(result, kwargs)
                conv.add_message("user", feedback_content)
            else:
                # No tool call, assume finished
                break

        result_dict = {
            item.img_name: {
                "result": final_response,
                "raw_response": final_response,
                "template_vars": context,
                "usage": total_usage,
                "tool_history": tool_history
            }
        }
        self.lifecycle_logger.log_cycle_end(item.img_name, should_log=should_log)
        return result_dict

    def _build_tool_feedback(self, result: Any, image_kwargs: Dict) -> Any:
        """
        Build feedback content for tool result.
        If result contains 'visualization' path, returns multimodal content (text + image).
        Otherwise, returns text-only content.
        """
        if isinstance(result, dict) and result.get("visualization"):
            vis_path = result["visualization"]
            # Construct multimodal content: image + text
            text_part = f"Tool Output:\n```json\n{json.dumps(result, indent=2)}\n```"
            
            # Return multimodal content list (OpenAI format)
            return [
                {"type": "image_url", "image_url": {"url": f"file://{vis_path}"}},
                {"type": "text", "text": text_part}
            ]
        else:
            # Plain text feedback
            return f"Tool Output: {result}"

    def _build_prompt(self, templates, context):
        """Helper to build prompt string from templates."""
        # Reuse logic from BasicAgent but extracted
        rendered_parts = {}
        for key, rel_path in templates.items():
            rendered_parts[key] = self.render_template(rel_path, context)
            
        if "full_prompt" in rendered_parts:
            return rendered_parts["full_prompt"]
        elif "usr" in rendered_parts:
             return rendered_parts["usr"]
        else:
            return "\n\n".join(rendered_parts.values())

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        """Try to parse tool call from text."""
        # Clean markdown code blocks
        clean_text = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(clean_text)
            if isinstance(data, dict) and "tool" in data:
                return data
        except:
            pass
        return None
