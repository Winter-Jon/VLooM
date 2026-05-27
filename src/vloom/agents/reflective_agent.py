import logging
import json
import asyncio
from typing import Any, Dict, List, Optional

from .basic_agent import BasicAgent
from ..core.image_processor import image_to_data_url
from ..core.parser import fix_json_content, parse_json
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

@AgentRegistry.register_agent("reflective")
@AgentRegistry.register_agent("reflection")
class ReflectiveAgent(BasicAgent):
    """
    反思Agent：
    1. 监控流式输出
    2. 发现特定token (如 <reflection>) 时打断
    3. 插入人工Prompt
    4. 继续生成
    """
    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "reflection_trigger": {
                "type": "str",
                "help": "Tag to trigger reflection (e.g., <reflection>).",
                "required": False,
                "default": "<reflection>"
            },
            "intervention_text": {
                "type": "str",
                "help": "Text to send when intervention is triggered.",
                "required": False,
                "default": "Wait, checks failed."
            }
        }

    async def run(self, item: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        # 1. 准备基础 Prompt
        templates = self.config.templates or self.config.template_files or {}
        context = item.template_vars
        rendered_parts = {}
        for key, rel_path in templates.items():
            rendered_parts[key] = self.render_template(rel_path, context)
            
        if "full_prompt" in rendered_parts:
            prompt = rendered_parts["full_prompt"]
        elif "usr" in rendered_parts:
             prompt = rendered_parts["usr"]
        else:
            prompt = "\n\n".join(rendered_parts.values())

        # 2. 构建初始消息
        # Create conversation
        conv = self.create_conversation()
        
        # Build initial content (text + image)
        # We reuse the logic from build_user_message but adapt to conversation
        # Since build_user_message returns a list of messages, we can validly just extend the conversation
        initial_msgs = await self.build_user_message(
            text=prompt,
            image_path=item.img_path,
            image_kwargs=kwargs
        )
        for msg in initial_msgs:
            conv.add_message(msg["role"], msg["content"])
            
        messages = conv.get_messages()

        # 3. 开始流式请求
        final_content = ""
        full_reasoning = ""
        found_reflection = False
        
        # 从 agent_kwargs 获取配置，或使用默认值
        trigger_token = self.config.agent_kwargs.get("reflection_trigger", "<reflection>")
        
        # Intervention Text: try template 'intervention', then agent_kwargs, then default
        if "intervention" in templates:
            intervention_text = self.render_template(templates["intervention"], context)
        else:
            intervention_text = self.config.agent_kwargs.get(
                "intervention_text", 
                "Wait, please reflect on your previous analysis. Are you sure? Please think again."
            )

        try:
            logger.info(f"ReflectiveAgent: sending stream request to {model_name}")
            stream = await self.llm.chat_stream(model=model_name, messages=messages, **kwargs)
            
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                reasoning_delta = getattr(chunk.choices[0].delta, 'reasoning_content', "") or ""
                
                final_content += delta
                full_reasoning += reasoning_delta
                
                # Check for trigger
                if trigger_token in final_content and not found_reflection:
                    logger.info(f"Trigger token '{trigger_token}' detected!")
                    found_reflection = True
                    break 
        
            if found_reflection:
                # Update conversation history
                conv.add_message("assistant", final_content)
                conv.add_message("user", intervention_text)
                
                messages = conv.get_messages()
                
                logger.info("Sending intervention request...")
                response = await self.llm.chat(model=model_name, messages=messages, **kwargs)
                second_content = response.choices[0].message.content.strip()

                final_content = second_content 
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    full_reasoning += "\n[Intervention]\n" + (response.choices[0].message.reasoning_content or "")

            # 4. 解析结果
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
            # If intervention happened, we have response.usage
            if found_reflection and 'response' in locals() and response.usage:
                usage["prompt_tokens"] = response.usage.prompt_tokens
                usage["completion_tokens"] = response.usage.completion_tokens
                usage["total_tokens"] = response.usage.total_tokens

            result_data = {
                "result": None,
                "raw_response": final_content,
                "thinking": full_reasoning,
                "template_vars": context,
                "prompt": prompt,
                "triggered_reflection": found_reflection,
                "usage": usage
            }
            
            try:
                result = parse_json(final_content)
                result_data["result"] = result
            except:
                pass
                
            return {item.img_name: result_data}

        except Exception as e:
            logger.error(f"Error in ReflectiveAgent for {item.img_name}: {e}")
            return {item.img_name: None}
