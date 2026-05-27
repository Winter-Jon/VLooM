import logging
import json
from typing import Any, Dict, List, Optional

from .base import BaseAgent
from ..core.image_processor import image_to_data_url
from ..core.parser import fix_json_content, parse_json
# Circular import note: agents/registry.py does NOT import agents anymore.
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

@AgentRegistry.register_agent("basic")
@AgentRegistry.register_agent("default") 
class BasicAgent(BaseAgent):
    """
    基础Agent：单轮对话，支持模板渲染和JSON解析
    """
    async def run(self, item: Any, model_name: str, **kwargs) -> Dict[str, Any]:
        """
        执行单次请求
        """
        should_log = self.lifecycle_logger.log_cycle_start(self.config.name, item.img_name)

        # 1. 准备 Prompt
        # Support both 'templates' and legacy 'template_files'
        templates = self.config.templates or self.config.template_files or {}
        context = item.template_vars
        rendered_parts = {}
        
        for key, rel_path in templates.items():
            rendered_parts[key] = self.render_template(rel_path, context)
            
        # Combine parts into full prompt
        if "full_prompt" in rendered_parts:
            prompt = rendered_parts["full_prompt"]
        elif "usr" in rendered_parts: # Common convention
             prompt = rendered_parts["usr"]
        else:
            prompt = "\n\n".join(rendered_parts.values())
            
        # 2. 构建与发送消息
        try:
            # 使用 BaseAgent 的 helper 构建消息
            # kwargs 可能包含 min_pixels 等，也传递给 build_user_message 作为 image_kwargs
            messages = await self.build_user_message(
                text=prompt, 
                image_path=item.img_path,
                image_kwargs=kwargs # Pass generation kwargs as potential image config override
            )
            
            
            logger.debug(f"Sending request to {model_name} for {item.img_name}")
            self.lifecycle_logger.log_question(messages, should_log=should_log)

            # Filter out image processing args from kwargs before passing to LLM
            llm_kwargs = {k: v for k, v in kwargs.items() 
                         if k not in ["min_pixels", "max_pixels", "max_image_size"]}
            
            response = await self.llm.chat(
                model=model_name,
                messages=messages,
                **llm_kwargs
            )
            
            content = response.choices[0].message.content.strip()
            
            thinking = None
            if hasattr(response.choices[0].message, 'reasoning_content'):
                thinking = response.choices[0].message.reasoning_content
            
            self.lifecycle_logger.log_response(model_name, content, thinking, should_log=should_log)

            result_data = {
                "result": None,
                "raw_response": content,
                "thinking": thinking,
                "template_vars": context,
                "prompt": prompt,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                } if response else {}
            }
            
            # 3. 解析结果
            try:
                result = parse_json(content)
                result_data["result"] = result
            except Exception as e:
                logger.warning(f"Failed to parse JSON for {item.img_name}: {content[:100]}...")
            
            return {item.img_name: result_data}
            
        except Exception as e:
            logger.error(f"Error processing {item.img_name}: {e}")
            return {item.img_name: None}
        finally:
            self.lifecycle_logger.log_cycle_end(item.img_name, should_log=should_log)
