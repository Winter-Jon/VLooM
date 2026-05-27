import logging
import asyncio
from typing import List, Dict, Any, AsyncGenerator, Optional
from abc import ABC, abstractmethod
from types import SimpleNamespace

from openai import AsyncClient, APITimeoutError, RateLimitError, APIConnectionError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)

class LLMStrategy(ABC):
    """
    LLM 调用策略抽象基类
    """
    @abstractmethod
    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        pass

    @abstractmethod
    async def chat_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator:
        pass
    
    @abstractmethod
    async def close(self):
        pass

class OpenAIStrategy(LLMStrategy):
    """
    真实的 OpenAI 接口调用策略
    """
    def __init__(self, api_key: str, base_url: str):
        self.client = AsyncClient(api_key=api_key, base_url=base_url)

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"LLM request failed (Final Attempt): {e}")
            raise

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def chat_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator:
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **kwargs
            )
            return stream
        except Exception as e:
            logger.error(f"LLM stream request failed (Final Attempt): {e}")
            raise

    async def close(self):
        await self.client.close()

class DryRunStrategy(LLMStrategy):
    """
    Dry Run 模拟策略
    """
    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        # logic moved to LifeCycleLogger in agents
        # Mock Response
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="[DRY RUN] This is a simulated response. \n {}",
                        reasoning_content="[DRY RUN] Simulated thinking process. \n {}"
                    )
                )
            ],
            usage=SimpleNamespace(total_tokens=0, prompt_tokens=0, completion_tokens=0)
        )

    async def chat_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator:
        logger.warning(f"DRY RUN (Stream): {model}")
        async def mock_stream():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="[DRY RUN] Simulated Stream"))]
            )
        return mock_stream()

    async def close(self):
        pass


class LLMClient:
    """
    LLM 客户端封装，使用策略模式 (Strategy Pattern) 来分离真实调用与模拟调用。
    
    支持通过 strategy_class 参数注入自定义策略，实现扩展性。
    """
    def __init__(
        self, 
        api_key: str, 
        base_url: str, 
        dry_run: bool = False,
        strategy_class: Optional[type] = None
    ):
        self.dry_run = dry_run
        
        if strategy_class is not None:
            # Use custom strategy class if provided
            self._strategy: LLMStrategy = strategy_class()
            self.dry_run = True  # Custom strategies are treated as dry run mode
        elif dry_run:
            self._strategy: LLMStrategy = DryRunStrategy()
        else:
            self._strategy: LLMStrategy = OpenAIStrategy(api_key, base_url)

    async def chat(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """非流式对话"""
        return await self._strategy.chat(model, messages, **kwargs)

    async def chat_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> AsyncGenerator:
        """流式对话"""
        return await self._strategy.chat_stream(model, messages, **kwargs)

    async def close(self):
        await self._strategy.close()
