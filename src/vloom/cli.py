import logging
import sys
import asyncio
import draccus
from pathlib import Path

from vloom.config import PipelineConfig, DefaultTaskConfig
from vloom.core.llm_client import LLMClient
from vloom.agents.registry import AgentRegistry
from vloom.dataset import create_dataset
from vloom.pipelines.batch_runner import run_batch_task
from vloom.utils import preload_imports

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")


def main():
    preload_imports()

    cfg: PipelineConfig = draccus.parse(config_class=PipelineConfig)

    logging.getLogger().setLevel(cfg.log_level.value)

    save_dir = cfg.save_dir
    config_out_path = save_dir / "config.yaml"
    with open(config_out_path, 'w', encoding='utf-8') as f:
        draccus.dump(cfg, f)

    logger.info("Starting Pipeline Execution...")
    try:
        asyncio.run(async_main(cfg))
    except KeyboardInterrupt:
        logger.info("User interrupted.")
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")


async def async_main(cfg: PipelineConfig):
    strategy_class = None
    if cfg.model.strategy_class:
        import importlib
        module_path, class_name = cfg.model.strategy_class.rsplit(".", 1)
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, class_name)
        logger.info(f"Using custom LLM strategy: {cfg.model.strategy_class}")

    llm_client = LLMClient(
        api_key=cfg.model.api_key,
        base_url=cfg.model.base_url,
        dry_run=cfg.dry_run or cfg.model.dry_run,
        strategy_class=strategy_class
    )

    registry = AgentRegistry(
        template_root=cfg.template_dir,
        llm_client=llm_client,
        log_interval=cfg.log_interval
    )

    registry.load_from_config(cfg)

    try:
        await pipeline(cfg, registry, llm_client)
    finally:
        await llm_client.close()


async def pipeline(cfg: PipelineConfig, registry: AgentRegistry, llm_client: LLMClient):
    datasets = [create_dataset(ds_cfg) for ds_cfg in cfg.datasets]
    tasks = cfg.run_tasks

    for dataset in datasets:
        logger.info(f"Dataset {dataset.name} has {len(dataset)} items.")

        for task_name in tasks:
            if not registry.get_agent(task_name):
                logger.error(f"Skipping unknown task: {task_name}")
                continue

            await run_batch_task(registry, dataset, task_name, cfg)
