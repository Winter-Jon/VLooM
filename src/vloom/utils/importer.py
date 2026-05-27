import importlib
import sys
import os
from pathlib import Path
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

def preload_imports():
    if "--config_path" not in sys.argv:
        return

    try:
        arg_idx = sys.argv.index("--config_path")
        config_path = sys.argv[arg_idx + 1]

        if not os.path.exists(config_path):
            logger.error(f"错误: 配置文件不存在: {config_path}")
            sys.exit(1)

        import yaml
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"错误: 配置文件 YAML 格式解析失败: {config_path}\n{e}")
            sys.exit(1)

        # 4. 模块导入
        imports = cfg_dict.get("imports", [])
        if imports:
            logger.info(f"正在预加载外部模块: {imports}")
            try:
                import_modules_from_strings(imports)
            except (ImportError, ModuleNotFoundError) as e:
                logger.error(f"致命错误: 无法导入配置文件中指定的模块。\n详细信息: {e}")
                logger.error("请检查 PYTHONPATH 或模块名称是否正确。")
                sys.exit(1)

    except Exception as e:
        # 捕获其他未预料到的错误
        logger.error(f"预加载阶段发生未知错误: {e}")
        sys.exit(1)

def import_modules_from_strings(imports, allow_failed_imports=False):
    """
    imports: list[str], e.g. ['main_repo.datasets.my_dataset']
    """
    if not imports:
        return

    # 关键：把当前执行目录加入 path，否则找不到主仓库的包
    sys.path.insert(0, os.getcwd())

    for module_name in imports:
        try:
            importlib.import_module(module_name)
            print(f"[Info] Successfully imported custom module: {module_name}")
        except ImportError as e:
            if allow_failed_imports:
                print(f"[Warning] Failed to import {module_name}: {e}")
            else:
                raise e

def import_modules_from_folder(folder_path: Path) -> None:
    """
    Import all .py files from the specified folder.
    This allows decorators in those files to run and register agents/tools.
    """
    path = Path(folder_path).resolve()
    if not path.exists():
        logger.warning(f"Plugin folder {path} does not exist. Skipping.")
        return

    # Add folder to sys.path so imports work relative to it if needed,
    # or just to allow importing them as top-level if they are simple scripts.
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.append(str_path)
    
    logger.info(f"Scanning for plugins in {path}...")
    
    count = 0
    for file in path.glob("*.py"):
        if file.name.startswith("__") or file.name.startswith("."):
            continue
        
        module_name = file.stem
        try:
            importlib.import_module(module_name)
            logger.info(f"Loaded plugin module: {module_name}")
            count += 1
        except Exception as e:
            logger.error(f"Failed to load plugin {module_name}: {e}")
            
    logger.info(f"Loaded {count} plugin modules from {folder_path}")
