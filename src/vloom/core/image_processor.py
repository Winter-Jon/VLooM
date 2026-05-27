import io
import asyncio
import base64
import logging
import aiofiles
from pathlib import Path
from PIL import Image
from typing import Optional, Union

logger = logging.getLogger(__name__)


def _sync_process_image(img_data: bytes, max_size: int = 1024) -> bytes:
    """同步处理图像逻辑 (IO bound -> CPU bound)"""
    try:
        # 使用PIL打开图像
        img = Image.open(io.BytesIO(img_data))
        
        # 转换为RGB模式（如果不是的话）
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # 获取原始尺寸
        width, height = img.size
        
        # 计算缩放比例
        if width > height:
            if width > max_size:
                new_width = max_size
                new_height = int(height * max_size / width)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"图像已从 {width}x{height} 调整为 {new_width}x{new_height}")
        else:
            if height > max_size:
                new_height = max_size
                new_width = int(width * max_size / height)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"图像已从 {width}x{height} 调整为 {new_width}x{new_height}")
        
        # 将图像保存到字节流
        output_buffer = io.BytesIO()
        img.save(output_buffer, format='JPEG', quality=95)
        return output_buffer.getvalue()
    except Exception as e:
        raise e

async def preprocess_image(img_path: Path, max_size: int = 1024) -> bytes:
    """预处理图像，调整最大边长 (非阻塞)
    
    Args:
        img_path: 图像文件路径
        max_size: 最大边长，默认为1024像素
    
    Returns:
        处理后的图像字节数据
    """
    img_path = Path(img_path)
    try:
        # 异步读取图像文件 (IO)
        async with aiofiles.open(img_path, 'rb') as f:
            img_data = await f.read()
        
        # 在 Executor 中运行 CPU 密集型图像处理
        loop = asyncio.get_running_loop()
        processed_data = await loop.run_in_executor(
            None, 
            _sync_process_image, 
            img_data, 
            max_size
        )
        
        return processed_data
        
    except Exception as e:
        logger.error(f"预处理图像 {img_path} 失败: {e}")
        # 如果预处理失败，返回原始图像数据
        try:
            async with aiofiles.open(img_path, 'rb') as f:
                return await f.read()
        except Exception as read_error:
            logger.error(f"读取原始图像 {img_path} 失败: {read_error}")
            raise e



async def image_to_data_url(img_path: Path, max_size: Optional[int] = 1024) -> Optional[str]:
    """将图片转换为base64数据URL（异步版本，支持预处理）
    
    Args:
        img_path: 图片文件路径
        max_size: 最大边长，如果为None则不进行预处理
    """
    img_path = Path(img_path)
    try:
        if max_size:
            data = await preprocess_image(img_path, max_size)
            if data is None:
                return None
        else:
            # 直接读取原始图片
            async with aiofiles.open(img_path, "rb") as f:
                data = await f.read()
        
        encoded_data = base64.b64encode(data).decode("utf-8")
        
        return f"data:image/jpeg;base64,{encoded_data}"

    except Exception as e:
        logger.error(f"转换图片 {img_path} 失败: {e}")
        return None

