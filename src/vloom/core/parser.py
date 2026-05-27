import json
import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)

def fix_json_content(content: str) -> Optional[str]:
    """尝试修复JSON内容"""
    try:
        # 移除可能的markdown格式
        content = content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        content = content.strip()
        
        # 尝试解析
        json.loads(content)
        return content
    except json.JSONDecodeError:
        lines = content.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('{') or in_json:
                in_json = True
                json_lines.append(line)
            if line.endswith('}'):
                break
        
        if json_lines:
            try:
                fixed_content = '\n'.join(json_lines)
                json.loads(fixed_content)
                return fixed_content
            except json.JSONDecodeError:
                pass
        
        return None

def parse_json(content: str) -> Dict[str, Any]:
    """解析JSON，如果失败则尝试修复"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fixed = fix_json_content(content)
        if fixed:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        raise
