import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ParsedJSONResponse:
    """Structured parse result for model responses that may include reasoning text."""

    result: Optional[Dict[str, Any]]
    thinking: str = ""
    repair_log: list[str] = field(default_factory=list)


def extract_think(content: str) -> tuple[str, str]:
    """Extract <think>...</think> content and return (thinking, remaining_text)."""
    match = re.search(r"<think>\s*(.*?)\s*</think>", content, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return "", content

    thinking = match.group(1).strip()
    remaining = content[:match.start()] + content[match.end():]
    return thinking, remaining.strip()


def clean_json_text(content: str) -> str:
    """Apply conservative repairs for common LLM JSON formatting issues."""
    content = re.sub(r"```(?:json)?\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL | re.IGNORECASE)
    content = content.strip()
    content = re.sub(
        r"^(?:Here is the JSON[:：]?\s*|JSON[:：]?\s*|Response[:：]?\s*)",
        "",
        content,
        flags=re.IGNORECASE,
    )
    content = re.sub(r"//[^\n]*", "", content)
    content = re.sub(r",(\s*[}\]])", r"\1", content)

    def _fix_quotes(match: re.Match) -> str:
        value = match.group(0)
        if value.startswith('"') and value.endswith('"'):
            return value
        if value.startswith("'") and value.endswith("'"):
            inner = value[1:-1].replace('"', '\\"')
            return f'"{inner}"'
        return value

    return re.sub(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'', _fix_quotes, content).strip()


def parse_json_response(content: str) -> ParsedJSONResponse:
    """Parse a JSON object from an LLM response.

    The parser accepts common provider wrappers such as Markdown JSON fences,
    leading explanatory labels, trailing commas, single-quoted strings, line
    comments, and optional <think>...</think> reasoning blocks.
    """
    thinking, remaining = extract_think(content)
    repair_log: list[str] = []

    candidates = [remaining.strip()]
    brace_match = re.search(r"\{.*\}", remaining, flags=re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0).strip())

    for candidate in list(candidates):
        cleaned = clean_json_text(candidate)
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            cleaned = clean_json_text(candidate)
            if cleaned == candidate:
                continue
            try:
                parsed = json.loads(cleaned)
                repair_log.append(f"cleaned_json: {exc} -> success")
            except json.JSONDecodeError:
                continue

        if isinstance(parsed, dict):
            return ParsedJSONResponse(result=parsed, thinking=thinking, repair_log=repair_log)

    if "{" in remaining and "}" not in remaining:
        repair_log.append("truncation_detected: JSON starts with { but no closing }")
    if "[" in remaining and "]" not in remaining:
        repair_log.append("truncation_detected: JSON starts with [ but no closing ]")

    return ParsedJSONResponse(result=None, thinking=thinking, repair_log=repair_log)

def fix_json_content(content: str) -> Optional[str]:
    """尝试修复JSON内容"""
    parsed = parse_json_response(content)
    if parsed.result is not None:
        return json.dumps(parsed.result, ensure_ascii=False)

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
    parsed = parse_json_response(content)
    if parsed.result is not None:
        return parsed.result

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
