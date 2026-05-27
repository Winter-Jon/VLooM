import logging
import json
from typing import Any, Dict, List, Optional
from datetime import datetime

class LifeCycleLogger:
    """
    Handles structured logging for the agent lifecycle.
    Outputs formatted logs for questions, responses, and tool usage.
    Designed to be used in both production and dry_run modes.
    """
    def __init__(self, dry_run: bool = False, log_interval: int = 10):
        self.dry_run = dry_run
        self.log_interval = log_interval
        self.logger = logging.getLogger("lifecycle")
        self.cycle_count = 0
        from threading import Lock
        self._lock = Lock()

    def _separator(self, char: str = "-", length: int = 60):
        self.logger.info(char * length)

    def log_cycle_start(self, task_name: str, item_id: str) -> bool:
        with self._lock:
            self.cycle_count += 1
            current_count = self.cycle_count
            
        # Log first 5, then every 'log_interval'
        should_log = (current_count <= 5) or (current_count % self.log_interval == 0)
        
        if should_log:
            self.logger.info(f"\n{'='*20} CYCLE START: {task_name} (ID: {item_id}) [Step {current_count}] {'='*20}")
            
        return should_log

    def log_question(self, messages: List[Dict[str, Any]], should_log: bool = True):
        if not should_log: 
            return

        self.logger.info("\n>>> [QUESTION]")
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if isinstance(content, list):
                # Handle list content (text + image)
                log_content = ""
                for part in content:
                    if part.get("type") == "text":
                        log_content += f"{part.get('text', '')}"
                    elif part.get("type") == "image_url":
                        url = part.get('image_url', {}).get('url', '')
                        log_content += f"\n[IMAGE] {url[:50]}..."
                self.logger.info(f"[{role.upper()}]: {log_content}")
            else:
                self.logger.info(f"[{role.upper()}]: {content}")
        self._separator()

    def log_response(self, model: str, content: str, thinking: Optional[str] = None, should_log: bool = True):
        if not should_log:
            return

        self.logger.info(f"\n<<< [RESPONSE] ({model})")
        if thinking:
             self.logger.info(f"[THINKING]:\n{thinking}\n")
        
        self.logger.info(f"[CONTENT]:\n{content}")
        self._separator()

    def log_tool_use(self, tool_name: str, args: Dict[str, Any], should_log: bool = True):
        if not should_log:
            return
            
        self.logger.info(f"\n[TOOL EXECUTION] {tool_name}")
        self.logger.info(f"Args: {json.dumps(args, ensure_ascii=False, indent=2)}")
        self._separator()

    def log_tool_result(self, result: Any, should_log: bool = True):
        if not should_log:
            return
            
        self.logger.info(f"\n[TOOL RESULT]")
        self.logger.info(str(result))
        self._separator()

    def log_cycle_end(self, item_id: str, should_log: bool = True):
        if should_log:
            self.logger.info(f"{'='*20} CYCLE END: {item_id} {'='*20}\n")
