"""
JARVIS Debug Logger
Logs debug information to a file for easy access
"""

import os
import logging
from datetime import datetime
from pathlib import Path

# Debug log directory
DEBUG_LOG_DIR = Path(__file__).parent / "debug_logs"
DEBUG_LOG_DIR.mkdir(exist_ok=True)

# Configure logging
LOG_FILE = DEBUG_LOG_DIR / f"jarvis_debug_{datetime.now().strftime('%Y-%m-%d')}.log"

# Create logger
logger = logging.getLogger("jarvis_debug")
logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# Console handler (also print to console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_debug(message: str, **kwargs):
    """Log debug message with optional context"""
    if kwargs:
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.debug(f"{message} | {context}")
    else:
        logger.debug(message)

def log_info(message: str, **kwargs):
    """Log info message with optional context"""
    if kwargs:
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.info(f"{message} | {context}")
    else:
        logger.info(message)

def log_warning(message: str, **kwargs):
    """Log warning message with optional context"""
    if kwargs:
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.warning(f"{message} | {context}")
    else:
        logger.warning(message)

def log_error(message: str, error: Exception = None, **kwargs):
    """Log error message with optional exception and context"""
    import traceback
    if error:
        error_details = f"{type(error).__name__}: {str(error)}"
        traceback_str = traceback.format_exc()
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            logger.error(f"{message} | {context} | {error_details}\n{traceback_str}")
        else:
            logger.error(f"{message} | {error_details}\n{traceback_str}")
    else:
        if kwargs:
            context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            logger.error(f"{message} | {context}")
        else:
            logger.error(message)

def get_latest_log() -> str:
    """Get the path to the latest log file"""
    return str(LOG_FILE)

def read_latest_log_lines(n: int = 100) -> list:
    """Read the last N lines from the latest log file"""
    try:
        if not os.path.exists(LOG_FILE):
            return [f"Log file does not exist: {LOG_FILE}"]
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n:] if len(lines) > n else lines
    except Exception as e:
        return [f"Error reading log: {e}"]

def get_error_summary(hours: int = 1) -> dict:
    """Get a summary of recent errors and warnings"""
    try:
        lines = read_latest_log_lines(1000)
        from datetime import datetime, timedelta
        import re
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        errors = []
        warnings = []
        
        for line in lines:
            if '[ERROR]' in line:
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        if log_time >= cutoff_time:
                            errors.append(line.strip())
                    except:
                        errors.append(line.strip())
            elif '[WARNING]' in line:
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        if log_time >= cutoff_time:
                            warnings.append(line.strip())
                    except:
                        warnings.append(line.strip())
        
        return {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors[-10:],  # Last 10 errors
            "warnings": warnings[-10:]  # Last 10 warnings
        }
    except Exception as e:
        return {"error": str(e)}

