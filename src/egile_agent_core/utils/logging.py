"""Logging configuration for Egile Agent Core."""

import logging
import sys
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def setup_logging(
    level: LogLevel = "INFO",
    format_string: str | None = None,
    stream: bool = True,
) -> None:
    """
    Set up logging for the Egile Agent Core.

    Args:
        level: Logging level.
        format_string: Custom format string. Uses default if None.
        stream: If True, log to stdout.
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Get the root egile logger
    logger = logging.getLogger("egile_agent_core")
    logger.setLevel(getattr(logging, level))

    # Clear existing handlers
    logger.handlers.clear()

    if stream:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the egile_agent_core prefix.

    Args:
        name: Logger name suffix.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"egile_agent_core.{name}")
