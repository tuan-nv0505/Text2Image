import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logger(logger_name: str = "ai_system", log_level: int = logging.INFO, log_file: str = "logs/app.log") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)
    log_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | [%(filename)s:%(funcName)s:%(lineno)d] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()
