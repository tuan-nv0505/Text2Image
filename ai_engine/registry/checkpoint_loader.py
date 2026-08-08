from pathlib import Path
from typing import Any

import logging
import torch

logger = logging.getLogger(__name__)


class CheckpointLoader:
    @staticmethod
    def load(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        if not path.is_file():
            raise ValueError(f"{path} is not a file.")
        logger.info("Loading checkpoint: %s", path)
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        return checkpoint