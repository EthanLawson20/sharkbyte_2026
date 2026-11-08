from pathlib import Path

import torch
from torch import Tensor

from lerobot.policies.act import ACTConfig, ACTPolicy


class ACTModel(ACTPolicy):
    """LeRobot's Action Chunking Transformer policy for a configured robot dataset."""

    def __init__(self, config: ACTConfig) -> None:
        super().__init__(config)

    def predict(self, observation: dict[str, Tensor]) -> Tensor:
        """Return the next action from a preprocessed, batched LeRobot observation."""
        with torch.inference_mode():
            return self.select_action(observation)

    def save(self, path: str | Path) -> None:
        """Save the policy weights and configuration in LeRobot checkpoint format."""
        self.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path) -> "ACTModel":
        """Load a policy checkpoint saved by LeRobot."""
        return cls.from_pretrained(path)
