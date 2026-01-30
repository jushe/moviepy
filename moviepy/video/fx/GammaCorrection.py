from dataclasses import dataclass

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class GammaCorrection(Effect):
    """Gamma-correction of a video clip."""

    gamma: float

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""

        def filter(im):
            # Convert to tensor, apply gamma correction, return numpy
            tensor = to_tensor(im, dtype=torch.float32)
            corrected = 255 * (tensor / 255) ** self.gamma
            return to_numpy(corrected.to(torch.uint8))

        return clip.image_transform(filter)
