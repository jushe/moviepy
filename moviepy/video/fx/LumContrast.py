from dataclasses import dataclass

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class LumContrast(Effect):
    """Luminosity-contrast correction of a clip."""

    lum: float = 0
    contrast: float = 0
    contrast_threshold: float = 127

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""

        def image_filter(im):
            # Convert to tensor and apply luminosity-contrast correction
            tensor = to_tensor(im, dtype=torch.float32)
            corrected = (
                tensor + self.lum + self.contrast * (tensor - float(self.contrast_threshold))
            )
            corrected = torch.clamp(corrected, 0, 255)
            return to_numpy(corrected.to(torch.uint8))

        return clip.image_transform(image_filter)
