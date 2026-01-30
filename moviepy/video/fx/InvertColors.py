from dataclasses import dataclass

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class InvertColors(Effect):
    """Returns the color-inversed clip.

    The values of all pixels are replaced with (255-v) or (1-v) for masks
    Black becomes white, green becomes purple, etc.
    """

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        maxi = 1.0 if clip.is_mask else 255
        
        def invert_func(frame):
            # Convert to tensor, perform inversion, return numpy for compatibility
            tensor = to_tensor(frame)
            inverted = maxi - tensor
            return to_numpy(inverted)
        
        return clip.image_transform(invert_func)
