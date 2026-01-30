from dataclasses import dataclass

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class MultiplyColor(Effect):
    """
    Multiplies the clip's colors by the given factor, can be used
    to decrease or increase the clip's brightness (is that the
    right word ?)
    """

    factor: float

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        
        def multiply_func(frame):
            # Convert to tensor, perform multiplication with clipping, return numpy
            tensor = to_tensor(frame, dtype=torch.float32)
            result = torch.clamp(self.factor * tensor, 0, 255)
            return to_numpy(result.to(torch.uint8))
        
        return clip.image_transform(multiply_func)
