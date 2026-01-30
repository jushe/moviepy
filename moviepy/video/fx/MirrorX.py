from dataclasses import dataclass
from typing import List, Union

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class MirrorX(Effect):
    """Flips the clip horizontally (and its mask too, by default)."""

    apply_to: Union[List, str] = "mask"

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        
        def mirror_func(frame):
            # Convert to tensor, flip horizontally, return numpy
            tensor = to_tensor(frame)
            flipped = torch.flip(tensor, dims=[1])
            return to_numpy(flipped)
        
        return clip.image_transform(mirror_func, apply_to=self.apply_to)
