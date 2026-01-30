from dataclasses import dataclass

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class MaskColor(Effect):
    """Returns a new clip with a mask for transparency where the original
    clip is of the given color.

    You can also have a "progressive" mask by specifying a non-null distance
    threshold ``threshold``. In this case, if the distance between a pixel and
    the given color is d, the transparency will be

    d**stiffness / (threshold**stiffness + d**stiffness)

    which is 1 when d>>threshold and 0 for d<<threshold, the stiffness of the
    effect being parametrized by ``stiffness``
    """

    color: tuple = (0, 0, 0)
    threshold: float = 0
    stiffness: float = 1

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        color_tensor = torch.tensor(self.color, dtype=torch.float32)

        def hill(x):
            if self.threshold:
                return x**self.stiffness / (
                    self.threshold**self.stiffness + x**self.stiffness
                )
            else:
                return 1.0 * (x != 0)

        def flim(im):
            # Convert to tensor
            tensor = to_tensor(im, dtype=torch.float32)
            
            # Move color to same device
            color = color_tensor.to(tensor.device)
            
            # Calculate euclidean distance from color
            diff = tensor - color
            distance = torch.sqrt((diff ** 2).sum(dim=2))
            
            # Apply hill function
            result = hill(distance)
            return to_numpy(result)

        mask = clip.image_transform(flim)
        mask.is_mask = True
        return clip.with_mask(mask)
