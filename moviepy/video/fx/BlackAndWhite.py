from dataclasses import dataclass

import torch

from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class BlackAndWhite(Effect):
    """Desaturates the picture, makes it black and white.
    Parameter RGB allows to set weights for the different color
    channels.
    If RBG is 'CRT_phosphor' a special set of values is used.
    preserve_luminosity maintains the sum of RGB to 1.
    """

    RGB: str = None
    preserve_luminosity: bool = True

    def apply(self, clip):
        """Apply the effect to the clip."""
        if self.RGB is None:
            self.RGB = [1, 1, 1]

        if self.RGB == "CRT_phosphor":
            self.RGB = [0.2125, 0.7154, 0.0721]

        rgb_weights = torch.tensor(self.RGB, dtype=torch.float32)
        if self.preserve_luminosity:
            rgb_weights = rgb_weights / rgb_weights.sum()

        def filter(im):
            # Convert to tensor
            tensor = to_tensor(im, dtype=torch.float32)
            
            # Move weights to same device as tensor
            weights = rgb_weights.to(tensor.device)
            
            # Apply weighted sum across color channels
            gray = (
                weights[0] * tensor[:, :, 0]
                + weights[1] * tensor[:, :, 1]
                + weights[2] * tensor[:, :, 2]
            )
            
            # Stack to create 3-channel grayscale image
            result = torch.stack([gray, gray, gray], dim=2)
            return to_numpy(result.to(torch.uint8))

        return clip.image_transform(filter)
