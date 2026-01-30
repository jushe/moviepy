from dataclasses import dataclass

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class FadeIn(Effect):
    """Makes the clip progressively appear from some color (black by default),
    over ``duration`` seconds at the beginning of the clip. Can be used for
    masks too, where the initial color must be a number between 0 and 1.

    For cross-fading (progressive appearance or disappearance of a clip
    over another clip, see ``CrossFadeIn``
    """

    duration: float
    initial_color: list = None

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if self.initial_color is None:
            self.initial_color = 0 if clip.is_mask else [0, 0, 0]

        initial_color_tensor = torch.tensor(self.initial_color, dtype=torch.float32)

        def filter(get_frame, t):
            if t >= self.duration:
                return get_frame(t)
            else:
                frame = get_frame(t)
                fading = 1.0 * t / self.duration
                
                # Convert to tensor
                tensor = to_tensor(frame, dtype=torch.float32)
                
                # Move initial_color to same device
                init_color = initial_color_tensor.to(tensor.device)
                
                # Apply fading
                result = fading * tensor + (1 - fading) * init_color
                return to_numpy(result)

        return clip.transform(filter)
