from dataclasses import dataclass

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class FadeOut(Effect):
    """Makes the clip progressively fade to some color (black by default),
    over ``duration`` seconds at the end of the clip. Can be used for masks too,
    where the final color must be a number between 0 and 1.

    For cross-fading (progressive appearance or disappearance of a clip over another
    clip), see ``CrossFadeOut``
    """

    duration: float
    final_color: list = None

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if clip.duration is None:
            raise ValueError("Attribute 'duration' not set")

        if self.final_color is None:
            self.final_color = 0 if clip.is_mask else [0, 0, 0]

        final_color_tensor = torch.tensor(self.final_color, dtype=torch.float32)

        def filter(get_frame, t):
            if (clip.duration - t) >= self.duration:
                return get_frame(t)
            else:
                frame = get_frame(t)
                fading = 1.0 * (clip.duration - t) / self.duration
                
                # Convert to tensor
                tensor = to_tensor(frame, dtype=torch.float32)
                
                # Move final_color to same device
                fin_color = final_color_tensor.to(tensor.device)
                
                # Apply fading
                result = fading * tensor + (1 - fading) * fin_color
                return to_numpy(result)

        return clip.transform(filter)
