from dataclasses import dataclass
from typing import Union

import torch

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.video.VideoClip import ImageClip
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class MasksOr(Effect):
    """Returns the logical 'or' (maximum pixel color values) between two masks.

    The result has the duration of the clip to which has been applied, if it has any.

    Parameters
    ----------

    other_clip ImageClip or np.ndarray
      Clip used to mask the original clip.

    Examples
    --------

    .. code:: python

        clip = ColorClip(color=(255, 0, 0), size=(1, 1))     # red
        mask = ColorClip(color=(0, 255, 0), size=(1, 1))     # green
        masked_clip = clip.with_effects([vfx.MasksOr(mask)]) # yellow
        masked_clip.get_frame(0)
        [[[255 255   0]]]
    """

    other_clip: Union[Clip, "np.ndarray"]

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        # to ensure that 'or' of two ImageClips will be an ImageClip
        if isinstance(self.other_clip, ImageClip):
            self.other_clip = self.other_clip.img

        if not hasattr(self.other_clip, 'get_frame'):
            # other_clip is a static array
            def filter_func(frame):
                tensor1 = to_tensor(frame)
                tensor2 = to_tensor(self.other_clip)
                result = torch.maximum(tensor1, tensor2)
                return to_numpy(result)
            
            return clip.image_transform(filter_func)
        else:
            # other_clip is a clip with get_frame method
            def filter_func(get_frame, t):
                frame1 = get_frame(t)
                frame2 = self.other_clip.get_frame(t)
                tensor1 = to_tensor(frame1)
                tensor2 = to_tensor(frame2)
                result = torch.maximum(tensor1, tensor2)
                return to_numpy(result)
            
            return clip.transform(filter_func)
