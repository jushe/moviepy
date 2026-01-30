import numbers
from dataclasses import dataclass
from typing import Union

import torch
import torchvision.transforms.functional as TF

from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class Resize(Effect):
    """Effect returning a video clip that is a resized version of the clip.

    Parameters
    ----------

    new_size : tuple or float or function, optional
        Can be either
        - ``(width, height)`` in pixels or a float representing
        - A scaling factor, like ``0.5``.
        - A function of time returning one of these.

    height : int, optional
        Height of the new clip in pixels. The width is then computed so
        that the width/height ratio is conserved.

    width : int, optional
        Width of the new clip in pixels. The height is then computed so
        that the width/height ratio is conserved.

    Examples
    --------

    .. code:: python

        clip.with_effects([vfx.Resize((460,720))]) # New resolution: (460,720)
        clip.with_effects([vfx.Resize(0.6)]) # width and height multiplied by 0.6
        clip.with_effects([vfx.Resize(width=800)]) # height computed automatically.
        clip.with_effects([vfx.Resize(lambda t : 1+0.02*t)]) # slow clip swelling
    """

    new_size: Union[tuple, float, callable] = None
    height: int = None
    width: int = None
    apply_to_mask: bool = True

    def resizer(self, pic, new_size):
        """Resize the image using PyTorch."""
        new_size = list(map(int, new_size))
        # Convert to tensor if needed
        tensor = to_tensor(pic)
        
        # Ensure tensor is in (C, H, W) format for torchvision
        if tensor.ndim == 3:
            # Convert from (H, W, C) to (C, H, W)
            tensor = tensor.permute(2, 0, 1)
        
        # Resize using torch - new_size is (width, height) but resize expects (height, width)
        resized = TF.resize(tensor, [new_size[1], new_size[0]], interpolation=TF.InterpolationMode.BILINEAR)
        
        # Convert back to (H, W, C) format
        if resized.ndim == 3:
            resized = resized.permute(1, 2, 0)
        
        return to_numpy(resized)

    def apply(self, clip):
        """Apply the effect to the clip."""
        w, h = clip.size

        if self.new_size is not None:

            def translate_new_size(new_size_):
                """Returns a [w, h] pair from `new_size_`. If `new_size_` is a
                scalar, then work out the correct pair using the clip's size.
                Otherwise just return `new_size_`
                """
                if isinstance(new_size_, numbers.Number):
                    return [new_size_ * w, new_size_ * h]
                else:
                    return new_size_

            if hasattr(self.new_size, "__call__"):
                # The resizing is a function of time

                def get_new_size(t):
                    return translate_new_size(self.new_size(t))

                if clip.is_mask:

                    def filter(get_frame, t):
                        return (
                            self.resizer(
                                (255 * get_frame(t)).astype("uint8"), get_new_size(t)
                            )
                            / 255.0
                        )

                else:

                    def filter(get_frame, t):
                        return self.resizer(
                            get_frame(t).astype("uint8"), get_new_size(t)
                        )

                newclip = clip.transform(
                    filter,
                    keep_duration=True,
                    apply_to=(["mask"] if self.apply_to_mask else []),
                )
                if self.apply_to_mask and clip.mask is not None:
                    newclip.mask = clip.mask.with_effects(
                        [Resize(self.new_size, apply_to_mask=False)]
                    )

                return newclip

            else:
                self.new_size = translate_new_size(self.new_size)

        elif self.height is not None:
            if hasattr(self.height, "__call__"):

                def func(t):
                    return 1.0 * int(self.height(t)) / h

                return clip.with_effects([Resize(func)])

            else:
                self.new_size = [w * self.height / h, self.height]

        elif self.width is not None:
            if hasattr(self.width, "__call__"):

                def func(t):
                    return 1.0 * self.width(t) / w

                return clip.with_effects([Resize(func)])

            else:
                self.new_size = [self.width, h * self.width / w]
        else:
            raise ValueError(
                "You must provide either 'new_size' or 'height' or 'width'"
            )

        # From here, the resizing is constant (not a function of time), size=newsize

        if clip.is_mask:

            def image_filter(pic):
                return (
                    1.0
                    * self.resizer((255 * pic).astype("uint8"), self.new_size)
                    / 255.0
                )

        else:

            def image_filter(pic):
                return self.resizer(pic.astype("uint8"), self.new_size)

        new_clip = clip.image_transform(image_filter)

        if self.apply_to_mask and clip.mask is not None:
            new_clip.mask = clip.mask.with_effects(
                [Resize(self.new_size, apply_to_mask=False)]
            )

        return new_clip
