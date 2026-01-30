import math
from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF

from moviepy.Clip import Clip
from moviepy.Effect import Effect
from moviepy.torch_utils import to_tensor, to_numpy


@dataclass
class Rotate(Effect):
    """
    Rotates the specified clip by ``angle`` degrees (or radians) anticlockwise
    If the angle is not a multiple of 90 (degrees) or ``center``, ``translate``,
    and ``bg_color`` are not ``None``, there will be black borders.
    You can make them transparent with:

    >>> new_clip = clip.with_mask().rotate(72)

    Parameters
    ----------

    clip : VideoClip
    A video clip.

    angle : float
    Either a value or a function angle(t) representing the angle of rotation.

    unit : str, optional
    Unit of parameter `angle` (either "deg" for degrees or "rad" for radians).

    resample : str, optional
    An optional resampling filter. One of "nearest", "bilinear", or "bicubic".

    expand : bool, optional
    If true, expands the output image to make it large enough to hold the
    entire rotated image. If false or omitted, make the output image the same
    size as the input image.

    translate : tuple, optional
    An optional post-rotate translation (a 2-tuple).

    center : tuple, optional
    Optional center of rotation (a 2-tuple). Origin is the upper left corner.

    bg_color : tuple, optional
    An optional color for area outside the rotated image. Only has effect if
    ``expand`` is true.
    """

    angle: float
    unit: str = "deg"
    resample: str = "bicubic"
    expand: bool = True
    center: tuple = None
    translate: tuple = None
    bg_color: tuple = None

    def apply(self, clip: Clip) -> Clip:
        """Apply the effect to the clip."""
        if hasattr(self.angle, "__call__"):
            get_angle = self.angle
        else:
            get_angle = lambda t: self.angle

        def filter(get_frame, t):
            angle = get_angle(t)
            im = get_frame(t)

            if self.unit == "rad":
                angle = math.degrees(angle)

            angle %= 360
            
            # Fast path for 90-degree rotations without special options
            if not self.center and not self.translate and not self.bg_color:
                if (angle == 0) and self.expand:
                    return im
                
                # Use torch for fast 90-degree rotations
                tensor = to_tensor(im)
                if (angle == 90) and self.expand:
                    # Rotate 90 degrees counterclockwise
                    result = torch.rot90(tensor, k=1, dims=[0, 1])
                    return to_numpy(result)
                elif (angle == 270) and self.expand:
                    # Rotate 270 degrees counterclockwise (or 90 clockwise)
                    result = torch.rot90(tensor, k=-1, dims=[0, 1])
                    return to_numpy(result)
                elif (angle == 180) and self.expand:
                    # Rotate 180 degrees
                    result = torch.rot90(tensor, k=2, dims=[0, 1])
                    return to_numpy(result)

            # For arbitrary angles, use torchvision's affine transformation
            tensor = to_tensor(im)
            
            # Handle mask images (float64)
            is_mask = im.dtype == "float64"
            if is_mask:
                tensor = tensor * 255.0
            
            # Convert from (H, W, C) to (C, H, W) for torchvision
            if tensor.ndim == 3:
                tensor = tensor.permute(2, 0, 1)
            
            # Rotate using torchvision
            # Note: torchvision rotates clockwise, so we negate the angle
            rotated = TF.rotate(
                tensor, 
                -angle,  # Negate for counterclockwise rotation
                interpolation=TF.InterpolationMode.BILINEAR,
                expand=self.expand,
                fill=list(self.bg_color) if self.bg_color else [0]
            )
            
            # Convert back to (H, W, C) format
            if rotated.ndim == 3:
                rotated = rotated.permute(1, 2, 0)
            
            # Convert back from mask format if needed
            if is_mask:
                rotated = rotated / 255.0
            
            return to_numpy(rotated)

        return clip.transform(filter, apply_to=["mask"])
