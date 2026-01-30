"""Utilities for PyTorch tensor operations and conversions for CUDA acceleration."""

import numpy as np
import torch


# Global device configuration
_USE_CUDA = torch.cuda.is_available()
_DEVICE = torch.device("cuda" if _USE_CUDA else "cpu")


def set_device(device):
    """Set the device for torch operations.
    
    Parameters
    ----------
    device : str or torch.device
        The device to use ('cuda', 'cpu', or a torch.device object).
    """
    global _DEVICE, _USE_CUDA
    if isinstance(device, str):
        _DEVICE = torch.device(device)
    else:
        _DEVICE = device
    _USE_CUDA = _DEVICE.type == "cuda"


def get_device():
    """Get the current device for torch operations.
    
    Returns
    -------
    torch.device
        The current device.
    """
    return _DEVICE


def use_cuda():
    """Check if CUDA is being used.
    
    Returns
    -------
    bool
        True if CUDA is available and enabled.
    """
    return _USE_CUDA


def to_tensor(array, dtype=None):
    """Convert a numpy array to a torch tensor on the configured device.
    
    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Input array or tensor.
    dtype : torch.dtype, optional
        The desired data type of the tensor.
    
    Returns
    -------
    torch.Tensor
        Tensor on the configured device.
    """
    if isinstance(array, torch.Tensor):
        tensor = array
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if tensor.device != _DEVICE:
            tensor = tensor.to(_DEVICE)
        return tensor
    
    if dtype is None:
        # Infer dtype from numpy array
        if array.dtype == np.uint8:
            dtype = torch.uint8
        elif array.dtype == np.float32:
            dtype = torch.float32
        elif array.dtype == np.float64:
            dtype = torch.float64
        else:
            dtype = torch.float32
    
    return torch.from_numpy(array).to(device=_DEVICE, dtype=dtype)


def to_numpy(tensor):
    """Convert a torch tensor to a numpy array.
    
    Parameters
    ----------
    tensor : torch.Tensor or np.ndarray
        Input tensor or array.
    
    Returns
    -------
    np.ndarray
        Numpy array on CPU.
    """
    if isinstance(tensor, np.ndarray):
        return tensor
    
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    
    return tensor.numpy()


def ensure_tensor(array_or_tensor, dtype=None):
    """Ensure input is a torch tensor on the configured device.
    
    This is an alias for to_tensor() for clarity in code.
    
    Parameters
    ----------
    array_or_tensor : np.ndarray or torch.Tensor
        Input array or tensor.
    dtype : torch.dtype, optional
        The desired data type of the tensor.
    
    Returns
    -------
    torch.Tensor
        Tensor on the configured device.
    """
    return to_tensor(array_or_tensor, dtype=dtype)


def ensure_numpy(array_or_tensor):
    """Ensure input is a numpy array.
    
    This is an alias for to_numpy() for clarity in code.
    
    Parameters
    ----------
    array_or_tensor : np.ndarray or torch.Tensor
        Input array or tensor.
    
    Returns
    -------
    np.ndarray
        Numpy array on CPU.
    """
    return to_numpy(array_or_tensor)


def process_frame_torch(frame, func, return_numpy=False):
    """Process a frame using a torch function.
    
    Parameters
    ----------
    frame : np.ndarray or torch.Tensor
        Input frame (H, W, C) format.
    func : callable
        Function that takes a torch tensor and returns a torch tensor.
    return_numpy : bool, optional
        If True, return a numpy array; otherwise return a tensor.
    
    Returns
    -------
    np.ndarray or torch.Tensor
        Processed frame.
    """
    tensor = to_tensor(frame)
    result = func(tensor)
    
    if return_numpy:
        return to_numpy(result)
    return result
