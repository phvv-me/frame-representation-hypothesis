from __future__ import annotations

from functools import reduce
from typing import (
    Iterable,
    Iterator,
    Literal,
)

import torch


def unsqueeze_like(
    tensor: torch.Tensor,
    like: torch.Tensor,
    direction: Literal["left", "right"] = "right",
) -> torch.Tensor:
    """
    Unsqueeze dimensions of tensor to match another tensor's number of dimensions.

    Args:
    tensor (torch.Tensor): tensor to unsqueeze
    like (torch.Tensor): tensor whose dimensions to match
    direction (Literal['left', 'right']): direction to add new dimensions.
                                          'left' adds at the beginning, 'right' adds at the end.

    Returns:
    torch.Tensor: Unsqueezed tensor

    Raises:
    ValueError: If tensor has more dimensions than 'like' tensor
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    elif direction == "right":
        return tensor[(...,) + (None,) * n_unsqueezes]
    elif direction == "left":
        return tensor[(None,) * n_unsqueezes + (...,)]
    else:
        raise ValueError("direction must be either 'left' or 'right'")


def cat(tensors: Iterator[torch.Tensor], dim=0) -> torch.Tensor:
    """
    Concatenates a sequence of tensors along a specified dimension.

    Args:
        tensors (Iterator[torch.Tensor]): An iterator of tensors to concatenate.
        dim (int): The dimension along which to concatenate the tensors. Default is 0.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    return reduce(lambda x, y: torch.cat((x, y), dim=dim), tensors)


def stack(tensors: Iterable | torch.Tensor, dim=0) -> torch.Tensor:
    """
    Converts a nested tuple structure of tensors into a single tensor along a specified dimension.

    Args:
        tensors (Union[Tuple, torch.Tensor]): A nested tuple structure or a tensor to stack.
        dim (int): The dimension along which to stack the tensors. Default is 0.

    Returns:
        torch.Tensor: The stacked tensor.
    """
    if isinstance(tensors, torch.Tensor):
        return tensors

    # Recursively stack tensors
    return torch.stack([stack(t, dim) for t in tensors], dim=dim)
