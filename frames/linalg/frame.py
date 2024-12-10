from __future__ import annotations

from functools import cached_property

import torch

from ..abstract import BaseModel
from ..linalg import solve_procrustes
from ..utils.tensor import unsqueeze_like


class Frame(BaseModel, arbitrary_types_allowed=True):
    """A class for representing Frames in a tensor space.

    Frames are sequences of linearly independent vectors, elements of Stiefel manifolds.
    """

    tensor: torch.Tensor  # shape: n (count) x d (dimension) x k (vectors)

    def __len__(self) -> int:
        """Get the length of the concept's frame."""
        return self.tensor.size(-1)

    def __rmul__(self, other: "Frame | torch.Tensor") -> torch.Tensor:
        """Compute the inner product of two concepts (right multiplication).

        Args:
            other: Another Frame object or torch.Tensor to multiply with

        Returns:
            torch.Tensor: The correlation between the two frames
        """
        other = Frame(tensor=other) if not isinstance(other, Frame) else other
        return self.correlation(other.tensor, self.tensor)

    def __mul__(self, other: "Frame") -> torch.Tensor:
        """Compute the inner product of two concepts (left multiplication).

        Args:
            other: Another Frame object to multiply with

        Returns:
            torch.Tensor: The correlation between the two frames
        """
        return other.__rmul__(self)

    def __sub__(self, other: "Frame") -> torch.Tensor:
        """Subtract one concept frame from another.

        Args:
            other: The Frame to subtract from this one

        Returns:
            torch.Tensor: A new tensor representing the orthogonal frame
                         closest to the frames' subtraction
        """
        return self._subtract_frames(self.tensor, other.tensor)

    @cached_property
    def rank(self) -> torch.Tensor:
        """Compute the rank of the concept's frame.

        Returns:
            torch.Tensor: The rank of the frame tensor
        """
        # This is an upper bound for torch.linalg.matrix_rank which is slow
        return self.vector_count(self.tensor)

    @staticmethod
    def vector_count(tensor: torch.Tensor) -> torch.Tensor:
        """Compute the rank of a tensor.

        Args:
            tensor: Input tensor to compute rank for

        Returns:
            torch.Tensor: Upper bound of the matrix rank

        Note:
            This is an upper bound for torch.linalg.matrix_rank which is slow
        """
        return tensor.ne(0).any(dim=-2).sum(-1)

    @classmethod
    def _geometric_mean_rank(
        cls, frame1: torch.Tensor, frame2: torch.Tensor, full_comparison: bool = True
    ) -> torch.Tensor:
        """Compute the geometric mean of the ranks of two frames."""
        rank1 = cls.vector_count(frame1)
        rank2 = cls.vector_count(frame2)

        if full_comparison:
            rank1 = rank1.unsqueeze(-1)
            rank2 = unsqueeze_like(rank2, rank1, direction="left")

        return torch.sqrt(rank1 * rank2)

    @classmethod
    def correlation(cls, frames1: torch.Tensor, frames2: torch.Tensor) -> torch.Tensor:
        """Compute correlation between two sets of frames.

        Args:
            frames1: First set of frames
            frames2: Second set of frames

        Returns:
            torch.Tensor: Correlation matrix between the frame sets
        """
        traces = torch.einsum("...mdk,ndk->...mn", frames1, frames2)
        return traces / cls._geometric_mean_rank(frames1, frames2)

    @classmethod
    def _correlation_diagonal(
        cls, frames1: torch.Tensor, frames2: torch.Tensor
    ) -> torch.Tensor:
        """Compute the correlation between two sets of frames in respective order.

        Useful for speeding up the computation only
        the correlation matrix main diagonal is required.
        """
        traces = torch.einsum("...ndk,...ndk->...n", frames1, frames2)
        return traces / cls._geometric_mean_rank(
            frames1, frames2, full_comparison=False
        )

    def similarity(self, other: Frame | torch.Tensor) -> torch.Tensor:
        # computes similarity (inner product) between list of frames in order
        #  so that the first frame in self is compared to the first frame in other
        #  and so on
        other_tensor = other.tensor if isinstance(other, Frame) else other
        return self._correlation_diagonal(self.tensor, other_tensor)

    @staticmethod
    def _subtract_frames(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
        """Find the orthogonal frame closest to the frames' subtraction."""
        return solve_procrustes(frame1 - frame2)

    @property
    def relative_rank(self) -> torch.Tensor:
        """Compute the matrices' proximity to being full-rank
        as matrix rank / num vectors"""
        return torch.linalg.matrix_rank(self.tensor) / self.rank

    @property
    def device(self) -> torch.device:
        """Get the device of the concept's frame."""
        return self.tensor.device
