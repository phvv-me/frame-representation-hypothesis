from functools import cached_property
from typing import List, Literal

import torch
from torch import Tensor

from ..abstract import BaseModel
from ..linalg import symsqrtinv
from ..models.hf import BaseHuggingFaceModel


class LinearUnembeddingRepresentation(BaseModel, arbitrary_types_allowed=True):
    """
    Basic manipulation of linear unembedding representations from a HuggingFace model.
    """

    model: BaseHuggingFaceModel

    meaningless_vector: Literal["pad", "mean"] | None = "mean"
    normalize_token_representations: bool = False
    set_padding_token_to_zero: bool = True

    @cached_property
    def _cov_unembedding_matrix(self) -> Tensor:
        """
        Compute the covariance matrix of the transposed unembedding matrix.

        Returns:
            Tensor: Covariance matrix of the transposed unembedding matrix.
        """
        return self.unembedding_matrix.mT.cov()

    @property
    def _meaningless(self) -> Tensor:
        """
        Get the meaningless vector based on the specified option.

        Returns:
            Tensor: The meaningless vector (pad, mean, or zero).
        """
        return self._get_meaningless_vector()

    def _get_meaningless_vector(self) -> int | Tensor:
        """
        Helper method to get the appropriate meaningless vector.

        Returns:
            Tensor: The selected meaningless vector.
        """
        match self.meaningless_vector:
            case "pad":
                return self.pad_vector
            case "mean":
                return self.mean_vector
            case _:
                return 0

    @cached_property
    def _euclidean_representations(self) -> Tensor:
        """
        Compute Euclidean representations of the unembedding matrix.

        Returns:
            Tensor: Euclidean representations.
        """
        return self.unembedding_matrix @ self._inv_sqrt_cov_unembedding_matrix

    @cached_property
    def _inv_sqrt_cov_unembedding_matrix(self) -> Tensor:
        """
        Compute the inverse square root of the covariance unembedding matrix.

        Returns:
            Tensor: Inverse square root of the covariance unembedding matrix.
        """
        return symsqrtinv(self._cov_unembedding_matrix)

    @cached_property
    def _token_representations(self) -> Tensor:
        """
        Compute token representations.

        Returns:
            Tensor: Token representations.
        """
        reprs = self._euclidean_representations - self._meaningless
        return self._apply_token_representation_options(reprs).to(self.model.dtype)

    def _apply_token_representation_options(self, reprs: Tensor) -> Tensor:
        """
        Apply optional transformations to token representations.

        Args:
            tokens (Tensor): Base token representations.

        Returns:
            Tensor: Transformed token representations.
        """
        if self.normalize_token_representations:
            reprs = self.normalize(reprs)
        if self.set_padding_token_to_zero:
            reprs[self.model._tokenizer.pad_token_id] = 0
        if reprs.isnan().any():
            raise ValueError("NaN values detected in token representations.")
        return reprs

    def get_token_representations(self, tokens: List[int]) -> Tensor:
        """
        Get token representations for given token IDs.

        Args:
            tokens (List[int]): List of token IDs.

        Returns:
            Tensor: Token representations for the given tokens.
        """
        return self._token_representations[tokens]

    @cached_property
    def mean_vector(self) -> Tensor:
        """
        Compute the mean vector of Euclidean representations.

        Returns:
            Tensor: Mean vector.
        """
        return self._euclidean_representations.mean(dim=0)

    def normalize(self, vectors: Tensor) -> Tensor:
        """
        Normalize the input vectors.

        Args:
            vectors (Tensor): Input vectors to normalize.

        Returns:
            Tensor: Normalized vectors.
        """
        return torch.nn.functional.normalize(vectors, p=2, dim=-1)

    @cached_property
    def pad_vector(self) -> Tensor:
        """
        Get the pad vector from Euclidean representations.

        Returns:
            Tensor: Pad vector.
        """
        return self._euclidean_representations[self.model._tokenizer.pad_token_id]

    @cached_property
    def unembedding_matrix(self) -> Tensor:
        """
        Get the unembedding matrix from the model.

        Returns:
            Tensor: Unembedding matrix.
        """
        return self.model.unembedding_matrix.float()
