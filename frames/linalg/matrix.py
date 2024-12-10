import torch


def symsqrtinv(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute the inverse of the square root of a Symmetric or
    Hermitian positive definite matrix or batch of matrices

    Args:
    matrix: A symmetric or Hermitian positive definite matrix or batch of matrices
    eps: A small number for numerical stability

    Returns:
    The inverse of the square root of the input matrix
    """
    L, V = torch.linalg.eigh(matrix)
    L_inv_sqrt = torch.where(L > eps, L.rsqrt(), 0).diag()
    return V @ L_inv_sqrt @ V.mH
