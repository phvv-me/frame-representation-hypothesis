import torch


def gram_schmidt(
    vectors: torch.Tensor, dim: int = -1, eps: float = 1e-3, *args, **kwargs
):
    """
    Gram-Schmidt orthogonalization process for a tensor of arbitrary shape.

    Args:
        vectors (torch.Tensor): Input tensor of shape [..., n, d], where n is the number of vectors
            and d is the dimension of each vector.
        dim (int): The dimension along which to perform orthogonalization. Default is -1.
        *args: Additional arguments to be passed to the normalize function.
        **kwargs: Additional keyword arguments to be passed to the normalize function.

    Returns:
        torch.Tensor: Orthogonalized tensor of the same shape as the input.
    """

    # Move the dimension to be orthogonalized to the last dimension
    vectors = vectors.transpose(dim, -1)

    n = vectors.size(-2)
    Q = torch.zeros_like(vectors)

    for i in range(n):
        v = vectors[..., i, :]

        if i > 0:
            U = Q[..., :i, :]
            v -= (U @ v.unsqueeze(-2).mT * U).sum(dim=-2)

        norm = v.norm(dim=-1, keepdim=True, *args, **kwargs)
        Q[..., i, :] = torch.where(norm > eps, v / norm, 0)

    # Move the dimension back to its original position
    return Q.transpose(dim, -1)


def solve_procrustes(vectors: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(vectors.float(), full_matrices=False)
    return (U @ Vh).to(vectors.dtype)
