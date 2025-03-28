import torch
import cuml

def run_umap(tensor: torch.FloatTensor, threshold: float = 1, *args, **kwargs):
    umap = cuml.UMAP(*args, **kwargs)
    embedding = torch.tensor(umap.fit_transform(tensor))

    # filtering outliers
    dists = (embedding - embedding.mean(0)).norm(p=2, dim=1)
    threshold = (dists.mean() + threshold * dists.std())
    outliers = dists > threshold

    return embedding[~outliers, :]