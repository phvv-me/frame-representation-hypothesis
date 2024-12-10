import contextlib
import gc

import torch


def garbage_collection_cuda():
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def gc_cuda():
    """Context manager to garbage collect Torch (CUDA) memory."""
    garbage_collection_cuda()

    try:
        yield
    finally:
        garbage_collection_cuda()
