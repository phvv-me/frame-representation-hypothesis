import socket
from functools import wraps
from itertools import islice
from typing import Any, Callable, Iterable, Iterator, List, TypeVar

T = TypeVar("T")


def mkdir(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that ensures the directory path returned by the wrapped function exists.

    Args:
        func: Function that returns a Path-like object

    Returns:
        Wrapper function that creates directory before returning path
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        path = func(*args, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    return wrapper


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[List[T]]:
    """Split an iterable into batches of specified size.

    Args:
        iterable: Input iterable to be batched
        batch_size: Size of each batch

    Returns:
        Iterator yielding lists of batch_size items

    Raises:
        ValueError: If batch_size is less than 1
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def batchedlist(iterable: Iterable[T], batch_size: int) -> List[List[T]]:
    """Convert an iterable into a list of batches.

    Args:
        iterable: Input iterable to be batched
        batch_size: Size of each batch

    Returns:
        List of lists, where each inner list has batch_size items
    """
    return list(batched(iterable, batch_size))


def is_online() -> bool:
    """Check if internet connection is available.

    Tests connection by attempting to reach Google's DNS server (8.8.8.8).

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False
