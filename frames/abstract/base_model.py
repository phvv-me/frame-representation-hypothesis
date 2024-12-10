"""Base model extending Pydantic with additional functionality and configuration.

This module provides an enhanced version of Pydantic's BaseModel with extra features
like hash support and special type handling.
"""

import functools
import hashlib
from typing import Any, TypeVar

from pydantic import BaseModel as OriginalBaseModel
from wirerope.rope import RopeCore

T = TypeVar("T")


class BaseModel(OriginalBaseModel):
    """Enhanced Pydantic BaseModel with additional functionalities.

    Features:
        - Forbids extra attributes by default
        - Handles special types in configuration (cached_property, lru_cache, RopeCore)
        - Implements consistent hashing based on model content

    Example:
        ```python
        class User(BaseModel):
            name: str
            age: int

        user = User(name="John", age=30)
        hash(user)  # Returns consistent hash based on content
        ```
    """

    class Config:
        extra: str = "forbid"
        ignored_types: tuple[Any, ...] = (
            functools.cached_property,
            functools._lru_cache_wrapper,
            RopeCore,
        )

    def __hash__(self) -> int:
        """Generate a consistent hash based on the model's content.

        Returns:
            int: A 64-bit hash value derived from SHA1 hash of the model's JSON representation.
        """
        sha1_hash = hashlib.sha1(self.model_dump_json().encode())
        # Convert the first 8 bytes of the hash to an integer
        return int.from_bytes(sha1_hash.digest()[:8], byteorder="big")
