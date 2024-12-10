from __future__ import annotations

from ..linalg import Frame


class Concept(Frame):
    """A single or set of concept frames."""

    synset: str | list[str]

    def __getitem__(self, synset: str) -> Concept:
        """Get a single concept by its synset."""
        idx = self._find_synset_index(synset)
        return Concept(synset=synset, tensor=self.tensor[[idx]])

    def __sub__(self, other: Concept) -> Concept:
        """Subtract one concept from another."""
        return Concept(
            synset=" - ".join([self.synset, other.synset]),
            tensor=super().__sub__(other),
        )

    def __str__(self) -> str:
        """Get a string representation of the concept."""
        count, dimension, rank = self.tensor.shape
        if count == 1:
            name = self.synset
            return f"{self.__class__.__name__}({name=}, {dimension=}, {rank=})"
        return f"{self.__class__.__name__}({count=}, {dimension=}, {rank=})"

    @property
    def name(self) -> str:
        """Get the name of the concept."""
        return self.synset

    def _find_synset_index(self, synset: str) -> int:
        """Find the index of a synset in the dataframe."""
        if isinstance(self.synset, str):
            raise ValueError("Concept has only one synset.")
        return self.synset.index(synset)
