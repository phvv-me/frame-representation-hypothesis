from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cudf
import numpy as np
import pandas as pd
import pycountry
import transformers
from methodtools import lru_cache
from nltk.corpus import wordnet2021 as wn2021
from nltk.corpus.reader import Lemma, Synset

from ..abstract import BaseModel

wn2021.add_exomw()


class SupportedLanguages(tuple, Enum):
    """LLM supported languages for WordNet synsets and lemmas."""

    # Gemma / Phi Technical papers indicate they are more reliable in english,
    # and we confirmed that using more languages can lead more easily to harmful outputs
    Default = "eng"

    # Languages supported by Llama 3.1
    Llama = ("deu_wikt", "eng", "fra", "hin_wikt", "ita_iwn", "por", "spa", "tha")

    @staticmethod
    def _get_language_from_code(code: str):
        return pycountry.languages.get(alpha_3=code.split("_")[0]).name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(map(self._get_language_from_code, self))

    @classmethod
    def from_model_id(cls, model_id: str):
        for group in cls:
            if group.name.lower() in model_id.lower():
                return group
        return cls.Default


class MultiLingualWordNetSynsets(BaseModel):
    padding: str = "max_length"
    truncation: bool = False
    max_length: int = 16 + 1
    pad_to_max_length: bool = True

    language_codes: SupportedLanguages = SupportedLanguages.Default

    # TODO: Add more variations
    variations: Optional[Tuple[str, ...]] = (" {term}", "{term} ", " {term} ")

    @property
    def languages(self):
        return self.language_codes.names

    @property
    def wordnet(self):
        return wn2021

    def get_all_lemmas(self, synset: Synset) -> List[Lemma]:
        """
        Retrieve all lemmas from all languages for a given synset.

        Args:
            synset (Synset): The synset for which to retrieve lemmas.

        Returns:
            List[Lemma]: A list of lemmas from all languages for the given synset.
        """
        return [
            lemma
            for lang in self.wordnet.langs()
            for lemma in synset.lemmas(lang)
            if lang in self.language_codes
        ]

    @staticmethod
    def get_all_antonyms(lemma: Lemma) -> List[str]:
        """
        Retrieve all antonyms for a given lemma.

        Args:
            lemma (Lemma): The lemma for which to retrieve antonyms.

        Returns:
            List[str]: A list of antonym names for the given lemma.
        """
        return [antonym.name() for antonym in lemma.antonyms()]

    @property
    def all_term_variations(self) -> Set[str]:
        """
        Get all term variations including the default one.

        Returns:
            Set[str]: A set of all term variations.
        """
        all_variations = set(self.variations if self.variations else ())
        all_variations.add("{term}")
        return all_variations

    def get_lemma_variations(self, lemma: str, variations: list[str]) -> List[str]:
        """
        Generate variations of a lemma name considering different spacings.

        Args:
            lemma (Lemma): The lemma for which to generate name variations.

        Returns:
            List[str]: A list of name variations for the given lemma.
        """
        return [variation.format(term=lemma.strip()) for variation in variations]

    def get_token_mapping(
        self, column: pd.Series, tokenizer: transformers.PreTrainedTokenizer
    ) -> Dict[str, List[int]]:
        """
        Generate token mappings for a series of terms using a specified tokenizer.

        Args:
            column (cudf.Series): A cuDF Series containing terms to be tokenized.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

        Returns:
            Dict[str, List[int]]: A dictionary mapping terms to their tokenized representations.
        """
        data = column.unique().tolist()

        padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"

        tokens = tokenizer(
            data,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            pad_to_max_length=self.pad_to_max_length,
            return_attention_mask=False,
        )["input_ids"]

        tokenizer.padding_side = padding_side

        return dict(zip(data, tokens))

    def get_wordnet_tokenized_dataframe(
        self, tokenizer: transformers.PreTrainedTokenizer
    ) -> cudf.DataFrame:
        """
        Create a DataFrame with tokenized representations of WordNet synsets and lemmas.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

        Returns:
            cudf.DataFrame: A DataFrame containing synsets, lemmas, languages, and their tokenized forms.
        """
        df = self._create_initial_dataframe()
        df = self._process_lemmas(df)
        df = self._tokenize_lemmas(df, tokenizer)

        return cudf.from_pandas(df)

    def _create_initial_dataframe(self) -> pd.DataFrame:
        """
        Create the initial DataFrame with synsets and their lemmas.

        Returns:
            cudf.DataFrame: A DataFrame with synsets and their corresponding lemmas.
        """
        df = pd.DataFrame({"synset": self.wordnet.all_synsets()})
        df["lemma"] = df["synset"].apply(self.get_all_lemmas)
        df["pos"] = df["synset"].apply(lambda synset: synset.pos())
        df["synset"] = df["synset"].apply(lambda synset: synset.name())
        return df.explode("lemma")

    def _process_lemmas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process lemmas to extract language and generate variations.

        Args:
            df (cudf.DataFrame): The input DataFrame with synsets and lemmas.

        Returns:
            cudf.DataFrame: A DataFrame with processed lemmas and their variations.
        """
        df["lang"] = df["lemma"].apply(lambda lemma: lemma.lang())
        df["lemma"] = df["lemma"].apply(lambda lemma: lemma.name().replace("_", " "))
        df["lemma"] = df["lemma"].apply(
            lambda lemma: self.get_lemma_variations(lemma, self.all_term_variations)
        )
        return df.explode("lemma")

    def _tokenize_lemmas(
        self, df: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer
    ) -> pd.DataFrame:
        """
        Tokenize lemmas using the provided tokenizer.

        Args:
            df (cudf.DataFrame): The input DataFrame with processed lemmas.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

        Returns:
            cudf.DataFrame: A DataFrame with tokenized lemmas.
        """
        df["tokens"] = (
            df["lemma"]
            .map(self.get_token_mapping(df["lemma"], tokenizer))
            .apply(lambda x: x[1:] if x[0] == tokenizer.bos_token_id else x)
        )
        return df

    def get_dataframe_filename(
        self, tokenizer: transformers.PreTrainedTokenizer
    ) -> Path:
        """
        Generate a filename for caching the tokenized WordNet DataFrame.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for generating tokens.

        Returns:
            str: The generated filename for caching.
        """
        filename = f".cache/wordnet/wordnet_{tokenizer.__class__.__name__.lower()}_{hash(self)}.parquet"
        filepath = Path.home() / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        return filepath

    @lru_cache()
    def get_cached_wordnet_tokenized_dataframe(
        self, tokenizer: transformers.PreTrainedTokenizer
    ) -> cudf.DataFrame:
        """
        Retrieve a cached tokenized WordNet DataFrame or generate and cache a new one.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.

        Returns:
            cudf.DataFrame: The tokenized WordNet DataFrame.
        """
        filepath = self.get_dataframe_filename(tokenizer)
        if filepath.exists():
            df = cudf.read_parquet(filepath)
        else:
            df = self.get_wordnet_tokenized_dataframe(tokenizer)
            df.to_parquet(filepath)

        return df.copy(deep=True)

    @lru_cache()
    def get_dataframe(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        min_lemmas_per_synset: int = 1,
        max_token_count: int = 99,
    ) -> cudf.DataFrame:
        """
        Retrieve a DataFrame with tokenized WordNet synsets, filtering out small synsets.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
            min_lemmas_per_synset (int): Minimum number of lemmas a synset must have to be included.
            single_token (bool): If True, only include single-token lemmas.

        Returns:
            cudf.DataFrame: The filtered tokenized WordNet DataFrame.
        """
        df = self.get_cached_wordnet_tokenized_dataframe(tokenizer)
        df = self._filter_and_process_dataframe(df, tokenizer)
        df = self._filter_by_token_count(df, max_token_count)
        df = self._filter_by_lemma_count(df, min_lemmas_per_synset)
        return self._assign_ids_and_sort(df).copy(deep=True)

    def _filter_and_process_dataframe(
        self,
        df: cudf.DataFrame,
        tokenizer: transformers.PreTrainedTokenizer,
    ) -> cudf.DataFrame:
        """
        Filter and process the DataFrame based on token length and count.

        Args:
            df (cudf.DataFrame): The input DataFrame.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for processing.
            single_token (bool): If True, only include single-token lemmas.

        Returns:
            cudf.DataFrame: The filtered and processed DataFrame.
        """
        df["token count"] = (
            df["tokens"]
            .list.index(tokenizer.pad_token_id)
            .replace(-1, self.max_length - 1)
        )
        return df.dropna().drop_duplicates(subset=["synset", "lemma"])

    def _filter_by_token_count(
        self, df: cudf.DataFrame, max_token_count: int
    ) -> cudf.DataFrame:
        """
        Apply the single-token filter to the DataFrame.

        Args:
            df (cudf.DataFrame): The input DataFrame.
            single_token (bool): If True, only include single-token lemmas.

        Returns:
            cudf.DataFrame: The filtered DataFrame.
        """
        if max_token_count < self.max_length:
            df = df[df["token count"] <= max_token_count]
            token_range = list(range(max_token_count))
            df["tokens"] = df["tokens"].list.take([token_range] * df.shape[0])

        return df

    def _filter_by_lemma_count(
        self, df: cudf.DataFrame, min_lemmas_per_synset: int
    ) -> cudf.DataFrame:
        """
        Apply the minimum lemmas per synset filter.

        Args:
            df (cudf.DataFrame): The input DataFrame.
            min_lemmas_per_synset (int): Minimum number of lemmas a synset must have to be included.

        Returns:
            cudf.DataFrame: The filtered DataFrame.
        """
        if min_lemmas_per_synset > 1:
            counts = df["synset"].value_counts()
            df = df[df["synset"].isin(counts[counts >= min_lemmas_per_synset].index)]
        return df

    def _assign_ids_and_sort(self, df: cudf.DataFrame) -> cudf.DataFrame:
        """
        Assign categorical IDs to synsets and lemmas, and sort the DataFrame.

        Args:
            df (cudf.DataFrame): The input DataFrame.

        Returns:
            cudf.DataFrame: The DataFrame with assigned IDs and sorted by synset ID.
        """
        df["synset id"] = df["synset"].astype("category").cat.codes
        df["lemma id"] = df["lemma"].astype("category").cat.codes
        return df.sort_values(by="synset id", ascending=True)

    @lru_cache()
    def get_all_synsets(
        self, tokenizer: transformers.PreTrainedTokenizer, *args, **kwargs
    ) -> pd.DataFrame:
        """
        Get all synsets with their tokenized representations.

        Args:
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for tokenization.
            *args: Additional arguments to pass to get_dataframe.
            **kwargs: Additional keyword arguments to pass to get_dataframe.

        Returns:
            cudf.DataFrame: A DataFrame containing synset IDs, synsets, and their tokenized representations.
        """
        df = self.get_dataframe(tokenizer, *args, **kwargs)
        return self._get_padded_synset_tokens(df, tokenizer)

    def _get_padded_synset_tokens(
        self, df: cudf.DataFrame, tokenizer: transformers.PreTrainedTokenizer
    ) -> cudf.Series:
        """
        Get padded token representations for each synset.

        Args:
            df (cudf.DataFrame): The input DataFrame.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer used for padding.

        Returns:
            cudf.Series: A series of padded token arrays for each synset.
        """
        max_size = df["synset id"].value_counts().max()
        return (
            df.to_pandas()
            .groupby("synset")["tokens"]
            .apply(
                lambda x: np.pad(
                    np.stack(x),
                    pad_width=[(0, max_size - x.shape[0]), (0, 0)],
                    mode="constant",
                    constant_values=tokenizer.pad_token_id,
                )
            )
            .reset_index()
        )
