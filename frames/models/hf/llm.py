"""
Module for working with HuggingFace models, providing a wrapper class with quantization support.
"""

from typing import Type

import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
)

from ...utils.stdlib import is_online
from .base import BaseHuggingFaceModel


class LanguageHuggingFaceModel(BaseHuggingFaceModel):
    tkn: Type[PreTrainedTokenizer] = AutoTokenizer
    use_chat_template: bool = False

    def load(self) -> None:
        """
        Load the model and tokenizer from HuggingFace Hub.
        Handles login if online and sets up the model with specified configuration.
        """
        super().load()
        self._tokenizer = self.tkn.from_pretrained(**self._tokenizer_kwargs())

        if self._is_mistral():
            self._fix_pad_token_in_mistral_model()

    def _tokenizer_kwargs(self):
        return dict(
            pretrained_model_name_or_path=self.id,
            trust_remote_code=self.trust_remote_code,
            local_files_only=not is_online(),
        )

    def _is_mistral(self):
        return self.id.startswith("mistalai/Mistral")

    def _fix_pad_token_in_mistral_model(self):
        self._fix_token(self.tokenizer.eos_token, "pad")

    def make_input(
        self, inputs: str | torch.IntTensor, *args, **kwargs
    ) -> torch.Tensor:
        if self.use_chat_template:
            messages = [[{"role": "user", "content": text}] for text in self._decode_if_tensor(inputs)]
            return self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True, *args, **kwargs).to(self.device)
        else:
            return self.tokenizer(
                text=self._decode_if_tensor(inputs),
                return_tensors="pt",
                *args,
                **kwargs,
            ).to(self.device)
        
    def _decode_if_tensor(self, inputs):
        return (
            self.decode(inputs.flatten(0, 1))
            if isinstance(inputs, torch.Tensor)
            else inputs
        )

    def _clean(self, text: str) -> str:
        """Clean the input text."""
        return (
            text.replace(self.tokenizer.bos_token, "")
            .replace(self.tokenizer.pad_token, "")
            .lstrip()
        )

    def tokenize(self, *args, **kwargs) -> torch.Tensor:
        return self.make_input(*args, **kwargs)["input_ids"]

    def decode(self, input_ids: torch.Tensor) -> list[str]:
        decoded = self._tokenizer.batch_decode(input_ids)
        return [self._clean(x) for x in decoded]

    def embed(self, input_text: str) -> torch.Tensor:
        return self.get_embeddings(self.tokenize(input_text))
    
    def generate(self, inputs: list[str], *args, **kwargs):
        return self.model.generate(*args, **self.make_input(inputs, padding=True), **kwargs)

    @property
    def unembedding_matrix(self) -> torch.Tensor:
        return self._model.lm_head.weight.data.detach()
