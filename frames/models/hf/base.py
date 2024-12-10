"""
Module for working with HuggingFace models, providing a wrapper class with quantization support.
"""

import os
import re
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional, Type, Union

import humanize
import torch
import transformers
from huggingface_hub import login
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AwqConfig,
    PreTrainedModel,
)
from transformers.utils import is_auto_awq_available

from ...abstract import BaseModel
from ...utils.stdlib import is_online

Q4BIT = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

Q8BIT = transformers.BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
)

AWQ = (
    AwqConfig(
        bits=4,
        fuse_max_seq_len=512,
        do_fuse=True,
    )
    if is_auto_awq_available() and torch.cuda.is_available()
    else None
)


class BaseHuggingFaceModel(BaseModel, ABC, arbitrary_types_allowed=True):
    """
    A wrapper class for HuggingFace models with support for various quantization methods.

    Attributes:
        id (str): The model identifier on HuggingFace Hub
        family (Optional[str]): The model family name
        cls (Type[AutoModelForCausalLM]): Model class to use
        tkn (Type[AutoTokenizer]): Tokenizer class to use
        device_map (Union[int, str]): Device mapping strategy
        torch_dtype (Union[torch.dtype, str]): Torch data type to use
        trust_remote_code (bool): Whether to trust remote code
        attn_implementation (str): Attention implementation to use
        quantization (Optional[Union[int, str]]): Quantization method
        compile (bool): Whether to compile the model
    """

    id: str
    family: Optional[str] = None
    cls: Type[PreTrainedModel] = AutoModelForCausalLM
    device_map: Union[int, str] = "auto"
    torch_dtype: Union[torch.dtype, str] = "auto"
    trust_remote_code: bool = True
    attn_implementation: str = "auto"
    quantization: Optional[Union[int, str]] = "auto"
    compile: bool = False

    _quantizations = {4: Q4BIT, 8: Q8BIT, "AWQ": AWQ}

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model and load it."""
        super().__init__(**kwargs)
        self.login_to_hf_if_online()
        self.load()

        logger.info(f"Loaded model: {self.id}")
        logger.warning(f"memory cost: {self.memory_footprint >> 20} Mb")

        if self._is_meta_llama():
            self._fix_llama_model()

        if self.compile:
            self._compile()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """Forward pass through the model."""
        return self._model(*args, **kwds)

    def __str__(self) -> str:
        """String representation of the model."""
        return self.id if self.family is None else self.name

    def _compile(self) -> None:
        """Compile the model using torch.compile() for potential speedup."""
        self._model.forward = torch.compile(
            self._model.forward, mode="reduce-overhead", fullgraph=True
        )

    @staticmethod
    def is_hf_online() -> bool:
        """Check if HuggingFace Hub is accessible.

        Returns:
            bool: True if HuggingFace Hub is accessible
        """
        return not bool(os.getenv("HF_HUB_OFFLINE"))

    @classmethod
    def login_to_hf_if_online(cls) -> None:
        """Login to HuggingFace Hub if online and token is available."""
        if is_online() and cls.is_hf_online():
            login(os.environ["HUGGING_FACE_LOGIN_TOKEN"], add_to_git_credential=True)

    def load(self) -> None:
        """Load the model and tokenizer from HuggingFace Hub.

        This method:
        1. Gathers model configuration kwargs
        2. Loads the model with proper quantization if specified
        3. Initializes the model on the specified device

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            self._model = self.cls.from_pretrained(**self._model_kwargs())
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _model_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for model initialization.

        Returns:
            Dict[str, Any]: Configuration kwargs for model loading
        """
        kwargs = dict(
            pretrained_model_name_or_path=self.id,
            device_map=self.device_map,
            torch_dtype=self._correct_torch_dtype,
            trust_remote_code=self.trust_remote_code,
            attn_implementation=self._get_attn_implementation(),
            local_files_only=not is_online(),
        )

        if quantization_config := self._get_quantization_config():
            kwargs["quantization_config"] = quantization_config

        return kwargs

    def _is_meta_llama(self) -> bool:
        """Check if current model is a Meta LLaMA model.

        Returns:
            bool: True if model is Meta-LLaMA-3
        """
        return "Meta-Llama-3" in self.id

    def _fix_llama_model(self) -> None:
        """Apply LLaMA-specific model fixes for tokens and padding."""
        self._fix_pad_token_in_llama_model()
        self._fix_eos_token_in_llama_model()
        self._tokenizer.padding_side = "left"

    def _fix_token(self, token_str: str, token_type: str) -> None:
        """Fix specific token in the tokenizer.

        Some tokens in LLaMA models are not correctly set by the tokenizer,
        so we must fix it manually.

        Args:
            token_str: Token string to fix
            token_type: Type of token (pad, eos, etc)
        """
        token_id = self._tokenizer.convert_tokens_to_ids(token_str)
        setattr(self._tokenizer, f"{token_type}_token", token_str)
        setattr(self._tokenizer, f"{token_type}_token_id", token_id)

    def _fix_pad_token_in_llama_model(self) -> None:
        """Fix padding token specifically for LLaMA models."""
        self._fix_token("<|eot_id|>", "pad")

    def _fix_eos_token_in_llama_model(self) -> None:
        """Fix end-of-sequence token specifically for LLaMA models."""
        self._fix_token("<|end_of_text|>", "eos")

    @property
    def model(self) -> PreTrainedModel:
        if not hasattr(self, "_model"):
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def memory_footprint(self) -> int:
        """Returns the current CUDA memory cost of the model in bytes."""
        return self.model.get_memory_footprint()

    @cached_property
    def parameter_count(self) -> int:
        """Returns the number of parameters in the model."""
        return sum(p.numel() for p in self.model.parameters())

    @property
    def is_instruction_finetuned(self) -> bool:
        return self.id.lower().endswith("it") or "instruct" in self.id.lower()

    @property
    def _parameterc_count_string_extracted_from_id(self):
        return re.compile(r"(\d+[Bb])").search(self.id).group(1).upper()

    @property
    def _parameter_count_string_from_actual_count(self):
        word = humanize.intword(self.parameter_count, format="%.f")
        i_alpha = next(i for i, char in enumerate(word) if char.isalpha())
        return word[: i_alpha + 1].upper().replace(" ", "")

    @property
    def parameter_count_string(self) -> str:
        try:
            return self._parameterc_count_string_extracted_from_id
        except AttributeError:
            return self._parameter_count_string_from_actual_count

    @property
    def is_instruction_finetuned_string(self):
        return "(Instruct)" if self.is_instruction_finetuned else ""

    @property
    def property_string(self):
        return f"{self.parameter_count_string} {self.is_instruction_finetuned_string}".strip()

    @property
    def name(self):
        return f"{self.family} {self.property_string}"

    @property
    def device(self) -> torch.device:
        """Returns the device the model is allocated to."""
        return self._model.device

    @property
    def dtype(self) -> torch.dtype:
        """Returns the model data type."""
        return self._model.dtype

    @property
    def get_embeddings(self) -> torch.Tensor:
        return self._model.get_input_embeddings()

    @property
    @abstractmethod
    def unembedding_matrix(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def _correct_torch_dtype(self) -> torch.dtype:
        if self.torch_dtype == "auto" and "AWQ" in self.id.upper():
            return torch.float16
        return self.torch_dtype

    def _get_quantization_config(self):
        return self._quantizations.get(self.quantization)

    def _get_attn_implementation(self):
        if self.attn_implementation != "auto":
            return self.attn_implementation
        return "sdpa" if torch.cuda.is_available() else "eager"

    def forward_last_hiden_state(self, *args, **kwargs) -> torch.Tensor:
        return self._model(*args, output_hidden_states=True, **kwargs).hidden_states[-1]

    def forward_next_token_logit(
        self, inputs_embeds: torch.FloatTensor, probes: torch.FloatTensor
    ) -> torch.Tensor:
        hs = self.forward_last_hiden_state(inputs_embeds=inputs_embeds)
        return hs.matmul(probes)[..., -1, :]
