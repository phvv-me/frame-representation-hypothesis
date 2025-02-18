from itertools import batched
import shelve
from typing import Any, Dict, List, Optional

import torch
from pydantic import BaseModel, Field
from tqdm import tqdm

from frames.utils.memory import gc_cuda

from ..data.figstep import MultilingualSafeBench
from ..representations import FrameUnembeddingRepresentation


class MultilingualModelGenerator(BaseModel):
    """Handles multilingual model generation with and without guidance.

    Attributes:
        db_path: Path to the results database
        models: List of model configurations
        query_types: List of query types to process
        languages: List of languages to process
        guide: Guide object for guided generation
        max_new_tokens: Maximum number of tokens to generate
        top_k: Number of tokens for top-k filtering
        output_hidden_states: Whether to output hidden states
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        min_lemmas_per_synset: Minimum lemmas per synset for guidance
        max_token_count: List of maximum token count values for guidance
        guidance_k: Top-k value for guidance
        guidance_steps: Number of steps for guidance
        batch_size: Batch size for processing
        sample_size: Number of samples to process from dataset
    """

    # Database and model configurations
    db_path: str
    models: List[Dict[str, Any]]
    query_types: List[str]
    languages: List[str]
    guide: Any

    # Standard generation config
    max_new_tokens: int = Field(
        default=42, description="Maximum number of tokens to generate"
    )
    top_k: Optional[int] = Field(
        default=None, description="Number of tokens for top-k filtering"
    )
    output_hidden_states: bool = Field(
        default=False, description="Whether to output hidden states"
    )
    do_sample: bool = Field(default=False, description="Whether to use sampling")
    temperature: Optional[float] = Field(
        default=None, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")

    # Guidance config
    min_lemmas_per_synset: int = Field(
        default=1, description="Minimum lemmas per synset for guidance"
    )
    max_token_count: List[int] = Field(
        default_factory=lambda: [3], description="List of maximum token count values for guidance"
    )
    guidance_k: int = Field(default=2, description="Top-k value for guidance")
    guidance_steps: int = Field(default=42, description="Number of steps for guidance")
    batch_size: int = Field(default=32, description="Batch size for processing")
    use_chat_template: bool = Field(
        default=False, description="Whether to use chat template for model input"
    )

    def _initialize_model(self, model_config: Dict[str, Any]) -> None:
        """Initialize model with given configuration."""
        self._manager = FrameUnembeddingRepresentation.from_model_id(device_map="auto", torch_dtype=torch.float16, use_chat_template=self.use_chat_template, **model_config)

    def _generate_with_guidance_for_token(self, data: List[Any], token_count: int) -> List[Any]:
        """Generate outputs using guided generation for a specific max_token_count."""
        return self._manager.quick_generate_with_topk_guide(
            data,
            guide=self.guide,
            min_lemmas_per_synset=self.min_lemmas_per_synset,
            max_token_count=token_count,
            k=self.guidance_k,
            steps=self.guidance_steps,
            batch_size=self.batch_size,
        )

    def _generate_without_guidance(self, data: List[Any]) -> List[str]:
        """Generate outputs without guidance."""
        return [out
            for inputs in batched(tqdm(data), self.batch_size)
            for out in self._manager.model.decode(
                self._manager.model.generate(
                    inputs,
                    max_new_tokens=self.max_new_tokens,
                    top_k=self.top_k,
                    output_hidden_states=self.output_hidden_states,
                    do_sample=self.do_sample,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            )
        ]

    def _get_result_key(
        self, model_id: str, query_type: str, language: str, use_guidance: bool, token_count: int | None = None
    ) -> str:
        """Generate unique key for storing results."""
        return f"{model_id}_{query_type}_{language}_{'guided' if use_guidance else 'default'}_{token_count or ''}_{'chat' if self.use_chat_template else ''}"

    def _save_results(self, key: str, results: List[Any]) -> None:
        """Save generation results to database."""
        with shelve.open(self.db_path) as db:
            if key not in db:
                print(f"Saving results for {key}")
                db[key] = {"results": results}

    def _process_configuration(
        self,
        model_config: Dict[str, Any],
        query_type: str,
        language: str,
        data: List[Any],
    ) -> None:
        """Process a configuration with and without guidance."""
        for use_guidance in [False, True]:
            if not use_guidance:
                key = self._get_result_key(model_config["id"], query_type, language, use_guidance)
                with shelve.open(self.db_path) as db:
                    if key in db:
                        continue
                results = self._generate_without_guidance(data)
                self._save_results(key, results)
            else:
                for token_count in self.max_token_count:
                    key = self._get_result_key(model_config["id"], query_type, language, use_guidance, token_count)
                    with shelve.open(self.db_path) as db:
                        if key in db:
                            continue
                    results = self._generate_with_guidance_for_token(data, token_count)
                    self._save_results(key, results)

    def generate_all(self) -> None:
        """Generate results for all configured combinations."""
        for model_config in self.models:
            with gc_cuda():
                self._initialize_model(model_config)
                for q in self.query_types:
                    for lang in self.languages:
                        data = MultilingualSafeBench(q, lang).to_list(return_flat_list=True)
                        self._process_configuration(model_config, q, lang, data)
