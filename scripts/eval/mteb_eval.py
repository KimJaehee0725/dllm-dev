import logging
from typing import Any, Dict, Iterable, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedTokenizer,
)

from scripts.utils import maybe_add_missing_special_tokens, register_useful_resolvers

log = logging.getLogger(__name__)


def _maybe_to_str(prompt_type: Any | None) -> Optional[str]:
    if prompt_type is None:
        return None
    if isinstance(prompt_type, str):
        return prompt_type
    return str(prompt_type).lower()


def _batch_iter(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


class BD3LMEncoder:
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        tokenizer: PreTrainedTokenizer,
        pretrained_model_revision: str | None = None,
        pooling: str = "mean",
        max_length: int | None = None,
        device: str | None = None,
        dtype: str | None = None,
        prompts: dict[str, str] | None = None,
        prompt: str | None = None,
        model_config_overrides: dict[str, Any] | None = None,
    ):
        self.pooling = pooling
        self.max_length = max_length
        self.prompts = prompts
        self.prompt = prompt or ""
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = maybe_add_missing_special_tokens(tokenizer)
        self.model = self._load_model(
            pretrained_model_name_or_path,
            pretrained_model_revision,
            model_config_overrides or {},
        ).to(self.device)
        self.model.eval()
        self.backbone = getattr(self.model, "backbone", self.model)
        if dtype is not None:
            self.model = self.model.to(dtype=getattr(torch, dtype))
            self.backbone = getattr(self.model, "backbone", self.model)

    def _load_model(
        self,
        pretrained_model_name_or_path: str,
        pretrained_model_revision: str | None,
        model_config_overrides: dict[str, Any],
    ):
        try:
            return AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                revision=pretrained_model_revision,
                **model_config_overrides,
            )
        except Exception:  # noqa: BLE001
            try:
                return AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )
            except Exception:  # noqa: BLE001
                return AutoModel.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=True,
                    revision=pretrained_model_revision,
                    **model_config_overrides,
                )

    def _resolve_prompt(self, task_name: str | None, prompt_type: Any | None) -> str:
        if not self.prompts:
            return self.prompt
        prompt_key = _maybe_to_str(prompt_type)
        if prompt_key is not None:
            task_prompt_key = f"{task_name}-{prompt_key}" if task_name else None
            if task_prompt_key and task_prompt_key in self.prompts:
                return self.prompts[task_prompt_key]
            if prompt_key in self.prompts:
                return self.prompts[prompt_key]
        if task_name and task_name in self.prompts:
            return self.prompts[task_name]
        return self.prompt

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int,
        pooling: str,
        max_length: int | None,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for batch in _batch_iter(texts, batch_size):
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = getattr(outputs, "last_hidden_state", None)
                if hidden is None:
                    hidden = outputs.hidden_states[-1]
            if pooling == "mean":
                mask = attention_mask.unsqueeze(-1).expand(hidden.shape).float()
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            elif pooling == "cls":
                pooled = hidden[:, 0]
            elif pooling == "last":
                last_indices = attention_mask.sum(dim=1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_indices]
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            if normalize_embeddings:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_embeddings.append(pooled.detach().cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    def _extract_texts_from_batch(self, batch: Any) -> List[str]:
        if isinstance(batch, list):
            if len(batch) == 0 or isinstance(batch[0], str):
                return batch
        if isinstance(batch, dict):
            for key in ("sentences", "sentence", "text", "texts"):
                if key in batch:
                    value = batch[key]
                    return value if isinstance(value, list) else [value]
        if hasattr(batch, "sentences"):
            value = batch.sentences
            return value if isinstance(value, list) else [value]
        return list(batch)

    def encode(self, inputs, task_name: str | None = None, prompt_type=None, **kwargs):
        pooling = kwargs.pop("pooling", self.pooling)
        batch_size = kwargs.pop("batch_size", 32)
        max_length = kwargs.pop("max_length", self.max_length)
        normalize_embeddings = kwargs.pop("normalize_embeddings", False)
        prompt = self._resolve_prompt(task_name, prompt_type)

        if isinstance(inputs, list) and (len(inputs) == 0 or isinstance(inputs[0], str)):
            texts = [prompt + text for text in inputs]
            return self._encode_texts(
                texts,
                batch_size=batch_size,
                pooling=pooling,
                max_length=max_length,
                normalize_embeddings=normalize_embeddings,
            )

        all_embeddings: List[np.ndarray] = []
        for batch in inputs:
            texts = self._extract_texts_from_batch(batch)
            texts = [prompt + text for text in texts]
            all_embeddings.append(
                self._encode_texts(
                    texts,
                    batch_size=batch_size,
                    pooling=pooling,
                    max_length=max_length,
                    normalize_embeddings=normalize_embeddings,
                )
            )
        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, queries, task_name: str | None = None, **kwargs):
        return self.encode(queries, task_name=task_name, prompt_type="query", **kwargs)

    def encode_corpus(self, corpus, task_name: str | None = None, **kwargs):
        if isinstance(corpus, list) and len(corpus) > 0 and isinstance(corpus[0], dict):
            texts = [
                (c.get("title", "") + " " + c.get("text", "")).strip() for c in corpus
            ]
            return self.encode(texts, task_name=task_name, prompt_type="passage", **kwargs)
        return self.encode(corpus, task_name=task_name, prompt_type="passage", **kwargs)


def run_mteb(
    model: BD3LMEncoder,
    tasks: list[str] | None = None,
    benchmark: str | None = None,
    eval_splits: list[str] | None = None,
    eval_subsets: list[str] | None = None,
    output_folder: str | None = None,
    overwrite_results: bool | None = None,
    verbosity: int | None = 2,
    encode_kwargs: dict[str, Any] | None = None,
):
    import mteb

    if benchmark is not None:
        selected_tasks = mteb.get_benchmark(benchmark)
    elif tasks is not None:
        selected_tasks = mteb.get_tasks(tasks=tasks)
    else:
        raise ValueError("Specify either benchmark or tasks for MTEB evaluation.")

    evaluation = mteb.MTEB(tasks=selected_tasks)
    run_kwargs: Dict[str, Any] = {}
    if output_folder is not None:
        run_kwargs["output_folder"] = output_folder
    if overwrite_results is not None:
        run_kwargs["overwrite_results"] = overwrite_results
    if verbosity is not None:
        run_kwargs["verbosity"] = verbosity
    if eval_splits is not None:
        run_kwargs["eval_splits"] = eval_splits
    if eval_subsets is not None:
        run_kwargs["eval_subsets"] = eval_subsets
    if encode_kwargs is not None:
        run_kwargs["encode_kwargs"] = encode_kwargs

    return evaluation.run(model, **run_kwargs)


@hydra.main(version_base=None, config_path="../../configs", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    model = hydra.utils.instantiate(cfg.task.model)
    results = hydra.utils.call(cfg.task, model=model)
    if results is not None:
        log.info("MTEB evaluation completed.")


if __name__ == "__main__":
    register_useful_resolvers()
    main()
