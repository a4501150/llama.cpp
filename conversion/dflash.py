from __future__ import annotations

import logging
from typing import Any, Iterable

from torch import Tensor

import gguf

from .base import ModelBase, TextModel

logger = logging.getLogger("dflash")


@ModelBase.register("DFlashDraftModel")
class DFlashDraftModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DFLASH_DRAFT

    def set_vocab(self):
        try:
            self._set_vocab_sentencepiece()
            return
        except FileNotFoundError:
            pass
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_causal_attention(False)

        head_dim = self.hparams.get("head_dim", 128)
        self.gguf_writer.add_rope_dimension_count(head_dim)

        arch = self.gguf_writer.arch

        dflash_cfg = self.hparams.get("dflash_config", {})

        def dflash_value(name: str, default: Any) -> Any:
            if name in dflash_cfg:
                return dflash_cfg[name]
            if name in self.hparams:
                return self.hparams[name]
            logger.warning("DFlashDraftModel: missing %s; using default %r", name, default)
            return default

        block_size = dflash_value("block_size", 16)
        self.gguf_writer.add_uint32(f"{arch}.dflash.block_size", block_size)

        mask_token_id = dflash_value("mask_token_id", 248070)
        self.gguf_writer.add_uint32(f"{arch}.dflash.mask_token_id", mask_token_id)

        target_layer_ids = dflash_value("target_layer_ids", [1, 16, 31, 46, 61])
        self.gguf_writer.add_array(f"{arch}.dflash.target_layer_ids", target_layer_ids)

        if "n_target_features" in dflash_cfg:
            n_target_features = dflash_cfg["n_target_features"]
        elif "n_target_features" in self.hparams:
            n_target_features = self.hparams["n_target_features"]
        else:
            n_target_features = self.hparams.get("hidden_size", 5120) * len(target_layer_ids)
            logger.warning(
                "DFlashDraftModel: inferred n_target_features=%d from hidden_size(%d) * n_target_layers(%d)",
                n_target_features,
                self.hparams.get("hidden_size", 5120),
                len(target_layer_ids),
            )

        self.gguf_writer.add_uint32(f"{arch}.dflash.n_target_features", n_target_features)

        logger.info(
            "DFlash metadata: block_size=%s mask_token_id=%s target_layer_ids=%s n_target_features=%s",
            block_size, mask_token_id, target_layer_ids, n_target_features,
        )

        if self.hparams.get("use_sliding_window") and self.hparams.get("sliding_window"):
            self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
            layer_types = self.hparams.get("layer_types", [])
            pattern = [t == "sliding_attention" for t in layer_types]
            if pattern:
                self.gguf_writer.add_sliding_window_pattern(pattern)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("model."):
            name = name[len("model."):]
        yield from super().modify_tensors(data_torch, name, bid)
