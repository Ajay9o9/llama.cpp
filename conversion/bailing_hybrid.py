from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BailingMoeV3ForCausalLM")
class BailingMoeV3Model(TextModel):
    """Ling 3.0 - hybrid KDA + MLA MoE (config model_type: bailing_hybrid)"""
    model_arch = gguf.MODEL_ARCH.BAILING_HYBRID

    _experts: list[dict[str, Tensor]] | None = None

    def is_mla_layer(self, il: int) -> bool:
        n_layer = self.hparams["num_hidden_layers"]
        group   = self.hparams["layer_group_size"]
        return (il + 1) % group == 0 or il >= (n_layer // group) * group

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        hparams = self.hparams

        # MLA is stored as MQA so the compressed kv cache can be used directly
        hparams["num_key_value_heads"] = 1

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        # n_head_kv == 0 marks a KDA (recurrent) layer, see llama_model_bailing_hybrid
        n_head_kv = [1 if self.is_mla_layer(il) else 0 for il in range(hparams["num_hidden_layers"])]
        self.gguf_writer.add_head_count_kv(n_head_kv)

        self.gguf_writer.add_ssm_conv_kernel(hparams["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(hparams["head_dim"])

        kv_lora_rank     = hparams["kv_lora_rank"]
        qk_rope_head_dim = hparams["qk_rope_head_dim"]
        qk_nope_head_dim = hparams["qk_nope_head_dim"]

        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        self.gguf_writer.add_value_length(kv_lora_rank)
        self.gguf_writer.add_key_length_mla(qk_nope_head_dim + qk_rope_head_dim)
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])

        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["moe_shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        self.gguf_writer.add_expert_group_count(hparams["n_group"])
        self.gguf_writer.add_expert_group_used_count(hparams["topk_group"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # the HF modeling code ignores these, but the official vLLM fork applies them as
        # silu(gate).clamp(max=limit) * up.clamp(-limit, limit) - 0 means no clamp
        n_layer = hparams["num_hidden_layers"]
        self.gguf_writer.add_swiglu_clamp_exp([float(v) for v in hparams["expert_swiglu_limit_list"][:n_layer]])
        self.gguf_writer.add_swiglu_clamp_shexp([float(v) for v in hparams["share_expert_swiglu_limit_list"][:n_layer]])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the MTP layer sits past the last real layer, skip it
        if bid is not None and bid >= self.hparams["num_hidden_layers"]:
            return
        if name.endswith((".eh_proj.weight", ".enorm.weight", ".hnorm.weight", ".final_layernorm.weight")):
            return

        # HF keeps conv1d as [d_inner, d_conv], ggml wants ne = [d_conv, 1, d_inner, 1]
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            if data_torch.ndim == 2:
                d_inner, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)
            elif data_torch.ndim == 3:
                d_inner, _, d_conv = data_torch.shape
                data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        # HF keeps A_log as [n_head], ggml wants ne = [1, n_head]
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch).reshape(-1, 1)

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        # both layer types call this g_proj: on KDA it is the output gate, on MLA it is the
        # head-wise attention gate. rename the KDA one so the two map to different tensors.
        if name.endswith(".attention.g_proj.weight"):
            assert bid is not None
            if not self.is_mla_layer(bid):
                name = name.replace(".attention.g_proj.weight", ".attention.g_proj_kda.weight")

        if "mlp.experts" in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from super().modify_tensors(data_torch, merged_name, bid)
            return

        # MLA absorption needs kv_b split, with k_b transposed
        if name.endswith("kv_b_proj.weight"):
            name_kb = name.replace("kv_b_proj", "k_b_proj")
            name_vb = name.replace("kv_b_proj", "v_b_proj")

            n_head_kv        = self.hparams["num_key_value_heads"]
            v_head_dim       = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]

            assert data_torch.shape[0] == n_head_kv * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head_kv, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2)

            yield from super().modify_tensors(k_b, name_kb, bid)
            yield from super().modify_tensors(v_b, name_vb, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
