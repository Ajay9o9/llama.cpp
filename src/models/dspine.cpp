#include "models.h"

#include "llama-impl.h"

void llama_model_dflash::load_tensors_dspine() {
    LLAMA_LOAD_LOCALS;

    // DSpine: code bridge after every layer, its last write is refined over the top-k lattice
    const struct ggml_tensor * codebook_meta = ml->get_tensor_meta("dspine_codebook.weight");
    const struct ggml_tensor * gate_meta     = ml->get_tensor_meta("dspine_gate.weight");
    if (!codebook_meta || !gate_meta) {
        throw std::runtime_error("DSpine model is missing its code bridge tensors");
    }

    const int64_t code  = codebook_meta->ne[0];
    const int64_t rank  = gate_meta->ne[1];
    const int64_t top_k = hparams.dflash_selector_top_k;
    if (top_k < 2 || n_embd < top_k * (top_k + 1) || hparams.dflash_block_size < 2) {
        throw std::runtime_error("invalid DSpine metadata");
    }

    dspine_codebook  = create_tensor(tn(LLM_TENSOR_DSPINE_CODEBOOK,  "weight"), { code, n_vocab }, 0);
    dspine_gate      = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "weight"), { code, rank }, 0);
    dspine_gate_b    = create_tensor(tn(LLM_TENSOR_DSPINE_GATE,      "bias"),   { rank }, 0);
    dspine_pred_gate = create_tensor(tn(LLM_TENSOR_DSPINE_PRED_GATE, "weight"), { code, rank }, 0);
    dspine_prev      = create_tensor(tn(LLM_TENSOR_DSPINE_PREV,      "weight"), { code, rank }, 0);
    for (int i = 0; i < n_layer; ++i) {
        layers[i].dspine_read  = create_tensor(tn(LLM_TENSOR_DSPINE_READ,  "weight", i), { n_embd, code }, 0);
        layers[i].dspine_write = create_tensor(tn(LLM_TENSOR_DSPINE_WRITE, "weight", i), { rank, n_embd }, 0);
    }

    LLAMA_LOG_INFO("%s: DSpine code bridge (code = %lld, rank = %lld, top-k = %lld)\n", __func__,
            (long long) code, (long long) rank, (long long) top_k);
}

// anchor (token 0) of every block
static ggml_tensor * dspine_anchor_ids(llm_graph_context & g, ggml_tensor * tokens, int64_t block, int64_t n_blocks) {
    return ggml_cont_1d(g.ctx0, ggml_view_2d(g.ctx0, tokens, 1, n_blocks, block * tokens->nb[0], 0), n_blocks);
}

// DSpine code bridge write after layer il: every row reads a unit code from its hidden state (the anchor row
// uses its token's codebook row), then row t gets a gated message from the code of row t-1
ggml_tensor * llama_model_dflash::build_dspine_write(llm_graph_context & g, const llama_model & model,
        ggml_tensor * cur, ggml_tensor * tokens, int il, dspine_stash * stash) {
    ggml_context * ctx0 = g.ctx0;

    const auto & layer = model.layers[il];

    const int64_t n_embd   = cur->ne[0];
    const int64_t n_tokens = cur->ne[1];
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    const int64_t code     = model.dspine_codebook->ne[0];
    const int64_t rank     = model.dspine_gate->ne[1];
    const float   eps      = g.hparams.f_norm_rms_eps;

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    const int64_t block = n_tokens / n_blocks;
    if (block < 2) {
        return cur;
    }

    // codes are L2-normalized, so the read is scale-invariant and only the rms is kept for the write
    ggml_tensor * rms = ggml_sqrt(ctx0, ggml_scale(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)), 1.0f/n_embd));

    ggml_tensor * codes = g.build_lora_mm(layer.dspine_read, ggml_rms_norm(ctx0, cur, eps));
    codes = ggml_scale(ctx0, ggml_l2_norm(ctx0, codes, eps), sqrtf((float) code));
    g.cb(codes, "dspine_codes", il);

    ggml_tensor * anchor_codes = ggml_get_rows(ctx0, model.dspine_codebook, dspine_anchor_ids(g, tokens, block, n_blocks));
    anchor_codes = ggml_scale(ctx0, ggml_reshape_3d(ctx0, anchor_codes, code, 1, n_blocks), sqrtf((float) code));

    codes = ggml_reshape_3d(ctx0, codes, code, block, n_blocks);
    codes = ggml_concat(ctx0, anchor_codes,
            ggml_cont(ctx0, ggml_view_3d(ctx0, codes, code, block - 1, n_blocks, codes->nb[1], codes->nb[2], codes->nb[1])), 1);
    codes = ggml_reshape_2d(ctx0, codes, code, n_tokens);

    ggml_tensor * q = ggml_reshape_3d(ctx0, g.build_lora_mm(model.dspine_gate,      codes), rank, block, n_blocks);
    ggml_tensor * s = ggml_reshape_3d(ctx0, g.build_lora_mm(model.dspine_pred_gate, codes), rank, block, n_blocks);
    ggml_tensor * v = ggml_reshape_3d(ctx0, g.build_lora_mm(model.dspine_prev,      codes), rank, block, n_blocks);

    const auto rows = [&](ggml_tensor * t, int64_t beg) {
        return ggml_view_3d(ctx0, t, rank, block - 1, n_blocks, t->nb[1], t->nb[2], beg * t->nb[1]);
    };

    // message(t) = sigmoid(Wg c_t + Wp c_(t-1) + b) * (Wv c_(t-1)), nothing for the anchor row
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, rows(q, 1), rows(s, 0)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_mul(ctx0, gate, rows(v, 0));

    ggml_tensor * delta = g.build_lora_mm(layer.dspine_write, ggml_reshape_2d(ctx0, msg, rank, (block - 1) * n_blocks));
    delta = ggml_pad_ext(ctx0, ggml_reshape_3d(ctx0, delta, n_embd, block - 1, n_blocks), 0, 0, 1, 0, 0, 0, 0, 0);
    delta = ggml_mul(ctx0, ggml_reshape_2d(ctx0, delta, n_embd, n_tokens), rms);

    if (stash) {
        stash->raw = cur;
        stash->rms = rms;
        stash->q   = ggml_reshape_2d(ctx0, q, rank, n_tokens);
    }

    return ggml_add(ctx0, cur, delta);
}

// DSpine serial refinement: redo the last bridge write once per exact predecessor candidate and score the top-k
// candidates of every row, packed like the DFlash2 lattice: [cand (k), scores (k x k, predecessor-major)]
static void build_dspine_lattice(llm_graph_context & g, const llama_model & model, ggml_tensor * tokens,
        ggml_tensor * output, const llama_model_dflash::dspine_stash & stash) {
    ggml_context * ctx0 = g.ctx0;
    auto         & res  = g.res;

    const int64_t n_embd   = g.n_embd;
    const int64_t n_tokens = g.n_tokens;
    const int64_t n_blocks = g.ubatch.n_seqs_unq;
    const int64_t top_k    = g.hparams.dflash_selector_top_k;
    const int64_t code     = model.dspine_codebook->ne[0];
    const int64_t rank     = model.dspine_gate->ne[1];

    GGML_ASSERT(n_blocks > 0 && n_tokens % n_blocks == 0);
    const int64_t block = n_tokens / n_blocks;
    const int64_t n_pos = block - 1;
    if (block < 2 || block > (int64_t) g.hparams.dflash_block_size || stash.raw == nullptr) {
        return;
    }

    // rows 1..block-1 of every block, with a broadcast candidate dim: [ne0, 1, n_pos, n_blocks]
    const auto pos_rows = [&](ggml_tensor * t) {
        ggml_tensor * t3 = ggml_reshape_3d(ctx0, t, t->ne[0], block, n_blocks);
        t3 = ggml_cont(ctx0, ggml_view_3d(ctx0, t3, t->ne[0], n_pos, n_blocks, t3->nb[1], t3->nb[2], t3->nb[1]));
        return ggml_reshape_4d(ctx0, t3, t->ne[0], 1, n_pos, n_blocks);
    };

    ggml_tensor * cand = ggml_reshape_3d(ctx0, ggml_top_k(ctx0, res->t_logits, top_k), top_k, block, n_blocks);
    ggml_tensor * cand_run = ggml_cont(ctx0, ggml_view_3d(ctx0, cand, top_k, n_pos, n_blocks, cand->nb[1], cand->nb[2], cand->nb[1]));

    // exact predecessor codes: the anchor for row 1, the candidates of row t-1 for row t
    ggml_tensor * pred = ggml_get_rows(ctx0, model.dspine_codebook, dspine_anchor_ids(g, tokens, block, n_blocks));
    pred = ggml_repeat_4d(ctx0, ggml_reshape_4d(ctx0, pred, code, 1, 1, n_blocks), code, top_k, 1, n_blocks);
    if (n_pos > 1) {
        ggml_tensor * prev_ids = ggml_cont(ctx0, ggml_view_3d(ctx0, cand_run, top_k, n_pos - 1, n_blocks, cand_run->nb[1], cand_run->nb[2], 0));
        ggml_tensor * prev = ggml_get_rows(ctx0, model.dspine_codebook, ggml_reshape_1d(ctx0, prev_ids, top_k * (n_pos - 1) * n_blocks));
        pred = ggml_concat(ctx0, pred, ggml_reshape_4d(ctx0, prev, code, top_k, n_pos - 1, n_blocks), 2);
    }
    pred = ggml_scale(ctx0, ggml_reshape_2d(ctx0, pred, code, top_k * n_pos * n_blocks), sqrtf((float) code));

    ggml_tensor * gp = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_pred_gate, pred), rank, top_k, n_pos, n_blocks);
    ggml_tensor * pv = ggml_reshape_4d(ctx0, g.build_lora_mm(model.dspine_prev,      pred), rank, top_k, n_pos, n_blocks);

    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_add(ctx0, gp, pos_rows(stash.q)), model.dspine_gate_b));
    ggml_tensor * msg  = ggml_mul(ctx0, gate, pv);

    const auto & last = model.layers[g.n_layer - 1];
    ggml_tensor * delta = g.build_lora_mm(last.dspine_write, ggml_reshape_2d(ctx0, msg, rank, top_k * n_pos * n_blocks));
    delta = ggml_reshape_4d(ctx0, delta, n_embd, top_k, n_pos, n_blocks);

    ggml_tensor * refined = ggml_add(ctx0, ggml_mul(ctx0, delta, pos_rows(stash.rms)), pos_rows(stash.raw));
    refined = g.build_norm(refined, model.output_norm, NULL, LLM_NORM_RMS, -1);
    g.cb(refined, "dspine_refined", -1);

    // table[k][j]: logit of candidate k after the write with predecessor candidate j
    ggml_tensor * head = ggml_get_rows(ctx0, output, ggml_reshape_1d(ctx0, cand_run, top_k * n_pos * n_blocks));
    head = ggml_reshape_4d(ctx0, head, n_embd, top_k, n_pos, n_blocks);
    ggml_tensor * table = ggml_mul_mat(ctx0, head, refined);

    ggml_tensor * row = ggml_concat(ctx0,
            ggml_cast(ctx0, cand_run, GGML_TYPE_F32),
            ggml_reshape_3d(ctx0, table, top_k * top_k, n_pos, n_blocks), 0);
    row = ggml_pad(ctx0, row, n_embd - top_k - top_k * top_k, 0, 0, 0);

    ggml_tensor * packed = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, 1, n_blocks), 0.0f);
    packed = ggml_reshape_2d(ctx0, ggml_concat(ctx0, packed, row, 1), n_embd, n_tokens);
    g.cb(packed, "dspine_lattice", -1);

    res->t_h_nextn = packed;
    ggml_build_forward_expand(g.gf, packed);
}

llama_model_dflash::graph_dspine::graph_dspine(const llm_graph_params & params) : graph_base(params) {
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd || ubatch.embd_row) {
        build_inject();
        return;
    }

    dspine_stash dspine;

    ggml_tensor * inp_tokens = build_inp_tokens();
    ggml_tensor * cur        = build_layers(inp_tokens, &dspine);

    build_logits(cur);

    build_dspine_lattice(*this, model, inp_tokens, get_lm_head(*this, model).first, dspine);
}
