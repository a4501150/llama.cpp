#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

#define LLAMA_MAX_SEQ 256

struct dflash_tape_gpu;
struct dflash_hidden_gpu;

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;        // number of recurrent-state snapshots per seq for rollback
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool embeddings_pre_norm;        // also extract the hidden state before the final output norm
    bool embeddings_pre_norm_masked; // extract for only rows where batch.logits != 0
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_context_type ctx_type;
    enum llama_pooling_type pooling_type;

    // DFlash: target layer indices to capture hidden states from (empty = disabled)
    std::vector<int> dflash_capture_layers;

    float dflash_sample_temp = 0.0f;
    int   dflash_topk = 1;

    bool dflash_verify_logits = false;
    int  dflash_verify_topk = 1;
    bool dflash_reduced_consumer_active = false;

    int dflash_cross_ctx = 512;
    int dflash_n_slots = 1;

    dflash_tape_gpu * tape_gpu = nullptr;
    dflash_tape_gpu * tape_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
    int tape_gpu_n_seqs = 0;

    dflash_hidden_gpu * hidden_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
    int hidden_gpu_n_seqs = 0;

    dflash_hidden_gpu * prefill_gpu_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};
    int prefill_gpu_n_seqs = 0;

    bool    dflash_prefill_capture_active = false;
    int32_t dflash_prefill_src_offset = 0;
    int32_t dflash_prefill_dst_offset = 0;
    int32_t dflash_prefill_n_tokens   = 0;
    int32_t dflash_prefill_src_offsets[LLAMA_DFLASH_MAX_SLOTS] = {};
    int32_t dflash_prefill_dst_offsets[LLAMA_DFLASH_MAX_SLOTS] = {};
    int32_t dflash_prefill_n_tokens_seqs[LLAMA_DFLASH_MAX_SLOTS] = {};

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
