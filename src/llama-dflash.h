#pragma once

// DFlash: cross-attention speculative decoding
// Struct definitions for hidden-state capture, tape recording, GPU staging,
// and drafter-side KV projection cache. Implementation in llama-dflash.cpp.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-arch.h"
#include "llama-graph.h"

#include <cstdint>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// DFlash constants (also as macros in llama.h)
#ifndef LLAMA_DFLASH_MAX_SLOTS
#define LLAMA_DFLASH_MAX_SLOTS         8
#define LLAMA_DFLASH_MAX_VERIFY_TOKENS 25
#define LLAMA_DFLASH_PER_SLOT_CTX      512
#endif

struct llama_ubatch;
struct llama_memory_recurrent;

// ---- arch helpers ----

static inline bool llama_dflash_gpu_hidden_supported_arch(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

static inline bool llama_dflash_gpu_tape_supported_arch(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return true;
        default:
            return false;
    }
}

static inline int llama_dflash_capture_tokens_per_seq(uint32_t n_tokens, uint32_t n_seq_tokens, uint32_t n_seqs_unq) {
    return n_seqs_unq > 1 ? (int) n_seq_tokens : (int) n_tokens;
}

static inline bool llama_dflash_suppress_callback_for_prefill_ubatch_for_test(
        bool prefill_plan_active,
        bool use_prefill_staging,
        bool has_intersection) {
    return prefill_plan_active && use_prefill_staging && !has_intersection;
}

static inline bool llama_dflash_prefill_plan_needs_staging_for_test(
        int planned_tokens,
        int current_ubatch_tokens) {
    GGML_UNUSED(current_ubatch_tokens);
    return planned_tokens > LLAMA_DFLASH_MAX_VERIFY_TOKENS;
}

// ---- data structs ----

struct dflash_layer_hidden_buf {
    std::vector<float> data;
    int64_t n_embd = 0;
    int64_t n_tokens = 0;
};

struct dflash_tape_layer {
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> gate;
    std::vector<float> beta;
    std::vector<float> qkv_mixed;
    int64_t S_k = 0, H_k = 0, S_v = 0, H_v = 0;
    int64_t conv_channels = 0;
    int n_tokens = 0;
    int n_seqs = 1;
    llama_seq_id seq_ids[LLAMA_DFLASH_MAX_SLOTS] = {};
};

struct dflash_tape_gpu_layer {
    ggml_tensor * k    = nullptr;
    ggml_tensor * v    = nullptr;
    ggml_tensor * gate = nullptr;
    ggml_tensor * beta = nullptr;
    ggml_tensor * qkv  = nullptr;
};

struct dflash_tape_gpu {
    std::vector<dflash_tape_gpu_layer> layers;
    std::vector<int32_t> layer_ids;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;
    int max_tokens = 0;
    int n_tokens = 0;

    ~dflash_tape_gpu() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

struct dflash_hidden_gpu {
    std::vector<ggml_tensor *> layers;
    std::vector<int32_t> layer_ids;
    ggml_backend_buffer_t buf = nullptr;
    ggml_context * ctx = nullptr;
    int64_t n_embd = 0;
    int max_tokens = 0;
    int n_tokens = 0;

    ~dflash_hidden_gpu() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

enum dflash_tape_type {
    DFLASH_TAPE_K    = 0,
    DFLASH_TAPE_V    = 1,
    DFLASH_TAPE_GATE = 2,
    DFLASH_TAPE_BETA = 3,
    DFLASH_TAPE_QKV  = 4,
};

struct llama_tree_mask {
    bool active = false;
    int n_tree_tokens = 0;
    std::vector<uint8_t> visibility;
};

struct dflash_prefill_capture_plan {
    bool active = false;
    llama_seq_id seq_id = -1;
    int32_t capture_begin = 0;
    int32_t capture_end   = 0;
    int32_t n_tokens = 0;
    int32_t n_written = 0;
};

struct dflash_capture_data {
    bool capture_active = true;

    std::vector<int32_t> layer_ids;
    std::vector<std::string> tensor_names;
    std::unordered_map<std::string, size_t> hidden_name_idx;
    std::vector<std::vector<dflash_layer_hidden_buf>> * hiddens = nullptr;

    bool tape_enabled = false;
    bool gpu_capture_enabled = true;
    std::vector<int32_t> recurrent_layer_ids;
    std::unordered_map<std::string, std::pair<int, int>> tape_name_map;
    std::vector<dflash_tape_layer> tape_layers;

    std::vector<std::unique_ptr<dflash_tape_gpu>> tapes;
    std::vector<std::unique_ptr<dflash_hidden_gpu>> hidden_gpu;
    std::vector<std::unique_ptr<dflash_hidden_gpu>> prefill_gpu;
    int prefill_gpu_max_tokens = 0;
    std::vector<dflash_prefill_capture_plan> prefill_plans;
    int active_tape_idx = 0;

    const llama_ubatch * ubatch = nullptr;
    std::vector<float> scatter_buf;

    uint32_t profile_flags = 0;
    bool profile = false;
    bool multi_gpu_capture_fallback_logged = false;
    bool multi_gpu_replay_fallback_logged = false;
    uint64_t profile_decode_us = 0;
    uint64_t profile_output_extract_us = 0;
    uint64_t profile_raw_logits_us = 0;
    uint64_t profile_raw_logits_bytes = 0;
    uint64_t profile_raw_logits_skipped = 0;
    uint64_t profile_reduced_logits_us = 0;
    uint64_t profile_reduced_logits_ids_us = 0;
    uint64_t profile_reduced_logits_probs_us = 0;
    uint64_t profile_reduced_logits_bytes = 0;
    uint64_t profile_verify_sync_split_us = 0;
    uint64_t profile_cb_ask = 0;
    uint64_t profile_cb_hidden_ask = 0;
    uint64_t profile_cb_tape_ask = 0;
    uint64_t profile_cb_qkv_ask = 0;
    uint64_t profile_cb_read = 0;
    uint64_t profile_cb_hidden_read = 0;
    uint64_t profile_cb_tape_read = 0;
    uint64_t profile_cb_qkv_read = 0;
    uint64_t profile_replay_wait_us = 0;
    uint64_t profile_replay_gdn_enqueue_us = 0;
    uint64_t profile_replay_gdn_wait_us = 0;
    uint64_t profile_replay_conv_enqueue_us = 0;
    uint64_t profile_replay_conv_wait_us = 0;
    uint64_t profile_replay_layers = 0;
    uint64_t profile_replay_sync_calls = 0;
    uint64_t profile_replay_direct_gpu = 0;
    uint64_t profile_replay_ggml_gpu = 0;
    uint64_t profile_replay_cpu_fallback = 0;
    uint64_t profile_conv_gpu_us = 0;
    uint64_t profile_conv_read_wait_us = 0;
    uint64_t profile_conv_cpu_us = 0;
    uint64_t profile_conv_write_wait_us = 0;
    std::unordered_map<std::string, uint64_t> profile_cb_names;

    using sync_backend_to_stream_fn_t = bool (*)(ggml_backend_t);
    sync_backend_to_stream_fn_t fn_sync_backend_to_stream = nullptr;
    ggml_backend_t sync_backend_to_stream_backend = nullptr;

    dflash_tape_gpu * active_tape() const {
        return (active_tape_idx >= 0 && active_tape_idx < (int) tapes.size())
                   ? tapes[active_tape_idx].get() : nullptr;
    }

    dflash_hidden_gpu * active_hidden_gpu() const {
        return (active_tape_idx >= 0 && active_tape_idx < (int) hidden_gpu.size())
                   ? hidden_gpu[active_tape_idx].get() : nullptr;
    }

    dflash_prefill_capture_plan * prefill_plan_for_seq(llama_seq_id seq_id) {
        return (seq_id >= 0 && seq_id < (llama_seq_id) prefill_plans.size())
                   ? &prefill_plans[(size_t) seq_id] : nullptr;
    }

    const dflash_prefill_capture_plan * prefill_plan_for_seq(llama_seq_id seq_id) const {
        return (seq_id >= 0 && seq_id < (llama_seq_id) prefill_plans.size())
                   ? &prefill_plans[(size_t) seq_id] : nullptr;
    }

    bool any_prefill_plan_active() const {
        for (const auto & plan : prefill_plans) {
            if (plan.active && plan.n_tokens > 0) return true;
        }
        return false;
    }

    int max_prefill_plan_tokens() const {
        int max_tokens = 0;
        for (const auto & plan : prefill_plans) {
            if (plan.active && plan.n_tokens > 0) {
                max_tokens = std::max(max_tokens, (int) plan.n_tokens);
            }
        }
        return max_tokens;
    }

    std::vector<dflash_layer_hidden_buf> * slot_hiddens(int slot) const {
        if (!hiddens || slot < 0 || slot >= (int) hiddens->size()) return nullptr;
        return &(*hiddens)[slot];
    }

    std::vector<dflash_layer_hidden_buf> * active_slot_hiddens() const {
        return slot_hiddens(active_tape_idx);
    }

    ggml_backend_buffer_t replay_buf = nullptr;
    size_t replay_buf_size = 0;
    std::vector<float> replay_zeros;

    bool replay_pending = false;
    ggml_backend_t replay_gpu_backend = nullptr;
    ggml_context * replay_graph_ctx = nullptr;
    ggml_backend_event_t replay_event = nullptr;
    bool replay_direct_gpu = false;
    const void * replay_sync_ptr = nullptr;
    std::vector<const void *> replay_sync_ptrs;
    int replay_sync_device = -1;
    int replay_n_accepted = 0;
    int32_t replay_cell_idx = -1;
    llama_seq_id replay_seq_id = 0;
    llama_memory_recurrent * replay_mem_recurrent = nullptr;

    ~dflash_capture_data() {
        if (replay_graph_ctx) ggml_free(replay_graph_ctx);
        if (replay_buf) ggml_backend_buffer_free(replay_buf);
        if (replay_event) ggml_backend_event_free(replay_event);
    }
};

struct dflash_kv_cache_data {
    llama_dflash_kv_cache_view view;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_backend_buffer_t update_buf = nullptr;
    ggml_backend_buffer_t shift_buf = nullptr;
    size_t update_buf_size = 0;
    size_t shift_buf_size = 0;
    void * shift_ptr = nullptr;

    std::vector<ggml_tensor *> k_ring;
    std::vector<ggml_tensor *> v_ring;

    int ring_size = 0;
    int write_pos = 0;
    int n_filled = 0;
    int n_elem = 0;

    using write_d2d_fn_t = bool (*)(void *, const void *, int, int, int, int);
    using append_d2d_fn_t = bool (*)(void *, const void *, int, int, int, int);
    using copy_d2d_fn_t = bool (*)(void *, const void *, size_t);
    using prepare_ptr_fn_t = bool (*)(const void *);
    using sync_ptr_fn_t = bool (*)(const void *);
    using sync_backend_stream_fn_t = bool (*)(ggml_backend_t);
    using interleave_fn_t = bool (*)(const void *, void *, int, int, int, int, int);
    write_d2d_fn_t fn_write_d2d = nullptr;
    write_d2d_fn_t fn_write_d2d_no_check = nullptr;
    append_d2d_fn_t fn_append_d2d = nullptr;
    append_d2d_fn_t fn_append_d2d_no_check = nullptr;
    copy_d2d_fn_t fn_copy_d2d = nullptr;
    copy_d2d_fn_t fn_copy_d2d_no_check = nullptr;
    prepare_ptr_fn_t fn_prepare_ptr = nullptr;
    sync_ptr_fn_t fn_sync_ptr = nullptr;
    sync_backend_stream_fn_t fn_wait_backend_stream = nullptr;
    sync_backend_stream_fn_t fn_wait_dflash_stream = nullptr;
    interleave_fn_t fn_interleave = nullptr;

    ~dflash_kv_cache_data() {
        if (update_buf) ggml_backend_buffer_free(update_buf);
        if (shift_buf) ggml_backend_buffer_free(shift_buf);
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

// DFlash ring buffer statistics (returned by speculative API)
struct common_dflash_ring_stats {
    int32_t ring_write_pos = 0;
    int32_t ring_capacity  = 0;
    int32_t ring_used      = 0;
};
