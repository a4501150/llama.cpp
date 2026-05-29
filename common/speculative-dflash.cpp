// DFlash cross-attention speculative decoding
// Adapted from BeeLlama's common_speculative_state_dflash to upstream's
// common_speculative_impl interface. #included from speculative.cpp.

// dflash-profile.h is in src/ — use relative path from common/
#include "../src/dflash-profile.h"

#include <cmath>
#include <limits>
#include <queue>

// ---------------------------------------------------------------------------
// static helpers
// ---------------------------------------------------------------------------

static bool common_dflash_debug_logs_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_DEBUG");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool common_dflash_kv_cache_disabled() {
    static const bool disabled = [] {
        const char * env = std::getenv("GGML_DFLASH_DISABLE_KV_CACHE");
        if (env && std::atoi(env) != 0) return true;
        const char * mode = std::getenv("GGML_DFLASH_KV_CACHE_MODE");
        return mode && (std::strcmp(mode, "off") == 0 || std::strcmp(mode, "none") == 0 || std::strcmp(mode, "disabled") == 0);
    }();
    return disabled;
}

static bool common_dflash_gpu_ring_env_enabled() {
    const char * env = std::getenv("GGML_DFLASH_GPU_RING");
    return env == nullptr || std::atoi(env) != 0;
}

static bool common_dflash_force_cpu_cross_ring() {
    static const bool v = [] {
        const char * e = getenv("GGML_DFLASH_FORCE_CPU_CROSS");
        return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

static bool common_dflash_log_contract_verbose() {
    static const bool v = [] {
        const char * e = getenv("GGML_DFLASH_VERBOSE_CONTRACT");
        return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    return v;
}

static bool common_dflash_gpu_ring_allowed(llama_context * ctx_tgt, llama_context * ctx_dft) {
    if (!common_dflash_gpu_ring_env_enabled()) {
        LOG_INF("dflash: GPU cross ring disabled by GGML_DFLASH_GPU_RING=0\n");
        return false;
    }
    const int32_t n_tgt = ctx_tgt ? llama_model_n_devices(llama_get_model(ctx_tgt)) : 1;
    const int32_t n_dft = ctx_dft ? llama_model_n_devices(llama_get_model(ctx_dft)) : 1;
    if (n_tgt > 1 || n_dft > 1) {
        LOG_INF("dflash: multi-GPU detected (target=%d drafter=%d); disabling GPU cross ring\n", n_tgt, n_dft);
        return false;
    }
    return true;
}

static bool common_dflash_argmax_token_valid(int32_t token_id, int n_vocab) {
    return token_id >= 0 && token_id < n_vocab;
}

static bool common_dflash_argmax_shape_valid(const char * where, int rows_available, int rows_required, int top_k) {
    if (top_k < 1 || rows_available < rows_required) {
        LOG_ERR("dflash: invalid reduced-logits shape in %s (rows=%d required=%d top_k=%d)\n",
                where, rows_available, rows_required, top_k);
        return false;
    }
    return true;
}

common_dflash_ring_write common_dflash_ring_write_plan(int ring_size, int ring_pos, int n_tokens) {
    if (ring_size <= 0 || n_tokens <= 0) {
        return { 0, 0, 0 };
    }
    int normalized_pos = ring_pos % ring_size;
    if (normalized_pos < 0) normalized_pos += ring_size;
    if (n_tokens <= ring_size) {
        return { normalized_pos, n_tokens, 0 };
    }
    const int skip = n_tokens - ring_size;
    normalized_pos = (normalized_pos + skip) % ring_size;
    return { normalized_pos, ring_size, skip };
}

static bool common_dflash_layer_ids_unique(const std::vector<int32_t> & ids) {
    std::vector<int32_t> sorted = ids;
    std::sort(sorted.begin(), sorted.end());
    return std::unique(sorted.begin(), sorted.end()) == sorted.end();
}

static std::string common_dflash_layer_ids_to_string(const std::vector<int32_t> & ids) {
    std::string result = "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) result += ",";
        result += std::to_string(ids[i]);
    }
    result += "]";
    return result;
}

// ---------------------------------------------------------------------------
// test helpers
// ---------------------------------------------------------------------------

bool common_dflash_prefill_capture_complete_for_test(int captured, int requested) {
    return captured >= requested;
}

bool common_dflash_cpu_ring_valid_after_write_for_test(
        bool was_valid, bool force_cpu_ring, bool has_gpu_ring, bool cpu_ring_written_all) {
    const bool cpu_ring_should_track = force_cpu_ring || !has_gpu_ring;
    if (!cpu_ring_should_track) return false;
    return cpu_ring_written_all;
    GGML_UNUSED(was_valid);
}

bool common_dflash_should_refuse_large_prefill_fallback_for_test(
        int requested, bool use_prefill_gpu, bool has_gpu_ring) {
    return !use_prefill_gpu && requested > LLAMA_DFLASH_MAX_VERIFY_TOKENS && has_gpu_ring;
}

bool common_dflash_cpu_ring_valid_after_source_write_for_test(
        bool was_valid, int source, bool force_cpu_ring, bool has_gpu_ring, bool cpu_data_all_layers) {
    const bool source_has_cpu = (source == 0);
    const bool cpu_ring_should_track = source_has_cpu && (force_cpu_ring || !has_gpu_ring);
    if (!cpu_ring_should_track) return false;
    return cpu_data_all_layers;
    GGML_UNUSED(was_valid);
}

bool common_dflash_tree_update_requires_cpu_hidden_for_test(bool has_cpu_hidden, bool has_gpu_ring) {
    return !has_cpu_hidden && has_gpu_ring;
}

// ---------------------------------------------------------------------------
// capture source enum
// ---------------------------------------------------------------------------

enum class dflash_capture_source {
    cpu_hidden,
    verify_gpu_hidden,
    prefill_gpu_hidden,
};

// ---------------------------------------------------------------------------
// common_speculative_impl_dflash
// ---------------------------------------------------------------------------

struct dflash_seq_state {
    std::vector<std::vector<float>> ring_buf;
    int ring_write_pos = 0;
    int ring_filled = 0;
    int committed_len = 0;
    bool cpu_ring_valid = true;
    bool prefill_flushed = false;
    bool prefill_flush_called = false;
    int  prefill_flush_requested = 0;
    int  prefill_flush_written = 0;
    bool prefill_suffix_seen = false;
    std::vector<float> cross_buf;
    std::vector<float> gpu_restore_staging;
    void * gpu_ring_handle = nullptr;
    bool kv_cache_init_attempted = false;
    bool kv_cache_enabled = false;
    bool ring_write_discarded = false;
};

struct common_speculative_impl_dflash : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    llama_model   * model_dft;
    bool            owns_ctx_dft;

    int block_size;
    llama_token mask_token_id;
    int n_target_layers;
    int n_embd;
    int n_target_features;

    static constexpr int RING_SIZE = 4096;

    int cross_ctx;

    uint32_t profile_flags = dflash_profile_flags();

    std::vector<int32_t> capture_layers;
    bool target_capture_enabled = true;
    bool gpu_capture_available = false;
    bool gpu_ring_enabled = false;

    llama_batch batch_dft;

    // params cached from construction
    int param_n_max = 8;
    int param_n_min = 1;
    float param_p_min = 0.0f;
    int param_n_max_base = 0;

    // per-seq state
    std::unordered_map<llama_seq_id, dflash_seq_state> seq_states;
    llama_seq_id active_seq_id = -1;
    int max_seqs = 1;

    dflash_seq_state & active_state() {
        auto it = seq_states.find(active_seq_id);
        GGML_ASSERT(it != seq_states.end());
        return it->second;
    }

    const dflash_seq_state & active_state() const {
        auto it = seq_states.find(active_seq_id);
        GGML_ASSERT(it != seq_states.end());
        return it->second;
    }

    dflash_seq_state & ensure_seq_state(llama_seq_id sid) {
        auto it = seq_states.find(sid);
        if (it != seq_states.end()) return it->second;
        auto & ss = seq_states[sid];
        ss.ring_buf.resize(n_target_layers);
        for (int i = 0; i < n_target_layers; ++i) {
            ss.ring_buf[i].resize((size_t)RING_SIZE * n_embd, 0.0f);
        }
        if (gpu_ring_enabled) {
            ss.gpu_ring_handle = llama_dflash_cross_ring_gpu_init(ctx_dft, n_target_layers, n_embd, cross_ctx);
            if (ss.gpu_ring_handle) {
                LOG_INF("dflash: GPU cross ring enabled for seq %d (%d layers x %d slots x %d embd)\n",
                        (int)sid, n_target_layers, cross_ctx, n_embd);
            }
        }
        return ss;
    }

    // ---- private helpers ----

    void discard_cross_ring(const char * reason) {
        if (reason && reason[0]) {
            LOG_WRN("dflash: discarding cross-ring state (seq %d): %s\n", (int)active_seq_id, reason);
        }
        auto & ss = active_state();
        ss.ring_write_pos = 0;
        ss.ring_filled = 0;
        ss.committed_len = 0;
        ss.cpu_ring_valid = true;
        ss.prefill_flushed = false;
        ss.prefill_flush_called = false;
        ss.prefill_flush_requested = 0;
        ss.prefill_flush_written = 0;
        ss.prefill_suffix_seen = false;
        ss.ring_write_discarded = true;
        ss.cross_buf.clear();
        llama_dflash_kv_cache_reset(ctx_dft);
    }

    bool validate_target_hiddens(const char * where) {
        const int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots != n_target_layers) {
            LOG_WRN("dflash: %s hidden slot count mismatch: got=%d expected=%d\n", where, (int) n_slots, n_target_layers);
            return false;
        }
        for (int i = 0; i < n_slots; ++i) {
            const int64_t h_embd = llama_get_layer_hidden_n_embd(ctx_tgt, i);
            const int64_t h_tok  = llama_get_layer_hidden_n_tokens(ctx_tgt, i);
            if (h_embd != n_embd || h_tok < 0) {
                LOG_WRN("dflash: %s hidden[%d] shape mismatch: embd=%lld expected=%d tokens=%lld\n",
                        where, i, (long long) h_embd, n_embd, (long long) h_tok);
                return false;
            }
        }
        return true;
    }

    bool profile_enabled_flag(uint32_t flags) const {
        return dflash_profile_has(profile_flags, flags);
    }

    int build_cross_data(llama_context * ctx) {
        auto & ss = active_state();
        if (ss.gpu_ring_handle) {
            int gpu_write_pos = ss.ring_write_pos % cross_ctx;
            int gpu_filled = std::min(ss.ring_filled, cross_ctx);
            llama_dflash_cross_ring_gpu_set_cross(ctx, ss.gpu_ring_handle, active_seq_id,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, cross_ctx);
            return gpu_filled;
        }
        if (!ss.cpu_ring_valid) {
            LOG_WRN("dflash: CPU cross ring is stale and GPU ring unavailable (seq %d)\n", (int)active_seq_id);
            return -1;
        }
        int cross_len = std::min(ss.ring_filled, cross_ctx > 0 ? cross_ctx : ss.ring_filled);
        ss.cross_buf.resize((size_t)n_target_features * cross_len);
        int read_start = (ss.ring_write_pos - cross_len + RING_SIZE) % RING_SIZE;
        for (int t = 0; t < cross_len; ++t) {
            int slot = (read_start + t) % RING_SIZE;
            for (int layer = 0; layer < n_target_layers; ++layer) {
                memcpy(&ss.cross_buf[(size_t)(layer * n_embd) + (size_t)t * n_target_features],
                       ss.ring_buf[layer].data() + (size_t)slot * n_embd,
                       n_embd * sizeof(float));
            }
        }
        llama_set_cross_data_seq(ctx, active_seq_id, ss.cross_buf.data(), n_target_features, cross_len);
        return cross_len;
    }

    int drafter_prefix_window() const {
        return std::max(0, (int) llama_n_ctx(ctx_dft) - block_size);
    }

    void trim_drafter_prefix_window() {
        auto & ss = active_state();
        auto * mem_dft = llama_get_memory(ctx_dft);
        if (!mem_dft) return;
        llama_memory_seq_rm(mem_dft, active_seq_id, ss.committed_len, -1);
        const int keep = drafter_prefix_window();
        if (keep <= 0 || ss.committed_len <= keep) return;
        const llama_pos trim_before = ss.committed_len - keep;
        llama_memory_seq_rm(mem_dft, active_seq_id, 0, trim_before);
    }

    void update_drafter_kv_cache(int n_written) {
        auto & ss = active_state();
        if (!ss.gpu_ring_handle || n_written <= 0) return;
        trim_drafter_prefix_window();
        const int n_update = std::min(n_written, cross_ctx);
        const int n_full_kv_update = std::min(n_update, drafter_prefix_window());
        const int gpu_write_pos = ss.ring_write_pos % cross_ctx;
        const int gpu_filled = std::min(ss.ring_filled, cross_ctx);

        if (n_full_kv_update > 0) {
            const llama_pos start_pos = ss.committed_len - n_full_kv_update;
            const bool full_kv_ok = llama_dflash_target_kv_cache_update_from_ring(
                ctx_dft, ss.gpu_ring_handle,
                gpu_write_pos, gpu_filled,
                n_target_layers, n_embd, n_full_kv_update,
                active_seq_id, start_pos);
            if (!full_kv_ok) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), active_seq_id, start_pos, -1);
            }
        }

        if (common_dflash_kv_cache_disabled()) return;

        if (!ss.kv_cache_init_attempted) {
            ss.kv_cache_init_attempted = true;
            ss.kv_cache_enabled = llama_dflash_kv_cache_init(ctx_dft, cross_ctx);
            if (ss.kv_cache_enabled) {
                LOG_INF("dflash: drafter K/V projection cache enabled (%d-token window)\n", cross_ctx);
            }
        }
        if (!ss.kv_cache_enabled) return;

        const bool ok = llama_dflash_kv_cache_update_from_ring(ctx_dft, ss.gpu_ring_handle,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, n_update);
        if (!ok) {
            llama_dflash_kv_cache_reset(ctx_dft);
            ss.kv_cache_enabled = false;
        }
    }

    void log_ring_profile(const char * name, int n_tokens, int actual_written,
            int64_t cpu_copy_us, int64_t gpu_enqueue_us, int64_t gpu_sync_us) const {
        if (!profile_enabled_flag(DFLASH_PROFILE_COPY)) return;
        const auto & ss = active_state();
        LOG_INF("dflash profile: %s seq=%d requested=%d written=%d cpu_copy=%.3f ms gpu_enqueue=%.3f ms gpu_sync=%.3f ms ring_filled=%d committed=%d gpu=%d\n",
                name, (int)active_seq_id, n_tokens, actual_written,
                cpu_copy_us / 1e3, gpu_enqueue_us / 1e3, gpu_sync_us / 1e3,
                ss.ring_filled, ss.committed_len, ss.gpu_ring_handle ? 1 : 0);
    }

    int ring_write(int n_tokens, int src_offset = 0, bool force_cpu_ring = false,
                   dflash_capture_source source = dflash_capture_source::cpu_hidden) {
        if (n_tokens <= 0) return 0;
        auto & ss = active_state();
        ss.ring_write_discarded = false;

        const bool use_prefill_gpu = (source == dflash_capture_source::prefill_gpu_hidden);
        const bool source_has_cpu_hidden = (source == dflash_capture_source::cpu_hidden);
        const bool cpu_ring_should_track = source_has_cpu_hidden && (force_cpu_ring || !ss.gpu_ring_handle);

        if (!cpu_ring_should_track) ss.cpu_ring_valid = false;

        const int32_t n_src_layers = use_prefill_gpu ? n_target_layers : llama_get_n_layer_hiddens(ctx_tgt);
        int actual_written = n_tokens;
        bool first_layer = true;
        bool any_layer = false;
        for (int layer = 0; layer < n_target_layers && layer < n_src_layers; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (use_prefill_gpu && !data) { any_layer = true; continue; }
            if ((!data && !ss.gpu_ring_handle) || ntok <= 0) continue;
            int to_write = std::min(n_tokens, (int)std::max((int64_t)0, ntok - src_offset));
            if (first_layer) { actual_written = to_write; first_layer = false; }
            else { actual_written = std::min(actual_written, to_write); }
            any_layer = true;
        }
        if (!any_layer || actual_written <= 0) return 0;

        bool gpu_upload_queued = false;
        bool gpu_d2d_failed = false;
        int64_t cpu_copy_us = 0, gpu_enqueue_us = 0;
        bool cpu_ring_written_all = cpu_ring_should_track;

        for (int layer = 0; layer < n_target_layers && layer < n_src_layers; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
            if (use_prefill_gpu && !data) { embd = n_embd; }
            else {
                int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
                if ((!data && !ss.gpu_ring_handle) || ntok <= 0) continue;
            }

            if (cpu_ring_should_track && data) {
                for (int t = 0; t < actual_written; ++t) {
                    int slot = (ss.ring_write_pos + t) % RING_SIZE;
                    memcpy(ss.ring_buf[layer].data() + (size_t)slot * embd,
                           data + (size_t)(src_offset + t) * embd,
                           embd * sizeof(float));
                }
            } else if (cpu_ring_should_track) {
                cpu_ring_written_all = false;
            }

            if (ss.gpu_ring_handle) {
                const auto plan = common_dflash_ring_write_plan(cross_ctx, ss.ring_write_pos, actual_written);
                if (plan.n_tokens > 0) {
                    bool used_d2d = false;
                    if (!data) {
                        if (use_prefill_gpu) {
                            used_d2d = llama_dflash_prefill_gpu_write_hidden(ss.gpu_ring_handle, ctx_tgt, active_seq_id, layer, plan.ring_pos,
                                src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            if (!used_d2d) {
                                used_d2d = llama_dflash_cross_ring_gpu_write_hidden(ss.gpu_ring_handle, ctx_tgt, layer, plan.ring_pos,
                                    src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            }
                        } else {
                            used_d2d = llama_dflash_cross_ring_gpu_write_hidden(ss.gpu_ring_handle, ctx_tgt, layer, plan.ring_pos,
                                src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            if (!used_d2d) {
                                used_d2d = llama_dflash_prefill_gpu_write_hidden(ss.gpu_ring_handle, ctx_tgt, active_seq_id, layer, plan.ring_pos,
                                    src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            }
                        }
                        gpu_d2d_failed = gpu_d2d_failed || !used_d2d;
                    }
                    if (!used_d2d && data) {
                        llama_dflash_cross_ring_gpu_write(ss.gpu_ring_handle, layer, plan.ring_pos,
                            data + (size_t)(src_offset + plan.src_token_offset) * embd,
                            plan.n_tokens, embd);
                    }
                    gpu_upload_queued = true;
                }
            }
        }
        if (gpu_d2d_failed) {
            discard_cross_ring("GPU hidden D2D ring write failed");
            return 0;
        }
        int64_t gpu_sync_us = 0;
        if (gpu_upload_queued) {
            llama_dflash_cross_ring_gpu_synchronize(ss.gpu_ring_handle);
        }
        log_ring_profile("ring_write", n_tokens, actual_written, cpu_copy_us, gpu_enqueue_us, gpu_sync_us);
        if (cpu_ring_should_track) { ss.cpu_ring_valid = cpu_ring_written_all; }
        else { ss.cpu_ring_valid = false; }
        ss.ring_write_pos = (ss.ring_write_pos + actual_written) % RING_SIZE;
        ss.ring_filled = std::min(ss.ring_filled + actual_written, RING_SIZE);
        return actual_written;
    }

    int ring_write_by_indices(const std::vector<int> & indices) {
        const int n_tokens = (int)indices.size();
        if (n_tokens <= 0) return 0;
        auto & ss = active_state();

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        int actual_written = n_tokens;
        bool first_layer = true, any_layer = false;
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (!data || ntok <= 0) continue;
            int wrote = 0;
            for (int t = 0; t < n_tokens; ++t) {
                if (indices[t] < 0 || indices[t] >= (int)ntok) break;
                wrote++;
            }
            if (first_layer) { actual_written = wrote; first_layer = false; }
            else { actual_written = std::min(actual_written, wrote); }
            any_layer = true;
        }
        if (!any_layer || actual_written <= 0) return 0;

        bool gpu_upload_queued = false;
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (!data || ntok <= 0) continue;
            for (int t = 0; t < actual_written; ++t) {
                int ring_slot = (ss.ring_write_pos + t) % RING_SIZE;
                memcpy(ss.ring_buf[layer].data() + (size_t)ring_slot * embd,
                       data + (size_t)indices[t] * embd, embd * sizeof(float));
                if (ss.gpu_ring_handle) {
                    int gpu_pos = (ss.ring_write_pos + t) % cross_ctx;
                    llama_dflash_cross_ring_gpu_write(ss.gpu_ring_handle, layer, gpu_pos,
                        data + (size_t)indices[t] * embd, 1, embd);
                    gpu_upload_queued = true;
                }
            }
        }
        if (gpu_upload_queued) {
            llama_dflash_cross_ring_gpu_synchronize(ss.gpu_ring_handle);
        }
        ss.ring_write_pos = (ss.ring_write_pos + actual_written) % RING_SIZE;
        ss.ring_filled = std::min(ss.ring_filled + actual_written, RING_SIZE);
        return actual_written;
    }

    void capture_target_hiddens() {
        llama_dflash_set_active_slot(ctx_tgt, active_seq_id);
        auto & ss = active_state();
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;
        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n_tokens <= 0) return;
        int start_offset = std::max(0, (int)n_tokens - RING_SIZE);
        int to_store = (int)n_tokens - start_offset;
        ss.ring_write_pos = 0;
        ss.ring_filled = 0;
        llama_dflash_kv_cache_reset(ctx_dft);
        const int actual_written = ring_write(to_store, start_offset, true);
        if (ss.ring_write_discarded) return;
        ss.committed_len = start_offset + actual_written;
        update_drafter_kv_cache(actual_written);
    }

    void append_target_hiddens(int n_accepted) {
        llama_dflash_set_active_slot(ctx_tgt, active_seq_id);
        auto & ss = active_state();
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || n_accepted <= 0) return;
        const int actual_written = ring_write(n_accepted);
        if (ss.ring_write_discarded) return;
        ss.committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
    }

    // ---- constructor ----

    common_speculative_impl_dflash(
            llama_context * ctx_tgt_,
            llama_context * ctx_dft_,
            llama_model   * model_dft_,
            bool            owns_ctx_dft_,
            int             cross_ctx_,
            int             n_max_,
            int             n_min_,
            float           p_min_,
            int             n_max_base_,
            int             n_seq_ = 1)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DFLASH, n_seq_)
        , ctx_tgt(ctx_tgt_)
        , ctx_dft(ctx_dft_)
        , model_dft(model_dft_)
        , owns_ctx_dft(owns_ctx_dft_)
        , cross_ctx(cross_ctx_ > 0 ? cross_ctx_ : 512)
        , batch_dft({})
        , param_n_max(n_max_)
        , param_n_min(n_min_)
        , param_p_min(p_min_)
        , param_n_max_base(n_max_base_)
        , max_seqs(n_seq_)
    {
        block_size        = llama_model_dflash_block_size(model_dft_);
        mask_token_id     = (llama_token) llama_model_dflash_mask_token_id(model_dft_);
        n_target_layers   = llama_model_dflash_n_target_layers(model_dft_);
        n_embd            = llama_model_n_embd(model_dft_);
        n_target_features = llama_model_dflash_n_target_features(model_dft_);

        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const int target_n_layer = model_tgt ? llama_model_n_layer(model_tgt) : 0;
        const int target_n_embd  = model_tgt ? llama_model_n_embd(model_tgt)  : 0;

        auto fail_contract = [&](const std::string & why) {
            throw std::runtime_error("dflash: invalid contract: " + why);
        };

        if (!model_tgt)          fail_contract("target model is null");
        if (block_size <= 1)     fail_contract("block_size must be > 1");
        if (mask_token_id < 0)   fail_contract("mask_token_id must be >= 0");
        if (n_target_layers <= 0) fail_contract("n_target_layers must be > 0");
        if (n_embd <= 0 || target_n_embd <= 0) fail_contract("embedding sizes must be > 0");
        if (n_embd != target_n_embd)            fail_contract("drafter/target n_embd mismatch");
        if (n_target_features != n_embd * n_target_layers) fail_contract("n_target_features inconsistent");

        capture_layers.assign(n_target_layers, 0);
        const int n_read = llama_model_dflash_target_layer_ids(model_dft_, capture_layers.data(), n_target_layers);
        if (n_read != n_target_layers) fail_contract("target_layer_ids read count mismatch");
        if (!common_dflash_layer_ids_unique(capture_layers)) fail_contract("duplicate target_layer_ids");
        for (const int32_t il : capture_layers) {
            if (il < 0 || il >= target_n_layer) fail_contract("target_layer_id out of range");
        }

        LOG_INF("dflash: contract ok: block_size=%d mask_token=%d layers=%s n_target_layers=%d n_embd=%d cross_ctx=%d n_seq=%d\n",
                block_size, (int) mask_token_id,
                common_dflash_layer_ids_to_string(capture_layers).c_str(),
                n_target_layers, n_embd, cross_ctx, max_seqs);

        llama_set_dflash_capture(ctx_tgt, capture_layers.data(), n_target_layers);
        target_capture_enabled = true;

        batch_dft = llama_batch_init(block_size * max_seqs, 0, max_seqs);

        const bool gpu_ring_allowed = common_dflash_gpu_ring_allowed(ctx_tgt, ctx_dft);
        const bool gpu_ring_forced_off = common_dflash_force_cpu_cross_ring();
        gpu_ring_enabled = gpu_ring_allowed && !gpu_ring_forced_off;

        llama_set_dflash_gpu_capture(ctx_tgt, gpu_ring_enabled);

        // pre-allocate seq_state for seq 0
        active_seq_id = 0;
        auto & ss0 = ensure_seq_state(0);
        if (ss0.gpu_ring_handle) {
            gpu_capture_available = true;
        } else if (gpu_ring_enabled) {
            llama_set_dflash_gpu_capture(ctx_tgt, false);
            gpu_ring_enabled = false;
            LOG_WRN("dflash: GPU cross ring unavailable; using CPU hidden capture\n");
        }
    }

    ~common_speculative_impl_dflash() override {
        for (auto & [sid, ss] : seq_states) {
            llama_dflash_cross_ring_gpu_free(ss.gpu_ring_handle);
        }
        llama_batch_free(batch_dft);
        if (owns_ctx_dft) {
            llama_free(ctx_dft);
        }
    }

    // ---- common_speculative_impl virtuals ----

    void begin(llama_seq_id seq_id_, const llama_tokens & prompt) override {
        active_seq_id = seq_id_;
        ensure_seq_state(active_seq_id);
        GGML_UNUSED(prompt);

        auto & ss = active_state();
        if (ss.prefill_suffix_seen && !ss.prefill_flush_called) {
            LOG_ERR("dflash: prefill suffix scheduled but flush_prefill() never called (seq %d)\n", (int)active_seq_id);
        }

        if (ss.prefill_flushed) {
            ss.prefill_flushed = false;
            ss.prefill_flush_called = false;
            ss.prefill_flush_requested = 0;
            ss.prefill_flush_written = 0;
            ss.prefill_suffix_seen = false;
            llama_dflash_prefill_capture_end(ctx_tgt);
            return;
        }
        if (!validate_target_hiddens("begin")) return;
        capture_target_hiddens();
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        struct ready_info {
            llama_seq_id sid;
            int n_draft;
            int batch_len;
            int draft_pos_base;
            int dp_idx;
        };
        std::vector<ready_info> ready;

        for (auto & [sid, ss] : seq_states) {
            if (sid < 0 || sid >= (int)dparams.size()) continue;
            if (!dparams[sid].drafting) continue;
            if (ss.committed_len == 0) continue;

            const int n_draft = std::min(block_size - 1, dparams[sid].n_max > 0 ? (int)dparams[sid].n_max : param_n_max);

            active_seq_id = sid;
            llama_memory_seq_rm(llama_get_memory(ctx_dft), sid, ss.committed_len, -1);
            int cross_len = build_cross_data(ctx_dft);
            if (cross_len <= 0) continue;

            ready.push_back({sid, n_draft, n_draft + 1, ss.committed_len, (int)sid});
        }
        if (ready.empty()) return;

        llama_set_dflash_n_slots(ctx_dft, (int)ready.size());

        common_batch_clear(batch_dft);
        for (auto & r : ready) {
            auto & dp = dparams[r.dp_idx];
            common_batch_add(batch_dft, dp.id_last, r.draft_pos_base, { r.sid }, true);
            for (int i = 1; i < r.batch_len; ++i) {
                common_batch_add(batch_dft, mask_token_id, r.draft_pos_base + i, { r.sid }, true);
            }
        }

        int ret = llama_decode(ctx_dft, batch_dft);
        if (ret != 0) {
            LOG_ERR("dflash: batched drafter decode failed with %d\n", ret);
            return;
        }

        const int32_t * argmax = llama_get_logits_argmax(ctx_dft);
        const float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
        const int K_flat = llama_get_logits_argmax_k(ctx_dft);
        const int argmax_rows = llama_get_logits_argmax_n(ctx_dft);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));

        int logit_offset = 0;
        for (auto & r : ready) {
            auto & dp = dparams[r.dp_idx];

            if (argmax) {
                if (!common_dflash_argmax_shape_valid(__func__, argmax_rows, logit_offset + r.batch_len, K_flat)) {
                    dp.result->clear();
                    logit_offset += r.batch_len;
                    dp.drafting = false;
                    continue;
                }
                for (int i = 1; i < r.batch_len && (int)dp.result->size() < r.n_draft; ++i) {
                    int row = logit_offset + i;
                    if (argmax_probs && param_p_min > 0.0f && (int)dp.result->size() >= param_n_min) {
                        if (argmax_probs[row * K_flat] < std::log(param_p_min)) break;
                    }
                    const int32_t token_raw = argmax[row * K_flat];
                    if (!common_dflash_argmax_token_valid(token_raw, n_vocab)) {
                        dp.result->clear();
                        break;
                    }
                    dp.result->push_back((llama_token)token_raw);
                }
            } else {
                for (int i = 1; i < r.batch_len && (int)dp.result->size() < r.n_draft; ++i) {
                    float * logits = llama_get_logits_ith(ctx_dft, logit_offset + i);
                    if (!logits) break;
                    dp.result->push_back((llama_token)(std::max_element(logits, logits + n_vocab) - logits));
                }
            }

            logit_offset += r.batch_len;
            dp.drafting = false;
        }
    }

    void accept(llama_seq_id /*seq_id_a*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // DFlash uses update_logits() instead
    }

    bool need_embd() const override { return false; }

    // ---- DFlash-specific virtuals ----

    void set_seq_id(llama_seq_id seq_id_) override {
        active_seq_id = seq_id_;
        ensure_seq_state(seq_id_);
    }

    void set_prefill_capture_enabled(bool enabled) override {
        if (target_capture_enabled == enabled) return;
        target_capture_enabled = enabled;
        llama_set_dflash_capture_active(ctx_tgt, enabled);
    }

    void discard_dflash_state(const char * reason) override { discard_cross_ring(reason); }
    void note_prefill_suffix_scheduled() override { active_state().prefill_suffix_seen = true; }

    int prepare_batch_draft(llama_context * ctx_dft_ext) override {
        if (active_state().committed_len == 0) return -1;
        return build_cross_data(ctx_dft_ext);
    }

    void update_logits(llama_context * /*ctx*/, const llama_tokens & /*batch_tokens*/, int n_accepted) override {
        append_target_hiddens(n_accepted);
    }

    void update_logits_by_indices(llama_context * /*ctx*/, const std::vector<int> & capture_indices) override {
        llama_dflash_set_active_slot(ctx_tgt, active_seq_id);
        auto & ss = active_state();
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || capture_indices.empty()) return;
        float * cpu_hidden0 = llama_get_layer_hidden(ctx_tgt, 0);
        if (!cpu_hidden0 && ss.gpu_ring_handle) {
            LOG_ERR("dflash tree update requires CPU hidden data but only GPU capture available (seq %d)\n", (int)active_seq_id);
            ss.ring_write_pos = 0; ss.ring_filled = 0; ss.committed_len = 0; ss.cpu_ring_valid = false;
            llama_dflash_kv_cache_reset(ctx_dft);
            return;
        }
        const int actual_written = ring_write_by_indices(capture_indices);
        ss.committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
    }

    int flush_prefill(int src_offset, int n_tokens) override {
        llama_dflash_set_active_slot(ctx_tgt, active_seq_id);
        auto & ss = active_state();
        ss.prefill_flush_called = true;
        ss.prefill_flush_requested += n_tokens;

        const bool use_prefill_gpu = llama_dflash_prefill_gpu_active(ctx_tgt);
        if (!use_prefill_gpu && !validate_target_hiddens("flush_prefill")) return 0;

        const int32_t n_src_layers = use_prefill_gpu ? n_target_layers : llama_get_n_layer_hiddens(ctx_tgt);
        if (n_src_layers == 0) return 0;

        dflash_capture_source source;
        int64_t captured = 0;
        int offset = 0;

        if (use_prefill_gpu) {
            source = dflash_capture_source::prefill_gpu_hidden;
            int32_t planned = 0, written = 0;
            if (llama_dflash_prefill_capture_info(ctx_tgt, active_seq_id, &planned, &written)) {
                captured = written;
            } else {
                captured = llama_dflash_prefill_gpu_n_tokens(ctx_tgt, active_seq_id);
            }
            offset = 0;
            if (n_tokens > 0 && captured < n_tokens) {
                LOG_ERR("dflash prefill flush: incomplete GPU capture %lld/%d (seq %d)\n", (long long)captured, n_tokens, (int)active_seq_id);
                return 0;
            }
        } else {
            if (n_tokens > LLAMA_DFLASH_MAX_VERIFY_TOKENS && ss.gpu_ring_handle) {
                LOG_ERR("dflash prefill flush: expected GPU staging for large span %d (seq %d)\n", n_tokens, (int)active_seq_id);
                return 0;
            }
            float * cpu_hidden0 = llama_get_layer_hidden(ctx_tgt, 0);
            captured = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
            if (captured <= 0) return 0;
            if (cpu_hidden0) source = dflash_capture_source::cpu_hidden;
            else if (ss.gpu_ring_handle) source = dflash_capture_source::verify_gpu_hidden;
            else return 0;
            offset = n_tokens > 0 ? src_offset : 0;
        }

        if (captured <= 0) return 0;
        int to_write = n_tokens > 0 ? n_tokens : (int)captured;
        if (offset < 0) offset = 0;
        if (offset + to_write > (int)captured) to_write = std::max(0, (int)captured - offset);
        if (to_write <= 0) return 0;

        if (!ss.prefill_flushed) {
            ss.ring_write_pos = 0;
            ss.ring_filled = 0;
            ss.committed_len = 0;
            llama_dflash_kv_cache_reset(ctx_dft);
        }

        const bool force_cpu_ring_for_flush = (source == dflash_capture_source::cpu_hidden);
        const int actual_written = ring_write(to_write, offset, force_cpu_ring_for_flush, source);
        if (actual_written != to_write) {
            discard_cross_ring("incomplete prefill flush");
            return 0;
        }
        ss.committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
        ss.prefill_flushed = true;
        ss.prefill_flush_written += actual_written;
        return actual_written;
    }

    size_t ring_state_size() const override {
        const auto & ss = active_state();
        if (!ss.cpu_ring_valid && !ss.gpu_ring_handle) return 0;
        int n_entries = ss.cpu_ring_valid
            ? std::min(ss.ring_filled, RING_SIZE)
            : std::min(ss.ring_filled, cross_ctx);
        return 6 * sizeof(int32_t) + (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
    }

    bool ring_state_save(uint8_t * buf, size_t size) const override {
        const auto & ss = active_state();
        if (!ss.cpu_ring_valid && !ss.gpu_ring_handle) return false;
        const bool compact = !ss.cpu_ring_valid && ss.gpu_ring_handle;
        int n_entries = compact ? std::min(ss.ring_filled, cross_ctx) : std::min(ss.ring_filled, RING_SIZE);
        size_t expected = 6 * sizeof(int32_t) + (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
        if (size < expected) return false;

        int32_t * hdr = (int32_t *)buf;
        hdr[0] = compact ? (n_entries % RING_SIZE) : ss.ring_write_pos;
        hdr[1] = compact ? n_entries : ss.ring_filled;
        hdr[2] = ss.committed_len;
        hdr[3] = compact ? -n_target_layers : n_target_layers;
        hdr[4] = n_embd;
        hdr[5] = n_entries;

        uint8_t * dst = buf + 6 * sizeof(int32_t);
        size_t layer_bytes = (size_t)n_entries * n_embd * sizeof(float);

        if (compact) {
            const int gpu_write_pos = cross_ctx > 0 ? ss.ring_write_pos % cross_ctx : 0;
            return llama_dflash_cross_ring_gpu_snapshot(ss.gpu_ring_handle,
                    gpu_write_pos, ss.ring_filled, cross_ctx,
                    (float *) dst, n_entries, n_target_layers, n_embd);
        }

        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(dst, ss.ring_buf[l].data(), layer_bytes);
            dst += layer_bytes;
        }
        return true;
    }

    bool ring_state_load(const uint8_t * buf, size_t size) override {
        auto & ss = active_state();
        if (size < 6 * sizeof(int32_t)) return false;
        const int32_t * hdr = (const int32_t *)buf;
        int saved_write_pos = hdr[0], saved_filled = hdr[1], saved_committed = hdr[2];
        int saved_layers_raw = hdr[3], saved_embd = hdr[4], saved_entries = hdr[5];
        const bool compact = saved_layers_raw < 0;
        int saved_layers = compact ? -saved_layers_raw : saved_layers_raw;

        if (saved_layers != n_target_layers || saved_embd != n_embd) return false;
        if (saved_write_pos < 0 || saved_write_pos >= RING_SIZE ||
            saved_filled < 0 || saved_filled > RING_SIZE ||
            saved_entries < 0 || saved_entries != saved_filled ||
            saved_committed < saved_filled) return false;

        size_t layer_bytes = (size_t)saved_entries * n_embd * sizeof(float);
        if (size < 6 * sizeof(int32_t) + layer_bytes * n_target_layers) return false;

        ss.ring_write_pos = saved_write_pos;
        ss.ring_filled = saved_filled;
        ss.committed_len = saved_committed;

        const uint8_t * src = buf + 6 * sizeof(int32_t);
        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(ss.ring_buf[l].data(), src, layer_bytes);
            src += layer_bytes;
        }

        if (ss.gpu_ring_handle) {
            int gpu_entries = std::min(ss.ring_filled, cross_ctx);
            const size_t layer_floats = (size_t) gpu_entries * n_embd;
            ss.gpu_restore_staging.resize(layer_floats * n_target_layers);
            for (int l = 0; l < n_target_layers; ++l) {
                float * tmp = ss.gpu_restore_staging.data() + layer_floats * l;
                for (int t = 0; t < gpu_entries; ++t) {
                    int cpu_slot = (ss.ring_write_pos - gpu_entries + t + RING_SIZE) % RING_SIZE;
                    memcpy(tmp + (size_t)t * n_embd,
                           ss.ring_buf[l].data() + (size_t)cpu_slot * n_embd,
                           n_embd * sizeof(float));
                }
                int gpu_pos = ((ss.ring_write_pos - gpu_entries) % cross_ctx + cross_ctx) % cross_ctx;
                llama_dflash_cross_ring_gpu_write(ss.gpu_ring_handle, l, gpu_pos, tmp, gpu_entries, n_embd);
            }
            llama_dflash_cross_ring_gpu_synchronize(ss.gpu_ring_handle);
            llama_dflash_kv_cache_reset(ctx_dft);
            update_drafter_kv_cache(gpu_entries);
        }

        ss.prefill_flushed = true;
        ss.cpu_ring_valid = true;
        return true;
    }

    common_dflash_ring_stats dflash_ring_stats() const override {
        const auto & ss = active_state();
        common_dflash_ring_stats stats;
        stats.ring_write_pos = ss.ring_write_pos;
        stats.ring_filled    = ss.ring_filled;
        stats.committed_len  = ss.committed_len;
        stats.cross_ctx      = cross_ctx;
        stats.cross_len      = std::min(ss.ring_filled, cross_ctx > 0 ? cross_ctx : ss.ring_filled);
        return stats;
    }

    void draft_tree(const common_params_speculative & params, const llama_tokens & prompt_tgt,
                    llama_token id_last, int tree_budget, common_speculative_tree & tree) override {
        auto & ss = active_state();
        const int n_draft = std::min(param_n_max, block_size - 1);
        if (n_draft <= 0 || ss.committed_len == 0) return;

        llama_memory_seq_rm(llama_get_memory(ctx_dft), active_seq_id, ss.committed_len, -1);
        int cross_len = build_cross_data(ctx_dft);
        if (cross_len <= 0) return;

        common_batch_clear(batch_dft);
        const int draft_pos_base = ss.committed_len;
        common_batch_add(batch_dft, id_last, draft_pos_base, { active_seq_id }, true);
        for (int i = 1; i < block_size; ++i) {
            common_batch_add(batch_dft, mask_token_id, draft_pos_base + i, { active_seq_id }, true);
        }
        if (llama_decode(ctx_dft, batch_dft) != 0) return;

        const int draft_horizon = std::min(n_draft, block_size - 1);
        const int K = llama_get_logits_argmax_k(ctx_dft);
        const bool can_branch = (K > 1);

        const int n_max_original = (param_n_max_base > 0) ? param_n_max_base : param_n_max;
        const int original_horizon = std::min(n_max_original, block_size - 1);

        int branch_budget = 0;
        if (can_branch && tree_budget > original_horizon) {
            const int base_bb = tree_budget - original_horizon;
            const float ratio = (float)base_bb / original_horizon;
            branch_budget = std::min((int)std::round(draft_horizon * ratio),
                                     std::max(0, tree_budget - draft_horizon));
            branch_budget = std::max(0, branch_budget);
        }

        const int main_path_len = std::min(draft_horizon, std::max(1, tree_budget - branch_budget));

        const int32_t * argmax = llama_get_logits_argmax(ctx_dft);
        if (!argmax) return;
        const float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
        const int argmax_rows = llama_get_logits_argmax_n(ctx_dft);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
        if (!common_dflash_argmax_shape_valid(__func__, argmax_rows, draft_horizon + 1, K)) return;

        tree.tokens.clear(); tree.parents.clear(); tree.depths.clear();
        tree.child_maps.clear(); tree.visibility.clear(); tree.log_probs.clear();
        tree.parents.push_back(-1);
        tree.child_maps.push_back({});
        tree.n_nodes = 0;
        tree.main_path_len = 0;

        // chain-seed backbone
        {
            int parent = 0;
            for (int d = 1; d <= draft_horizon && d <= main_path_len && tree.n_nodes < tree_budget; ++d) {
                const int32_t token_raw = argmax[d * K];
                if (!common_dflash_argmax_token_valid(token_raw, n_vocab)) break;
                float log_prob = argmax_probs ? argmax_probs[d * K] : -std::numeric_limits<float>::infinity();
                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back((llama_token) token_raw);
                tree.parents.push_back(parent);
                tree.depths.push_back(d);
                tree.log_probs.push_back(log_prob);
                tree.child_maps.push_back({});
                tree.child_maps[parent][(llama_token) token_raw] = current_idx;
                tree.n_nodes++;
                parent = current_idx;
            }
            tree.main_path_len = tree.n_nodes;
        }

        // best-first heap expansion
        if (K > 1 && branch_budget > 0 && argmax_probs) {
            struct heap_entry {
                float log_w; int parent_idx; int depth; int rank;
                bool operator<(const heap_entry & o) const { return log_w < o.log_w; }
            };
            std::priority_queue<heap_entry> heap;
            float cum = 0.0f;
            for (int d = 1; d <= main_path_len; ++d) {
                heap.push({cum + argmax_probs[d * K + 1], (d == 1) ? 0 : d - 1, d, 1});
                cum += argmax_probs[d * K];
            }
            while (!heap.empty() && (tree.n_nodes - tree.main_path_len) < branch_budget) {
                auto top = heap.top(); heap.pop();
                const int32_t tr = argmax[top.depth * K + top.rank];
                if (!common_dflash_argmax_token_valid(tr, n_vocab)) continue;
                if (tree.child_maps[top.parent_idx].count((llama_token) tr)) continue;
                int ci = tree.n_nodes + 1;
                tree.tokens.push_back((llama_token) tr);
                tree.parents.push_back(top.parent_idx);
                tree.depths.push_back(top.depth);
                tree.log_probs.push_back(argmax_probs[top.depth * K + top.rank]);
                tree.child_maps.push_back({});
                tree.child_maps[top.parent_idx][(llama_token) tr] = ci;
                tree.n_nodes++;
                if (top.rank + 1 < K) {
                    heap.push({top.log_w - argmax_probs[top.depth * K + top.rank] + argmax_probs[top.depth * K + top.rank + 1],
                               top.parent_idx, top.depth, top.rank + 1});
                }
                if (top.depth < draft_horizon) {
                    heap.push({top.log_w + argmax_probs[(top.depth + 1) * K], ci, top.depth + 1, 0});
                }
            }
        }

        // visibility matrix
        int n = tree.n_nodes + 1;
        tree.visibility.assign(n * n, false);
        tree.visibility[0] = true;
        for (int i = 1; i < n; ++i) {
            int parent = tree.parents[i];
            for (int j = 0; j < i; ++j) tree.visibility[i * n + j] = tree.visibility[parent * n + j];
            tree.visibility[i * n + i] = true;
        }

        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
    }
};

// ---------------------------------------------------------------------------
// common_speculative_create_ctx_dft
// ---------------------------------------------------------------------------

llama_context * common_speculative_create_ctx_dft(const common_params_speculative & params, int dflash_n_slots) {
    if (!params.model_dft) return nullptr;
    llama_context_params cparams_dft = params.cparams_dft;
    cparams_dft.dflash_n_slots = dflash_n_slots;
    cparams_dft.dflash_cross_ctx = params.dflash_cross_ctx;
    if (dflash_n_slots > (int)cparams_dft.n_seq_max) {
        cparams_dft.n_seq_max = dflash_n_slots;
    }
    llama_context * ctx_dft = llama_init_from_model(params.model_dft, cparams_dft);
    if (!ctx_dft) {
        LOG_ERR("failed to create draft context\n");
        return nullptr;
    }
    if (params.draft_topk > 1) {
        llama_set_dflash_topk(ctx_dft, params.draft_topk);
    }
    if (params.sample_temp > 0.0f) {
        llama_set_dflash_sample_temp(ctx_dft, params.sample_temp);
    }

    // warmup
    {
        const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx_dft));
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);
        llama_token tmp[2]; int n_tmp = 0;
        if (bos != LLAMA_TOKEN_NULL) tmp[n_tmp++] = bos;
        if (eos != LLAMA_TOKEN_NULL) tmp[n_tmp++] = eos;
        if (n_tmp == 0) tmp[n_tmp++] = 0;

        llama_set_warmup(ctx_dft, true);
        llama_decode(ctx_dft, llama_batch_get_one(tmp, n_tmp));
        llama_memory_t mem = llama_get_memory(ctx_dft);
        if (mem) llama_memory_clear(mem, true);
        llama_synchronize(ctx_dft);
        llama_perf_context_reset(ctx_dft);
        llama_set_warmup(ctx_dft, false);
    }

    return ctx_dft;
}
