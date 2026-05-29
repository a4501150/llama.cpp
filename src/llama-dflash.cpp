// DFlash: cross-attention speculative decoding
// Split translation unit for llama_context DFlash member definitions.
// Ported from beellama.cpp/src/llama-context.cpp.

#include "llama-dflash.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-iswa.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "dflash-profile.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

// ---------------------------------------------------------------------------
// static helpers
// ---------------------------------------------------------------------------

static llama_memory_recurrent * get_recurrent_mem(llama_memory_i * mem) {
    if (auto * h = dynamic_cast<llama_memory_hybrid *>(mem))      return h->get_mem_recr();
    if (auto * h = dynamic_cast<llama_memory_hybrid_iswa *>(mem)) return h->get_mem_recr();
    return dynamic_cast<llama_memory_recurrent *>(mem);
}

static void dflash_read_tensor_to(struct ggml_tensor * t, float * dst, size_t n_floats) {
    if (ggml_is_contiguous(t)) {
        const size_t n_bytes = n_floats * sizeof(float);
        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(dst, t->data, n_bytes);
        } else {
            ggml_backend_tensor_get(t, dst, 0, n_bytes);
        }
        return;
    }

    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const size_t esz = ggml_element_size(t);

    size_t contig_elems = ne0;
    if (t->nb[1] == ne0 * esz) {
        contig_elems = ne0 * ne1;
        if (t->nb[2] == ne0 * ne1 * esz) {
            contig_elems = ne0 * ne1 * ne2;
        }
    }

    size_t dst_off = 0;
    size_t n_chunks = n_floats / contig_elems;
    const size_t chunk_bytes = contig_elems * sizeof(float);

    for (size_t i = 0; i < n_chunks; ++i) {
        size_t src_off = 0;
        size_t idx = i;
        if (contig_elems == (size_t)(ne0)) {
            int64_t i1 = idx % ne1; idx /= ne1;
            int64_t i2 = idx % ne2; idx /= ne2;
            int64_t i3 = idx;
            src_off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        } else if (contig_elems == (size_t)(ne0 * ne1)) {
            int64_t i2 = idx % ne2; idx /= ne2;
            int64_t i3 = idx;
            src_off = i2 * t->nb[2] + i3 * t->nb[3];
        } else {
            int64_t i3 = idx;
            src_off = i3 * t->nb[3];
        }

        if (ggml_backend_buffer_is_host(t->buffer)) {
            memcpy(dst + dst_off, (const char *)t->data + src_off, chunk_bytes);
        } else {
            ggml_backend_tensor_get(t, dst + dst_off, src_off, chunk_bytes);
        }
        dst_off += contig_elems;
    }
}

static void dflash_read_tensor(struct ggml_tensor * t, std::vector<float> & dst, size_t n_floats) {
    dst.resize(n_floats);
    dflash_read_tensor_to(t, dst.data(), n_floats);
}

static bool dflash_diagnostic_debug_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_DEBUG");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool dflash_crash_trace_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_CRASH_TRACE");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool dflash_profile_sync_split_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_PROFILE_SYNC_SPLIT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static void dflash_log_decode_seq_state(
        const char * where,
        const llama_ubatch & ubatch,
        const dflash_capture_data * cap,
        const llama_cparams & cparams) {
    if (!dflash_crash_trace_enabled() || cap == nullptr) {
        return;
    }

    const int n_log = std::min((int) ubatch.n_seqs_unq, 4);
    for (int s = 0; s < n_log; ++s) {
        const llama_seq_id seq = ubatch.seq_id_unq[s];

        int tok_count = 0;
        int32_t pos_min = INT32_MAX;
        int32_t pos_max = INT32_MIN;

        for (uint32_t t = 0; t < ubatch.n_tokens; ++t) {
            for (uint32_t k = 0; k < ubatch.n_seq_id[t]; ++k) {
                if (ubatch.seq_id[t][k] != seq) {
                    continue;
                }
                ++tok_count;
                pos_min = std::min(pos_min, (int32_t) ubatch.pos[t]);
                pos_max = std::max(pos_max, (int32_t) ubatch.pos[t]);
            }
        }

        auto * plan = cap->prefill_plan_for_seq(seq);

        dflash_hidden_gpu * hidden = nullptr;
        if (seq >= 0 && seq < (llama_seq_id) cap->hidden_gpu.size()) {
            hidden = cap->hidden_gpu[(size_t) seq].get();
        }

        dflash_hidden_gpu * prefill = nullptr;
        if (seq >= 0 && seq < (llama_seq_id) cap->prefill_gpu.size()) {
            prefill = cap->prefill_gpu[(size_t) seq].get();
        }

        dflash_tape_gpu * tape = nullptr;
        if (seq >= 0 && seq < (llama_seq_id) cap->tapes.size()) {
            tape = cap->tapes[(size_t) seq].get();
        }

        const int pos_begin = tok_count > 0 ? pos_min : -1;
        const int pos_end   = tok_count > 0 ? pos_max + 1 : -1;

        LLAMA_LOG_INFO(
            "%s: dflash decode seq[%d]: seq=%d tok_count=%d pos=[%d,%d) plan=%d cap=[%d,%d) written=%d "
            "hidden=%p htok=%d/%d prefill=%p ptok=%d/%d tape=%p ttok=%d/%d prefill_seq=%p src_off=%d dst_off=%d copy=%d\n",
            where,
            s,
            (int) seq,
            tok_count,
            pos_begin,
            pos_end,
            plan && plan->active ? 1 : 0,
            plan ? (int) plan->capture_begin : -1,
            plan ? (int) plan->capture_end : -1,
            plan ? (int) plan->n_written : -1,
            (void *) hidden,
            hidden ? hidden->n_tokens : -1,
            hidden ? hidden->max_tokens : -1,
            (void *) prefill,
            prefill ? prefill->n_tokens : -1,
            prefill ? prefill->max_tokens : -1,
            (void *) tape,
            tape ? tape->n_tokens : -1,
            tape ? tape->max_tokens : -1,
            (void *) cparams.prefill_gpu_seqs[s],
            cparams.dflash_prefill_src_offsets[s],
            cparams.dflash_prefill_dst_offsets[s],
            cparams.dflash_prefill_n_tokens_seqs[s]);
    }
}

static void dflash_clear_prefill_cparams(llama_cparams & cparams) {
    cparams.prefill_gpu_n_seqs = 0;
    cparams.dflash_prefill_capture_active = false;
    cparams.dflash_prefill_src_offset = 0;
    cparams.dflash_prefill_dst_offset = 0;
    cparams.dflash_prefill_n_tokens   = 0;
    for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
        cparams.prefill_gpu_seqs[s] = nullptr;
        cparams.dflash_prefill_src_offsets[s] = 0;
        cparams.dflash_prefill_dst_offsets[s] = 0;
        cparams.dflash_prefill_n_tokens_seqs[s] = 0;
    }
}

static void dflash_profile_reset(dflash_capture_data & cap) {
    cap.profile_decode_us = 0;
    cap.profile_output_extract_us = 0;
    cap.profile_raw_logits_us = 0;
    cap.profile_raw_logits_bytes = 0;
    cap.profile_raw_logits_skipped = 0;
    cap.profile_reduced_logits_us = 0;
    cap.profile_reduced_logits_ids_us = 0;
    cap.profile_reduced_logits_probs_us = 0;
    cap.profile_reduced_logits_bytes = 0;
    cap.profile_verify_sync_split_us = 0;
    cap.profile_cb_ask = 0;
    cap.profile_cb_hidden_ask = 0;
    cap.profile_cb_tape_ask = 0;
    cap.profile_cb_qkv_ask = 0;
    cap.profile_cb_read = 0;
    cap.profile_cb_hidden_read = 0;
    cap.profile_cb_tape_read = 0;
    cap.profile_cb_qkv_read = 0;
    cap.profile_replay_wait_us = 0;
    cap.profile_replay_gdn_enqueue_us = 0;
    cap.profile_replay_gdn_wait_us = 0;
    cap.profile_replay_conv_enqueue_us = 0;
    cap.profile_replay_conv_wait_us = 0;
    cap.profile_replay_layers = 0;
    cap.profile_replay_sync_calls = 0;
    cap.profile_replay_direct_gpu = 0;
    cap.profile_replay_ggml_gpu = 0;
    cap.profile_replay_cpu_fallback = 0;
    cap.profile_conv_gpu_us = 0;
    cap.profile_conv_read_wait_us = 0;
    cap.profile_conv_cpu_us = 0;
    cap.profile_conv_write_wait_us = 0;
    cap.profile_cb_names.clear();
}

static void dflash_profile_cb_name(dflash_capture_data & cap, const ggml_tensor * t, const char * phase) {
    if (!cap.profile) {
        return;
    }
    cap.profile_cb_names[std::string(phase) + ":" + t->name] += 1;
}

static void dflash_profile_log(const dflash_capture_data & cap, const char * func, int32_t n_vocab) {
    if (!cap.profile) {
        return;
    }

    const uint64_t skipped_bytes_est =
        cap.profile_raw_logits_skipped * (uint64_t) std::max(0, n_vocab) * sizeof(float);

    if (dflash_profile_has(cap.profile_flags, DFLASH_PROFILE_SUMMARY | DFLASH_PROFILE_VERIFY)) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: decode=%.3f ms output_extract=%.3f ms "
            "raw_logits=%.3f ms raw_logits_bytes=%.3f MiB raw_logits_skipped=%" PRIu64
            " raw_logits_skipped_bytes_est=%.3f MiB "
            "reduced_logits=%.3f ms reduced_logits_ids=%.3f ms reduced_logits_probs=%.3f ms "
            "reduced_logits_bytes=%.3f KiB verify_sync_split=%.3f ms "
            "cb ask=%" PRIu64 " hidden=%" PRIu64 " tape=%" PRIu64 " qkv=%" PRIu64
            " read=%" PRIu64 " hidden=%" PRIu64 " tape=%" PRIu64 " qkv=%" PRIu64 "\n",
            func,
            cap.profile_decode_us / 1000.0,
            cap.profile_output_extract_us / 1000.0,
            cap.profile_raw_logits_us / 1000.0,
            cap.profile_raw_logits_bytes / (1024.0 * 1024.0),
            cap.profile_raw_logits_skipped,
            skipped_bytes_est / (1024.0 * 1024.0),
            cap.profile_reduced_logits_us / 1000.0,
            cap.profile_reduced_logits_ids_us / 1000.0,
            cap.profile_reduced_logits_probs_us / 1000.0,
            cap.profile_reduced_logits_bytes / 1024.0,
            cap.profile_verify_sync_split_us / 1000.0,
            cap.profile_cb_ask,
            cap.profile_cb_hidden_ask,
            cap.profile_cb_tape_ask,
            cap.profile_cb_qkv_ask,
            cap.profile_cb_read,
            cap.profile_cb_hidden_read,
            cap.profile_cb_tape_read,
            cap.profile_cb_qkv_read);
    }

    if (dflash_profile_has(cap.profile_flags, DFLASH_PROFILE_REPLAY) &&
        (cap.profile_replay_wait_us || cap.profile_replay_gdn_enqueue_us || cap.profile_replay_gdn_wait_us ||
        cap.profile_replay_conv_enqueue_us || cap.profile_replay_conv_wait_us ||
        cap.profile_conv_gpu_us || cap.profile_conv_read_wait_us ||
        cap.profile_conv_cpu_us || cap.profile_conv_write_wait_us ||
        cap.profile_replay_layers || cap.profile_replay_sync_calls ||
        cap.profile_replay_direct_gpu || cap.profile_replay_ggml_gpu || cap.profile_replay_cpu_fallback)) {
        LLAMA_LOG_INFO(
            "%s: dflash profile: replay_path=direct-gpu:%" PRIu64 " replay_path=ggml-gpu:%" PRIu64
            " replay_path=cpu-fallback:%" PRIu64 " replay_layers=%" PRIu64 " replay_sync_calls=%" PRIu64
            " gdn_enqueue=%.3f ms gdn_wait=%.3f ms conv_enqueue=%.3f ms conv_wait=%.3f ms "
            "legacy_replay_wait=%.3f ms legacy_conv_gpu_enqueue=%.3f ms "
            "legacy_conv_read_wait=%.3f ms legacy_conv_write_wait=%.3f ms conv_cpu=%.3f ms\n",
            func,
            cap.profile_replay_direct_gpu,
            cap.profile_replay_ggml_gpu,
            cap.profile_replay_cpu_fallback,
            cap.profile_replay_layers,
            cap.profile_replay_sync_calls,
            cap.profile_replay_gdn_enqueue_us / 1000.0,
            cap.profile_replay_gdn_wait_us / 1000.0,
            cap.profile_replay_conv_enqueue_us / 1000.0,
            cap.profile_replay_conv_wait_us / 1000.0,
            cap.profile_replay_wait_us / 1000.0,
            cap.profile_conv_gpu_us / 1000.0,
            cap.profile_conv_read_wait_us / 1000.0,
            cap.profile_conv_write_wait_us / 1000.0,
            cap.profile_conv_cpu_us / 1000.0);
    }

    if (dflash_profile_has(cap.profile_flags, DFLASH_PROFILE_TRACE) && !cap.profile_cb_names.empty()) {
        LLAMA_LOG_INFO("%s: dflash profile: callback tensors:\n", func);
        for (const auto & hit : cap.profile_cb_names) {
            LLAMA_LOG_INFO("%s: dflash profile:   %" PRIu64 " %s\n", func, hit.second, hit.first.c_str());
        }
    }
}

// ---------------------------------------------------------------------------
// eval callback
// ---------------------------------------------------------------------------

static bool dflash_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cap = (dflash_capture_data *) user_data;
    const llama_ubatch * ub = cap->ubatch;
    const uint32_t n_seqs_unq = ub ? ub->n_seqs_unq : 0;

    auto h_it = cap->hidden_name_idx.find(t->name);

    if (ask) {
        if (cap->profile) {
            cap->profile_cb_ask++;
        }
        if (h_it != cap->hidden_name_idx.end()) {
            if (cap->profile) {
                cap->profile_cb_hidden_ask++;
                dflash_profile_cb_name(*cap, t, "ask");
            }
            return true;
        }
        if (cap->tape_enabled && cap->tape_name_map.count(t->name)) {
            if (cap->profile) {
                cap->profile_cb_tape_ask++;
                dflash_profile_cb_name(*cap, t, "ask");
            }
            auto * active_tape = cap->active_tape();
            const bool gpu_tape_fits = active_tape && (!ub || (int) ub->n_seq_tokens <= active_tape->max_tokens);
            if (gpu_tape_fits) {
                auto it = cap->tape_name_map.find(t->name);
                if (it != cap->tape_name_map.end() && it->second.second == DFLASH_TAPE_QKV && cap->profile) {
                    cap->profile_cb_qkv_ask++;
                }
                return false;
            }
            if (n_seqs_unq > 1) {
                return false;
            }
            return true;
        }
        return false;
    }

    // ask=false: tensor data is ready
    if (h_it != cap->hidden_name_idx.end()) {
        if (cap->profile) {
            cap->profile_cb_read++;
            cap->profile_cb_hidden_read++;
            dflash_profile_cb_name(*cap, t, "read");
        }
        const int64_t new_embd = t->ne[0];
        const int64_t new_n = t->ne[1];
        const size_t h_idx = h_it->second;

        if (n_seqs_unq <= 1) {
            const int slot = ub ? ub->seq_id_unq[0] : -1;
            auto * sh = cap->slot_hiddens(slot);
            if (!sh) {
                return true;
            }
            GGML_ASSERT(h_idx < sh->size());
            auto & buf = (*sh)[h_idx];
            buf.n_embd = new_embd;
            const size_t old_elems = (size_t) buf.n_tokens * (size_t) new_embd;
            const size_t add_elems = (size_t) new_n * (size_t) new_embd;
            buf.data.resize(old_elems + add_elems);
            dflash_read_tensor_to(t, buf.data.data() + old_elems, add_elems);
            buf.n_tokens += new_n;
            return true;
        }

        GGML_ASSERT(ub && (int64_t) ub->n_tokens == new_n);
        cap->scatter_buf.resize((size_t) new_embd * (size_t) new_n);
        dflash_read_tensor_to(t, cap->scatter_buf.data(), cap->scatter_buf.size());

        const int n_slots = cap->hiddens ? (int) cap->hiddens->size() : 0;
        for (uint32_t s = 0; s < n_seqs_unq; ++s) {
            const llama_seq_id seq = ub->seq_id_unq[s];
            if (seq < 0 || seq >= n_slots) continue;
            auto & slot_bufs = (*cap->hiddens)[seq];
            if (h_idx >= slot_bufs.size()) continue;
            auto & buf = slot_bufs[h_idx];
            buf.n_embd = new_embd;
            buf.data.reserve((size_t) (buf.n_tokens + new_n) * (size_t) new_embd);
        }

        for (int64_t i = 0; i < new_n; ++i) {
            const llama_seq_id seq = ub->seq_id[i][0];
            if (seq < 0 || seq >= n_slots) continue;
            auto & slot_bufs = (*cap->hiddens)[seq];
            if (h_idx >= slot_bufs.size()) continue;
            auto & buf = slot_bufs[h_idx];
            const size_t old_elems = (size_t) buf.n_tokens * (size_t) new_embd;
            buf.data.resize(old_elems + (size_t) new_embd);
            std::memcpy(buf.data.data() + old_elems,
                        cap->scatter_buf.data() + (size_t) i * (size_t) new_embd,
                        (size_t) new_embd * sizeof(float));
            buf.n_tokens += 1;
        }
        return true;
    }

    // tape recording
    if (cap->tape_enabled) {
        auto it = cap->tape_name_map.find(t->name);
        if (it != cap->tape_name_map.end()) {
            int layer_idx = it->second.first;
            int type      = it->second.second;
            auto & tape   = cap->tape_layers[layer_idx];

            if (cap->profile) {
                cap->profile_cb_read++;
                cap->profile_cb_tape_read++;
                if (type == DFLASH_TAPE_QKV) {
                    cap->profile_cb_qkv_read++;
                }
                dflash_profile_cb_name(*cap, t, "read");
            }

            auto * active_tape = cap->active_tape();
            const bool gpu_tape_fits = active_tape && (!ub || (int) ub->n_seq_tokens <= active_tape->max_tokens);
            if (gpu_tape_fits) {
                return true;
            }

            size_t n_elem = ggml_nelements(t);

            switch (type) {
                case DFLASH_TAPE_K:
                    tape.S_k = t->ne[0];
                    tape.H_k = t->ne[1];
                    tape.n_tokens = (int) t->ne[2];
                    dflash_read_tensor(t, tape.k, n_elem);
                    break;
                case DFLASH_TAPE_V:
                    tape.S_v = t->ne[0];
                    tape.H_v = t->ne[1];
                    dflash_read_tensor(t, tape.v, n_elem);
                    break;
                case DFLASH_TAPE_GATE:
                    dflash_read_tensor(t, tape.gate, n_elem);
                    break;
                case DFLASH_TAPE_BETA:
                    dflash_read_tensor(t, tape.beta, n_elem);
                    break;
                case DFLASH_TAPE_QKV:
                    tape.conv_channels = t->ne[0];
                    tape.n_tokens = (int) t->ne[1];
                    if (ub && n_seqs_unq > 1) {
                        tape.n_seqs = std::min((int) n_seqs_unq, (int) LLAMA_DFLASH_MAX_SLOTS);
                        for (int s = 0; s < tape.n_seqs; ++s) {
                            tape.seq_ids[s] = ub->seq_id_unq[s];
                        }
                    } else {
                        tape.n_seqs = 1;
                        tape.seq_ids[0] = ub ? ub->seq_id_unq[0] : 0;
                    }
                    dflash_read_tensor(t, tape.qkv_mixed, n_elem);
                    break;
            }
            return true;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// hidden state accessors
// ---------------------------------------------------------------------------

float * llama_context::get_layer_hidden(int layer_idx) {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    if (!sh || layer_idx < 0 || layer_idx >= (int) sh->size()) {
        return nullptr;
    }
    auto & buf = (*sh)[layer_idx];
    if (buf.n_tokens <= 0 || buf.data.empty()) {
        return nullptr;
    }
    return buf.data.data();
}

int64_t llama_context::get_layer_hidden_n_tokens(int layer_idx) const {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    if (sh && layer_idx >= 0 && layer_idx < (int) sh->size() && (*sh)[layer_idx].n_tokens > 0) {
        return (*sh)[layer_idx].n_tokens;
    }
    auto * hgpu = dflash_capture ? dflash_capture->active_hidden_gpu() : nullptr;
    if (hgpu && layer_idx >= 0 && layer_idx < (int) hgpu->layers.size()) {
        return hgpu->n_tokens;
    }
    if (!sh || layer_idx < 0 || layer_idx >= (int) sh->size()) {
        return 0;
    }
    return (*sh)[layer_idx].n_tokens;
}

int64_t llama_context::get_layer_hidden_n_embd(int layer_idx) const {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    if (sh && layer_idx >= 0 && layer_idx < (int) sh->size() &&
            (*sh)[layer_idx].n_tokens > 0 && (*sh)[layer_idx].n_embd > 0) {
        return (*sh)[layer_idx].n_embd;
    }
    auto * hgpu = dflash_capture ? dflash_capture->active_hidden_gpu() : nullptr;
    if (hgpu && layer_idx >= 0 && layer_idx < (int) hgpu->layers.size()) {
        return hgpu->n_embd;
    }
    if (!sh || layer_idx < 0 || layer_idx >= (int) sh->size()) {
        return 0;
    }
    return (*sh)[layer_idx].n_embd;
}

int32_t llama_context::get_n_layer_hiddens() const {
    auto * sh = dflash_capture ? dflash_capture->active_slot_hiddens() : nullptr;
    return sh ? (int32_t) sh->size() : 0;
}

// ---------------------------------------------------------------------------
// capture config & simple setters
// ---------------------------------------------------------------------------

void llama_context::set_dflash_sample_temp(float temp) {
    cparams.dflash_sample_temp = temp;
}

void llama_context::set_dflash_topk(int k) {
    cparams.dflash_topk = (k >= 1) ? k : 1;
    gf_res_prev->reset();
}

void llama_context::set_dflash_verify_logits(bool enabled, int top_k) {
    const int clamped_top_k = std::max(1, std::min(top_k, 64));
    if (cparams.dflash_verify_logits == enabled && cparams.dflash_verify_topk == clamped_top_k) {
        return;
    }
    cparams.dflash_verify_logits = enabled;
    cparams.dflash_verify_topk = clamped_top_k;
}

void llama_context::set_dflash_consume_reduced(bool enabled) {
    cparams.dflash_reduced_consumer_active = enabled;
}

void llama_context::set_dflash_n_slots(int n) {
    const int clamped = std::max(1, std::min(n, (int) LLAMA_DFLASH_MAX_SLOTS));
    if (cparams.dflash_n_slots == clamped) {
        return;
    }
    cparams.dflash_n_slots = clamped;
    sched_need_reserve = true;
    gf_res_prev->reset();
}

void llama_context::set_dflash_capture(const int32_t * layer_ids, int32_t n_layers) {
    if (layer_ids == nullptr || n_layers <= 0) {
        cparams.dflash_capture_layers.clear();

        if (dflash_capture) {
            dflash_capture->layer_ids.clear();
            dflash_capture->hidden_name_idx.clear();
            dflash_capture->tensor_names.clear();
            dflash_capture->capture_active = false;
        }

        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;

        return;
    }

    cparams.dflash_capture_layers.clear();
    for (int32_t i = 0; i < n_layers; ++i) {
        cparams.dflash_capture_layers.push_back(layer_ids[i]);
    }

    if (!dflash_capture) {
        dflash_capture = std::make_unique<dflash_capture_data>();
        dflash_capture->hiddens = &layer_hiddens;
        dflash_capture->profile_flags = dflash_profile_flags();
        dflash_capture->profile = dflash_capture->profile_flags != 0;
    }

    dflash_capture->capture_active = true;
    dflash_capture->layer_ids.clear();
    dflash_capture->hidden_name_idx.clear();
    dflash_capture->tensor_names.clear();

    layer_hiddens.assign(1, std::vector<dflash_layer_hidden_buf>(n_layers));

    for (int32_t i = 0; i < n_layers; ++i) {
        dflash_capture->layer_ids.push_back(layer_ids[i]);
        std::string name = "l_out-" + std::to_string(layer_ids[i]);
        dflash_capture->hidden_name_idx[name] = i;
        dflash_capture->tensor_names.push_back(std::move(name));
    }

    cparams.cb_eval = dflash_eval_callback;
    cparams.cb_eval_user_data = dflash_capture.get();

    if (memory) {
        memory->set_force_split_seq(true);
    }
}

void llama_context::set_dflash_capture_active(bool active) {
    if (!dflash_capture) {
        return;
    }

    if (dflash_capture->capture_active == active) {
        return;
    }

    dflash_capture->capture_active = active;

    if (active) {
        if (!dflash_capture->layer_ids.empty()) {
            cparams.cb_eval = dflash_eval_callback;
            cparams.cb_eval_user_data = dflash_capture.get();
        }
        if (memory) {
            memory->set_force_split_seq(true);
        }
    } else {
        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;
        cparams.hidden_gpu_n_seqs = 0;
        dflash_clear_prefill_cparams(cparams);
        cparams.tape_gpu_n_seqs = 0;
        cparams.tape_gpu = nullptr;
        for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.hidden_gpu_seqs[s] = nullptr;
            cparams.tape_gpu_seqs[s] = nullptr;
        }
    }

    cparams.dflash_capture_layers.clear();
    if (active && !dflash_capture->layer_ids.empty()) {
        for (auto lid : dflash_capture->layer_ids) {
            cparams.dflash_capture_layers.push_back(lid);
        }
    }
}

void llama_context::set_dflash_gpu_capture(bool enabled) {
    if (!dflash_capture) {
        return;
    }

    dflash_capture->gpu_capture_enabled = enabled;

    cparams.hidden_gpu_n_seqs = 0;
    dflash_clear_prefill_cparams(cparams);
    cparams.tape_gpu_n_seqs = 0;
    cparams.tape_gpu = nullptr;
    for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
        cparams.tape_gpu_seqs[s] = nullptr;
        cparams.hidden_gpu_seqs[s] = nullptr;
    }

    if (!enabled) {
        if (dflash_capture->capture_active && !dflash_capture->layer_ids.empty()) {
            cparams.cb_eval = dflash_eval_callback;
            cparams.cb_eval_user_data = dflash_capture.get();
        }
    }
}

void llama_context::dflash_reset_hidden_capture() {
    if (!dflash_capture) {
        return;
    }
    for (auto & slot_bufs : layer_hiddens) {
        for (auto & buf : slot_bufs) {
            buf.n_tokens = 0;
            std::vector<float>().swap(buf.data);
        }
    }
    for (auto & hidden : dflash_capture->hidden_gpu) {
        if (hidden) {
            hidden->n_tokens = 0;
        }
    }
    for (auto & pf : dflash_capture->prefill_gpu) {
        if (pf) {
            pf->n_tokens = 0;
        }
    }
    for (auto & plan : dflash_capture->prefill_plans) {
        plan.n_written = 0;
    }
    for (auto & tl : dflash_capture->tape_layers) {
        tl.n_tokens = 0;
    }
    std::vector<float>().swap(dflash_capture->scatter_buf);
    if (dflash_capture->profile) {
        dflash_profile_reset(*dflash_capture);
    }
    dflash_capture->ubatch = nullptr;
}

void llama_context::dflash_ensure_recurrent_setup() {
    if (!dflash_capture || !dflash_capture->recurrent_layer_ids.empty()) {
        return;
    }
    const auto & hparams = model.hparams;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        if (hparams.is_recurrent(il)) {
            int idx = (int) dflash_capture->recurrent_layer_ids.size();
            dflash_capture->recurrent_layer_ids.push_back(il);

            std::string il_str = std::to_string(il);
            dflash_capture->tape_name_map["k_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_K};
            dflash_capture->tape_name_map["v_conv_predelta-" + il_str]        = {idx, DFLASH_TAPE_V};
            dflash_capture->tape_name_map["gate-" + il_str]                   = {idx, DFLASH_TAPE_GATE};
            dflash_capture->tape_name_map["beta-" + il_str]                   = {idx, DFLASH_TAPE_BETA};
            dflash_capture->tape_name_map["qkv_mixed_pretranspose-" + il_str] = {idx, DFLASH_TAPE_QKV};
        }
    }
    dflash_capture->tape_layers.resize(dflash_capture->recurrent_layer_ids.size());
}

void llama_context::set_tape_recording(bool enable) {
    if (!dflash_capture) {
        return;
    }

    dflash_capture->tape_enabled = enable;

    if (enable) {
        dflash_ensure_recurrent_setup();
        if (dflash_capture->tapes.empty()) {
            allocate_tape_gpu(1, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
        }
        for (auto & tape : dflash_capture->tapes) {
            if (tape) {
                tape->n_tokens = 0;
            }
        }
        for (auto & hidden : dflash_capture->hidden_gpu) {
            if (hidden) {
                hidden->n_tokens = 0;
            }
        }
    }

    if (enable && !dflash_capture->tapes.empty()) {
        const int n_tapes = (int) dflash_capture->tapes.size();
        cparams.tape_gpu = dflash_capture->tapes[0].get();
        cparams.tape_gpu_n_seqs = n_tapes;
        for (int s = 0; s < n_tapes && s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = dflash_capture->tapes[s].get();
        }
        for (int s = n_tapes; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = nullptr;
        }
    } else {
        cparams.tape_gpu = nullptr;
        cparams.tape_gpu_n_seqs = 0;
        cparams.hidden_gpu_n_seqs = 0;
        dflash_clear_prefill_cparams(cparams);
        for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = nullptr;
            cparams.hidden_gpu_seqs[s] = nullptr;
        }

        if (dflash_capture->capture_active && !dflash_capture->layer_ids.empty()) {
            cparams.cb_eval = dflash_eval_callback;
            cparams.cb_eval_user_data = dflash_capture.get();
        } else {
            cparams.cb_eval = nullptr;
            cparams.cb_eval_user_data = nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// GPU allocation
// ---------------------------------------------------------------------------

void llama_context::allocate_tape_gpu(int max_tokens) {
    allocate_tape_gpu(1, max_tokens);
}

void llama_context::allocate_tape_gpu(int n_slots, int max_tokens) {
    if (!dflash_capture) {
        return;
    }
    if (n_slots < 1) {
        n_slots = 1;
    }

    if (!layer_hiddens.empty() && (int) layer_hiddens.size() != n_slots) {
        const size_t n_capture_layers = layer_hiddens.front().size();
        layer_hiddens.resize(n_slots);
        for (auto & slot_bufs : layer_hiddens) {
            if (slot_bufs.size() != n_capture_layers) {
                slot_bufs.resize(n_capture_layers);
            }
        }
    }

    dflash_ensure_recurrent_setup();

    if (model.n_devices() > 1) {
        if (!dflash_capture->multi_gpu_capture_fallback_logged) {
            LLAMA_LOG_INFO("%s: multi-GPU target detected (%zu devices); GPU tape/hidden capture disabled\n",
                __func__, model.n_devices());
            dflash_capture->multi_gpu_capture_fallback_logged = true;
        }
        return;
    }

    allocate_hidden_gpu(n_slots, max_tokens);

    if (dflash_capture->recurrent_layer_ids.empty()) {
        return;
    }
    if (!llama_dflash_gpu_tape_supported_arch(model.arch)) {
        dflash_capture->tapes.clear();
        LLAMA_LOG_INFO("%s: GPU tape disabled for model arch %s; using eval-callback tape fallback\n",
            __func__, llm_arch_name(model.arch));
        return;
    }

    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            break;
        }
    }
    if (!gpu_backend) {
        return;
    }

    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const int n_rec = (int) rec_ids.size();

    const int64_t S = hparams.ssm_d_state;
    const int64_t H_v = hparams.ssm_dt_rank;
    const int64_t H_k = (cparams.fused_gdn_ar && cparams.fused_gdn_ch)
                       ? (int64_t) hparams.ssm_n_group
                       : H_v;

    dflash_capture->tapes.clear();
    dflash_capture->tapes.reserve(n_slots);

    size_t total_size = 0;

    for (int slot = 0; slot < n_slots; ++slot) {
        size_t ctx_mem = ggml_tensor_overhead() * (n_rec * 5 + 2);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * tape_ctx = ggml_init(ctx_params);
        if (!tape_ctx) {
            LLAMA_LOG_WARN("%s: failed to create GPU tape context for slot %d, falling back to CPU tape\n",
                __func__, slot);
            dflash_capture->tapes.clear();
            return;
        }

        auto tape = std::make_unique<dflash_tape_gpu>();
        tape->layers.resize(n_rec);
        tape->layer_ids = dflash_capture->recurrent_layer_ids;
        tape->max_tokens = max_tokens;
        tape->ctx = tape_ctx;

        for (int li = 0; li < n_rec; ++li) {
            const int il = rec_ids[li];
            const auto * conv_kernel = model.layers[il].ssm_conv1d;
            GGML_ASSERT(conv_kernel != nullptr);
            const int64_t conv_window = conv_kernel->ne[0] - 1;
            GGML_ASSERT(conv_window > 0 && hparams.n_embd_r() % conv_window == 0);
            const int64_t conv_ch = hparams.n_embd_r() / conv_window;

            auto & tl = tape->layers[li];
            tl.k    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_k, (int64_t)max_tokens);
            tl.v    = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, S, H_v, (int64_t)max_tokens);
            tl.gate = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
            tl.beta = ggml_new_tensor_3d(tape_ctx, GGML_TYPE_F32, (int64_t)1, H_v, (int64_t)max_tokens);
            tl.qkv  = ggml_new_tensor_2d(tape_ctx, GGML_TYPE_F32, conv_ch, (int64_t)max_tokens);
        }

        tape->buf = ggml_backend_alloc_ctx_tensors(tape_ctx, gpu_backend);

        if (!tape->buf) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU tape buffer for slot %d, falling back to CPU tape\n",
                __func__, slot);
            dflash_capture->tapes.clear();
            return;
        }

        total_size += ggml_backend_buffer_get_size(tape->buf);
        dflash_capture->tapes.push_back(std::move(tape));
    }

    dflash_capture->active_tape_idx = 0;

    LLAMA_LOG_INFO("%s: allocated GPU tape buffers: %.1f MB total (%d slot%s, %d layers, %d max tokens)\n",
        __func__, total_size / (1024.0 * 1024.0), n_slots, n_slots == 1 ? "" : "s", n_rec, max_tokens);
}

void llama_context::allocate_hidden_gpu(int n_slots, int max_tokens) {
    if (!dflash_capture || dflash_capture->layer_ids.empty()) {
        return;
    }
    if (n_slots < 1) {
        n_slots = 1;
    }
    if (!dflash_capture->gpu_capture_enabled || model.n_devices() > 1 ||
            !llama_dflash_gpu_hidden_supported_arch(model.arch)) {
        dflash_capture->hidden_gpu.clear();
        dflash_capture->fn_sync_backend_to_stream = nullptr;
        dflash_capture->sync_backend_to_stream_backend = nullptr;
        return;
    }

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_reg_t cuda_reg = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            cuda_reg = ggml_backend_dev_backend_reg(dev);
            break;
        }
    }
    if (!gpu_backend) {
        dflash_capture->hidden_gpu.clear();
        dflash_capture->fn_sync_backend_to_stream = nullptr;
        dflash_capture->sync_backend_to_stream_backend = nullptr;
        return;
    }

    dflash_capture->fn_sync_backend_to_stream = cuda_reg
        ? (dflash_capture_data::sync_backend_to_stream_fn_t)
            ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_stream")
        : nullptr;
    dflash_capture->sync_backend_to_stream_backend =
        dflash_capture->fn_sync_backend_to_stream ? gpu_backend : nullptr;

    const int n_layers = (int) dflash_capture->layer_ids.size();
    const int64_t n_embd = model.hparams.n_embd;

    dflash_capture->hidden_gpu.clear();
    dflash_capture->hidden_gpu.reserve(n_slots);

    size_t total_size = 0;
    for (int slot = 0; slot < n_slots; ++slot) {
        const size_t ctx_mem = ggml_tensor_overhead() * ((size_t) n_layers + 2);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * hidden_ctx = ggml_init(ctx_params);
        if (!hidden_ctx) {
            LLAMA_LOG_WARN("%s: failed to create GPU hidden context for slot %d; using callback hidden fallback\n",
                __func__, slot);
            dflash_capture->hidden_gpu.clear();
            return;
        }

        auto hidden = std::make_unique<dflash_hidden_gpu>();
        hidden->layers.resize(n_layers);
        hidden->layer_ids = dflash_capture->layer_ids;
        hidden->n_embd = n_embd;
        hidden->max_tokens = max_tokens;
        hidden->ctx = hidden_ctx;

        for (int i = 0; i < n_layers; ++i) {
            hidden->layers[i] = ggml_new_tensor_2d(hidden_ctx, GGML_TYPE_F32, n_embd, (int64_t) max_tokens);
        }

        hidden->buf = ggml_backend_alloc_ctx_tensors(hidden_ctx, gpu_backend);
        if (!hidden->buf) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU hidden buffer for slot %d; using callback hidden fallback\n",
                __func__, slot);
            dflash_capture->hidden_gpu.clear();
            return;
        }

        total_size += ggml_backend_buffer_get_size(hidden->buf);
        dflash_capture->hidden_gpu.push_back(std::move(hidden));
    }

    LLAMA_LOG_INFO("%s: allocated GPU hidden buffers: %.1f MB total (%d slot%s, %d layers, %d max tokens)\n",
        __func__, total_size / (1024.0 * 1024.0), n_slots, n_slots == 1 ? "" : "s", n_layers, max_tokens);
}

bool llama_context::allocate_prefill_gpu(int n_slots, int max_tokens) {
    if (!dflash_capture || dflash_capture->layer_ids.empty()) {
        return false;
    }
    if (n_slots < 1) {
        n_slots = 1;
    }
    if (!dflash_capture->gpu_capture_enabled || model.n_devices() > 1 ||
            !llama_dflash_gpu_hidden_supported_arch(model.arch)) {
        return false;
    }

    if (!dflash_capture->prefill_gpu.empty() &&
        (int) dflash_capture->prefill_gpu.size() >= n_slots &&
        dflash_capture->prefill_gpu_max_tokens >= max_tokens) {
        for (int s = 0; s < n_slots && s < (int) dflash_capture->prefill_gpu.size(); ++s) {
            dflash_capture->prefill_gpu[s]->n_tokens = 0;
        }
        return true;
    }

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_reg_t cuda_reg = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            cuda_reg = ggml_backend_dev_backend_reg(dev);
            break;
        }
    }
    if (!gpu_backend) {
        return false;
    }
    if (!dflash_capture->fn_sync_backend_to_stream && cuda_reg) {
        dflash_capture->fn_sync_backend_to_stream =
            (dflash_capture_data::sync_backend_to_stream_fn_t)
                ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_stream");
        dflash_capture->sync_backend_to_stream_backend =
            dflash_capture->fn_sync_backend_to_stream ? gpu_backend : nullptr;
    }

    const int n_layers = (int) dflash_capture->layer_ids.size();
    const int64_t n_embd = model.hparams.n_embd;

    dflash_capture->prefill_gpu.clear();
    dflash_capture->prefill_gpu.reserve(n_slots);

    size_t total_size = 0;
    for (int slot = 0; slot < n_slots; ++slot) {
        const size_t ctx_mem = ggml_tensor_overhead() * ((size_t) n_layers + 2);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * hidden_ctx = ggml_init(ctx_params);
        if (!hidden_ctx) {
            LLAMA_LOG_WARN("%s: failed to create prefill GPU context for slot %d; using callback fallback\n",
                __func__, slot);
            dflash_capture->prefill_gpu.clear();
            dflash_capture->prefill_gpu_max_tokens = 0;
            return false;
        }

        auto hidden = std::make_unique<dflash_hidden_gpu>();
        hidden->layers.resize(n_layers);
        hidden->layer_ids = dflash_capture->layer_ids;
        hidden->n_embd = n_embd;
        hidden->max_tokens = max_tokens;
        hidden->n_tokens = 0;
        hidden->ctx = hidden_ctx;

        for (int i = 0; i < n_layers; ++i) {
            hidden->layers[i] = ggml_new_tensor_2d(hidden_ctx, GGML_TYPE_F32, n_embd, (int64_t) max_tokens);
        }

        hidden->buf = ggml_backend_alloc_ctx_tensors(hidden_ctx, gpu_backend);
        if (!hidden->buf) {
            LLAMA_LOG_WARN("%s: failed to allocate prefill GPU buffer for slot %d; using callback fallback\n",
                __func__, slot);
            dflash_capture->prefill_gpu.clear();
            dflash_capture->prefill_gpu_max_tokens = 0;
            return false;
        }

        total_size += ggml_backend_buffer_get_size(hidden->buf);
        dflash_capture->prefill_gpu.push_back(std::move(hidden));
    }

    dflash_capture->prefill_gpu_max_tokens = max_tokens;

    LLAMA_LOG_INFO("%s: allocated prefill GPU staging buffers: %.1f MB total (%d slot%s, %d layers, %d max tokens)\n",
        __func__, total_size / (1024.0 * 1024.0), n_slots, n_slots == 1 ? "" : "s", n_layers, max_tokens);
    return true;
}

bool llama_context::dflash_wait_for_gpu_capture_stream() {
    if (!dflash_capture) {
        return false;
    }

    const bool graph_gpu_capture_active =
        cparams.hidden_gpu_n_seqs > 0 ||
        cparams.prefill_gpu_n_seqs > 0 ||
        cparams.tape_gpu_n_seqs > 0;

    if (!graph_gpu_capture_active) {
        return false;
    }

    if (dflash_capture->fn_sync_backend_to_stream &&
        dflash_capture->sync_backend_to_stream_backend) {
        return dflash_capture->fn_sync_backend_to_stream(
            dflash_capture->sync_backend_to_stream_backend);
    }

    return false;
}

// ---------------------------------------------------------------------------
// memory utilities
// ---------------------------------------------------------------------------

bool llama_context::dflash_memory_seq_cp_recurrent_ordered(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos    p0,
        llama_pos    p1) {
    if (model.n_devices() > 1) {
        return false;
    }

    llama_memory_recurrent * mem_recr = get_recurrent_mem(memory.get());
    if (!mem_recr) {
        return false;
    }

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_reg_t cuda_reg = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            cuda_reg = ggml_backend_dev_backend_reg(dev);
            break;
        }
    }
    if (!gpu_backend || !cuda_reg) {
        return false;
    }

    using sync_dflash_stream_to_backend_fn_t = bool (*)(ggml_backend_t);
    auto fn_wait_backend = (sync_dflash_stream_to_backend_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_dflash_stream");
    if (!fn_wait_backend) {
        return false;
    }

    // Use seq_cp (synchronized) since seq_cp_recurrent_no_sync is not available upstream
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (!fn_wait_backend(gpu_backend)) {
        LLAMA_LOG_ERROR("%s: failed to order DFlash recurrent backup stream before verifier compute\n", __func__);
        GGML_ABORT("failed to order DFlash recurrent backup stream before verifier compute");
    }

    return true;
}

void llama_context::set_active_dflash_slot(int slot_idx) {
    if (!dflash_capture) {
        return;
    }
    int n_slots = (int) layer_hiddens.size();
    n_slots = std::max(n_slots, (int) dflash_capture->tapes.size());
    n_slots = std::max(n_slots, (int) dflash_capture->hidden_gpu.size());
    n_slots = std::max(n_slots, (int) dflash_capture->prefill_gpu.size());
    if (slot_idx < 0 || slot_idx >= n_slots) {
        LLAMA_LOG_WARN("%s: slot %d out of range [0, %d); ignoring\n",
            __func__, slot_idx, n_slots);
        return;
    }
    if (slot_idx == dflash_capture->active_tape_idx) {
        return;
    }
    dflash_capture->active_tape_idx = slot_idx;
    cparams.tape_gpu = dflash_capture->active_tape();
    cparams.tape_gpu_seqs[0] = cparams.tape_gpu;
    cparams.tape_gpu_n_seqs = cparams.tape_gpu ? 1 : 0;
    cparams.hidden_gpu_seqs[0] = dflash_capture->active_hidden_gpu();
    cparams.hidden_gpu_n_seqs = cparams.hidden_gpu_seqs[0] ? 1 : 0;
    cparams.prefill_gpu_seqs[0] =
        (slot_idx >= 0 && slot_idx < (int) dflash_capture->prefill_gpu.size())
            ? dflash_capture->prefill_gpu[slot_idx].get()
            : nullptr;
    cparams.prefill_gpu_n_seqs = cparams.prefill_gpu_seqs[0] ? 1 : 0;
    if (gf_res_prev) {
        gf_res_prev->reset();
    }
}

// ---------------------------------------------------------------------------
// prefill capture
// ---------------------------------------------------------------------------

void llama_context::dflash_prefill_capture_begin(llama_seq_id seq_id, int32_t capture_begin, int32_t capture_end) {
    if (!dflash_capture) {
        return;
    }

    if (seq_id < 0 || seq_id >= (llama_seq_id) LLAMA_DFLASH_MAX_SLOTS) {
        LLAMA_LOG_WARN("%s: slot %d out of range [0, %d); ignoring prefill capture plan\n",
            __func__, (int) seq_id, (int) LLAMA_DFLASH_MAX_SLOTS);
        return;
    }

    if ((int) dflash_capture->prefill_plans.size() <= seq_id) {
        dflash_capture->prefill_plans.resize((size_t) seq_id + 1);
    }

    auto & plan = dflash_capture->prefill_plans[(size_t) seq_id];

    if (capture_end <= capture_begin) {
        plan = {};
        return;
    }

    plan.active = true;
    plan.seq_id = seq_id;
    plan.capture_begin = capture_begin;
    plan.capture_end = capture_end;
    plan.n_tokens = capture_end - capture_begin;
    plan.n_written = 0;

    for (auto & pf : dflash_capture->prefill_gpu) {
        if (pf) {
            pf->n_tokens = 0;
        }
    }

    if (dflash_profile_enabled(DFLASH_PROFILE_PREFILL)) {
        LLAMA_LOG_INFO("%s: dflash prefill plan: slot=%d capture_begin=%d capture_end=%d n_tokens=%d\n",
            __func__, (int) seq_id, (int) capture_begin, (int) capture_end, (int) plan.n_tokens);
    }
}

void llama_context::dflash_prefill_capture_end() {
    if (!dflash_capture) {
        return;
    }
    for (auto & plan : dflash_capture->prefill_plans) {
        plan.active = false;
    }
}

bool llama_context::dflash_prefill_capture_info(llama_seq_id seq_id, int32_t * n_tokens, int32_t * n_written) const {
    if (!dflash_capture) {
        return false;
    }

    const auto * plan = dflash_capture->prefill_plan_for_seq(seq_id);
    if (!plan) {
        return false;
    }

    if (!plan->active && plan->n_tokens <= 0) {
        return false;
    }

    if (plan->seq_id != seq_id) {
        return false;
    }

    if (n_tokens)  *n_tokens  = plan->n_tokens;
    if (n_written) *n_written = plan->n_written;
    return true;
}

// ---------------------------------------------------------------------------
// rollback / branch
// ---------------------------------------------------------------------------

void llama_context::dflash_rollback(llama_seq_id seq_id, llama_seq_id seq_backup, llama_pos n_past_before, int n_accepted) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) {
        LLAMA_LOG_WARN("%s: dflash_rollback requires hybrid memory\n", __func__);
        return;
    }

    auto * mem_attn = mem_hybrid->get_mem_attn();
    auto * mem_recr = mem_hybrid->get_mem_recr();

    // Flat mode: no duplicate entries at same position, safe to keep accepted KV
    int kv_keep_pos = n_past_before + n_accepted;
    mem_attn->seq_rm(seq_id, kv_keep_pos, -1);
    mem_attn->seq_rm(seq_backup, -1, -1);

    // Recurrent state: restore from backup, then tape replay
    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp(seq_backup, seq_id, -1, -1);
    mem_recr->seq_rm(seq_backup, -1, -1);

    tape_replay(seq_id, n_accepted);
}

void llama_context::dflash_prepare_branch(llama_seq_id seq_id, llama_seq_id seq_backup, int depth) {
    auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
    if (!mem_hybrid) {
        LLAMA_LOG_WARN("%s: dflash_prepare_branch requires hybrid memory\n", __func__);
        return;
    }

    auto * mem_recr = mem_hybrid->get_mem_recr();

    mem_recr->seq_rm(seq_id, -1, -1);
    mem_recr->seq_cp(seq_backup, seq_id, -1, -1);

    tape_replay(seq_id, depth);
}

// ---------------------------------------------------------------------------
// cross-attention data
// ---------------------------------------------------------------------------

static int64_t cross_bucket(int64_t n) {
    if (n <= 16) return 16;
    if (n <= 128) {
        int64_t b = 1;
        while (b < n) b <<= 1;
        return b;
    }
    const int64_t step = 128;
    return ((n + step - 1) / step) * step;
}

static int64_t dflash_max_cross_ctx() {
    static const int64_t max_ctx = [] {
        const char * e = getenv("GGML_DFLASH_MAX_CTX");
        return e ? (int64_t) atoi(e) : (int64_t) 4096;
    }();
    return max_ctx;
}

void llama_context::set_cross_data(const float * data, int64_t n_embd, int64_t n_tokens) {
    const int64_t max_ctx = dflash_max_cross_ctx();
    const int64_t capped = (max_ctx > 0 && n_tokens > max_ctx) ? max_ctx : n_tokens;
    const int64_t bucket = cross_bucket(capped);

    if (cross.n_enc != bucket) {
        sched_need_reserve = true;
    }
    cross.n_embd    = n_embd;
    cross.n_enc     = bucket;
    cross.n_enc_real = n_tokens;
    cross.v_embd_gpu = nullptr;
    cross.v_embd_gpu_n_enc_real = 0;
    cross.fn_set_tensor_d2d = nullptr;
    cross.dflash_kv_cache = nullptr;
    cross.v_embd.resize(n_embd * n_tokens);
    if (data) {
        memcpy(cross.v_embd.data(), data, n_embd * n_tokens * sizeof(float));
    }
}

void llama_context::set_cross_data_seq(llama_seq_id seq_id, const float * data, int64_t n_embd, int64_t n_tokens) {
    if (seq_id < 0) {
        set_cross_data(data, n_embd, n_tokens);
        return;
    }

    set_cross_data(data, n_embd, n_tokens);

    auto & entry = cross.v_embd_per_seq[seq_id];
    entry.n_enc      = cross.n_enc;
    entry.n_enc_real = n_tokens;
    entry.v_embd_gpu = nullptr;
    entry.v_embd_gpu_n_enc_real = 0;
    entry.v_embd.resize(n_embd * n_tokens);
    if (data) {
        memcpy(entry.v_embd.data(), data, n_embd * n_tokens * sizeof(float));
    }
}

void llama_context::set_cross_data_gpu(
        llama_seq_id seq_id, const void * d_staging, int cross_len,
        int n_layers, int n_embd_layer, set_tensor_d2d_fn_t fn_d2d) {
    int64_t n_target_features = (int64_t)n_layers * n_embd_layer;

    const int64_t max_ctx = dflash_max_cross_ctx();
    const int64_t capped = (max_ctx > 0 && cross_len > max_ctx) ? max_ctx : cross_len;
    const int64_t bucket = cross_bucket(capped);

    if (cross.n_enc != bucket) {
        sched_need_reserve = true;
    }
    cross.n_embd     = n_target_features;
    cross.n_enc      = bucket;
    cross.n_enc_real = cross_len;
    cross.v_embd_gpu = d_staging;
    cross.v_embd_gpu_n_enc_real = cross_len;
    cross.fn_set_tensor_d2d = fn_d2d;
    cross.dflash_kv_cache = nullptr;

    const bool use_gpu_only = d_staging != nullptr && fn_d2d != nullptr && cparams.dflash_n_slots <= 1;
    if (use_gpu_only) {
        std::vector<float>().swap(cross.v_embd);
    } else {
        if (cross.v_embd.size() != (size_t)(n_target_features * cross_len)) {
            cross.v_embd.resize(n_target_features * cross_len);
        }
    }

    if (seq_id >= 0) {
        auto & entry = cross.v_embd_per_seq[seq_id];
        entry.n_enc      = bucket;
        entry.n_enc_real = cross_len;
        entry.v_embd_gpu = d_staging;
        entry.v_embd_gpu_n_enc_real = cross_len;
        if (use_gpu_only) {
            std::vector<float>().swap(entry.v_embd);
        } else {
            if (entry.v_embd.size() != (size_t)(n_target_features * cross_len)) {
                entry.v_embd.resize(n_target_features * cross_len);
            }
        }
    }

    if (dflash_kv_cache && dflash_kv_cache_prepare((int) cross.n_enc)) {
        cross.dflash_kv_cache = &dflash_kv_cache->view;
    }
}

// ---------------------------------------------------------------------------
// DFlash drafter K/V projection cache
// ---------------------------------------------------------------------------

static bool dflash_kv_cache_multi_gpu_logged = false;

bool llama_context::dflash_kv_cache_init(int ctx_size) {
    if (ctx_size <= 0 || !llm_arch_is_dflash_drafter(model.arch)) {
        return false;
    }
    if (model.n_devices() > 1) {
        dflash_kv_cache.reset();
        cross.dflash_kv_cache = nullptr;
        if (!dflash_kv_cache_multi_gpu_logged) {
            LLAMA_LOG_INFO("%s: multi-GPU drafter detected (%zu devices); disabling DFlash drafter K/V projection cache\n",
                __func__, model.n_devices());
            dflash_kv_cache_multi_gpu_logged = true;
        }
        return false;
    }
    if (dflash_kv_cache && dflash_kv_cache->ring_size == ctx_size) {
        return true;
    }

    ggml_backend_t gpu_backend = nullptr;
    ggml_backend_reg_t cuda_reg = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            cuda_reg = ggml_backend_dev_backend_reg(dev);
            break;
        }
    }
    if (!gpu_backend || !cuda_reg) {
        return false;
    }

    auto fn_write_d2d = (dflash_kv_cache_data::write_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_write_d2d");
    auto fn_write_d2d_no_check = (dflash_kv_cache_data::write_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_write_d2d_no_check");
    auto fn_append_d2d = (dflash_kv_cache_data::append_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_append_d2d");
    auto fn_append_d2d_no_check = (dflash_kv_cache_data::append_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_append_d2d_no_check");
    auto fn_copy_d2d = (dflash_kv_cache_data::copy_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_copy_d2d");
    auto fn_copy_d2d_no_check = (dflash_kv_cache_data::copy_d2d_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_copy_d2d_no_check");
    auto fn_prepare_ptr = (dflash_kv_cache_data::prepare_ptr_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_sync_ptr = (dflash_kv_cache_data::sync_ptr_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    auto fn_wait_backend_stream = (dflash_kv_cache_data::sync_backend_stream_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_stream");
    auto fn_wait_dflash_stream = (dflash_kv_cache_data::sync_backend_stream_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_backend_wait_for_dflash_stream");
    auto fn_interleave = (dflash_kv_cache_data::interleave_fn_t)
        ggml_backend_reg_get_proc_address(cuda_reg, "dflash_kv_cache_interleave");
    if (!fn_write_d2d || !fn_append_d2d || !fn_interleave) {
        return false;
    }

    const int n_layers = (int) model.hparams.n_layer;
    const int64_t n_embd_head = model.hparams.n_embd_head_v();
    const int64_t n_head_kv = model.hparams.n_head_kv();
    const int n_elem = (int) (n_embd_head * n_head_kv);
    if (n_layers <= 0 || n_elem <= 0) {
        return false;
    }

    const size_t ctx_mem = ggml_tensor_overhead() * ((size_t) n_layers * 2 + 8);
    struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
    ggml_context * cache_ctx = ggml_init(ctx_params);
    if (!cache_ctx) {
        return false;
    }

    auto cache = std::make_unique<dflash_kv_cache_data>();
    cache->ctx = cache_ctx;
    cache->ring_size = ctx_size;
    cache->n_elem = n_elem;
    cache->fn_write_d2d = fn_write_d2d;
    cache->fn_write_d2d_no_check = fn_write_d2d_no_check;
    cache->fn_append_d2d = fn_append_d2d;
    cache->fn_append_d2d_no_check = fn_append_d2d_no_check;
    cache->fn_copy_d2d = fn_copy_d2d;
    cache->fn_copy_d2d_no_check = fn_copy_d2d_no_check;
    cache->fn_prepare_ptr = fn_prepare_ptr;
    cache->fn_sync_ptr = fn_sync_ptr;
    cache->fn_wait_backend_stream = fn_wait_backend_stream;
    cache->fn_wait_dflash_stream = fn_wait_dflash_stream;
    cache->fn_interleave = fn_interleave;
    cache->view.n_layers = n_layers;
    cache->view.n_embd_head = n_embd_head;
    cache->view.n_head_kv = n_head_kv;
    cache->view.ctx_len = ctx_size;
    cache->view.ring_size = ctx_size;
    cache->k_ring.resize(n_layers);
    cache->v_ring.resize(n_layers);
    cache->view.k_ring.resize(n_layers);
    cache->view.v_ring.resize(n_layers);

    for (int il = 0; il < n_layers; ++il) {
        cache->k_ring[il] = ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, ctx_size);
        cache->v_ring[il] = ggml_new_tensor_3d(cache_ctx, GGML_TYPE_F32, n_embd_head, n_head_kv, ctx_size);
        cache->view.k_ring[il] = cache->k_ring[il];
        cache->view.v_ring[il] = cache->v_ring[il];
    }

    cache->buf = ggml_backend_alloc_ctx_tensors(cache_ctx, gpu_backend);
    if (!cache->buf) {
        return false;
    }

    for (int il = 0; il < n_layers; ++il) {
        ggml_backend_tensor_memset(cache->k_ring[il], 0, 0, ggml_nbytes(cache->k_ring[il]));
        ggml_backend_tensor_memset(cache->v_ring[il], 0, 0, ggml_nbytes(cache->v_ring[il]));
    }

    const size_t total_size = ggml_backend_buffer_get_size(cache->buf);
    LLAMA_LOG_INFO("%s: allocated DFlash drafter K/V cache: %.1f MB (%d layers, %d tokens, %d elems/token)\n",
        __func__, total_size / (1024.0 * 1024.0), n_layers, ctx_size, n_elem);

    dflash_kv_cache = std::move(cache);
    cross.dflash_kv_cache = nullptr;
    return true;
}

void llama_context::dflash_kv_cache_reset() {
    if (!dflash_kv_cache) {
        return;
    }
    dflash_kv_cache->write_pos = 0;
    dflash_kv_cache->n_filled = 0;
    dflash_kv_cache->view.n_filled = 0;
    dflash_kv_cache->view.write_pos = 0;
    cross.dflash_kv_cache = nullptr;
}

bool llama_context::dflash_kv_cache_prepare(int ctx_window) {
    if (!dflash_kv_cache || ctx_window <= 0 || ctx_window > dflash_kv_cache->ring_size) {
        return false;
    }
    if (dflash_kv_cache->n_filled <= 0) {
        return false;
    }

    dflash_kv_cache->view.n_filled = std::min(dflash_kv_cache->n_filled, ctx_window);
    dflash_kv_cache->view.write_pos = dflash_kv_cache->write_pos;
    return true;
}

bool llama_context::dflash_kv_cache_update(int n_tokens) {
    if (!dflash_kv_cache || n_tokens <= 0 || !llm_arch_is_dflash_drafter(model.arch)) {
        return false;
    }
    if (model.n_devices() > 1) {
        dflash_kv_cache.reset();
        cross.dflash_kv_cache = nullptr;
        return false;
    }
    if (!cross.dflash_kv_update_gpu && !cross.v_embd_gpu && cross.v_embd.empty()) {
        return false;
    }

    n_tokens = std::min(n_tokens, dflash_kv_cache->ring_size);

    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            break;
        }
    }
    if (!gpu_backend) {
        return false;
    }

    const size_t max_nodes = graph_max_nodes(n_tokens);
    auto res = std::make_unique<llm_graph_result>(max_nodes);

    llama_ubatch ubatch = {};
    ubatch.n_tokens = (uint32_t) n_tokens;
    ubatch.n_seq_tokens = (uint32_t) n_tokens;
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;

    const auto gparams = graph_params(res.get(), ubatch, nullptr, LLM_GRAPH_TYPE_DFLASH_KV_UPDATE);
    ggml_cgraph * gf = model.build_graph(gparams);
    if (!gf) {
        return false;
    }

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
    const size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(res->get_ctx(), gpu_buft);

    if (needed > dflash_kv_cache->update_buf_size) {
        if (dflash_kv_cache->update_buf) {
            ggml_backend_buffer_free(dflash_kv_cache->update_buf);
        }
        dflash_kv_cache->update_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
        dflash_kv_cache->update_buf_size = dflash_kv_cache->update_buf
            ? ggml_backend_buffer_get_size(dflash_kv_cache->update_buf) : 0;
    }
    if (!dflash_kv_cache->update_buf) {
        return false;
    }

    {
        struct ggml_tallocr talloc = ggml_tallocr_new(dflash_kv_cache->update_buf);
        struct ggml_tensor * t = ggml_get_first_tensor(res->get_ctx());
        while (t) {
            if (t->data == nullptr && t->view_src == nullptr) {
                ggml_tallocr_alloc(&talloc, t);
            } else if (t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
            t = ggml_get_next_tensor(res->get_ctx(), t);
        }
    }

    res->set_inputs(&ubatch);

    const ggml_status status = ggml_backend_graph_compute_async(gpu_backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        return false;
    }
    if (!dflash_kv_cache->fn_wait_backend_stream ||
            !dflash_kv_cache->fn_wait_backend_stream(gpu_backend)) {
        ggml_backend_synchronize(gpu_backend);
    }

    const int n_layers = dflash_kv_cache->view.n_layers;
    if ((int) res->dflash_k_update.size() < n_layers || (int) res->dflash_v_update.size() < n_layers) {
        return false;
    }

    const int filled_before = dflash_kv_cache->n_filled;
    const int total = filled_before + n_tokens;
    const bool needs_shift = total > dflash_kv_cache->ring_size;
    const bool fast_append =
        dflash_kv_cache->fn_append_d2d_no_check &&
        dflash_kv_cache->fn_prepare_ptr &&
        !dflash_kv_cache->k_ring.empty() &&
        dflash_kv_cache->fn_prepare_ptr(dflash_kv_cache->k_ring[0]->data);
    const auto fn_append = fast_append ? dflash_kv_cache->fn_append_d2d_no_check
                                       : dflash_kv_cache->fn_append_d2d;
    bool used_async_copy = false;

    const size_t stride = (size_t) dflash_kv_cache->n_elem * sizeof(float);
    const size_t shift_bytes = (size_t) dflash_kv_cache->ring_size * stride;
    if (needs_shift && shift_bytes > dflash_kv_cache->shift_buf_size) {
        if (dflash_kv_cache->shift_buf) {
            ggml_backend_buffer_free(dflash_kv_cache->shift_buf);
            dflash_kv_cache->shift_buf = nullptr;
            dflash_kv_cache->shift_buf_size = 0;
            dflash_kv_cache->shift_ptr = nullptr;
        }
        dflash_kv_cache->shift_buf = ggml_backend_buft_alloc_buffer(gpu_buft, shift_bytes);
        dflash_kv_cache->shift_buf_size = dflash_kv_cache->shift_buf
            ? ggml_backend_buffer_get_size(dflash_kv_cache->shift_buf) : 0;
        dflash_kv_cache->shift_ptr = dflash_kv_cache->shift_buf
            ? ggml_backend_buffer_get_base(dflash_kv_cache->shift_buf) : nullptr;
    }
    if (needs_shift && (!dflash_kv_cache->shift_buf || !dflash_kv_cache->shift_ptr)) {
        return false;
    }

    const bool fast_copy =
        needs_shift &&
        dflash_kv_cache->fn_copy_d2d_no_check &&
        dflash_kv_cache->fn_prepare_ptr &&
        dflash_kv_cache->shift_ptr &&
        dflash_kv_cache->fn_prepare_ptr(dflash_kv_cache->k_ring[0]->data) &&
        dflash_kv_cache->fn_prepare_ptr(dflash_kv_cache->shift_ptr);
    const auto fn_copy = fast_copy ? dflash_kv_cache->fn_copy_d2d_no_check
                                   : dflash_kv_cache->fn_copy_d2d;

    for (int il = 0; il < n_layers; ++il) {
        if (!needs_shift) {
            if (!fn_append(
                    dflash_kv_cache->k_ring[il]->data,
                    res->dflash_k_update[il]->data,
                    dflash_kv_cache->ring_size,
                    filled_before,
                    n_tokens,
                    dflash_kv_cache->n_elem)) {
                return false;
            }
            if (!fn_append(
                    dflash_kv_cache->v_ring[il]->data,
                    res->dflash_v_update[il]->data,
                    dflash_kv_cache->ring_size,
                    filled_before,
                    n_tokens,
                    dflash_kv_cache->n_elem)) {
                return false;
            }
            continue;
        }

        auto shift_and_append = [&](void * dst_cache, const void * src_update) -> bool {
            char * cache_ptr = (char *) dst_cache;
            const char * src = (const char *) src_update;

            if (n_tokens >= dflash_kv_cache->ring_size) {
                src += (size_t) (n_tokens - dflash_kv_cache->ring_size) * stride;
                used_async_copy = true;
                return fn_copy(cache_ptr, src, shift_bytes);
            }

            const int drop = total - dflash_kv_cache->ring_size;
            const int keep = filled_before - drop;
            if (keep > 0) {
                const size_t keep_bytes = (size_t) keep * stride;
                if (!fn_copy(dflash_kv_cache->shift_ptr, cache_ptr + (size_t) drop * stride, keep_bytes)) {
                    return false;
                }
                if (!fn_copy(cache_ptr, dflash_kv_cache->shift_ptr, keep_bytes)) {
                    return false;
                }
            }

            used_async_copy = true;
            return fn_copy(cache_ptr + (size_t) keep * stride, src, (size_t) n_tokens * stride);
        };

        if (!shift_and_append(dflash_kv_cache->k_ring[il]->data, res->dflash_k_update[il]->data)) {
            return false;
        }
        if (!shift_and_append(dflash_kv_cache->v_ring[il]->data, res->dflash_v_update[il]->data)) {
            return false;
        }
    }

    if (used_async_copy || (!needs_shift && fast_append)) {
        if (dflash_kv_cache->fn_wait_dflash_stream) {
            if (!dflash_kv_cache->fn_wait_dflash_stream(gpu_backend)) {
                return false;
            }
        } else if (!dflash_kv_cache->fn_sync_ptr || !dflash_kv_cache->fn_sync_ptr(dflash_kv_cache->k_ring[0]->data)) {
            return false;
        }
    }

    dflash_kv_cache->write_pos = 0;
    dflash_kv_cache->n_filled = std::min(filled_before + n_tokens, dflash_kv_cache->ring_size);
    dflash_kv_cache->view.n_filled = dflash_kv_cache->n_filled;
    dflash_kv_cache->view.write_pos = dflash_kv_cache->write_pos;
    return true;
}

bool llama_context::dflash_kv_cache_update_gpu(
        const void * d_hidden,
        int n_tokens,
        int n_layers,
        int n_embd_layer,
        set_tensor_d2d_fn_t fn_d2d) {
    if (!d_hidden || n_tokens <= 0 || n_layers <= 0 || n_embd_layer <= 0 || !fn_d2d) {
        return false;
    }

    const int64_t n_target_features = (int64_t) n_layers * n_embd_layer;
    if (n_target_features != (int64_t) model.hparams.dflash_n_target_features) {
        return false;
    }

    cross.dflash_kv_update_gpu = d_hidden;
    cross.dflash_kv_update_n_embd = n_target_features;
    cross.dflash_kv_update_n_enc_real = n_tokens;
    cross.dflash_kv_update_fn_set_tensor_d2d = fn_d2d;

    const bool ok = dflash_kv_cache_update(n_tokens);

    cross.dflash_kv_update_gpu = nullptr;
    cross.dflash_kv_update_n_embd = 0;
    cross.dflash_kv_update_n_enc_real = 0;
    cross.dflash_kv_update_fn_set_tensor_d2d = nullptr;

    return ok;
}

bool llama_context::dflash_target_kv_cache_update_gpu(
        llama_seq_id seq_id,
        llama_pos start_pos,
        const void * d_hidden,
        int n_tokens,
        int n_layers,
        int n_embd_layer,
        set_tensor_d2d_fn_t fn_d2d) {
    if (!d_hidden || n_tokens <= 0 || n_layers <= 0 || n_embd_layer <= 0 || !fn_d2d) {
        return false;
    }
    if (!llm_arch_is_dflash_drafter(model.arch) || !memory) {
        return false;
    }
    if (model.n_devices() > 1) {
        return false;
    }

    const int64_t n_target_features = (int64_t) n_layers * n_embd_layer;
    if (n_target_features != (int64_t) model.hparams.dflash_n_target_features) {
        return false;
    }

    // Get the base KV cache for slot finding
    llama_kv_cache * base_kv = nullptr;
    if (auto * kv = dynamic_cast<llama_kv_cache *>(memory.get())) {
        base_kv = kv;
    } else if (auto * kv_iswa = dynamic_cast<llama_kv_cache_iswa *>(memory.get())) {
        base_kv = kv_iswa->get_base();
    }
    if (!base_kv) {
        return false;
    }

    base_kv->seq_rm(seq_id, start_pos, -1);

    // Build a synthetic ubatch for the commit
    auto ubatch_data = std::make_shared<llama_ubatch::data_t>();
    const uint32_t n_pos = model.hparams.n_pos_per_embd();
    ubatch_data->token.resize(n_tokens, 0);
    ubatch_data->pos.resize((size_t) n_tokens * n_pos, 0);
    ubatch_data->n_seq_id.resize(n_tokens, 1);
    ubatch_data->seq_id.resize(n_tokens);
    ubatch_data->seq_idx.resize(LLAMA_MAX_SEQ, -1);
    ubatch_data->output.resize(n_tokens, 0);
    ubatch_data->seq_id_data.resize(n_tokens, seq_id);
    ubatch_data->seq_id_unq.push_back(seq_id);

    if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
        ubatch_data->seq_idx[(size_t) seq_id] = 0;
    }

    for (int i = 0; i < n_tokens; ++i) {
        ubatch_data->seq_id[i] = &ubatch_data->seq_id_data[i];
        ubatch_data->pos[i] = start_pos + (llama_pos) i;
    }
    if (!ubatch_data->output.empty()) {
        ubatch_data->output.back() = 1;
    }

    llama_ubatch ubatch = {
        /*.b_equal_seqs =*/ true,
        /*.n_tokens     =*/ (uint32_t) n_tokens,
        /*.n_seq_tokens =*/ (uint32_t) n_tokens,
        /*.n_seqs       =*/ 1,
        /*.n_seqs_unq   =*/ 1,
        /*.n_pos        =*/ n_pos,
        /*.token        =*/ ubatch_data->token.data(),
        /*.embd         =*/ nullptr,
        /*.pos          =*/ ubatch_data->pos.data(),
        /*.n_seq_id     =*/ ubatch_data->n_seq_id.data(),
        /*.seq_id       =*/ ubatch_data->seq_id.data(),
        /*.seq_id_unq   =*/ ubatch_data->seq_id_unq.data(),
        /*.seq_idx      =*/ ubatch_data->seq_idx.data(),
        /*.output       =*/ ubatch_data->output.data(),
        /*.data         =*/ std::move(ubatch_data),
    };

    auto sinfo = base_kv->find_slot(ubatch, false);
    if (sinfo.empty()) {
        return false;
    }
    base_kv->apply_ubatch(sinfo, ubatch);

    std::vector<llama_ubatch> ubatches = { ubatch };
    llama_kv_cache::slot_info_vec_t sinfos;
    sinfos.push_back(std::move(sinfo));
    auto mctx = std::make_unique<llama_kv_cache_context>(base_kv, std::move(sinfos), std::move(ubatches));

    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            break;
        }
    }
    if (!gpu_backend) {
        return false;
    }

    const size_t max_nodes = graph_max_nodes(n_tokens);
    auto res = std::make_unique<llm_graph_result>(max_nodes);

    const auto gparams = graph_params(res.get(), ubatch, mctx.get(), LLM_GRAPH_TYPE_DFLASH_KV_UPDATE);
    ggml_cgraph * gf = model.build_graph(gparams);
    if (!gf) {
        return false;
    }

    ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
    const size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(res->get_ctx(), gpu_buft);

    ggml_backend_buffer_t alloc_buf = nullptr;
    const bool reuse_update_buf = dflash_kv_cache != nullptr;
    if (reuse_update_buf) {
        if (needed > dflash_kv_cache->update_buf_size) {
            if (dflash_kv_cache->update_buf) {
                ggml_backend_buffer_free(dflash_kv_cache->update_buf);
            }
            dflash_kv_cache->update_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
            dflash_kv_cache->update_buf_size = dflash_kv_cache->update_buf
                ? ggml_backend_buffer_get_size(dflash_kv_cache->update_buf) : 0;
        }
        alloc_buf = dflash_kv_cache->update_buf;
    } else {
        alloc_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
    }
    if (!alloc_buf) {
        return false;
    }

    {
        struct ggml_tallocr talloc = ggml_tallocr_new(alloc_buf);
        struct ggml_tensor * t = ggml_get_first_tensor(res->get_ctx());
        while (t) {
            if (t->data == nullptr && t->view_src == nullptr) {
                ggml_tallocr_alloc(&talloc, t);
            } else if (t->view_src != nullptr && t->buffer == nullptr) {
                ggml_backend_view_init(t);
            }
            t = ggml_get_next_tensor(res->get_ctx(), t);
        }
    }

    cross.dflash_kv_update_gpu = d_hidden;
    cross.dflash_kv_update_n_embd = n_target_features;
    cross.dflash_kv_update_n_enc_real = n_tokens;
    cross.dflash_kv_update_fn_set_tensor_d2d = fn_d2d;

    res->set_inputs(&ubatch);
    const ggml_status status = ggml_backend_graph_compute_async(gpu_backend, gf);

    cross.dflash_kv_update_gpu = nullptr;
    cross.dflash_kv_update_n_embd = 0;
    cross.dflash_kv_update_n_enc_real = 0;
    cross.dflash_kv_update_fn_set_tensor_d2d = nullptr;

    bool synchronized = false;
    if (status == GGML_STATUS_SUCCESS) {
        const bool ordered =
            reuse_update_buf &&
            dflash_kv_cache &&
            dflash_kv_cache->fn_wait_backend_stream &&
            dflash_kv_cache->fn_wait_backend_stream(gpu_backend);
        if (!ordered) {
            ggml_backend_synchronize(gpu_backend);
            synchronized = true;
        }
    }

    if (!reuse_update_buf) {
        if (!synchronized) {
            ggml_backend_synchronize(gpu_backend);
        }
        ggml_backend_buffer_free(alloc_buf);
    }

    return status == GGML_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// tape replay — CPU fallback
// ---------------------------------------------------------------------------

void llama_context::tape_replay_cpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers   = dflash_capture->tape_layers;
    const uint32_t n_embd_s = hparams.n_embd_s();
    if (dflash_capture->profile) {
        dflash_capture->profile_replay_cpu_fallback += 1;
        dflash_capture->profile_replay_layers += rec_ids.size();
    }

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers[li];

        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;

        const int64_t S = tape.S_k;
        const int64_t H_k = tape.H_k;
        const int64_t H_v = tape.H_v;
        if (S <= 0 || H_k <= 0 || H_v <= 0) continue;
        if ((size_t) S * (size_t) S * (size_t) H_v != (size_t) n_embd_s) continue;

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        const size_t s_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
        std::vector<float> state(n_embd_s);
        ggml_backend_tensor_get(s_tensor, state.data(), s_offset, n_embd_s * sizeof(float));

        for (int tok = 0; tok < n_accepted; ++tok) {
            for (int64_t hv = 0; hv < H_v; ++hv) {
                const int64_t hk = hv % H_k;
                float g_val = exp2f(tape.gate[tok * H_v + hv] * 1.442695041f);
                float b_val = 1.0f / (1.0f + expf(-tape.beta[tok * H_v + hv]));

                float * S_h = state.data() + hv * S * S;
                const float * k_t = tape.k.data() + tok * (S * H_k) + hk * S;
                const float * v_t = tape.v.data() + tok * (S * H_v) + hv * S;

                for (int64_t col = 0; col < S; ++col) {
                    float kv = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        kv += S_h[col * S + row] * k_t[row];
                    }
                    float delta_col = (v_t[col] - g_val * kv) * b_val;
                    for (int64_t row = 0; row < S; ++row) {
                        S_h[col * S + row] = g_val * S_h[col * S + row] + k_t[row] * delta_col;
                    }
                }
            }
        }

        ggml_backend_tensor_set(s_tensor, state.data(), s_offset, n_embd_s * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// tape replay — GPU GDN direct
// ---------------------------------------------------------------------------

bool llama_context::tape_replay_gdn_direct_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    if (!cuda_reg) {
        return false;
    }
    using ptr_device_fn_t = bool (*)(const void *, int *);
    using replay_fn_t = bool (*)(void *, const void *, const void *, const void *, const void *, int, int, int, int);
    auto fn_ptr_device = (ptr_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_ptr_device");
    auto fn_replay = (replay_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_replay_gdn_state_no_check");
    if (!fn_ptr_device || !fn_replay) {
        return false;
    }

    dflash_tape_gpu * gpu_tape = dflash_capture->active_tape();
    if (!gpu_tape || n_accepted > gpu_tape->max_tokens || n_accepted > gpu_tape->n_tokens) {
        return false;
    }

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const uint32_t n_embd_s = model.hparams.n_embd_s();

    struct replay_launch {
        void * state;
        const void * k;
        const void * v;
        const void * gate;
        const void * beta;
        int S, H_k, H_v, device;
    };
    std::vector<replay_launch> launches;
    launches.reserve(rec_ids.size());

    auto is_cuda_tensor = [](const ggml_tensor * t) {
        if (!t || !t->data || !t->buffer) return false;
        const char * name = ggml_backend_buffer_name(t->buffer);
        return name && std::strncmp(name, "CUDA", 4) == 0;
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        const int il = rec_ids[li];
        if (li >= gpu_tape->layers.size()) return false;

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        auto & tl = gpu_tape->layers[li];
        if (!is_cuda_tensor(s_tensor) || !is_cuda_tensor(tl.k) || !is_cuda_tensor(tl.v) ||
                !is_cuda_tensor(tl.gate) || !is_cuda_tensor(tl.beta)) {
            return false;
        }

        const int64_t S_i = tl.v->ne[0], H_v_i = tl.v->ne[1], H_k_i = tl.k->ne[1];
        if (S_i <= 0 || H_k_i <= 0 || H_v_i <= 0) return false;
        if (!(S_i == 16 || S_i == 32 || S_i == 64 || S_i == 128)) return false;

        const size_t s_offset = (size_t) cell_idx * n_embd_s * ggml_element_size(s_tensor);
        launches.push_back({
            (char *) s_tensor->data + s_offset,
            tl.k->data, tl.v->data, tl.gate->data, tl.beta->data,
            (int) S_i, (int) H_k_i, (int) H_v_i, -1,
        });
    }

    if (launches.empty()) return false;

    int replay_device = -1;
    for (auto & launch : launches) {
        int device = -1;
        if (!fn_ptr_device(launch.state, &device)) return false;
        if (replay_device < 0) replay_device = device;
        else if (device != replay_device) return false;
        launch.device = device;
    }

    dflash_capture->replay_sync_ptrs.clear();
    dflash_capture->replay_sync_device = replay_device;
    for (const auto & launch : launches) {
        if (!fn_replay(launch.state, launch.k, launch.v, launch.gate, launch.beta,
                    n_accepted, launch.S, launch.H_k, launch.H_v)) {
            GGML_ABORT("DFlash direct GPU GDN replay launch failed after validation\n");
        }
    }
    dflash_capture->replay_sync_ptr = launches.back().state;
    return true;
}

bool llama_context::tape_replay_gdn_direct_from_cpu_tape(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    if (!cuda_reg) return false;

    using prepare_ptr_fn_t = bool (*)(const void *);
    using replay_fn_t = bool (*)(void *, const void *, const void *, const void *, const void *, int, int, int, int);
    using sync_ptr_fn_t = bool (*)(const void *);
    auto fn_prepare = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_replay = (replay_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_replay_gdn_state_no_check");
    auto fn_sync = (sync_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    if (!fn_prepare || !fn_replay || !fn_sync) return false;

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers_ref = dflash_capture->tape_layers;
    if (rec_ids.empty() || tape_layers_ref.empty()) return false;

    const uint32_t n_embd_s = model.hparams.n_embd_s();

    struct replay_upload {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        void * state = nullptr;
        ggml_tensor * k = nullptr, * v = nullptr, * gate = nullptr, * beta = nullptr;
        int S = 0, H_k = 0, H_v = 0;
    };
    std::vector<replay_upload> uploads;
    uploads.reserve(rec_ids.size());

    auto cleanup = [&]() {
        for (auto & u : uploads) {
            if (u.buf) ggml_backend_buffer_free(u.buf);
            if (u.ctx) ggml_free(u.ctx);
        }
    };

    auto is_cuda_tensor = [](const ggml_tensor * t) {
        if (!t || !t->data || !t->buffer || ggml_backend_buffer_is_host(t->buffer)) return false;
        const char * name = ggml_backend_buffer_name(t->buffer);
        return name && std::strncmp(name, "CUDA", 4) == 0;
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        if (li >= tape_layers_ref.size()) { cleanup(); return false; }
        const int il = rec_ids[li];
        const auto & tape = tape_layers_ref[li];
        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) { cleanup(); return false; }

        const int64_t S_i = tape.S_k, H_k_i = tape.H_k, H_v_i = tape.H_v;
        if (S_i <= 0 || H_k_i <= 0 || H_v_i <= 0) { cleanup(); return false; }
        if (!(S_i == 16 || S_i == 32 || S_i == 64 || S_i == 128)) { cleanup(); return false; }
        if ((size_t)S_i * (size_t)S_i * (size_t)H_v_i != (size_t)n_embd_s) { cleanup(); return false; }

        ggml_tensor * s_tensor = mem_recurrent->s_l[il];
        if (!is_cuda_tensor(s_tensor)) { cleanup(); return false; }

        const size_t k_elems = (size_t)S_i * (size_t)H_k_i * (size_t)n_accepted;
        const size_t v_elems = (size_t)S_i * (size_t)H_v_i * (size_t)n_accepted;
        const size_t gb_elems = (size_t)H_v_i * (size_t)n_accepted;

        if (tape.k.size() < k_elems || tape.v.size() < v_elems ||
                tape.gate.size() < gb_elems || tape.beta.size() < gb_elems) {
            cleanup(); return false;
        }

        const size_t ctx_mem = ggml_tensor_overhead() * 4;
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        ggml_context * ctx = ggml_init(ctx_params);
        if (!ctx) { cleanup(); return false; }

        replay_upload u;
        u.ctx = ctx;
        u.k    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)k_elems);
        u.v    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)v_elems);
        u.gate = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)gb_elems);
        u.beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)gb_elems);

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(s_tensor->buffer);
        u.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!u.buf) { uploads.push_back(u); cleanup(); return false; }

        ggml_backend_tensor_set(u.k,    tape.k.data(),    0, k_elems  * sizeof(float));
        ggml_backend_tensor_set(u.v,    tape.v.data(),    0, v_elems  * sizeof(float));
        ggml_backend_tensor_set(u.gate, tape.gate.data(), 0, gb_elems * sizeof(float));
        ggml_backend_tensor_set(u.beta, tape.beta.data(), 0, gb_elems * sizeof(float));

        const size_t s_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
        u.state = (char *)s_tensor->data + s_offset;
        u.S = (int)S_i; u.H_k = (int)H_k_i; u.H_v = (int)H_v_i;
        uploads.push_back(u);
    }

    if (uploads.empty()) { cleanup(); return false; }

    for (const auto & u : uploads) {
        if (!fn_prepare(u.state)) { cleanup(); return false; }
    }
    for (const auto & u : uploads) {
        if (!fn_replay(u.state, u.k->data, u.v->data, u.gate->data, u.beta->data,
                    n_accepted, u.S, u.H_k, u.H_v)) {
            cleanup();
            GGML_ABORT("DFlash direct CPU-tape GDN replay launch failed\n");
        }
    }
    bool synced = true;
    for (const auto & u : uploads) {
        synced = fn_sync(u.state) && synced;
    }
    cleanup();
    if (!synced) {
        GGML_ABORT("DFlash direct CPU-tape GDN replay sync failed\n");
    }
    return true;
}

// ---------------------------------------------------------------------------
// tape replay — conv state rebuild
// ---------------------------------------------------------------------------

bool llama_context::tape_replay_conv_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted) {
    return tape_replay_conv_gpu(mem_recurrent, cell_idx, n_accepted, true);
}

bool llama_context::tape_replay_conv_gpu(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, bool advance_pos) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0 || model.n_devices() > 1) {
        return false;
    }

    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    if (!cuda_reg) return false;
    using rebuild_fn_t = bool (*)(void *, const void *, int, int, int);
    auto fn_rebuild = (rebuild_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_rebuild_conv_state");
    if (!fn_rebuild) return false;

    dflash_tape_gpu * gpu_tape = dflash_capture->active_tape();
    if (!gpu_tape || n_accepted > gpu_tape->max_tokens || n_accepted > gpu_tape->n_tokens) {
        return false;
    }

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    const uint32_t n_embd_r = model.hparams.n_embd_r();

    struct conv_launch { void * r_state; const void * qkv; int conv_ch; int conv_window; };
    std::vector<conv_launch> launches;
    launches.reserve(rec_ids.size());

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        const int il = rec_ids[li];
        if (li >= gpu_tape->layers.size()) return false;

        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        ggml_tensor * qkv_tensor = gpu_tape->layers[li].qkv;
        if (!r_tensor || !r_tensor->data || !qkv_tensor || !qkv_tensor->data) return false;

        const int64_t conv_ch = qkv_tensor->ne[0];
        if (conv_ch <= 0 || n_embd_r % conv_ch != 0) return false;
        const int64_t conv_window = n_embd_r / conv_ch;

        const size_t r_offset = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);
        launches.push_back({ (char *)r_tensor->data + r_offset, qkv_tensor->data, (int)conv_ch, (int)conv_window });
    }

    if (launches.empty()) return false;

    for (const auto & l : launches) {
        if (!fn_rebuild(l.r_state, l.qkv, n_accepted, l.conv_ch, l.conv_window)) return false;
    }

    if (advance_pos) {
        mem_recurrent->cells[cell_idx].pos += n_accepted;
    }
    return true;
}

bool llama_context::tape_replay_conv_gpu_from_cpu_tape(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id) {
    if (!dflash_capture || !mem_recurrent || n_accepted <= 0) return false;

    ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
    if (!cuda_reg) return false;
    using prepare_ptr_fn_t = bool (*)(const void *);
    using rebuild_fn_t = bool (*)(void *, const void *, int, int, int);
    using sync_ptr_fn_t = bool (*)(const void *);
    auto fn_prepare = (prepare_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_prepare_ptr");
    auto fn_rebuild = (rebuild_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_rebuild_conv_state");
    auto fn_sync = (sync_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr");
    if (!fn_prepare || !fn_rebuild || !fn_sync) return false;

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers_ref = dflash_capture->tape_layers;
    if (rec_ids.empty() || tape_layers_ref.empty()) return false;

    const uint32_t n_embd_r = model.hparams.n_embd_r();

    auto is_cuda_tensor = [](const ggml_tensor * t) {
        if (!t || !t->data || !t->buffer || ggml_backend_buffer_is_host(t->buffer)) return false;
        const char * name = ggml_backend_buffer_name(t->buffer);
        return name && std::strncmp(name, "CUDA", 4) == 0;
    };

    struct conv_upload {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        void * r_state = nullptr;
        ggml_tensor * qkv = nullptr;
        int conv_ch = 0, conv_window = 0;
    };
    std::vector<conv_upload> uploads;
    uploads.reserve(rec_ids.size());

    auto cleanup = [&]() {
        for (auto & u : uploads) {
            if (u.buf) ggml_backend_buffer_free(u.buf);
            if (u.ctx) ggml_free(u.ctx);
        }
    };

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        if (li >= tape_layers_ref.size()) { cleanup(); return false; }
        const int il = rec_ids[li];
        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        if (!r_tensor) continue;
        if (!is_cuda_tensor(r_tensor)) { cleanup(); return false; }

        const auto & tape = tape_layers_ref[li];
        if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens || tape.qkv_mixed.empty()) { cleanup(); return false; }

        size_t qkv_seq_offset = 0;
        if (tape.n_seqs > 1) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                if (tape.seq_ids[s] == seq_id) { found = true; break; }
                qkv_seq_offset += (size_t)tape.n_tokens * (size_t)tape.conv_channels;
            }
            if (!found) { cleanup(); return false; }
        }

        const int64_t conv_ch = tape.conv_channels;
        if (conv_ch <= 0 || n_embd_r % conv_ch != 0) { cleanup(); return false; }
        const int64_t conv_window = n_embd_r / conv_ch;

        const size_t qkv_elems = (size_t)n_accepted * (size_t)conv_ch;
        if (tape.qkv_mixed.size() < qkv_seq_offset + qkv_elems) { cleanup(); return false; }

        const size_t ctx_mem = ggml_tensor_overhead();
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        ggml_context * ctx = ggml_init(ctx_params);
        if (!ctx) { cleanup(); return false; }

        conv_upload u;
        u.ctx = ctx;
        u.qkv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)qkv_elems);

        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(r_tensor->buffer);
        u.buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!u.buf) { uploads.push_back(u); cleanup(); return false; }

        ggml_backend_tensor_set(u.qkv, tape.qkv_mixed.data() + qkv_seq_offset, 0, qkv_elems * sizeof(float));

        const size_t r_offset = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);
        u.r_state = (char *)r_tensor->data + r_offset;
        u.conv_ch = (int)conv_ch;
        u.conv_window = (int)conv_window;
        uploads.push_back(u);
    }

    if (uploads.empty()) { cleanup(); return false; }

    for (const auto & u : uploads) { if (!fn_prepare(u.r_state)) { cleanup(); return false; } }
    for (const auto & u : uploads) {
        if (!fn_rebuild(u.r_state, u.qkv->data, n_accepted, u.conv_ch, u.conv_window)) {
            cleanup(); GGML_ABORT("DFlash CPU-tape conv replay failed\n");
        }
    }
    bool synced = true;
    for (const auto & u : uploads) { synced = fn_sync(u.r_state) && synced; }
    cleanup();
    if (!synced) { GGML_ABORT("DFlash CPU-tape conv replay sync failed\n"); }

    mem_recurrent->cells[cell_idx].pos += n_accepted;
    return true;
}

void llama_context::tape_replay_conv(llama_memory_recurrent * mem_recurrent, int32_t cell_idx, int n_accepted, llama_seq_id seq_id) {
    const auto & hparams = model.hparams;
    const auto & rec_ids = dflash_capture->recurrent_layer_ids;
    auto & tape_layers_ref = dflash_capture->tape_layers;
    const uint32_t n_embd_r = hparams.n_embd_r();

    if (model.n_devices() <= 1 && tape_replay_conv_gpu(mem_recurrent, cell_idx, n_accepted)) {
        return;
    }
    if (model.n_devices() > 1 && tape_replay_conv_gpu_from_cpu_tape(mem_recurrent, cell_idx, n_accepted, seq_id)) {
        return;
    }

    // CPU conv rebuild fallback
    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            break;
        }
    }
    const bool use_async = gpu_backend && model.n_devices() <= 1;

    struct conv_layer_data {
        size_t tape_li;
        ggml_tensor * r_tensor;
        size_t r_offset;
        int64_t conv_ch, conv_window;
        std::vector<float> old_window, qkv_mixed, new_conv;
    };
    std::vector<conv_layer_data> layers;
    layers.reserve(rec_ids.size());

    for (size_t li = 0; li < rec_ids.size(); ++li) {
        int il = rec_ids[li];
        auto & tape = tape_layers_ref[li];
        dflash_tape_gpu * gpu_tape = dflash_capture->active_tape();
        dflash_tape_gpu_layer * gpu_layer = nullptr;
        if (gpu_tape && li < gpu_tape->layers.size() &&
                n_accepted <= gpu_tape->max_tokens && n_accepted <= gpu_tape->n_tokens) {
            gpu_layer = &gpu_tape->layers[li];
        }

        if (!mem_recurrent->r_l[il]) continue;
        const bool use_gpu_qkv = gpu_backend && gpu_layer && gpu_layer->qkv;
        if (!use_gpu_qkv) {
            if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens || tape.qkv_mixed.empty()) continue;
        }

        size_t qkv_seq_offset = 0;
        if (!use_gpu_qkv && tape.n_seqs > 1) {
            bool found = false;
            for (int s = 0; s < tape.n_seqs; ++s) {
                if (tape.seq_ids[s] == seq_id) { found = true; break; }
                qkv_seq_offset += (size_t)tape.n_tokens * (size_t)tape.conv_channels;
            }
            GGML_ASSERT(found);
        }

        ggml_tensor * r_tensor = mem_recurrent->r_l[il];
        const size_t r_offset = (size_t)cell_idx * n_embd_r * ggml_element_size(r_tensor);
        const int64_t conv_ch = use_gpu_qkv ? gpu_layer->qkv->ne[0] : tape.conv_channels;
        GGML_ASSERT(conv_ch > 0 && n_embd_r % conv_ch == 0);
        const int64_t conv_window = (int64_t)(n_embd_r / conv_ch);

        conv_layer_data & d = layers.emplace_back();
        d.tape_li = li;
        d.r_tensor = r_tensor;
        d.r_offset = r_offset;
        d.conv_ch = conv_ch;
        d.conv_window = conv_window;
        d.old_window.resize(n_embd_r);
        d.qkv_mixed.resize((size_t)n_accepted * (size_t)conv_ch);
        d.new_conv.resize(n_embd_r);

        if (use_async) {
            ggml_backend_tensor_get_async(gpu_backend, r_tensor, d.old_window.data(), r_offset, n_embd_r * sizeof(float));
        } else {
            ggml_backend_tensor_get(r_tensor, d.old_window.data(), r_offset, n_embd_r * sizeof(float));
        }
        if (use_gpu_qkv) {
            if (use_async) {
                ggml_backend_tensor_get_async(gpu_backend, gpu_layer->qkv, d.qkv_mixed.data(), 0, d.qkv_mixed.size() * sizeof(float));
            } else {
                ggml_backend_tensor_get(gpu_layer->qkv, d.qkv_mixed.data(), 0, d.qkv_mixed.size() * sizeof(float));
            }
        } else {
            std::memcpy(d.qkv_mixed.data(), tape.qkv_mixed.data() + qkv_seq_offset, d.qkv_mixed.size() * sizeof(float));
        }
    }

    if (use_async && !layers.empty()) {
        ggml_backend_synchronize(gpu_backend);
    }

    for (auto & d : layers) {
        for (int64_t w = 0; w < d.conv_window; ++w) {
            int src_pos = n_accepted + (int)w;
            for (int64_t ch = 0; ch < d.conv_ch; ++ch) {
                float val;
                if (src_pos < (int)d.conv_window) {
                    val = d.old_window[ch * d.conv_window + src_pos];
                } else {
                    val = d.qkv_mixed[(src_pos - d.conv_window) * d.conv_ch + ch];
                }
                d.new_conv[ch * d.conv_window + w] = val;
            }
        }
    }

    for (auto & d : layers) {
        if (use_async) {
            ggml_backend_tensor_set_async(gpu_backend, d.r_tensor, d.new_conv.data(), d.r_offset, n_embd_r * sizeof(float));
        } else {
            ggml_backend_tensor_set(d.r_tensor, d.new_conv.data(), d.r_offset, n_embd_r * sizeof(float));
        }
    }
    if (use_async && !layers.empty()) {
        ggml_backend_synchronize(gpu_backend);
    }

    mem_recurrent->cells[cell_idx].pos += n_accepted;
}

// ---------------------------------------------------------------------------
// tape replay — sync
// ---------------------------------------------------------------------------

void llama_context::tape_replay_sync() {
    if (!dflash_capture || !dflash_capture->replay_pending) {
        return;
    }

    auto * backend = dflash_capture->replay_gpu_backend;
    if (backend) {
        if (!dflash_capture->replay_event) {
            auto * dev = ggml_backend_get_device(backend);
            if (dev) {
                dflash_capture->replay_event = ggml_backend_event_new(dev);
            }
        }
        if (dflash_capture->replay_event) {
            ggml_backend_event_record(dflash_capture->replay_event, backend);
            ggml_backend_event_synchronize(dflash_capture->replay_event);
        } else {
            ggml_backend_synchronize(backend);
        }
    } else if (dflash_capture->replay_direct_gpu &&
            (!dflash_capture->replay_sync_ptrs.empty() || dflash_capture->replay_sync_ptr)) {
        ggml_backend_reg_t cuda_reg = ggml_backend_reg_by_name("CUDA");
        using sync_ptr_fn_t = bool (*)(const void *);
        using sync_device_fn_t = bool (*)(int);
        auto fn_sync_ptr = cuda_reg
            ? (sync_ptr_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_ptr")
            : nullptr;
        auto fn_sync_device = cuda_reg
            ? (sync_device_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cuda_synchronize_device")
            : nullptr;
        bool synced = false;
        if (fn_sync_device && dflash_capture->replay_sync_device >= 0) {
            synced = fn_sync_device(dflash_capture->replay_sync_device);
        } else if (fn_sync_ptr) {
            if (dflash_capture->replay_sync_ptr) {
                synced = fn_sync_ptr(dflash_capture->replay_sync_ptr);
            } else {
                synced = true;
                for (const void * ptr : dflash_capture->replay_sync_ptrs) {
                    synced = fn_sync_ptr(ptr) && synced;
                }
            }
        }
        if (!synced) {
            LLAMA_LOG_WARN("%s: direct GPU tape replay sync failed\n", __func__);
        }
    }

    if (dflash_capture->replay_graph_ctx) {
        ggml_free(dflash_capture->replay_graph_ctx);
        dflash_capture->replay_graph_ctx = nullptr;
    }

    tape_replay_conv(dflash_capture->replay_mem_recurrent,
                     dflash_capture->replay_cell_idx,
                     dflash_capture->replay_n_accepted,
                     dflash_capture->replay_seq_id);

    dflash_capture->replay_pending = false;
    dflash_capture->replay_direct_gpu = false;
    dflash_capture->replay_sync_ptr = nullptr;
    dflash_capture->replay_sync_ptrs.clear();
    dflash_capture->replay_sync_device = -1;
}

// ---------------------------------------------------------------------------
// tape replay — main dispatch
// ---------------------------------------------------------------------------

void llama_context::tape_replay(llama_seq_id seq_id, int n_accepted) {
    if (!dflash_capture || n_accepted <= 0) {
        return;
    }

    tape_replay_sync();

    dflash_tape_gpu * const gpu_tape = dflash_capture->active_tape();
    const bool use_gpu_tape = (gpu_tape != nullptr &&
                               n_accepted <= gpu_tape->max_tokens &&
                               n_accepted <= gpu_tape->n_tokens);

    if (!use_gpu_tape && dflash_capture->tape_layers.empty()) {
        return;
    }

    auto * mem_recurrent = dynamic_cast<llama_memory_recurrent *>(memory.get());
    if (!mem_recurrent) {
        auto * mem_hybrid = dynamic_cast<llama_memory_hybrid *>(memory.get());
        if (mem_hybrid) {
            mem_recurrent = mem_hybrid->get_mem_recr();
        }
    }
    if (!mem_recurrent) {
        LLAMA_LOG_WARN("%s: tape replay requires recurrent memory\n", __func__);
        return;
    }

    const auto & rec_ids = dflash_capture->recurrent_layer_ids;

    int32_t cell_idx = -1;
    if (seq_id >= 0 && (uint32_t) seq_id < mem_recurrent->size) {
        int32_t tail = mem_recurrent->cells[seq_id].tail;
        if (tail >= 0) {
            cell_idx = tail;
        }
    }
    if (cell_idx < 0) {
        LLAMA_LOG_WARN("%s: no active cell for seq %d\n", __func__, seq_id);
        return;
    }

    ggml_backend_t gpu_backend = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_backend = backend.get();
            break;
        }
    }

    if (!gpu_backend) {
        tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return;
    }

    // partial offload: CPU-resident state → CPU replay
    for (int li = 0; li < (int) rec_ids.size(); ++li) {
        ggml_tensor * s_tensor = mem_recurrent->s_l[rec_ids[li]];
        if (s_tensor && s_tensor->buffer && ggml_backend_buffer_is_host(s_tensor->buffer)) {
            tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return;
        }
    }

    const bool multi_gpu_target = model.n_devices() > 1;
    if (multi_gpu_target) {
        if (tape_replay_gdn_direct_from_cpu_tape(mem_recurrent, cell_idx, n_accepted)) {
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return;
        }
        if (!dflash_capture->multi_gpu_replay_fallback_logged) {
            LLAMA_LOG_WARN("%s: multi-GPU target; using CPU recurrent replay fallback\n", __func__);
            dflash_capture->multi_gpu_replay_fallback_logged = true;
        }
        tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return;
    }

    if (use_gpu_tape && tape_replay_gdn_direct_gpu(mem_recurrent, cell_idx, n_accepted)) {
        dflash_capture->replay_pending = true;
        dflash_capture->replay_gpu_backend = nullptr;
        dflash_capture->replay_graph_ctx = nullptr;
        dflash_capture->replay_direct_gpu = true;
        dflash_capture->replay_n_accepted = n_accepted;
        dflash_capture->replay_cell_idx = cell_idx;
        dflash_capture->replay_seq_id = seq_id;
        dflash_capture->replay_mem_recurrent = mem_recurrent;
        return;
    }

    // ggml graph-based GPU replay
    const int n_rec = (int) rec_ids.size();
    if (n_rec == 0) {
        tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
        return;
    }

    {
        const uint32_t n_embd_s = model.hparams.n_embd_s();
        size_t ctx_mem = ggml_tensor_overhead() * ((size_t)n_rec * 14 + 4) + ggml_graph_overhead_custom(n_rec * 12, false);
        struct ggml_init_params ctx_params = { ctx_mem, nullptr, true };
        struct ggml_context * ctx = ggml_init(ctx_params);
        struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_rec * 12, false);

        struct replay_input {
            ggml_tensor * q, * k, * v, * g, * b;
            size_t tape_li;
        };
        std::vector<replay_input> inputs;
        inputs.reserve(n_rec);

        for (int li = 0; li < n_rec; ++li) {
            int il = rec_ids[li];
            int64_t S, H_k, H_v;

            if (use_gpu_tape) {
                auto & tl = gpu_tape->layers[li];
                S = tl.k->ne[0]; H_k = tl.k->ne[1]; H_v = tl.v->ne[1];
            } else {
                auto & tape = dflash_capture->tape_layers[li];
                if (tape.n_tokens <= 0 || n_accepted > tape.n_tokens) continue;
                S = tape.S_k; H_k = tape.H_k; H_v = tape.H_v;
            }

            ggml_tensor * k_in, * v_in, * g_in, * b_in;

            if (use_gpu_tape) {
                auto & tl = gpu_tape->layers[li];
                k_in = ggml_view_4d(ctx, tl.k, S, H_k, (int64_t)n_accepted, 1LL,
                    tl.k->nb[1], tl.k->nb[2], tl.k->nb[2] * n_accepted, 0);
                v_in = ggml_view_4d(ctx, tl.v, S, H_v, (int64_t)n_accepted, 1LL,
                    tl.v->nb[1], tl.v->nb[2], tl.v->nb[2] * n_accepted, 0);
                g_in = ggml_view_4d(ctx, tl.gate, 1LL, H_v, (int64_t)n_accepted, 1LL,
                    tl.gate->nb[1], tl.gate->nb[2], tl.gate->nb[2] * n_accepted, 0);
                b_in = ggml_view_4d(ctx, tl.beta, 1LL, H_v, (int64_t)n_accepted, 1LL,
                    tl.beta->nb[1], tl.beta->nb[2], tl.beta->nb[2] * n_accepted, 0);
            } else {
                k_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t)n_accepted, 1LL);
                v_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, (int64_t)n_accepted, 1LL);
                g_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1LL, H_v, (int64_t)n_accepted, 1LL);
                b_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1LL, H_v, (int64_t)n_accepted, 1LL);
                ggml_set_input(k_in); ggml_set_input(v_in);
                ggml_set_input(g_in); ggml_set_input(b_in);
            }

            ggml_tensor * q_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, (int64_t)n_accepted, 1LL);
            ggml_set_input(q_in);

            ggml_tensor * b_sigmoid = ggml_sigmoid(ctx, b_in);

            ggml_tensor * s_tensor = mem_recurrent->s_l[il];
            size_t s_byte_offset = (size_t)cell_idx * n_embd_s * ggml_element_size(s_tensor);
            ggml_tensor * s_view = ggml_view_4d(ctx, s_tensor, S, S, H_v, 1LL,
                S * ggml_element_size(s_tensor),
                S * S * ggml_element_size(s_tensor),
                S * S * H_v * ggml_element_size(s_tensor),
                s_byte_offset);

            ggml_tensor * result = ggml_gated_delta_net(ctx, q_in, k_in, v_in, g_in, b_sigmoid, s_view);

            size_t attn_bytes = (size_t)(S * H_v * n_accepted) * ggml_element_size(result);
            ggml_tensor * result_state = ggml_view_1d(ctx, result, n_embd_s, attn_bytes);
            ggml_tensor * s_write = ggml_view_1d(ctx, s_tensor, n_embd_s, s_byte_offset);
            ggml_tensor * cpy = ggml_cpy(ctx, result_state, s_write);
            ggml_build_forward_expand(graph, cpy);

            inputs.push_back({ q_in, k_in, v_in, g_in, b_in, (size_t)li });
        }

        if (inputs.empty()) {
            ggml_free(ctx);
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return;
        }

        ggml_backend_buffer_type_t gpu_buft = ggml_backend_get_default_buffer_type(gpu_backend);
        size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, gpu_buft);

        if (needed > dflash_capture->replay_buf_size) {
            if (dflash_capture->replay_buf) {
                ggml_backend_buffer_free(dflash_capture->replay_buf);
            }
            dflash_capture->replay_buf = ggml_backend_buft_alloc_buffer(gpu_buft, needed);
            dflash_capture->replay_buf_size = dflash_capture->replay_buf
                ? ggml_backend_buffer_get_size(dflash_capture->replay_buf) : 0;
        }

        if (!dflash_capture->replay_buf) {
            ggml_free(ctx);
            tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
            tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            return;
        }

        {
            struct ggml_tallocr talloc = ggml_tallocr_new(dflash_capture->replay_buf);
            struct ggml_tensor * t = ggml_get_first_tensor(ctx);
            while (t) {
                if (t->data == nullptr && t->view_src == nullptr) {
                    ggml_tallocr_alloc(&talloc, t);
                } else if (t->view_src != nullptr && t->buffer == nullptr) {
                    ggml_backend_view_init(t);
                }
                t = ggml_get_next_tensor(ctx, t);
            }
        }

        for (auto & inp : inputs) {
            {
                const int64_t S = inp.q->ne[0];
                const int64_t H = inp.q->ne[1];
                size_t q_size = (size_t)(S * H * n_accepted);
                if (dflash_capture->replay_zeros.size() < q_size) {
                    dflash_capture->replay_zeros.resize(q_size, 0.0f);
                }
                ggml_backend_tensor_set(inp.q, dflash_capture->replay_zeros.data(), 0, ggml_nbytes(inp.q));
            }

            if (!use_gpu_tape) {
                auto & tape = dflash_capture->tape_layers[inp.tape_li];
                const int64_t S = tape.S_k;
                const int64_t H_k = tape.H_k;
                const int64_t H_v = tape.H_v;
                ggml_backend_tensor_set(inp.k, tape.k.data(),    0, S * H_k * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.v, tape.v.data(),    0, S * H_v * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.g, tape.gate.data(), 0, H_v * n_accepted * sizeof(float));
                ggml_backend_tensor_set(inp.b, tape.beta.data(), 0, H_v * n_accepted * sizeof(float));
            }
        }

        const ggml_status replay_status = ggml_backend_graph_compute_async(gpu_backend, graph);
        if (replay_status != GGML_STATUS_SUCCESS) {
            ggml_free(ctx);
            if (!use_gpu_tape) {
                tape_replay_cpu(mem_recurrent, cell_idx, n_accepted);
                tape_replay_conv(mem_recurrent, cell_idx, n_accepted, seq_id);
            }
            return;
        }

        dflash_capture->replay_pending = true;
        dflash_capture->replay_gpu_backend = gpu_backend;
        dflash_capture->replay_graph_ctx = ctx;
        dflash_capture->replay_direct_gpu = false;
        dflash_capture->replay_sync_device = -1;
        dflash_capture->replay_n_accepted = n_accepted;
        dflash_capture->replay_cell_idx = cell_idx;
        dflash_capture->replay_seq_id = seq_id;
        dflash_capture->replay_mem_recurrent = mem_recurrent;
    }
}

// ---------------------------------------------------------------------------
// cross ring GPU
// ---------------------------------------------------------------------------

struct dflash_cross_ring_handle {
    void * gpu_ring;
    void   (*fn_free)(void *);
    void   (*fn_write)(void *, int, int, const float *, int, int);
    bool   (*fn_write_d2d)(void *, int, int, const void *, int, int);
    void   (*fn_synchronize)(void *);
    bool   (*fn_snapshot)(void *, int, int, int, float *, int, int, int);
    const float * (*fn_interleave)(void *, int, int, int);
    void   (*fn_set_tensor)(void *, const void *, size_t, size_t);
};

void * llama_context::init_cross_ring_gpu(int n_layers, int n_embd, int ring_size) {
    ggml_backend_reg_t cuda_reg = nullptr;
    for (auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            cuda_reg = ggml_backend_dev_backend_reg(dev);
            break;
        }
    }
    if (!cuda_reg) return nullptr;

    using alloc_fn_t      = void * (*)(int, int, int);
    using free_fn_t       = void   (*)(void *);
    using write_fn_t      = void   (*)(void *, int, int, const float *, int, int);
    using write_d2d_fn_t  = bool   (*)(void *, int, int, const void *, int, int);
    using sync_fn_t       = void   (*)(void *);
    using snapshot_fn_t   = bool   (*)(void *, int, int, int, float *, int, int, int);
    using interleave_fn_t = const float * (*)(void *, int, int, int);
    using set_tensor_fn_t = void   (*)(void *, const void *, size_t, size_t);

    auto fn_alloc      = (alloc_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_alloc");
    auto fn_free       = (free_fn_t)       ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_free");
    auto fn_write      = (write_fn_t)      ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_write");
    auto fn_write_d2d  = (write_d2d_fn_t)  ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_write_d2d");
    auto fn_sync       = (sync_fn_t)       ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_synchronize");
    auto fn_snapshot   = (snapshot_fn_t)   ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_snapshot");
    auto fn_interleave = (interleave_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_interleave");
    auto fn_set_tensor = (set_tensor_fn_t) ggml_backend_reg_get_proc_address(cuda_reg, "dflash_cross_ring_gpu_set_tensor");

    if (!fn_alloc || !fn_free || !fn_write || !fn_write_d2d || !fn_sync || !fn_snapshot || !fn_interleave || !fn_set_tensor) {
        return nullptr;
    }

    void * gpu_ring = fn_alloc(n_layers, n_embd, ring_size);
    if (!gpu_ring) return nullptr;

    auto * handle = new dflash_cross_ring_handle();
    handle->gpu_ring      = gpu_ring;
    handle->fn_free       = fn_free;
    handle->fn_write      = fn_write;
    handle->fn_write_d2d  = fn_write_d2d;
    handle->fn_synchronize = fn_sync;
    handle->fn_snapshot   = fn_snapshot;
    handle->fn_interleave = fn_interleave;
    handle->fn_set_tensor = fn_set_tensor;
    return handle;
}

bool llama_context::cross_ring_gpu_write_hidden(void * handle, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd) {
    if (!handle || !dflash_capture) return false;
    auto * hgpu = dflash_capture->active_hidden_gpu();
    if (!hgpu || layer < 0 || layer >= (int) hgpu->layers.size()) return false;
    if (src_offset < 0 || n_tokens <= 0 || n_embd != hgpu->n_embd || src_offset + n_tokens > hgpu->n_tokens) return false;

    auto * tensor = hgpu->layers[layer];
    if (!tensor || !tensor->data) return false;

    auto * h = (dflash_cross_ring_handle *)handle;
    const size_t src_offset_bytes = (size_t)src_offset * (size_t)n_embd * sizeof(float);
    const void * src = (const char *)tensor->data + src_offset_bytes;
    if (h->fn_write_d2d(h->gpu_ring, layer, ring_pos, src, n_tokens, n_embd)) {
        return true;
    }

    const size_t n_bytes = (size_t)n_tokens * (size_t)n_embd * sizeof(float);
    std::vector<float> staging((size_t)n_tokens * (size_t)n_embd);
    ggml_backend_tensor_get(tensor, staging.data(), src_offset_bytes, n_bytes);
    h->fn_write(h->gpu_ring, layer, ring_pos, staging.data(), n_tokens, n_embd);
    h->fn_synchronize(h->gpu_ring);
    return true;
}

bool llama_context::prefill_gpu_write_hidden(void * handle, int slot, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd) {
    if (!handle || !dflash_capture) return false;
    if (slot < 0 || slot >= (int) dflash_capture->prefill_gpu.size()) return false;
    auto * pgpu = dflash_capture->prefill_gpu[slot].get();
    if (!pgpu || layer < 0 || layer >= (int) pgpu->layers.size()) return false;
    if (src_offset < 0 || n_tokens <= 0 || n_embd != pgpu->n_embd || src_offset + n_tokens > pgpu->n_tokens) return false;

    auto * tensor = pgpu->layers[layer];
    if (!tensor || !tensor->data) return false;

    auto * h = (dflash_cross_ring_handle *)handle;
    const size_t src_offset_bytes = (size_t)src_offset * (size_t)n_embd * sizeof(float);
    const void * src = (const char *)tensor->data + src_offset_bytes;
    if (h->fn_write_d2d(h->gpu_ring, layer, ring_pos, src, n_tokens, n_embd)) {
        return true;
    }

    const size_t n_bytes = (size_t)n_tokens * (size_t)n_embd * sizeof(float);
    std::vector<float> staging((size_t)n_tokens * (size_t)n_embd);
    ggml_backend_tensor_get(tensor, staging.data(), src_offset_bytes, n_bytes);
    h->fn_write(h->gpu_ring, layer, ring_pos, staging.data(), n_tokens, n_embd);
    h->fn_synchronize(h->gpu_ring);
    return true;
}

// ---------------------------------------------------------------------------
// C-API wrappers
// ---------------------------------------------------------------------------

void llama_set_dflash_capture(llama_context * ctx, const int32_t * layer_ids, int32_t n_layers) {
    ctx->set_dflash_capture(layer_ids, n_layers);
}
void llama_set_dflash_capture_active(llama_context * ctx, bool active) {
    ctx->set_dflash_capture_active(active);
}
void llama_set_dflash_gpu_capture(llama_context * ctx, bool enabled) {
    ctx->set_dflash_gpu_capture(enabled);
}
void llama_set_dflash_sample_temp(llama_context * ctx, float temp) {
    ctx->set_dflash_sample_temp(temp);
}
void llama_set_dflash_topk(llama_context * ctx, int k) {
    ctx->set_dflash_topk(k);
}
void llama_set_dflash_verify_logits(llama_context * ctx, bool enabled, int top_k) {
    ctx->set_dflash_verify_logits(enabled, top_k);
}
void llama_set_dflash_consume_reduced(llama_context * ctx, bool enabled) {
    ctx->set_dflash_consume_reduced(enabled);
}
void llama_set_dflash_n_slots(llama_context * ctx, int n) {
    ctx->set_dflash_n_slots(n);
}
void llama_set_tape_recording(llama_context * ctx, bool enable) {
    ctx->set_tape_recording(enable);
}
void llama_set_force_split_seq(llama_context * ctx, bool force) {
    auto * mem = llama_get_memory(ctx);
    if (mem) {
        mem->set_force_split_seq(force);
    }
}
void llama_dflash_allocate_slots(llama_context * ctx, int n_slots) {
    ctx->allocate_tape_gpu(n_slots, LLAMA_DFLASH_MAX_VERIFY_TOKENS);
}
void llama_dflash_set_active_slot(llama_context * ctx, int slot_idx) {
    ctx->set_active_dflash_slot(slot_idx);
}
void llama_tape_replay(llama_context * ctx, llama_seq_id seq_id, int n_accepted) {
    ctx->tape_replay(seq_id, n_accepted);
}
void llama_tape_replay_sync(llama_context * ctx) {
    ctx->tape_replay_sync();
}
bool llama_dflash_memory_seq_cp_recurrent_ordered(
        llama_context * ctx, llama_seq_id src, llama_seq_id dst, llama_pos p0, llama_pos p1) {
    return ctx ? ctx->dflash_memory_seq_cp_recurrent_ordered(src, dst, p0, p1) : false;
}
void llama_dflash_rollback(llama_context * ctx, llama_seq_id seq_id, llama_seq_id seq_backup, int n_past_before, int n_accepted) {
    ctx->dflash_rollback(seq_id, seq_backup, n_past_before, n_accepted);
}
void llama_dflash_prepare_branch(llama_context * ctx, llama_seq_id seq_id, llama_seq_id seq_backup, int depth) {
    ctx->dflash_prepare_branch(seq_id, seq_backup, depth);
}
void llama_set_cross_data(llama_context * ctx, const float * data, int64_t n_embd, int64_t n_tokens) {
    ctx->set_cross_data(data, n_embd, n_tokens);
}
void llama_set_cross_data_seq(llama_context * ctx, llama_seq_id seq_id, const float * data, int64_t n_embd, int64_t n_tokens) {
    ctx->set_cross_data_seq(seq_id, data, n_embd, n_tokens);
}

// cross ring GPU C-API
void * llama_dflash_cross_ring_gpu_init(llama_context * ctx, int n_layers, int n_embd, int ring_size) {
    return ctx->init_cross_ring_gpu(n_layers, n_embd, ring_size);
}
void llama_dflash_cross_ring_gpu_free(void * handle) {
    if (!handle) return;
    auto * h = (dflash_cross_ring_handle *)handle;
    h->fn_free(h->gpu_ring);
    delete h;
}
void llama_dflash_cross_ring_gpu_write(void * handle, int layer, int ring_pos, const float * data, int n_tokens, int n_embd) {
    if (!handle) return;
    auto * h = (dflash_cross_ring_handle *)handle;
    h->fn_write(h->gpu_ring, layer, ring_pos, data, n_tokens, n_embd);
}
bool llama_dflash_cross_ring_gpu_write_hidden(void * handle, llama_context * ctx, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd) {
    if (!ctx) return false;
    return ctx->cross_ring_gpu_write_hidden(handle, layer, ring_pos, src_offset, n_tokens, n_embd);
}
bool llama_dflash_prefill_gpu_write_hidden(void * handle, llama_context * ctx, int slot, int layer, int ring_pos, int src_offset, int n_tokens, int n_embd) {
    if (!ctx) return false;
    return ctx->prefill_gpu_write_hidden(handle, slot, layer, ring_pos, src_offset, n_tokens, n_embd);
}
bool llama_dflash_prefill_gpu_active(llama_context * ctx) {
    if (!ctx) return false;
    return ctx->prefill_gpu_active();
}
int64_t llama_dflash_prefill_gpu_n_tokens(llama_context * ctx, int slot) {
    if (!ctx) return 0;
    return ctx->prefill_gpu_n_tokens(slot);
}
void llama_dflash_prefill_capture_begin(llama_context * ctx, llama_seq_id seq_id, int32_t capture_begin, int32_t capture_end) {
    if (!ctx) return;
    ctx->dflash_prefill_capture_begin(seq_id, capture_begin, capture_end);
}
void llama_dflash_prefill_capture_end(llama_context * ctx) {
    if (!ctx) return;
    ctx->dflash_prefill_capture_end();
}
bool llama_dflash_prefill_capture_info(llama_context * ctx, llama_seq_id seq_id, int32_t * n_tokens, int32_t * n_written) {
    if (!ctx) return false;
    return ctx->dflash_prefill_capture_info(seq_id, n_tokens, n_written);
}
void llama_dflash_cross_ring_gpu_synchronize(void * handle) {
    if (!handle) return;
    auto * h = (dflash_cross_ring_handle *)handle;
    h->fn_synchronize(h->gpu_ring);
}
bool llama_dflash_cross_ring_gpu_snapshot(
        void * handle, int ring_write_pos, int ring_filled, int ctx_window,
        float * data, int n_tokens, int n_layers, int n_embd) {
    if (!handle) return false;
    auto * h = (dflash_cross_ring_handle *)handle;
    return h->fn_snapshot(h->gpu_ring, ring_write_pos, ring_filled, ctx_window,
            data, n_tokens, n_layers, n_embd);
}
void llama_dflash_cross_ring_gpu_set_cross(
        llama_context * ctx, void * handle, llama_seq_id seq_id,
        int ring_write_pos, int ring_filled,
        int n_layers, int n_embd, int ctx_window) {
    if (!handle || !ctx) return;
    auto * h = (dflash_cross_ring_handle *)handle;
    const float * d_staging = h->fn_interleave(h->gpu_ring, ring_write_pos, ring_filled, ctx_window);
    if (!d_staging) return;
    int cross_len = ring_filled < ctx_window ? ring_filled : ctx_window;
    ctx->set_cross_data_gpu(seq_id, d_staging, cross_len, n_layers, n_embd, h->fn_set_tensor);
}
bool llama_dflash_kv_cache_init(llama_context * ctx, int ctx_size) {
    if (!ctx) return false;
    return ctx->dflash_kv_cache_init(ctx_size);
}
void llama_dflash_kv_cache_reset(llama_context * ctx) {
    if (!ctx) return;
    ctx->dflash_kv_cache_reset();
}
bool llama_dflash_kv_cache_update(llama_context * ctx, int n_tokens) {
    if (!ctx) return false;
    return ctx->dflash_kv_cache_update(n_tokens);
}
bool llama_dflash_kv_cache_update_from_ring(
        llama_context * ctx, void * handle,
        int ring_write_pos, int ring_filled,
        int n_layers, int n_embd, int n_tokens) {
    if (!ctx || !handle || n_tokens <= 0) return false;
    auto * h = (dflash_cross_ring_handle *)handle;
    const float * d_staging = h->fn_interleave(h->gpu_ring, ring_write_pos, ring_filled, n_tokens);
    if (!d_staging) return false;
    const int n_update = std::min(ring_filled, n_tokens);
    return ctx->dflash_kv_cache_update_gpu(d_staging, n_update, n_layers, n_embd, h->fn_set_tensor);
}
bool llama_dflash_target_kv_cache_update_from_ring(
        llama_context * ctx, void * handle,
        int ring_write_pos, int ring_filled,
        int n_layers, int n_embd, int n_tokens,
        llama_seq_id seq_id, llama_pos start_pos) {
    if (!ctx || !handle || n_tokens <= 0) return false;
    auto * h = (dflash_cross_ring_handle *)handle;
    const float * d_staging = h->fn_interleave(h->gpu_ring, ring_write_pos, ring_filled, n_tokens);
    if (!d_staging) return false;
    const int n_update = std::min(ring_filled, n_tokens);
    return ctx->dflash_target_kv_cache_update_gpu(
        seq_id, start_pos, d_staging, n_update, n_layers, n_embd, h->fn_set_tensor);
}

// ---------------------------------------------------------------------------
// decode loop shims — called from the do-while in llama_context::decode()
// ---------------------------------------------------------------------------

void llama_context::dflash_prepare_ubatch_impl(const llama_ubatch & ubatch) {
    if (!dflash_capture) {
        return;
    }

    if (!dflash_capture->capture_active) {
        cparams.cb_eval = nullptr;
        cparams.cb_eval_user_data = nullptr;
        cparams.hidden_gpu_n_seqs = 0;
        dflash_clear_prefill_cparams(cparams);
        cparams.tape_gpu_n_seqs = 0;
        cparams.tape_gpu = nullptr;
        for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.tape_gpu_seqs[s] = nullptr;
            cparams.hidden_gpu_seqs[s] = nullptr;
        }
        dflash_capture->ubatch = &ubatch;
        return;
    }

    const bool dflash_gpu_capture_ready = model.n_devices() <= 1 && dflash_capture->gpu_capture_enabled;
    dflash_capture->ubatch = &ubatch;
    cparams.hidden_gpu_n_seqs = 0;
    dflash_clear_prefill_cparams(cparams);
    for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
        cparams.hidden_gpu_seqs[s] = nullptr;
    }

    const int ns = std::min((int) ubatch.n_seqs_unq, (int) LLAMA_DFLASH_MAX_SLOTS);

    const int dflash_capture_n_seqs =
        ubatch.n_seqs_unq > 1 ? (int) ubatch.n_seqs_unq : 1;

    const int dflash_capture_n_tokens =
        ubatch.n_seqs_unq > 1 ? (int) ubatch.n_seq_tokens
                               : (int) ubatch.n_tokens;

    const bool dflash_prefill_plan_active = dflash_capture->any_prefill_plan_active();
    const int dflash_prefill_plan_max_tokens = dflash_capture->max_prefill_plan_tokens();
    const bool prefill_plan_needs_staging =
        dflash_prefill_plan_active &&
        dflash_prefill_plan_max_tokens > LLAMA_DFLASH_MAX_VERIFY_TOKENS;
    const bool dflash_use_prefill_staging = prefill_plan_needs_staging;

    bool dflash_graph_hidden_ready = false;
    bool dflash_suppress_callback_for_view = false;

    if (dflash_capture && dflash_diagnostic_debug_enabled() && dflash_prefill_plan_active) {
        LLAMA_LOG_INFO(
            "%s: dflash capture route: n_tokens=%u n_seq_tokens=%u n_seqs=%u n_seqs_unq=%u capture_n_tokens=%d use_prefill_staging=%d\n",
            __func__,
            ubatch.n_tokens,
            ubatch.n_seq_tokens,
            ubatch.n_seqs,
            ubatch.n_seqs_unq,
            dflash_capture_n_tokens,
            dflash_use_prefill_staging ? 1 : 0);
    }

    if (dflash_use_prefill_staging && dflash_gpu_capture_ready) {
        const int needed_tokens = dflash_prefill_plan_max_tokens;
        const int needed_slots = std::max(ns, (int) dflash_capture->prefill_plans.size());
        if (dflash_capture->prefill_gpu.empty() ||
            dflash_capture->prefill_gpu_max_tokens < needed_tokens) {
            allocate_prefill_gpu(needed_slots, needed_tokens);
        }

        const bool prefill_fits = !dflash_capture->prefill_gpu.empty() &&
            needed_tokens <= dflash_capture->prefill_gpu_max_tokens;

        if (prefill_fits) {
            int32_t ubatch_pos_min[LLAMA_DFLASH_MAX_SLOTS];
            int32_t ubatch_pos_max[LLAMA_DFLASH_MAX_SLOTS];
            bool ubatch_seq_seen[LLAMA_DFLASH_MAX_SLOTS] = {};
            for (int s = 0; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
                ubatch_pos_min[s] = INT32_MAX;
                ubatch_pos_max[s] = INT32_MIN;
            }
            for (uint32_t t = 0; t < ubatch.n_tokens; ++t) {
                for (uint32_t k = 0; k < ubatch.n_seq_id[t]; ++k) {
                    const llama_seq_id tok_seq = ubatch.seq_id[t][k];
                    for (int s = 0; s < ns; ++s) {
                        if (ubatch.seq_id_unq[s] != tok_seq) {
                            continue;
                        }
                        ubatch_seq_seen[s] = true;
                        ubatch_pos_min[s] = std::min(ubatch_pos_min[s], (int32_t) ubatch.pos[t]);
                        ubatch_pos_max[s] = std::max(ubatch_pos_max[s], (int32_t) ubatch.pos[t]);
                    }
                }
            }

            bool any_intersection = false;
            bool all_intersections_have_buffer = true;
            for (int s = 0; s < ns; ++s) {
                const llama_seq_id seq = ubatch.seq_id_unq[s];
                cparams.prefill_gpu_seqs[s] = nullptr;

                auto * plan = dflash_capture->prefill_plan_for_seq(seq);
                if (!plan || !plan->active || plan->n_tokens <= 0 || !ubatch_seq_seen[s]) {
                    continue;
                }

                const int32_t ubatch_end = ubatch_pos_max[s] + 1;
                const int32_t inter_begin = std::max(ubatch_pos_min[s], plan->capture_begin);
                const int32_t inter_end   = std::min(ubatch_end, plan->capture_end);
                if (inter_begin >= inter_end) {
                    if (dflash_diagnostic_debug_enabled()) {
                        LLAMA_LOG_INFO("%s: dflash prefill capture ubatch: slot=%d ubatch=[%d,%d) no intersection with capture=[%d,%d)\n",
                            __func__, (int) seq, (int) ubatch_pos_min[s], (int) ubatch_end,
                            (int) plan->capture_begin, (int) plan->capture_end);
                    }
                    continue;
                }

                const int src_offset = inter_begin - ubatch_pos_min[s];
                const int dst_offset = inter_begin - plan->capture_begin;
                const int n_copy     = inter_end - inter_begin;
                any_intersection = true;

                dflash_hidden_gpu * hp = nullptr;
                if (seq >= 0 && seq < (int) dflash_capture->prefill_gpu.size()) {
                    hp = dflash_capture->prefill_gpu[seq].get();
                }
                if (!hp) {
                    all_intersections_have_buffer = false;
                    continue;
                }

                cparams.prefill_gpu_seqs[s] = hp;
                cparams.dflash_prefill_src_offsets[s] = src_offset;
                cparams.dflash_prefill_dst_offsets[s] = dst_offset;
                cparams.dflash_prefill_n_tokens_seqs[s] = n_copy;
                if (cparams.dflash_prefill_n_tokens <= 0) {
                    cparams.dflash_prefill_src_offset = src_offset;
                    cparams.dflash_prefill_dst_offset = dst_offset;
                    cparams.dflash_prefill_n_tokens   = n_copy;
                } else {
                    cparams.dflash_prefill_n_tokens = std::max(cparams.dflash_prefill_n_tokens, n_copy);
                }

                if (dflash_diagnostic_debug_enabled()) {
                    LLAMA_LOG_INFO("%s: dflash prefill capture ubatch: slot=%d ubatch=[%d,%d) inter=[%d,%d) src_offset=%d dst_offset=%d n_tokens=%d\n",
                        __func__,
                        (int) seq,
                        (int) ubatch_pos_min[s], (int) ubatch_end,
                        (int) inter_begin, (int) inter_end,
                        src_offset, dst_offset, n_copy);
                }
            }
            for (int s = ns; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
                cparams.prefill_gpu_seqs[s] = nullptr;
            }
            if (any_intersection && all_intersections_have_buffer) {
                dflash_graph_hidden_ready = true;
                cparams.prefill_gpu_n_seqs = ns;
                cparams.dflash_prefill_capture_active = true;
            } else if (!any_intersection) {
                cparams.prefill_gpu_n_seqs = 0;
                cparams.dflash_prefill_capture_active = false;
                dflash_suppress_callback_for_view = true;
                dflash_graph_hidden_ready = false;
            } else {
                dflash_clear_prefill_cparams(cparams);
                dflash_graph_hidden_ready = false;
            }
        }
    } else {
        dflash_clear_prefill_cparams(cparams);
    }

    // verify-sized hidden_gpu path
    if (!dflash_graph_hidden_ready) {
        dflash_graph_hidden_ready =
            !dflash_capture->hidden_gpu.empty() &&
            dflash_gpu_capture_ready &&
            dflash_capture_n_tokens <= LLAMA_DFLASH_MAX_VERIFY_TOKENS;

        for (int s = 0; s < ns; ++s) {
            const llama_seq_id seq = ubatch.seq_id_unq[s];
            dflash_hidden_gpu * hp = nullptr;
            if (seq >= 0 && seq < (int) dflash_capture->hidden_gpu.size()) {
                hp = dflash_capture->hidden_gpu[seq].get();
                if (hp) {
                    hp->n_tokens = dflash_capture_n_tokens <= hp->max_tokens ? dflash_capture_n_tokens : 0;
                }
            }
            cparams.hidden_gpu_seqs[s] = hp;
            dflash_graph_hidden_ready = dflash_graph_hidden_ready && hp && hp->n_tokens > 0;
        }
        for (int s = ns; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
            cparams.hidden_gpu_seqs[s] = nullptr;
        }
        if (dflash_graph_hidden_ready) {
            cparams.hidden_gpu_n_seqs = ns;
        }
    }

    // tape readiness
    bool dflash_graph_tape_ready =
        !dflash_use_prefill_staging &&
        !dflash_capture->tapes.empty() &&
        dflash_gpu_capture_ready &&
        dflash_capture->tape_enabled &&
        dflash_capture_n_tokens <= LLAMA_DFLASH_MAX_VERIFY_TOKENS;
    const int tape_ns = dflash_graph_tape_ready ? ns : 0;
    const int prev_tape_ns = cparams.tape_gpu_n_seqs;

    bool seqs_changed = (tape_ns != prev_tape_ns);

    for (int s = 0; s < ns; ++s) {
        const llama_seq_id seq = ubatch.seq_id_unq[s];
        dflash_tape_gpu * tp = nullptr;
        if (seq >= 0 && seq < (int) dflash_capture->tapes.size()) {
            tp = dflash_capture->tapes[seq].get();
            if (tp) {
                tp->n_tokens = dflash_capture_n_tokens <= tp->max_tokens ? dflash_capture_n_tokens : 0;
            }
        }
        dflash_tape_gpu * graph_tp = dflash_graph_tape_ready ? tp : nullptr;
        if (graph_tp != cparams.tape_gpu_seqs[s]) {
            seqs_changed = true;
        }
        cparams.tape_gpu_seqs[s] = graph_tp;
    }
    for (int s = ns; s < (int) LLAMA_DFLASH_MAX_SLOTS; ++s) {
        if (cparams.tape_gpu_seqs[s] != nullptr) {
            seqs_changed = true;
        }
        cparams.tape_gpu_seqs[s] = nullptr;
    }
    cparams.tape_gpu_n_seqs = tape_ns;
    cparams.tape_gpu = tape_ns > 0 ? cparams.tape_gpu_seqs[0] : nullptr;

    if (seqs_changed && gf_res_prev) {
        if (dflash_diagnostic_debug_enabled()) {
            LLAMA_LOG_INFO(
                "%s: dflash decode graph reset: seqs_changed=1 tape_ns=%d prev_tape_ns=%d hidden_ns=%d use_prefill=%d prefill_active=%d capture_active=%d\n",
                __func__,
                tape_ns,
                prev_tape_ns,
                cparams.hidden_gpu_n_seqs,
                dflash_use_prefill_staging ? 1 : 0,
                cparams.dflash_prefill_capture_active ? 1 : 0,
                dflash_capture->capture_active ? 1 : 0);
        }
        gf_res_prev->reset();
    }

    // track active slot for single-seq ubatches
    if (ubatch.n_seqs_unq == 1) {
        const llama_seq_id seq = ubatch.seq_id_unq[0];
        if (seq >= 0 && seq < (int) dflash_capture->tapes.size()) {
            dflash_capture->active_tape_idx = seq;
        }
    }

    const bool dflash_skip_eval_callback =
        dflash_graph_hidden_ready || dflash_suppress_callback_for_view;

    cparams.cb_eval = dflash_skip_eval_callback ? nullptr : dflash_eval_callback;
    cparams.cb_eval_user_data = dflash_skip_eval_callback ? nullptr : dflash_capture.get();

    GGML_UNUSED(dflash_capture_n_seqs);
}

void llama_context::dflash_post_ubatch_impl(const llama_ubatch & ubatch, const llm_graph_result * res, ggml_status /*status*/, int n_out, int64_t /*n_vocab_unused*/) {
    if (!dflash_capture || !res) {
        return;
    }

    // GPU capture stream sync
    const bool gpu_stream_ready = dflash_wait_for_gpu_capture_stream();
    if (!gpu_stream_ready) {
        const int64_t t_sync_start_us = dflash_capture->profile ? ggml_time_us() : 0;
        ggml_backend_sched_synchronize(sched.get());
        if (dflash_capture->profile && dflash_profile_sync_split_enabled()) {
            dflash_capture->profile_verify_sync_split_us += ggml_time_us() - t_sync_start_us;
        }
    } else if (dflash_capture->profile && dflash_profile_sync_split_enabled()) {
        const int64_t t_sync_start_us = ggml_time_us();
        ggml_backend_sched_synchronize(sched.get());
        dflash_capture->profile_verify_sync_split_us += ggml_time_us() - t_sync_start_us;
    }

    // prefill plan accounting
    if (cparams.dflash_prefill_capture_active && cparams.dflash_prefill_n_tokens > 0) {
        const int n_seqs_ub = std::min((int) ubatch.n_seqs_unq, (int) LLAMA_DFLASH_MAX_SLOTS);
        for (int s = 0; s < n_seqs_ub; ++s) {
            const int n_written_seq = cparams.dflash_prefill_n_tokens_seqs[s];
            if (n_written_seq <= 0) {
                continue;
            }
            const llama_seq_id seq = ubatch.seq_id_unq[s];
            auto * plan = dflash_capture->prefill_plan_for_seq(seq);
            if (!plan || !plan->active) {
                continue;
            }

            const int32_t new_written = cparams.dflash_prefill_dst_offsets[s] + n_written_seq;
            if (new_written > plan->n_written) {
                plan->n_written = new_written;
            }

            if (seq == plan->seq_id && seq >= 0 && seq < (int) dflash_capture->prefill_gpu.size()) {
                auto * pgpu = dflash_capture->prefill_gpu[seq].get();
                if (pgpu) {
                    pgpu->n_tokens = std::max(pgpu->n_tokens, (int) plan->n_written);
                }
            }

            if (dflash_profile_enabled(DFLASH_PROFILE_PREFILL)) {
                LLAMA_LOG_INFO("%s: dflash prefill capture complete: slot=%d planned=%d written=%d\n",
                    __func__, (int) plan->seq_id, (int) plan->n_tokens, (int) plan->n_written);
            }
        }
    }

    // argmax/topK readback
    auto * t_argmax = res->t_logits_argmax;
    if (t_argmax && n_out > 0) {
        ggml_backend_t backend_argmax = ggml_backend_sched_get_tensor_backend(sched.get(), t_argmax);
        GGML_ASSERT(backend_argmax != nullptr);
        const int64_t total_elems = ggml_nelements(t_argmax);
        const int K = (int)(total_elems / (2 * n_out));
        const int n_ids = K * n_out;
        const size_t ids_bytes = (size_t) n_ids * sizeof(int32_t);
        const size_t probs_bytes = (size_t) n_ids * sizeof(float);
        const bool prof = dflash_capture->profile;
        const int64_t t_start_us = prof ? ggml_time_us() : 0;
        logits_argmax_buf.resize(n_ids);
        const int64_t t_ids_start_us = prof ? ggml_time_us() : 0;
        ggml_backend_tensor_get_async(backend_argmax, t_argmax, logits_argmax_buf.data(), 0, ids_bytes);
        if (prof) {
            dflash_capture->profile_reduced_logits_ids_us += ggml_time_us() - t_ids_start_us;
        }
        logits_argmax_prob_buf.resize(n_ids);
        const int64_t t_probs_start_us = prof ? ggml_time_us() : 0;
        ggml_backend_tensor_get_async(backend_argmax, t_argmax, logits_argmax_prob_buf.data(), ids_bytes, probs_bytes);
        if (prof) {
            dflash_capture->profile_reduced_logits_probs_us += ggml_time_us() - t_probs_start_us;
            const int64_t elapsed_us = ggml_time_us() - t_start_us;
            dflash_capture->profile_reduced_logits_us += elapsed_us;
            dflash_capture->profile_output_extract_us += elapsed_us;
            dflash_capture->profile_reduced_logits_bytes += ids_bytes + probs_bytes;
        }
        logits_argmax_count = n_out;
        logits_argmax_k = K;
    }
}

void llama_context::dflash_post_decode_profile(int64_t n_vocab_val) {
    if (dflash_capture && dflash_capture->profile) {
        dflash_profile_log(*dflash_capture, __func__, (int32_t) n_vocab_val);
    }
}

// ---------------------------------------------------------------------------
// hidden state getters (C-API)
// ---------------------------------------------------------------------------

float * llama_get_layer_hidden(llama_context * ctx, int slot) {
    if (!ctx) return nullptr;
    return ctx->get_layer_hidden(slot);
}

int64_t llama_get_layer_hidden_n_tokens(llama_context * ctx, int slot) {
    if (!ctx) return 0;
    return ctx->get_layer_hidden_n_tokens(slot);
}

int64_t llama_get_layer_hidden_n_embd(llama_context * ctx, int slot) {
    if (!ctx) return 0;
    return ctx->get_layer_hidden_n_embd(slot);
}

int32_t llama_get_n_layer_hiddens(llama_context * ctx) {
    if (!ctx) return 0;
    return ctx->get_n_layer_hiddens();
}

// ---------------------------------------------------------------------------
// argmax getters (C-API)
// ---------------------------------------------------------------------------

const int32_t * llama_get_logits_argmax(llama_context * ctx) {
    if (!ctx || ctx->get_logits_argmax_count() <= 0) return nullptr;
    return ctx->get_logits_argmax_data();
}

const int32_t * llama_get_logits_argmax_ith(llama_context * ctx, int32_t i) {
    if (!ctx || i < 0) return nullptr;
    const int K = ctx->get_logits_argmax_k_val();
    if (K <= 0) return nullptr;
    const int32_t * data = ctx->get_logits_argmax_data();
    if (!data) return nullptr;
    return data + (size_t) i * K;
}

int32_t llama_get_logits_argmax_n(llama_context * ctx) {
    if (!ctx) return 0;
    return ctx->get_logits_argmax_count();
}

int32_t llama_get_logits_argmax_k(llama_context * ctx) {
    if (!ctx) return 0;
    return ctx->get_logits_argmax_k_val();
}

const float * llama_get_logits_argmax_probs(llama_context * ctx) {
    if (!ctx || ctx->get_logits_argmax_count() <= 0) return nullptr;
    return ctx->get_logits_argmax_probs_data();
}

const float * llama_get_logits_argmax_probs_ith(llama_context * ctx, int32_t i) {
    if (!ctx || i < 0) return nullptr;
    const int K = ctx->get_logits_argmax_k_val();
    if (K <= 0) return nullptr;
    const float * data = ctx->get_logits_argmax_probs_data();
    if (!data) return nullptr;
    return data + (size_t) i * K;
}
