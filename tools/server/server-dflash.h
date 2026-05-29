#pragma once

// DFlash server-side helpers
// Static functions and types used by the server's update_slots() for DFlash
// speculative decoding integration. Included from server-context.cpp.

#include "sampling.h"
#include "speculative.h"
#include "../../src/dflash-profile.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// env-based feature flags
// ---------------------------------------------------------------------------

static bool dflash_server_profile_enabled(uint32_t flags) {
    return dflash_profile_enabled(flags);
}

static bool dflash_server_crash_trace_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_CRASH_TRACE");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool dflash_verify_padding_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("GGML_DFLASH_VERIFY_PAD");
        return env && env[0] != '\0' && strcmp(env, "0") != 0;
    }();
    return enabled;
}

// ---------------------------------------------------------------------------
// reduced verify plan
// ---------------------------------------------------------------------------

struct dflash_reduced_verify_plan {
    bool enabled = false;
    int top_k = 1;
    const char * reason = "disabled";
};

static bool dflash_reduced_sampler_chain_supported(
        const common_params_sampling & sampling,
        bool                           stochastic,
        const char                  ** reason) {
    auto reject = [&](const char * why) {
        if (reason) *reason = why;
        return false;
    };

    bool saw_top_k = false;
    for (const auto sampler : sampling.samplers) {
        switch (sampler) {
            case COMMON_SAMPLER_TYPE_NONE:
            case COMMON_SAMPLER_TYPE_TEMPERATURE:
                break;
            case COMMON_SAMPLER_TYPE_PENALTIES:
                if (!(sampling.penalty_repeat == 1.0f && sampling.penalty_freq == 0.0f && sampling.penalty_present == 0.0f))
                    return reject("penalties");
                break;
            case COMMON_SAMPLER_TYPE_DRY:
                if (sampling.dry_multiplier != 0.0f && sampling.dry_penalty_last_n != 0)
                    return reject("dry");
                break;
            case COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                if (sampling.top_n_sigma >= 0.0f) return reject("top-n-sigma");
                break;
            case COMMON_SAMPLER_TYPE_TOP_K:
                saw_top_k = true;
                break;
            case COMMON_SAMPLER_TYPE_TYPICAL_P:
                if (sampling.typ_p < 1.0f) {
                    if (!stochastic) return reject("typical");
                    if (!saw_top_k)  return reject("sampler-order");
                }
                break;
            case COMMON_SAMPLER_TYPE_TOP_P:
                if (stochastic && sampling.top_p < 1.0f && !saw_top_k) return reject("sampler-order");
                break;
            case COMMON_SAMPLER_TYPE_MIN_P:
                if (stochastic && sampling.min_p > 0.0f && !saw_top_k) return reject("sampler-order");
                break;
            case COMMON_SAMPLER_TYPE_XTC:
                if (sampling.xtc_probability > 0.0f) return reject("xtc");
                break;
            case COMMON_SAMPLER_TYPE_INFILL:
                return reject("infill");
            case COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                return reject("adaptive");
        }
    }
    if (stochastic && !saw_top_k) return reject("top-k-sampler");
    return true;
}

static dflash_reduced_verify_plan dflash_select_reduced_verify_plan(
        const common_params_sampling    & sampling,
        const common_params_speculative & speculative,
        bool                              use_rejection,
        bool                              has_tree) {
    dflash_reduced_verify_plan plan;

    if (has_tree)       { plan.reason = "tree";       return plan; }
    if (use_rejection)  { plan.reason = "rejection";  return plan; }
    if (sampling.n_probs > 0) { plan.reason = "prob-reporting"; return plan; }
    if (!sampling.grammar.empty() || sampling.grammar_lazy || !sampling.grammar_triggers.empty()) {
        plan.reason = "grammar"; return plan;
    }
    if (sampling.has_logit_bias() || sampling.ignore_eos) { plan.reason = "logit-bias"; return plan; }
    if (!(sampling.penalty_repeat == 1.0f && sampling.penalty_freq == 0.0f && sampling.penalty_present == 0.0f)) {
        plan.reason = "penalties"; return plan;
    }
    if (sampling.dry_multiplier != 0.0f && sampling.dry_penalty_last_n != 0) { plan.reason = "dry"; return plan; }
    if (sampling.top_n_sigma >= 0.0f) { plan.reason = "top-n-sigma"; return plan; }
    if (sampling.xtc_probability > 0.0f) { plan.reason = "xtc"; return plan; }
    if (sampling.mirostat != 0) { plan.reason = "mirostat"; return plan; }
    if (sampling.dynatemp_range > 0.0f) { plan.reason = "dynamic-temp"; return plan; }
    if (sampling.reasoning_budget_tokens >= 0) { plan.reason = "finite-reasoning-budget"; return plan; }

    bool is_dflash = false;
    for (auto t : speculative.types) {
        if (t == COMMON_SPECULATIVE_TYPE_DFLASH) { is_dflash = true; break; }
    }
    if (!is_dflash) { plan.reason = "not-dflash"; return plan; }

    const bool stochastic = sampling.temp > 0.0f;
    const char * sampler_reason = nullptr;
    if (!dflash_reduced_sampler_chain_supported(sampling, stochastic, &sampler_reason)) {
        plan.reason = sampler_reason;
        return plan;
    }

    if (sampling.temp <= 0.0f) {
        plan.enabled = true;
        plan.top_k = 1;
        plan.reason = "greedy";
        return plan;
    }
    if (sampling.top_k <= 0)  { plan.reason = "unbounded-top-k"; return plan; }
    if (sampling.top_k > 64)  { plan.reason = "top-k-too-large"; return plan; }

    plan.enabled = true;
    plan.top_k = std::max(1, sampling.top_k);
    plan.reason = "top-k";
    return plan;
}

// ---------------------------------------------------------------------------
// DFlash effective draft max (block_size - 1)
// ---------------------------------------------------------------------------

static int dflash_flat_effective_draft_max(const llama_model * model_dft, int n_draft_max) {
    if (!model_dft) return n_draft_max;
    const int block_size = llama_model_dflash_block_size(model_dft);
    if (block_size <= 1) return n_draft_max;
    return std::min(n_draft_max, block_size - 1);
}

// ---------------------------------------------------------------------------
// reduced verify sampling
// ---------------------------------------------------------------------------

static std::vector<llama_token> dflash_sample_reduced_verify(
        struct common_sampler * smpl,
        struct llama_context  * ctx,
        const std::vector<int> & idxs,
        const llama_tokens    & draft,
        int                     top_k) {
    if ((int) idxs.size() != (int) draft.size() + 1 || top_k <= 0) {
        return {};
    }
    if (llama_get_logits_argmax_k(ctx) != top_k ||
            llama_get_logits_argmax_n(ctx) < (int32_t) idxs.size()) {
        return {};
    }

    std::vector<llama_token> candidate_ids((size_t) idxs.size() * (size_t) top_k);
    std::vector<float> candidate_logits((size_t) idxs.size() * (size_t) top_k);

    for (size_t row = 0; row < idxs.size(); ++row) {
        const int32_t * ids = llama_get_logits_argmax_ith(ctx, idxs[row]);
        const float * logits = llama_get_logits_argmax_probs_ith(ctx, idxs[row]);
        if (!ids || !logits) {
            return {};
        }
        for (int k = 0; k < top_k; ++k) {
            candidate_ids[row * (size_t) top_k + (size_t) k] = (llama_token) ids[k];
            candidate_logits[row * (size_t) top_k + (size_t) k] = logits[k];
        }
    }

    return common_sampler_sample_reduced_and_accept_n(
        smpl, candidate_ids.data(), candidate_logits.data(), (int32_t) idxs.size(), top_k, draft);
}

// ---------------------------------------------------------------------------
// reduced verify logging
// ---------------------------------------------------------------------------

static void dflash_log_reduced_verify_decision(
        bool        graph_enabled,
        bool        view_enabled,
        int32_t     view_start,
        int32_t     n_tokens,
        int         top_k,
        const char * reason) {
    if (!dflash_server_profile_enabled(DFLASH_PROFILE_VERIFY)) {
        return;
    }
    SRV_INF("dflash reduced verifier decision: graph_enabled=%d view_enabled=%d view_start=%d n_tokens=%d top_k=%d reason=%s\n",
            graph_enabled ? 1 : 0, view_enabled ? 1 : 0, view_start, n_tokens, top_k, reason ? reason : "unknown");
}

// ---------------------------------------------------------------------------
// DFlash slot statistics
// ---------------------------------------------------------------------------

struct server_slot_dflash_stats {
    int64_t dflash_cycle_count = 0;
    int64_t dflash_requested_total = 0;
    int64_t dflash_produced_total = 0;
    int64_t dflash_accepted_total = 0;
    std::array<int64_t, 5> dflash_accept_hist = {};

    void reset() {
        dflash_cycle_count = 0;
        dflash_requested_total = 0;
        dflash_produced_total = 0;
        dflash_accepted_total = 0;
        dflash_accept_hist = {};
    }

    static int accept_bin(int n_accepted) {
        if (n_accepted <= 0) return 0;
        if (n_accepted <= 2) return 1;
        if (n_accepted <= 4) return 2;
        if (n_accepted <= 6) return 3;
        return 4;
    }

    void note_cycle(int n_draft, int n_produced, int n_accepted) {
        dflash_cycle_count++;
        dflash_requested_total += n_draft;
        dflash_produced_total += n_produced;
        dflash_accepted_total += n_accepted;
        dflash_accept_hist[accept_bin(n_accepted)]++;
    }
};
