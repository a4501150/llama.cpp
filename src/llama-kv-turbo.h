#pragma once

#include "ggml.h"

#include <cstdint>

// TurboQuant KV cache type helpers

static inline bool ggml_type_is_turbo_tcq(ggml_type t) {
    return t == GGML_TYPE_TURBO3_TCQ || t == GGML_TYPE_TURBO2_TCQ;
}

static inline bool ggml_type_is_turbo_kv(ggml_type t) {
    return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0
        || t == GGML_TYPE_TURBO3_TCQ || t == GGML_TYPE_TURBO2_TCQ;
}

static inline uint32_t turbo_padded_head_dim(uint32_t n_embd, ggml_type type_k, ggml_type type_v) {
    if (!ggml_type_is_turbo_kv(type_k) && !ggml_type_is_turbo_kv(type_v)) {
        return n_embd;
    }
    // TurboQuant rotation operates on 128-element groups
    return ((n_embd + 127) / 128) * 128;
}
