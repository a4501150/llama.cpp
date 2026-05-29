#pragma once

// DFlash: cross-attention speculative decoding
// This header provides forward declarations and struct definitions for DFlash.
// Implementation lives in llama-dflash.cpp (split TU for llama_context methods).

#include "ggml.h"
#include "llama.h"

#include <cstdint>
#include <vector>
#include <string>

// DFlash constants (also declared in llama.h)
#ifndef LLAMA_DFLASH_MAX_SLOTS
#define LLAMA_DFLASH_MAX_SLOTS         8
#define LLAMA_DFLASH_MAX_VERIFY_TOKENS 25
#define LLAMA_DFLASH_PER_SLOT_CTX      512
#endif

// Forward declarations for DFlash internal types
struct dflash_layer_hidden_buf;
struct dflash_tape_layer;
struct dflash_tape_gpu;
struct dflash_hidden_gpu;
struct dflash_prefill_capture_plan;
struct dflash_capture_data;
struct dflash_kv_cache_data;

// DFlash ring buffer statistics (returned by speculative API)
struct common_dflash_ring_stats {
    int32_t ring_write_pos = 0;
    int32_t ring_capacity  = 0;
    int32_t ring_used      = 0;
};

// TODO: Full struct definitions should be ported from beellama.cpp/src/llama-context.h
// The following structs need complete definitions:
// - dflash_layer_hidden_buf  (per-layer hidden state buffer)
// - dflash_tape_layer        (per-layer tape recording state)
// - dflash_tape_gpu          (GPU tape state per slot)
// - dflash_hidden_gpu        (GPU hidden state capture per slot)
// - dflash_prefill_capture_plan  (prefill capture scheduling)
// - dflash_capture_data      (main capture state container)
// - dflash_kv_cache_data     (DFlash-specific KV cache state)
//
// Source: /home/jinyang/src/beellama.cpp/src/llama-context.h lines 77-400
