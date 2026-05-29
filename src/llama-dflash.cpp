// DFlash: cross-attention speculative decoding
// Split translation unit for llama_context DFlash member definitions.
//
// This file contains all DFlash-specific methods that were originally
// interleaved in llama-context.cpp in the BeeLlama fork.
//
// TODO: Port the following from beellama.cpp/src/llama-context.cpp:
//
// 1. DFlash method implementations (~2,500 lines):
//    - llama_context::set_dflash_capture / set_dflash_capture_active / set_dflash_gpu_capture
//    - llama_context::set_dflash_sample_temp / set_dflash_topk / set_dflash_verify_logits
//    - llama_context::dflash_reset_hidden_capture / dflash_ensure_recurrent_setup
//    - llama_context::dflash_wait_for_gpu_capture_stream
//    - llama_context::dflash_prefill_capture_begin / _end / _info
//    - llama_context::set_active_dflash_slot
//    - llama_context::dflash_rollback / dflash_prepare_branch
//    - llama_context::dflash_kv_cache_init / _reset / _prepare / _update / _update_gpu
//    - llama_context::set_tape_recording / allocate_tape_gpu / allocate_prefill_gpu / init_cross_ring_gpu
//
// 2. C-API wrappers:
//    - llama_set_dflash_* / llama_dflash_* functions
//    - llama_get_logits_argmax / llama_get_logits_argmax_ith
//
// 3. Static helpers:
//    - dflash_read_tensor_to / dflash_read_tensor
//    - dflash_eval_callback
//    - dflash_profile_reset / dflash_profile_log
//    - dflash_log_decode_seq_state
//    - dflash_clear_prefill_cparams
//
// 4. Decode loop shim functions:
//    - dflash_prepare_ubatch() — called before process_ubatch() in the decode loop
//    - dflash_post_ubatch()   — called after process_ubatch() in the decode loop
//
// Source: beellama.cpp/src/llama-context.cpp (113 diff hunks, ~4,580 lines)

#include "llama-dflash.h"
// #include "llama-context.h"  // needed for llama_context member definitions
