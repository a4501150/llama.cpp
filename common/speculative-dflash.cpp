// DFlash speculative decoding implementation
// Adapted from BeeLlama.cpp's common_speculative_state_dflash to
// upstream's common_speculative_impl interface.
//
// This file is #included from speculative.cpp after the base class definition.
//
// TODO: Port from beellama.cpp/common/speculative.cpp:
//
// 1. common_speculative_impl_dflash struct (~1,600 lines):
//    Source: beellama lines 1653-3263 (common_speculative_state_dflash)
//    Adapt base class from common_speculative_state -> common_speculative_impl
//    Key methods: begin, draft, accept, plus DFlash-specific:
//    - ring buffer management (CPU + optional GPU mirror)
//    - cross-attention context preparation
//    - batch draft for multi-slot operation
//
// 2. Static helpers (~120 lines):
//    - common_dflash_ring_write_plan() (lines 140-155)
//
// 3. Public dispatch functions:
//    - common_speculative_discard_dflash_state
//    - common_speculative_ring_state_save / _load / _size
//    - common_speculative_dflash_ring_stats
//    - common_speculative_set_prefill_capture_enabled
//    - common_speculative_note_prefill_suffix_scheduled
//    - common_speculative_flush_prefill
//    - common_speculative_set_seq_id
//    - common_speculative_draft_batch
//    - common_speculative_update_logits_by_indices
//
// Source: beellama.cpp/common/speculative.cpp (363 diff hunks)
