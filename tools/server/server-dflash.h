#pragma once

// DFlash server-side helpers
// Extracted from beellama.cpp/tools/server/server-context.cpp
//
// TODO: Port the following from beellama.cpp/tools/server/server-context.cpp:
//
// 1. Static helper functions (~600 lines, lines 48-581 in beellama):
//    - dflash_server_profile_enabled / dflash_server_crash_trace_enabled
//    - struct dflash_reduced_verify_plan
//    - dflash_reduced_sampler_chain_supported
//    - dflash_select_reduced_verify_plan
//    - dflash_sample_reduced_verify
//    - dflash_slot_in_view / dflash_view_has_unexpected_prompt_logits
//    - dflash_log_reduced_verify_decision / dflash_batch_view_is_reduced_verify
//    - dflash_select_batch_reduced_verify_plan
//    - dflash_flat_effective_draft_max
//
// 2. Shim functions for update_slots() (~6 call sites):
//    - dflash_slots_init(...)
//    - dflash_prepare_batch(...)
//    - dflash_post_draft(...)
//    - dflash_configure_verify(...)
//    - dflash_post_verify(...)
//    - dflash_post_accept(...)
//
// 3. DFlash slot state struct:
//    - server_slot_dflash_state (base struct for server_slot to inherit)
//    - Fields: dflash_cycle_count, dflash_requested_total, dflash_produced_total,
//      dflash_accepted_total, dflash_accept_hist, etc.
//
// Source: beellama.cpp/tools/server/server-context.cpp (296 diff hunks, ~2,475 lines)
