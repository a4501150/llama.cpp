// CopySpec / Suffix / Recycle speculative decoding implementations
// Ported from BeeLlama.cpp, adapted to upstream common_speculative_impl interface
// This file is #included from speculative.cpp after the base class definition

// ============================================================================
// Suffix-tree based speculative decoding
// ============================================================================

struct common_speculative_impl_suffix : public common_speculative_impl {
    SuffixTree tree;
    static constexpr int SEQ_ID = 1;

    int32_t max_depth;
    int32_t n_draft_max;
    float   spec_factor;
    float   spec_offset;
    float   min_prob;

    size_t tree_size = 0;

    common_speculative_impl_suffix(
            common_speculative_type type,
            uint32_t n_seq,
            int32_t max_depth,
            int32_t n_draft_max,
            float   spec_factor,
            float   spec_offset,
            float   min_prob)
        : common_speculative_impl(type, n_seq)
        , tree(max_depth)
        , max_depth(max_depth)
        , n_draft_max(n_draft_max)
        , spec_factor(spec_factor)
        , spec_offset(spec_offset)
        , min_prob(min_prob)
    {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        tree = SuffixTree(max_depth);
        tree_size = 0;
        if (!prompt.empty()) {
            tree.extend(SEQ_ID, prompt.data(), prompt.size());
            tree_size = prompt.size();
        }
    }

    bool process(const llama_batch & /*batch*/) override { return false; }
    bool need_embd() const override { return false; }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (uint32_t s = 0; s < n_seq; ++s) {
            auto & dp = dparams[s];
            if (!dp.drafting || !dp.prompt || !dp.result) continue;

            const auto & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;

            // feed new tokens incrementally
            if (tree_size < prompt_tgt.size() + 1) {
                for (size_t j = tree_size; j < prompt_tgt.size(); ++j) {
                    tree.append(SEQ_ID, prompt_tgt[j]);
                }
                tree.append(SEQ_ID, id_last);
                tree_size = prompt_tgt.size() + 1;
            }

            // build context for pattern matching
            std::vector<int32_t> context;
            context.reserve(prompt_tgt.size() + 1);
            for (const auto & t : prompt_tgt) { context.push_back(t); }
            context.push_back(id_last);
            if (context.size() < 2) continue;

            SuffixDraft sd = tree.speculate(
                context.data(), context.size(),
                n_draft_max, spec_factor, spec_offset, min_prob, false);

            for (const auto & tok : sd.token_ids) {
                dp.result->push_back(tok);
            }
            if (!dp.result->empty()) {
                dp.drafting = false;
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {}
};

// ============================================================================
// CopySpec: model-free speculation via rolling-hash suffix matching
// ============================================================================

struct common_speculative_impl_copyspec : public common_speculative_impl {
    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    int32_t gamma;

    std::unordered_multimap<uint64_t, int32_t> index;
    llama_tokens prompt_tokens;
    int32_t original_prompt_size = 0;

    common_speculative_impl_copyspec(common_speculative_type type, uint32_t n_seq, int32_t gamma)
        : common_speculative_impl(type, n_seq), gamma(gamma) {}

    static uint64_t hash_window(const llama_token * tokens, int32_t len) {
        uint64_t h = FNV_OFFSET;
        for (int32_t i = 0; i < len; i++) {
            h ^= (uint64_t)(uint32_t)tokens[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        index.clear();
        prompt_tokens = prompt;
        original_prompt_size = (int32_t)prompt.size();
        if ((int32_t)prompt.size() <= gamma) return;
        for (int32_t i = 0; i <= (int32_t)prompt.size() - gamma; i++) {
            index.emplace(hash_window(prompt.data() + i, gamma), i + gamma);
        }
    }

    bool process(const llama_batch & /*batch*/) override { return false; }
    bool need_embd() const override { return false; }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (uint32_t s = 0; s < n_seq; ++s) {
            auto & dp = dparams[s];
            if (!dp.drafting || !dp.prompt || !dp.result) continue;

            const auto & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            const int32_t n_max = dp.n_max > 0 ? dp.n_max : 16;

            // need at least gamma tokens of context
            if ((int32_t)(prompt_tgt.size() + 1) < gamma) continue;

            // build context tail for hashing
            std::vector<llama_token> ctx_tail;
            ctx_tail.reserve(gamma);
            int32_t ctx_total = (int32_t)prompt_tgt.size() + 1;
            int32_t start = ctx_total - gamma;
            for (int32_t i = start; i < (int32_t)prompt_tgt.size(); i++) {
                ctx_tail.push_back(prompt_tgt[i]);
            }
            ctx_tail.push_back(id_last);

            uint64_t h = hash_window(ctx_tail.data(), gamma);
            auto range = index.equal_range(h);

            int32_t best_pos = -1;
            int32_t best_avail = 0;

            for (auto it = range.first; it != range.second; ++it) {
                int32_t pos = it->second;
                if (pos < gamma || pos > (int32_t)prompt_tokens.size()) continue;

                // verify hash collision
                bool match = true;
                for (int32_t j = 0; j < gamma; j++) {
                    if (prompt_tokens[pos - gamma + j] != ctx_tail[j]) {
                        match = false;
                        break;
                    }
                }
                if (!match) continue;

                int32_t avail = (int32_t)prompt_tokens.size() - pos;
                if (avail > best_avail) {
                    best_avail = avail;
                    best_pos = pos;
                }
            }

            if (best_pos >= 0 && best_avail > 0) {
                int32_t n_copy = std::min(best_avail, n_max);
                for (int32_t i = 0; i < n_copy; i++) {
                    dp.result->push_back(prompt_tokens[best_pos + i]);
                }
                dp.drafting = false;
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // update_logits equivalent: incrementally extend index
        // This is called per-accept but actual token feeding happens via prompt updates
    }
};

// ============================================================================
// Recycle: bigram/logit adjacency speculation
// ============================================================================

struct common_speculative_impl_recycle : public common_speculative_impl {
    int32_t k;

    std::unordered_map<llama_token, std::vector<std::pair<float, llama_token>>> adj;
    size_t n_fed = 0;

    common_speculative_impl_recycle(common_speculative_type type, uint32_t n_seq, int32_t k)
        : common_speculative_impl(type, n_seq), k(k) {}

    void add_bigram(llama_token a, llama_token b) {
        auto & vec = adj[a];
        for (auto & p : vec) {
            if (p.second == b) {
                p.first += 1.0f;
                // bubble up
                for (size_t i = &p - vec.data(); i > 0; --i) {
                    if (vec[i].first > vec[i-1].first) {
                        std::swap(vec[i], vec[i-1]);
                    } else break;
                }
                return;
            }
        }
        if ((int32_t)vec.size() < k) {
            vec.emplace_back(1.0f, b);
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        adj.clear();
        n_fed = 0;
        for (size_t i = 0; i + 1 < prompt.size(); i++) {
            add_bigram(prompt[i], prompt[i+1]);
        }
        n_fed = prompt.size();
    }

    bool process(const llama_batch & /*batch*/) override { return false; }
    bool need_embd() const override { return false; }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (uint32_t s = 0; s < n_seq; ++s) {
            auto & dp = dparams[s];
            if (!dp.drafting || !dp.prompt || !dp.result) continue;

            const auto & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            const int32_t n_max = dp.n_max > 0 ? dp.n_max : 16;

            // feed any new bigrams
            if (n_fed < prompt_tgt.size() + 1) {
                for (size_t j = (n_fed > 0 ? n_fed - 1 : 0); j < prompt_tgt.size(); ++j) {
                    llama_token a = (j == 0 && n_fed == 0) ? prompt_tgt[0] : prompt_tgt[j];
                    llama_token b = (j + 1 < prompt_tgt.size()) ? prompt_tgt[j+1] : id_last;
                    if (j + 1 <= prompt_tgt.size()) {
                        add_bigram(a, b);
                    }
                }
                n_fed = prompt_tgt.size() + 1;
            }

            // greedy walk
            llama_token cur = id_last;
            for (int32_t i = 0; i < n_max; i++) {
                auto it = adj.find(cur);
                if (it == adj.end() || it->second.empty()) break;
                cur = it->second[0].second;
                dp.result->push_back(cur);
            }

            if (!dp.result->empty()) {
                dp.drafting = false;
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {}
};
