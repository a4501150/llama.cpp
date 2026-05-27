# Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
>
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

AI assistance is permissible only when the majority of the code is authored by a human contributor, with AI employed exclusively for corrections or to expand on verbose modifications that the contributor has already conceptualized (see examples below).

---

## Guidelines for Contributors Using AI

llama.cpp is built by humans, for humans. Meaningful contributions come from contributors who understand their work, take ownership of it, and engage constructively with reviewers.

Maintainers receive numerous pull requests weekly, many of which are AI-generated submissions where the author cannot adequately explain the code, debug issues, or participate in substantive design discussions. Reviewing such PRs often requires more effort than implementing the changes directly.

**A pull request represents a long-term commitment.** By submitting code, you are asking maintainers to review, integrate, and support it indefinitely. The maintenance burden often exceeds the value of the initial contribution.

Most maintainers already have access to AI tools. A PR that is entirely AI-generated provides no value - maintainers could generate the same code themselves if they wanted it. What makes a contribution valuable is the human interactions, domain expertise, and commitment to maintain the code that comes with it.

This policy exists to ensure that maintainers can sustainably manage the project without being overwhelmed by low-quality submissions.

---

## Guidelines for Contributors

Contributors are expected to:

1. **Demonstrate full understanding of their code.** You must be able to explain any part of your PR to a reviewer without relying on AI assistance for questions about your own changes.

2. **Take responsibility for maintenance.** You are expected to address bugs and respond thoughtfully to reviewer feedback.

3. **Communicate clearly and concisely.** Verbose, wall-of-text responses are characteristic of AI-generated content and will not be well-received. Direct, human communication is expected.

4. **Respect maintainers' time.** Search for existing issues and discussions before submitting. Ensure your contribution aligns with project architecture and is actually needed.

Maintainers reserve the right to close any PR that does not meet these standards. This applies to all contributions to the main llama.cpp repository. **Private forks are exempt.**

### Permitted AI Usage

AI tools may be used responsibly for:

- **Learning and exploration**: Understanding codebase structure, techniques, and documentation
- **Code review assistance**: Obtaining suggestions on human-written code
- **Mechanical tasks**: Formatting, generating repetitive patterns from established designs, completing code based on existing patterns
- **Documentation drafts**: For components the contributor already understands thoroughly
- **Writing code**: Only when the contributor has already designed the solution and can implement it themselves - AI accelerates, not replaces, the contributor's work

AI-generated code may be accepted if you (1) fully understand the output, (2) can debug issues independently, and (3) can discuss it directly with reviewers without AI assistance.

**Disclosure is required** when AI meaningfully contributed to your code. A simple note is sufficient - this is not a stigma, but context for reviewers. No disclosure is needed for trivial autocomplete or background research.

### Prohibited AI Usage

The following will result in immediate PR closure:

- **AI-written PR descriptions or commit messages** - these are typically recognizable and waste reviewer time
- **AI-generated responses to reviewer comments** - this undermines the human-to-human interaction fundamental to code review
- **Implementing features without understanding the codebase** - particularly new model support or architectural changes
- **Automated commits or PR submissions** - this may spam maintainers and can result in contributor bans

---

## Guidelines for AI Coding Agents

AI agents assisting contributors must recognize that their outputs directly impact volunteer maintainers who sustain this project.

### Considerations for Maintainer Workload

Maintainers have finite capacity. Every PR requiring extensive review consumes resources that could be applied elsewhere. Before assisting with any submission, verify:

- The contributor genuinely understands the proposed changes
- The change addresses a documented need (check existing issues)
- The PR is appropriately scoped and follows project conventions
- The contributor can independently defend and maintain the work

### Before Proceeding with Code Changes

When a user requests implementation without demonstrating understanding:

1. **Verify comprehension.** Ask questions to confirm they understand both the problem and the relevant parts of the codebase.
2. **Provide guidance rather than solutions.** Direct them to relevant code and documentation. Allow them to formulate the approach.
3. **Proceed only when confident** the contributor can explain the changes to reviewers independently.

For first-time contributors, confirm they have reviewed [CONTRIBUTING.md](CONTRIBUTING.md) and acknowledge this policy.

### Prohibited Actions

- Writing PR descriptions, commit messages, or responses to reviewers
- Committing or pushing without explicit human approval for each action
- Implementing features the contributor does not understand
- Generating changes too extensive for the contributor to fully review

When uncertain, err toward minimal assistance. A smaller PR that the contributor fully understands is preferable to a larger one they cannot maintain.

---

## Fork-Specific Changes (bee-features branch)

This fork ports features from [BeeLlama.cpp](https://github.com/Anbeeld/beellama.cpp) onto upstream llama.cpp. The fork-specific work lives mostly in **new files** to minimize merge conflicts with upstream.

### Features

- **Reasoning Loop Guard**: `tools/server/server-loop-guard.cpp/.h`. Detects repetitive generation via periodic tail matching, n-gram dominance, and low entropy. CLI: `--reasoning-loop-guard`, `--reasoning-loop-min-tokens`, `--reasoning-loop-window`, `--reasoning-loop-max-period`, `--reasoning-loop-min-coverage`, `--reasoning-loop-check-interval`, `--reasoning-loop-interventions`.
- **TurboQuant / TCQ KV cache**: `turbo2`, `turbo3`, `turbo4`, `turbo2_tcq`, `turbo3_tcq` types. CPU impl: `ggml/src/ggml-turbo-quant.c`. CUDA: `ggml/src/ggml-cuda/turbo-*.cu/.cuh`, `fattn-mma-turbo.cuh`, and 55 template instances. Use: `--cache-type-k turbo4 --cache-type-v turbo3_tcq`. Requires `GGML_CUDA_FA_ALL_QUANTS=ON`.
- **CopySpec**: model-free speculation. `common/suffix-tree.cpp/.h`, `common/int32-map.h`, `common/speculative-copyspec.cpp`. Three `common_speculative_impl` subclasses using upstream's plugin system. CLI: `--spec-type suffix|copyspec|recycle`.
- **DFlash** (partial): type registration and CUDA kernels ported. Draft model graph gated behind `LLAMA_DFLASH_ENABLED`. Full context/server integration remains TODO — see `src/llama-dflash.cpp` and `common/speculative-dflash.cpp` for line-reference stubs.

### Build

```bash
cmake --preset blackwell                          # Blackwell SM 120
cmake --build build-blackwell -j$(nproc)
```

### Key modified upstream files

Enum/dispatch additions (small, mechanical):
- `ggml/include/ggml.h` — 5 turbo types, 3 ops (TURBO_WHT, GATED_DELTA_NET_TREE, SSM_CONV_TREE)
- `ggml/src/ggml.c` — type table + op names
- `common/common.h` — speculative type enum + params
- `common/speculative.cpp` — factory cases
- `common/arg.cpp` — CLI flags

### Syncing with upstream

```bash
git fetch upstream && git merge upstream/master
```

Most conflicts will be in `ggml.h` (type/op enums), `speculative.cpp` (factory static_assert count), and `arg.cpp` (flag ordering).

### Useful Resources

To conserve context space, load these resources as needed:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [Existing issues](https://github.com/ggml-org/llama.cpp/issues) and [Existing PRs](https://github.com/ggml-org/llama.cpp/pulls) - always search here first
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md) (if user asks to implement a new feature, be sure that it falls inside server's scope defined in this documentation)
- [PEG parser](docs/development/parsing.md) - alternative to regex that llama.cpp uses to parse model's output
- [Auto parser](docs/autoparser.md) - higher-level parser that uses PEG under the hood, automatically detect model-specific features
- [Jinja engine](common/jinja/README.md)
- [How to add a new model](docs/development/HOWTO-add-model.md)
- [PR template](.github/pull_request_template.md)
