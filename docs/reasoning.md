<a id="reasoning"></a>
# [To be reviewed] Reasoning

[Handbook contents](../README.md#table-of-contents) | [Previous: QA Systems](qa.md) | [Next: Recommendation Systems](recommendation-systems.md)

## Contents

- [Reasoning, search, supervision, and evaluation](#reasoning-overview)
- [Information seeking](#information-seeking)


<a id="reasoning-overview"></a>

- **Reasoning setup:** specify the problem representation, allowed tools, answer verifier, and compute budget. Multi-step prompting can improve some tasks; a persuasive written rationale is not independent proof of a correct answer or a faithful account of internal computation. [Chain-of-thought prompting](https://arxiv.org/abs/2201.11903)
- **Search and selection:** sample multiple candidates, decompose a problem, or search over partial solutions. A useful verifier can turn additional computation into better answer selection; a biased verifier can instead amplify a systematic error. Compare search to ordinary resampling at matched generated-token and tool budgets. [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
- **Supervision:** outcome rewards assess the final result; process supervision labels intermediate steps. Verifiable arithmetic, tests, or formal proof checking provide more direct feedback than a style preference, though each verifier has a limited specification. [Process supervision](https://arxiv.org/abs/2305.20050)
- **Example:** a generated program may pass visible tests while failing hidden edge cases. Keep evaluation tests out of the policy's feedback loop when claiming held-out generalization; otherwise the system is adapting to those tests.
- **Measurement:** report pass@1 and a specified sampling/selection protocol. With $`n`$ sampled candidates and $`c`$ successes, the common estimator is $`\widehat{\operatorname{pass@k}}=1-\binom{n-c}{k}/\binom{n}{k}`$ for $`n\geq k`$, using zero for impossible numerator combinations. Oracle pass@k is different from the success rate of a deployable selector that cannot see correctness labels. [Code-model evaluation](https://arxiv.org/abs/2107.03374)
- Sweep inference budgets, report latency and cost alongside correctness, and test transfer to new problem families. Additional tokens or a higher reward on the training distribution do not establish a general reasoning capability.

<a id="information-seeking"></a>

- [To be reviewed] Information seeking

    - Turn an information need into successive evidence-gathering actions: formulate a query, inspect sources, update the evidence state, and decide whether another query is useful. ReAct interleaves reasoning-oriented generation with environment actions and observations. [ReAct](https://arxiv.org/abs/2210.03629)
    - **Example:** answering a disputed release date may require the original announcement and a later correction. Track source dates and claim provenance, and stop when the available evidence resolves the question or a stated search budget is exhausted. Repeating equivalent searches is additional activity rather than additional evidence.

For math/coding/agent benchmark families, saturation, and the distinction between pass@k search opportunity and pass^k repeated-run reliability, see [Benchmarking & Evaluation](benchmarking-and-evaluation.md#eval-metrics).

For turning reasoning into tool actions, dependency-aware plans, and multi-agent coordination, see [Agents](agents.md#agent-planning). Runtime continuity and recursive improvement are covered in [RSI & Harness](rsi-and-harness.md).

[To be reviewed] For learning from a final verifier or intermediate step labels, see [credit assignment and GAE](large-language-models.md#rl-credit-assignment) and [outcome/process rewards and shaping](large-language-models.md#rl-reward-design). A process score supplies another measurement; it does not automatically establish causal token attribution or immunity to [reward hacking](large-language-models.md#rl-reward-hacking).

[To be reviewed] For the difference between a generated rationale and evidence about internal computation, see [LLM Interpretability](llm-interpretability.md#interp-scope). [Patching](llm-interpretability.md#interp-patching) and [circuit analysis](llm-interpretability.md#interp-circuits) test specific internal mechanisms rather than inferring them from a persuasive reasoning trace.
