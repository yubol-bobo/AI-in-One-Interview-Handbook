# Large Language Models

[Handbook contents](../README.md#table-of-contents) | [Previous: NLP (Pre-LLM Era)](nlp.md) | [Next: Agents](agents.md)

## Contents

- [LLM Foundations](#llm-basic)
    - [Architecture and Visibility](#architecture)
    - [Representations: Token States and Retrieval Vectors](#embedding)
    - [Tokenization](#tokenization)
- [Training Overview: Three Stages](#stages)
- [Pre-training](#pre-training)
    - [Objective Families and Loss](#to-be-reviewed-objective-families-and-loss)
    - [Causal Next-token Loss Example](#causal-loss-example)
    - [Corpus Preparation and Mixture Design](#to-be-reviewed-corpus-preparation-and-mixture-design)
    - [Scaling, Budget, and Recovery](#to-be-reviewed-scaling-budget-and-recovery)
- [Mid-training](#mid-training)
    - [Retention, Replay, and Behavior Restoration](#to-be-reviewed-retention-replay-and-behavior-restoration)
- [Post-training](#post-training)
    - [SFT, RL, and OPD: Relationships and Differences](#sft-rl-opd-comparison)
    - [SFT and Instruction Tuning](#post-sft)
        - [Task and Provider Examples](#to-be-reviewed-task-and-provider-examples)
    - [RLHF, RLAIF, RLVR, and RLCD: A Side-by-Side Comparison](#rl-feedback-comparison)
        - [What Changes Mathematically, and What Can Be Shared?](#to-be-reviewed-what-changes-mathematically-and-what-can-be-shared)
        - [Same-task Example: A Code Assistant](#to-be-reviewed-same-task-example-a-code-assistant)
        - [Relationships, Hybrids, and Evaluation](#to-be-reviewed-relationships-hybrids-and-evaluation)
    - [Online RL: Objectives, PPO, and GRPO](#online-rl)
        - [PPO Update Details](#to-be-reviewed-ppo-update-details)
        - [DAPO: Clipping, Sampling, and Token Weighting](#rl-dapo)
        - [GSPO: Sequence Ratios and Sequence Clipping](#rl-gspo)
        - [GRPO, DAPO, and GSPO Comparison](#rl-method-comparison)
        - [Credit Assignment: Returns, Baselines, and GAE](#rl-credit-assignment)
        - [Reward Hacking: Proxy Gains Versus Intended Outcomes](#rl-reward-hacking)
        - [Outcome Rewards, Process Rewards, and Reward Shaping](#rl-reward-design)
        - [Exploration, Sparse Rewards, and Critic-Free Estimators](#rl-exploration)
        - [KL Control, PPO Clipping, Entropy, and Length](#rl-kl-control)
        - [On-Policy, Off-Policy, and Offline Learning](#rl-policy-data)
        - [Interview Discussion and Debugging](#rl-interview)
            - [A Practical RL Debugging Sequence](#rl-debugging)
    - [Preferences, Reward Models, and AI Feedback](#preference-optimization)
        - [Direct Preference Optimization (DPO)](#to-be-reviewed-direct-preference-optimization-dpo)
        - [Feedback Collection and Reward-model Evaluation](#to-be-reviewed-feedback-collection-and-reward-model-evaluation)
        - [RLAIF and Judge Bias](#to-be-reviewed-rlaif-and-judge-bias)
        - [Unresolved Terminology: KPO](#to-be-reviewed-unresolved-terminology-kpo)
    - [Distillation: Offline Traces and On-policy Distillation (OPD)](#distillation-and-opd)
        - [OPD Objectives and the Gradient Distinction](#to-be-reviewed-opd-objectives-and-the-gradient-distinction)
        - [OPD Implementation and Evaluation](#to-be-reviewed-opd-implementation-and-evaluation)
        - [OPSD: Self-Distillation with Privileged Solutions](#opd-opsd)
    - [Synthetic Data, Rejection Sampling, and Iterative Recipes](#to-be-reviewed-synthetic-data-rejection-sampling-and-iterative-recipes)
    - [Stage Gates and Reproducible Comparisons](#to-be-reviewed-stage-gates-and-reproducible-comparisons)
- [Shared Training Systems](#llm-engineering)
    - [Mixed Precision](#mixed-precision)
    - [Distributed Training](#distributed-training)
    - [Update Parameterization and PEFT](#fine-tuning)
    - [Quantization and Storage Precision](#llm-quantization)
    - [Training Toolchains](#to-be-reviewed-training-toolchains)
    - [Accumulation, Checkpointing, and Communication](#optimization-and-efficiency)
- [Inference and Serving](#llm-inference)
    - [Decoding Strategies](#llm-decoding)
    - [KV Cache Management and Kernels](#llm-cache-management)
    - [Speculative Decoding](#llm-speculative)
    - [Scheduling and Serving Performance](#llm-serving-systems)
- [Evaluation and Monitoring](#evaluation-and-monitoring)
- [Recent interesting papers](#recent-interesting-papers)
- [Related chapters](#related-chapters)


<a id="llm-basic"></a>
<a id="llm-concept"></a>
## [To be reviewed] LLM Foundations

An LLM combines an architecture, tokenizer, learned weights, training history, and a runtime policy. Distinguish the model's raw conditional distribution from a chat template, decoding rule, retrieval system, or external tool wrapper. The training chapters below follow pre-training, mid-training, and post-training; architecture and inference mechanisms have their own canonical explanations.

<a id="architecture"></a>
### [To be reviewed] Architecture and Visibility

The [Transformer chapter](transformer.md#transformer) explains encoder-only, encoder–decoder, decoder-only, and prefix-LM designs. Architecture family, attention visibility, and training objective are separate choices. A causal model predicts continuations, a prefix LM can expose its supplied prefix bidirectionally, and an encoder–decoder supplies source memory through cross-attention. BERT, RoBERTa, T5, XLNet, and GPT-style models are historical examples of these choices, not a ranking of current frontier models.

Decoder-only models offer a uniform next-token training/generation interface. They do not have a universal zero-shot advantage over encoder–decoder systems, and bidirectional attention does not inherently force an unusable low-rank matrix. Compare objectives, data, scale, compute, and deployment workload. The [cache discussion](transformer.md#tf-cache) covers both causal self-attention and source cross-attention reuse.

<a id="embedding"></a>
### [To be reviewed] Representations: Token States and Retrieval Vectors

A [token embedding](transformer.md#tf-embedding) is a learned input lookup; a contextual token state incorporates the permitted surrounding sequence. A sentence/document embedding additionally specifies pooling, padding masks, layer selection, normalization, and possibly query/document instructions. Raw `[CLS]` or mean-pooled causal states do not automatically have a useful retrieval geometry. Position encodings and their length-generalization limits are explained once in [position information](transformer.md#tf-position).

For paired queries and positive passages, a contrastive retrieval objective can minimize $`-\log\frac{\exp(s(q,d^+)/\tau)}{\sum_{d\in\mathcal{B}}\exp(s(q,d)/\tau)}`$. In-batch and mined hard negatives teach different distinctions; exclude known alternative positives where possible. The objective differs from ordinary next-token prediction. [Sentence-BERT](https://aclanthology.org/D19-1410/), [DPR](https://arxiv.org/abs/2004.04906)

**Example:** “treatment failed” and “treatment succeeded” share vocabulary but should not be interchangeable answers. Evaluate real retrieval with difficult negatives, and keep query/index encoders and preprocessing compatible. [Embedding model evaluation](rag.md#embedding-models) covers index versioning, task-specific choices, recall definitions, and operational trade-offs.

<a id="tokenization"></a>
### [To be reviewed] Tokenization
- [To be reviewed] Types

    - Tokenization defines the model's discrete alphabet and its mapping to text. BPE learns merge rules; WordPiece uses a subword vocabulary and segmentation procedure; unigram tokenization scores candidate segmentations under a learned token distribution. A tokenizer library and a tokenization algorithm are separate concepts.
    - **Example:** SentencePiece can implement BPE or a unigram model. “SentencePiece versus BPE” therefore does not specify two disjoint algorithms. Freeze normalization, whitespace handling, special-token IDs, and chat serialization with the tokenizer artifact. [SentencePiece](https://aclanthology.org/D18-2012/)

- [To be reviewed] Vocabulary choice implications

    - With vocabulary size $`V`$ and hidden width $`d`$, an embedding table has $`Vd`$ parameters; an untied output matrix adds another $`Vd`$. Increasing vocabulary can shorten sequences but raises embedding/softmax storage and changes rare-token estimation.
    - **Illustrative calculation:** at width 4,096, adding 10,000 vocabulary entries adds 40.96 million parameters to one table. Compare token fertility by language and domain, inference cost per user-visible character, and quality at a matched training budget; vocabulary size alone is not an efficiency metric.

- [To be reviewed] Tokenization impact on model performance (OOV handling, multilingual scenarios)

    - Unknown-word behavior depends on the base alphabet and fallback rules. Byte-level coverage can represent arbitrary byte sequences, but it does not ensure economical segmentation, semantic understanding, or easy copying of an unfamiliar identifier.
    - **Example:** one tokenizer might use 5 tokens for an identifier while another uses 20, changing context occupancy and generation cost. Compare multilingual and domain slices. Per-token perplexities from different tokenizers are not directly comparable; use a shared unit such as bits per byte when the text encoding and evaluation protocol support it.

<a id="stages"></a>
<a id="llm-training-lifecycle"></a>
<a id="stage-map"></a>
## [To be reviewed] Training Overview: Three Stages

The main progression is **pre-training → mid-training → post-training**. Boundaries are conventions: late data annealing or context extension may be called continued pre-training in one report and mid-training in another. Identify the **starting checkpoint, data distribution, objective, trainable parameters, and evaluation target** before assigning a name. A model can revisit an earlier kind of training and then restore its downstream behavior. [Qwen3 stages](https://arxiv.org/html/2505.09388v1), [adaptation/restoration case study](https://thinkingmachines.ai/blog/on-policy-distillation/)

| Stage | Primary question | Typical supervision | Stage-specific evidence |
| --- | --- | --- | --- |
| Pre-training | What broad representations and capabilities can be learned from the chosen corpus and compute budget? | Causal, masked, denoising, FIM, or auxiliary token-prediction targets | Domain/language loss, capability tests, contamination controls, optimization stability |
| Mid-training | How should an existing model adapt to a domain, capability emphasis, modality, or context regime? | Often continued LM learning on a changed mixture; transition-specific objectives can differ | Target gain versus matched additional training, general retention, length/position sweeps |
| Post-training | How should the model behave or choose actions/responses for the intended tasks? | Demonstrations, preference labels, rewards, teacher feedback | Independent task success, generalization, retention, feedback validity, total cost |

There is one detailed home for each topic below. [Pre-training](#pre-training) owns corpus preparation and the base LM objective; [mid-training](#mid-training) owns adaptation/retention; [post-training](#post-training) owns SFT, RL, preferences, and OPD. [Shared training systems](#llm-engineering) owns PEFT, precision, sharding, accumulation, and checkpoint execution; these controls can apply in several stages. [Inference](#llm-inference), [RAG](rag.md#rag), and [evaluation](#evaluation-and-monitoring) support deployment and measurement rather than defining additional training stages.

<a id="pre-training"></a>
## [To be reviewed] Pre-training

### [To be reviewed] Objective Families and Loss

**Objective.** A causal LM commonly minimizes $`\mathcal{L}_{\mathrm{CLM}}(\theta)=-\mathbb{E}_{x\sim D}\sum_{t\in\mathcal{M}(x)}\log p_\theta(x_t\mid x_{<t})`$, with an explicit target mask $`\mathcal{M}`$ and normalization convention. Masked/span-denoising objectives use different visibility and targets. Next-token prediction, document packing, and loss masking must agree: accidental cross-document attention or training on unintended padding changes the problem. [Llama 3 training report](https://arxiv.org/abs/2407.21783)

**Objective branches.** Fill-in-the-middle (FIM) training serializes a code prefix and suffix before asking the model to generate the missing middle, typically as a mixture with ordinary causal examples. Multi-token prediction adds training targets or modules for more than one future token; specify whether those predictions are independent auxiliary heads or a sequential dependency design. Neither method implies that arbitrary full outputs can be generated in parallel without an appropriate decoding algorithm. [FIM](https://arxiv.org/abs/2207.14255), [multi-token prediction](https://arxiv.org/abs/2404.19737)

<a id="causal-loss-example"></a>
### [To be reviewed] Causal Next-token Loss Example

For an unshifted labeled sequence, the following example aligns each logit with its next-token target. It assumes at least one valid target; a model returning its own causal-LM loss may already perform this shift. General MLE is covered in [Statistics](statistics.md#mle).

```python
import torch.nn.functional as F
# [To be reviewed] logits: [B, T, V]; labels: unshifted [B, T].
# labels use -100 for ignored targets; require at least one valid target.
# Do not shift again if the model's built-in loss already shifts labels.
loss = F.cross_entropy(
    logits[:, :-1, :].reshape(-1, logits.size(-1)),
    labels[:, 1:].reshape(-1),
    ignore_index=-100,
)
loss.backward()

```

### [To be reviewed] Corpus Preparation and Mixture Design

**Data is part of the model specification.** Record provenance and permissions, extraction, language/domain identification, exact and near deduplication, filtering, mixture weights, and benchmark decontamination. Prefer source- or time-separated evaluations; randomly separating duplicated web pages does not test independent generalization. Filtering for “educational quality” can improve a chosen capability while changing domain coverage, so inspect both retained and rejected slices. [FineWeb data ablations](https://arxiv.org/abs/2406.17557)

Normalize encodings, remove extraction boilerplate, inspect malformed records, and preserve meaningful code, tables, equations, and whitespace. Store transform versions and sample rejected data. Split by original document/source before producing chunks or paraphrases; leakage control and benchmark decontamination are part of the data protocol, not an opportunity to tune on benchmark answers. A filter can improve average cleanliness while removing dialects or specialist topics. [RefinedWeb](https://arxiv.org/abs/2306.01116)

### [To be reviewed] Scaling, Budget, and Recovery

**Compute/data trade-off.** The approximation $`C\approx6ND`$ for training a dense Transformer with $`N`$ parameters on $`D`$ tokens is a rough parameter-matmul FLOP model; attention, vocabulary work, context length, recomputation, and implementation overhead matter. It is not a complete model for sparse MoE systems. Compute-optimal scaling results describe a specified training budget and empirical regime. Training a smaller model for more tokens may be preferable when lifetime inference cost matters; a single tokens-per-parameter rule is not universally optimal. [Chinchilla scaling study](https://arxiv.org/abs/2203.15556)

**Execution and recovery.** Record the optimizer, learning-rate/token schedule, global batch, sequence-length schedule, precision, parallelism, and packing policy. A training-resume checkpoint may need optimizer/scheduler/scaler state, random-number states, and data-sampler position in addition to weights. A weights-only restart changes the optimization/data trajectory. Exact reproduction can additionally depend on hardware, kernels, and deterministic execution settings.

**Example experiment.** Compare two domain-mixture policies using the same architecture, tokenizer, number of processed tokens, and optimization budget. Report weighted aggregate loss *and* each domain's loss, contamination-resistant capability evaluations, and useful tokens/second. Changing the evaluation mixture can create an apparent improvement without improving any model conditional distribution. Do not assign a fixed “99% of all training compute” share to this stage: report the actual accounting, including post-training rollouts and teacher/judge inference where applicable.

<a id="mid-training"></a>
## [To be reviewed] Mid-training

Mid-training commonly continues from a pretrained checkpoint while changing the data mixture, context regime, or capability emphasis. The objective can remain causal LM loss; the scientific distinction is the distribution and capability being adapted. **Domain-adaptive pre-training (DAPT)** uses domain text; **task-adaptive pre-training (TAPT)** uses a task's unlabeled input distribution. Their names do not imply supervised response training. The original DAPT/TAPT results concern particular encoder models and classification tasks; extrapolating to a modern decoder requires a new experiment. [DAPT/TAPT](https://arxiv.org/abs/2004.10964)

- **Data annealing / specialization:** increase high-quality code, mathematics, multilingual, or target-domain exposure while controlling forgetting. Compare the reweighted mixture against simply training longer at the old mixture. DeepSeekMath provides a concrete continued-pretraining example, rather than proof that every model needs the same recipe. [DeepSeekMath](https://arxiv.org/abs/2402.03300)
- **Long-context adaptation:** change sequence lengths, position encoding/scaling, and the training mixture together when needed. Merely increasing an advertised context limit does not establish reliable long-range use. Evaluate evidence retrieval at multiple positions, multi-document synthesis, code dependencies, short-context retention, and memory/latency. Synthetic needle tests measure a limited subproblem. [Long-context model report](https://arxiv.org/abs/2407.21783), [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- **Other transitions:** adding modalities, token vocabularies, or architectural components changes more than data distribution. Initialize and align new components, specify which weights train, and budget the required adaptation. Such runs may be called additional pre-training, mid-training, or multimodal alignment in different reports.

**Example.** Continue an instruction-tuned model on internal technical manuals using LM loss. Domain prediction can improve while conversation behavior degrades because document continuation and assistant response generation induce different training distributions. Compare the original checkpoint, continued-pretraining checkpoint, and a checkpoint with behavior-restoring SFT/OPD. Evaluate held-out manual questions with independent evidence; memorizing a document is not the same as updating a retrieval corpus. The published OPD case study illustrates this adaptation/behavior-restoration pattern. [Case study](https://thinkingmachines.ai/blog/on-policy-distillation/)

### [To be reviewed] Retention, Replay, and Behavior Restoration

- **Retention protocol:** mixing general-domain examples is one option, alongside replay, regularization toward a reference, or task-isolated adapters. There is no universal replay ratio: tune the target/general mixture under a fixed token budget and evaluate both domains. Report the chosen replay ratio and its validation protocol.
- **Example:** adapt a model to specialist terminology while retaining a held-out general instruction and language-understanding suite. A ratio that preserves average score can still harm rare capabilities; inspect slices and compare to the unchanged base model. [Continual-learning forgetting study](https://arxiv.org/abs/1312.6211)



<a id="post-training"></a>
<a id="adaptation-and-preference-methods"></a>
## [To be reviewed] Post-training

Post-training changes task behavior using demonstrations, preferences, rewards, teacher feedback, or a mixture. **SFT, online RL, and OPD answer different questions about supervision and the states visited during learning.** They are compatible components, not three mandatory consecutive steps. Full tuning and PEFT specify which parameters can change; either can implement a suitable SFT, RL, or distillation objective.

For applying these objectives to stateful tool/environment trajectories, see [Agent Training](agents.md#agent-training). Prompt, memory, and runtime-code optimization are compared separately in [RSI & Harness](rsi-and-harness.md#rsi-learning).

<a id="sft-rl-opd-comparison"></a>
### [To be reviewed] SFT, RL, and OPD: Relationships and Differences

| Dimension | SFT on demonstrations | Online RL, e.g. PPO/GRPO | On-policy distillation (OPD) |
| --- | --- | --- | --- |
| Prefixes used for the update | Prefixes in the supplied demonstration dataset | Prefixes/actions from policy/environment rollouts, subject to the algorithm's lag/correction policy | Prefixes generated by the rollout student |
| Feedback | A desired target token/response; human or synthetic origin | Rewards from a verifier, humans/AI-derived reward model, or environment | Teacher conditionals or a derived token-level teacher signal on student states |
| Typical objective | Demonstration negative log-likelihood | Expected reward with the chosen advantage estimator, clipping, and optional regularization | Specified conditional-divergence or teacher-feedback objective; KL direction varies |
| Main mismatch or failure | Student-generated states can differ from demonstration states; incorrect labels are imitated | Reward misspecification, exploration/credit assignment, variance, stale rollouts | Unreliable teacher on student states, token/support mismatch, teacher cost, estimator bias |
| What it does not require by definition | A separately fitted reward model or fresh policy rollouts for each update | A teacher next-token distribution | A human preference pair or scalar correctness reward |

**Same-task example: implementing `inc(x)`.** SFT imitates a verified demonstration such as `return x + 1`. Online RL samples candidate programs and uses test outcomes to prefer successful behavior; the reward says which behavior succeeds, not necessarily which next token to write. OPD samples the student's partial program and asks the teacher how to continue **that exact partial program**, possibly using dense distributional feedback. The teacher may favor a plausible continuation that still fails a test, so evaluate correctness independently.

**How they combine:** SFT can initialize a policy before RL or OPD; RL can produce a stronger teacher or additional verified demonstrations; OPD can transfer behavior into another student or restore behavior after [mid-training](#mid-training). Rejection-sampled successful rollouts can become a later SFT dataset. Once that dataset is fixed, its cross-entropy training remains SFT, even though its data came from an earlier policy. [GKD and RL integration](https://arxiv.org/abs/2306.13649), [multi-stage example](https://arxiv.org/abs/2501.12948)

**DPO belongs alongside these methods as a direct preference objective.** Its basic offline formulation trains on preferred/rejected response pairs without requiring an online rollout loop or a fitted reward model. RLHF/RLAIF identify feedback provenance and a pipeline family; PPO/GRPO identify update algorithms. Keeping these axes separate prevents labels such as “AI feedback,” “synthetic data,” “LoRA,” and “RL” from being treated as interchangeable stages.

<a id="post-sft"></a>
### [To be reviewed] SFT and Instruction Tuning

SFT is supervised adaptation to target outputs; instruction fine-tuning is its instruction-conditioned subset. Classification labels, translations, structured records, and tool-use demonstrations can all define supervised targets. A typical response objective is $`\mathcal{L}_{\mathrm{SFT}}=-\mathbb{E}_{(x,y)\sim D}\sum_t m_t\log\pi_\theta(y_t\mid x,y_{<t})`$. State the target mask and whether reduction weights tokens, responses, or examples: equal weighting of two response means is different from a global token mean when lengths differ. [Instruction tuning](https://arxiv.org/abs/2109.01652), [SFT trainer](https://huggingface.co/docs/trl/sft_trainer)

**Data and serialization.** Validate roles, chat templates, tool boundaries, EOS, padding, and the intended assistant-turn masks. Include difficult valid examples, grounded answers, appropriate abstentions, and realistic multi-turn trajectories. A teacher-written tool observation is not proof that a real tool would return it. When packing samples, specify whether attention can cross example boundaries. The [causal-loss example](#causal-loss-example) explains the distinct label-shift requirement.

**Splits and labels.** Separate source problems, documents, templates, users, or time periods as appropriate before generating paraphrases or chunks. Check synthetic tasks and their answers, not only surface fluency. A random row split can leak equivalent math problems into evaluation. Maintain task/length/language coverage rather than improving only the most frequent easy template.

**Concrete evaluation.** For evidence-grounded support, verify every cited training answer against its source and include answerable and unanswerable questions. Compare task correctness, citation support, format validity, and abstention on an untouched evaluation set. Decreasing demonstration loss can coexist with worse unsupported-answer behavior. Compare a prompted base model, frozen-backbone baseline when applicable, and the fine-tuned model with matched development effort.

**Reasoning demonstrations.** Correct solution traces can teach useful patterns, but a convincing rationale is not a correctness certificate or a faithful description of internal computation. Compare answer-only and trace-conditioned training under clearly accounted token budgets, and keep hidden tests or independent solution checks outside the training feedback loop.

#### [To be reviewed] Task and Provider Examples

- [To be reviewed] BERT Fine-Tuning

    - Attach a task head or span-prediction head, then fine-tune the encoder with task-appropriate supervision. Classification typically uses a pooled or designated-token representation; extractive QA predicts start/end positions with an explicit no-answer design when needed.
    - **Example:** for document classification, split at the document or author level and assess truncation sensitivity. Report a frozen-encoder baseline, learning-rate sensitivity, and variability across seeds. Encoder weights pretrained on generic text do not remove domain shift. [BERT paper](https://aclanthology.org/N19-1423/)

[[YouTube](https://www.youtube.com/watch?v=ujubwa_oa-0&t=129s)]

- [To be reviewed] GPT Fine-tuning

    - **Evaluation-first workflow:** define a representative held-out set and a prompted base-model baseline, prepare schema-valid examples with desired responses, train a customized model when supported, and compare on the untouched evaluation set. Keep the inference instructions and formatting compatible with training. [Official OpenAI SFT guide](https://developers.openai.com/api/docs/guides/supervised-fine-tuning)
    - **Example:** for support-ticket classification, track per-class precision/recall, invalid-label rate, and performance on new product categories. A lower training loss does not establish better deployment performance. Inspect label quality and distribution coverage before adding more epochs. [Fine-tuning best practices](https://developers.openai.com/api/docs/guides/fine-tuning-best-practices)
    - **Dated availability note, 2026-10-26:** the official SFT guide states that the fine-tuning platform is winding down and is closed to new users. The original model version listed below is historical context; verify account eligibility, base-model lifecycle, and the documented retirement timeline before attempting a provider-hosted run.

    - Version: gpt-4o-2024-08-06
- [To be reviewed] Claude Fine-Tuning

    - The linked historical example concerns **Claude 3 Haiku fine-tuning through Amazon Bedrock**. Scope any reproduction to the exact base-model revision, hosting platform, region, data schema, and supported customization path. A model-family name alone does not establish which weights or parameters a managed service updates. [Anthropic's historical announcement](https://claude.com/resources/articles/fine-tune-claude-3-haiku)
    - **Example:** compare a customized domain classifier with the same base model using an optimized prompt, keeping the held-out examples and output schema fixed. Report correctness, calibration where meaningful, latency, and total inference cost. Retain the training-data version and service configuration; current availability must be established from the relevant platform documentation.

[[Claude 3 Haiku](https://www.youtube.com/watch?v=BfURFJbEQLA)]

<a id="rl-feedback-comparison"></a>
### [To be reviewed] RLHF, RLAIF, RLVR, and RLCD: A Side-by-Side Comparison

These labels describe different parts of a feedback pipeline. **RLHF/RLAIF emphasize who supplies judgments; RLVR emphasizes how outcomes are verified; RLCD specifies how synthetic preference pairs are constructed.** They are not four mutually exclusive policy-gradient algorithms. In this section, RLCD means **Reinforcement Learning from Contrastive Distillation**, following Yang et al., rather than contrastive decoding. [RLCD paper](https://arxiv.org/abs/2307.12950)

| Dimension | RLHF | RLAIF | RLVR | RLCD, original recipe |
| --- | --- | --- | --- | --- |
| Full name | Reinforcement Learning from Human Feedback | Reinforcement Learning from AI Feedback | Reinforcement Learning with Verifiable Rewards | Reinforcement Learning from Contrastive Distillation |
| Source of the learning signal | Human preferences, ratings, or corrections | AI-judge preferences or scores under a rubric | Executable outcome/constraint checks against a specification | Positive/negative prompting generates a pair; the positive-prompt output is labeled preferred |
| Typical route to an RL reward | Human comparisons → fitted reward model → policy optimization | AI comparisons → fitted reward model; direct-RLAIF can instead query the judge during RL | Policy rollout → verifier → reward | Contrastive pairs → fitted preference/reward model → PPO |
| Separate learned reward model required? | Used in the canonical comparison-based pipeline; not required by every human-feedback RL design | Used in canonical RLAIF; omitted in direct-RLAIF | Unnecessary for a pure verifier reward; hybrids can add learned rewards | Yes in the original pipeline |
| Natural application | Open-ended helpfulness and preferences requiring human judgment | Rubric-based evaluation when an AI judge is reliable enough | Testable code, checkable answers/proofs, explicit instruction constraints | A desired attribute that contrastive prompts can reliably separate |
| Main scaling cost | Annotation, adjudication, reward fitting, and rollouts | Judge inference, reward fitting when used, and rollouts | Rollouts, execution/proof-checking, and verifier construction | Two response-generation conditions, reward fitting, and rollouts |
| Characteristic failure | Annotator disagreement; reward-model extrapolation and preference proxies | Judge errors, position/style bias, or judge-directed reward hacking | Incomplete specifications, weak tests, parser exploits, sparse successes | Prompt polarity fails to establish the intended preference; shortcut learning |

The RLHF column describes the canonical [InstructGPT pipeline](https://arxiv.org/abs/2203.02155). The AI-feedback distinction follows [RLAIF and direct-RLAIF](https://arxiv.org/abs/2309.00267); the verifier route follows [Tülu 3, Section 6](https://arxiv.org/html/2411.15124v5#S6). RLCD follows its [original method, Section 3](https://arxiv.org/html/2307.12950v3#S3). These are pipeline descriptions; they do not establish a universal ranking of performance or total cost.

#### [To be reviewed] What Changes Mathematically, and What Can Be Shared?

In a hard-label pairwise reward-model formulation, a preferred/rejected pair can train the scalar score using $`\mathcal{L}_{\mathrm{RM}}=-\mathbb{E}_{(x,y_w,y_l)\sim D}\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))`$. Human labeling, AI judging, and contrastive construction change **D and the validity of its labels**, even when this fitting loss is shared. Soft-label variants use an appropriate cross-entropy target instead. [Comparison-based reward modeling](https://arxiv.org/html/2203.02155v1#S3), [RLAIF soft-label formulation](https://arxiv.org/html/2309.00267v3#S2.SS2.SSS1)

RLVR can replace that learned score with, for example, $`R_{\mathrm{verify}}(x,y)=\mathbf{1}\{\text{the specified checker accepts }(x,y)\}`$. This illustrative binary reward certifies acceptance by that checker, not arbitrary real-world correctness. A final-answer check also does not certify every reasoning step. Continuous/partial-credit rewards and learned-verifier variants need their own definitions and validation. [Tülu 3 verifier formulation](https://arxiv.org/html/2411.15124v5#S6)

**The update algorithm is a separate choice.** Tülu 3 uses PPO for RLVR; DeepSeek-R1-Zero uses GRPO with rule-based rewards. Neither “RLVR = GRPO” nor “RLHF = PPO by definition” is correct. The [online RL section](#online-rl) specifies policy objectives and estimators. [Tülu 3](https://arxiv.org/html/2411.15124v5#S6), [DeepSeek-R1-Zero](https://arxiv.org/html/2501.12948v1#S2.SS2)

#### [To be reviewed] Same-task Example: A Code Assistant

Consider a function that must sort a list while preserving the original input. This is an illustrative comparison, not a reported experiment:

- **RLHF:** reviewers compare answers for correctness, clarity, and adherence to the nonmutation requirement. Those comparisons can supervise a reward model; human judgments still need checks on difficult code.
- **RLAIF:** a judge compares the same candidates under the same rubric. Specify whether RL uses a fitted reward model or fresh judge scores, and audit both factual mistakes and preference bias.
- **RLVR:** execute checks for sorted output, permutation preservation, nonmutation, and edge cases. An incomplete suite that ignores input mutation can accept an implementation that sorts the input in place and returns it, despite the task requirement.
- **RLCD:** generate one candidate under an instruction encouraging compliance and another under a contrasting instruction; assign the first the preferred label by construction. The label is wrong if the supposed positive candidate violates the requirement. Audit actual outputs rather than trusting prompt polarity.

#### [To be reviewed] Relationships, Hybrids, and Evaluation

As a design inference from these definitions, a coding assistant can combine human-rated communication, AI-rated rubric compliance, and executable correctness rewards. Record each component, scale, and aggregation rule: a weighted sum can trade correctness against style, whereas a hard correctness gate imposes a different objective. Human-written rubrics, tests, and contrastive prompts remain human design inputs even when individual outputs are labeled automatically.

**Keep the supervision axes separate.** RLCD constructs response-level preferences; [OPD](#distillation-and-opd) instead queries teacher guidance on student-generated states. RLCD is also different from contrastive decoding at inference. Training DPO on human-, AI-, or contrastively generated pairs is a direct-preference variant; pair provenance alone does not make that update an online RL loop or the original RLCD procedure. See [DPO](#preference-optimization) and the [SFT/RL/OPD comparison](#sft-rl-opd-comparison).

For a fair comparison, hold the starting checkpoint, prompt splits, evaluation decoding, and resource accounting fixed where the causal question permits. Report human annotation, judge calls, verifier execution, reward-model training, and rollout costs separately. Evaluate using independent human judgments and unseen executable tests, inspect disagreement slices, and measure correctness separately from style. A method's own training reward cannot serve as the sole evidence that it is better.

<a id="online-rl"></a>
### [To be reviewed] Online RL: Objectives, PPO, and GRPO

An illustrative sequence-level objective is $`J(\theta)=\mathbb{E}_{x,y\sim\pi_\theta}[R(x,y)]-\beta\,\mathbb{E}_x D_{\mathrm{KL}}(\pi_\theta(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x))`$. A particular algorithm may approximate, clip, omit, or differently implement the KL regularizer. Separate the **current policy**, **rollout/old policy**, **reference policy**, **reward model/verifier**, and an optional **value estimator**. Conflating these roles leads to incorrect loss implementations. [PPO](https://arxiv.org/abs/1707.06347), [RLHF pipeline](https://arxiv.org/abs/2203.02155)

Choose and validate the reward source using the [RLHF/RLAIF/RLVR/RLCD comparison](#rl-feedback-comparison). For multi-step tool environments, rewards may also include terminal success, intermediate events, cost, and invalid-action penalties; these choices change behavior.

**GRPO.** Sample a group of responses to a prompt and form group-relative advantages, for example $`A_i=(r_i-\bar r)/(s_r+\epsilon)`$ under one explicit normalization convention. This can remove the need for a separately learned value model. Define the standard-deviation convention, zero-variance behavior, token/sequence reduction, clipping, and any KL term. Equal-reward groups provide no relative learning signal; group normalization also changes how prompts contribute, so it is not an unbiased universal replacement for a critic. [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

**Failure modes and instrumentation.** Track reward components, independent correctness, entropy, length/truncation, advantage distribution, clipping frequency, policy/reference KL, rollout age, and useful samples per unit compute. A rising learned reward with flat independent accuracy suggests proxy exploitation or poor generalization. Old rollouts need a justified off-policy correction or controlled policy lag; merely computing a likelihood ratio does not guarantee stability. [DAPO systems/optimization study](https://arxiv.org/abs/2503.14476)

**Example.** A code model gets reward 1 for passing a visible test suite. Keep a separate suite of unseen edge cases and track syntax errors, timeouts, and sandbox violations as well as pass rate. Compare SFT, best-of-N sampling, PPO/GRPO, and a no-update baseline at a matched rollout and evaluation budget. RL's additional search cost belongs in the comparison.

#### [To be reviewed] PPO Update Details

- PPO uses an importance ratio $`r_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)`$ and a clipped surrogate $`\mathbb{E}[\min(r_t A_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]`$. In language generation, actions are tokens and states are prefixes.
- A language-model RLHF implementation also needs rollout generation, advantage estimation, masking, and a chosen reference-policy regularization scheme. Clipping limits parts of the surrogate objective; it is not a hard guarantee on policy KL. **Example:** reward gain accompanied by collapsing output diversity signals the need to examine optimization and the reward proxy. [PPO](https://arxiv.org/abs/1707.06347)

<a id="rl-dapo"></a>
#### [To be reviewed] DAPO: Clipping, Sampling, and Token Weighting

**Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)** is a GRPO-related optimization and sampling recipe. It retains group-relative outcome advantages and token probability ratios, with four coordinated changes. [Paper, v2](https://arxiv.org/html/2503.14476v2), [authors' overview](https://dapo-sia.github.io/)

1. **Clip-Higher:** decouple the lower/upper ratio thresholds, with a larger upper allowance. This leaves more room to increase probabilities of positively advantaged, initially unlikely tokens; it does not directly impose a minimum entropy or guarantee exploration.
2. **Dynamic sampling:** discard all-correct/all-incorrect prompt groups in the binary-outcome setting and replenish the batch with mixed groups. Retain both successful and failed responses within a mixed group. This is different from training only on successful traces.
3. **Token-level reduction:** pool token contributions rather than average each response's token mean equally. Longer responses consequently have more aggregate weight; the effect includes penalizing undesirable patterns in long failed responses.
4. **Overlong reward shaping:** distinguish a truncated solution from a demonstrated reasoning error. The paper studies masking truncated-response loss and a soft length penalty near the generation cap. These choices change which behavior is rewarded; document the exact masking and EOS conventions. [DAPO Sections 3.1–3.4](https://arxiv.org/html/2503.14476v2#S3), [implementation controls](https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html)

Write $`\rho_{i,t}=\pi_\theta(y_{i,t}\mid x,y_{i,<t})/\pi_{\mathrm{old}}(y_{i,t}\mid x,y_{i,<t})`$, with response length $`T_i=|y_i|`$ and detached group advantage $`A_i`$. A retained-group surrogate to maximize is:

$`J_{\mathrm{DAPO}}=\mathbb{E}\left[\frac{\sum_{i=1}^{G}\sum_{t=1}^{T_i}\min\left(\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}})A_i\right)}{\sum_{i=1}^{G}T_i}\right]`$.

The expectation includes the sampling/filtering procedure. Specify whether an implementation pools over each group, minibatch, or global batch: averaging separately normalized group losses differs from pooling all tokens when groups have unequal total lengths. Distributed workers must reproduce the intended global denominator, including response masks. The paper's recipe omits reference KL in its studied reasoning setup; this is not evidence that reference regularization is universally unnecessary.

**Worked distinctions.** For a positive advantage, ratio 1.25 reaches the flat part of a surrogate with upper threshold 1.20, but remains active with threshold 1.28. A negative advantage has the opposite relevant clipping direction. For responses of lengths 2 and 8, equal-response reduction gives aggregate weights 1/2 and 1/2; pooled-token reduction gives 1/5 and 4/5, before accounting for different token gradients. Neither reduction identifies which step caused success.

**Sampling cost and selection:** if iid responses succeed with probability $`p`$, a binary-reward group of size $`G`$ is mixed with probability $`1-p^G-(1-p)^G`$. At $`p=0.99,G=8`$, this is about 7.73%; idealized replenishment needs about 12.94 sampled groups per retained group. Filtering therefore changes prompt weighting and can consume substantial rollout work. Report discarded tokens, generation rounds, wall time, and independent accuracy, not only optimizer steps. See [exploration](#rl-exploration), [reward hacking](#rl-reward-hacking), and [NeMo's sampling diagnostics](https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html#dynamic-sampling).

<a id="rl-gspo"></a>
#### [To be reviewed] GSPO: Sequence Ratios and Sequence Clipping

**Group Sequence Policy Optimization (GSPO)** retains a group-relative response advantage but replaces individual token ratios in the surrogate with a shared, length-normalized sequence ratio. With the same notation:

$`s_i(\theta)=\left(\frac{\pi_\theta(y_i\mid x)}{\pi_{\mathrm{old}}(y_i\mid x)}\right)^{1/T_i}=\exp\left(\frac{1}{T_i}\sum_{t=1}^{T_i}\log\rho_{i,t}\right)`$,

$`J_{\mathrm{GSPO}}=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(s_i A_i,\operatorname{clip}(s_i,1-\epsilon_-,1+\epsilon_+)A_i\right)\right]`$.

Clipping is decided for the response as a whole in the relevant advantage direction. Reference KL is omitted from these displayed formulas; the GSPO paper omits it for exposition, whereas the DAPO recipe explicitly removes it. [GSPO paper, v2](https://arxiv.org/html/2507.18071v2#S4), [Qwen's method description](https://qwenlm.github.io/blog/gspo/)

**What the ratio means:** $`s_i`$ is the geometric mean of token ratios, not the raw sequence importance weight $`\prod_t\rho_{i,t}`$. Taking its length root changes the objective; clipping changes it further. Do not describe this as an unbiased exact importance-sampling estimator of arbitrary current-policy sequence rewards. Compute it from summed log-probability differences for numerical stability.

**Gradient and example.** On an unclipped branch with detached advantage, a response contributes a gradient proportional to $`(s_i A_i/T_i)\sum_t\nabla_\theta\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})`$. Every token shares the same scalar coefficient, although token score gradients differ. For token ratios 1.44 and 0.81, the raw product is 1.1664 and the geometric mean is 1.08. With positive advantage and illustrative bounds [0.8, 1.2], tokenwise clipping flattens the first token's direct contribution, while sequence clipping leaves the GSPO response active. Real GSPO clipping widths require separate tuning; PPO's token-ratio thresholds are not transferable defaults.

The authors report improved stability in their long-response/MoE experiments and reduced dependence on routing replay. Treat these as results under their configurations; routing behavior and inference/training numerical differences still require monitoring. **GSPO-token** is a distinct implementation variant: its stop-gradient construction preserves a shared sequence-ratio value while allowing token-specific advantages; with identical response advantages it matches GSPO's objective and gradient. Ordinary GSPO does not itself supply finer causal [credit assignment](#rl-credit-assignment). [Gradient analysis and variant](https://arxiv.org/html/2507.18071v2#S4.SS2), [reported systems implications](https://qwenlm.github.io/blog/gspo/#benefits-for-moe-rl-and-infrastructure)

<a id="rl-method-comparison"></a>
#### [To be reviewed] GRPO, DAPO, and GSPO: What Actually Changes?

| Axis | Original outcome-supervised GRPO | DAPO recipe | GSPO |
| --- | --- | --- | --- |
| Advantage | Response reward centered/standardized within a prompt group | Group-relative response advantage | Group-relative response advantage |
| Ratio and clipping | Per-token ratio; per-token clipping | Per-token ratio; decoupled lower/upper clipping | Geometric-mean sequence ratio; response-level clipping |
| Reduction in displayed objectives | Mean of response token means | Pooled token contributions within the specified reduction unit | Mean of response-level surrogates; gradient includes inverse response length |
| Handling all-equal groups | Outcome-based group advantage vanishes | Replenish filtered prompt groups | No automatic remedy from changing the ratio alone |
| Main diagnostic | Group variance, token clipping, KL/entropy | Also acceptance rate, discarded generation, length weighting/truncation | Also sequence-ratio/clipping distribution, MoE routing, numerical mismatch |

These are update/sampling choices, while [RLHF/RLAIF/RLVR](#rl-feedback-comparison) describe supervision. Compare methods with matched data access and total rollout compute, recording all sampling, masking, reward, KL, and reduction differences. A win after changing five recipe components does not isolate the ratio estimator's effect. [OPSD](#opd-opsd) changes the source and density of supervision; it is not another group-reward clipping rule.

<a id="rl-credit-assignment"></a>
#### [To be reviewed] Credit Assignment: Returns, Baselines, and GAE

**The question is which earlier decisions should change after a delayed outcome.** In an LLM, a state can be a prompt-plus-prefix and an action a token; in an agent, an action can contain many tokens and trigger an environment transition. A terminal test result does not identify a unique responsible token or tool call. Separate temporal credit assignment, variance reduction, and causal attribution.

**Contextual bandit versus MDP:** for a single-turn prompt with only response-level feedback, the complete response can be treated as one contextual-bandit action. Autoregressive factorization still gives tokenwise log-probability gradients. A multi-turn tool agent additionally acts on an environment and receives new observations; model that sequential structure explicitly rather than treating tool results as tokens controlled by the policy.

For rewards r_t on transitions and an episode ending after T actions, define reward-to-go as $`G_t=\sum_{l=0}^{T-t-1}\gamma^l r_{t+l}`$. In the **undiscounted** episodic case, $`J(\theta)=\mathbb E[\sum_t r_t]`$ has the score-function gradient $`\nabla_\theta J=\mathbb E[\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)(G_t-b(s_t))]`$ under the usual regularity/support conditions and policy-independent environment dynamics. An action-independent baseline has zero expected score contribution. Hold the sampled advantage/baseline fixed in the actor surrogate; train a learned value function with its own objective. For a discounted start-state objective, retain the corresponding discount weighting or explicitly use a discounted-occupancy formulation; silently reusing the undiscounted identity changes the estimand.

**Actor–critic.** A state value estimates expected future return under the policy; an advantage measures an action's return relative to that state baseline. A common TD residual is $`\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)`$, where d_t indicates a genuine terminal transition. Generalized Advantage Estimation uses $`\widehat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}`$, stopping at episode boundaries. It combines bootstrapped residuals: smaller lambda can reduce variance but increases dependence on value accuracy; lambda = 1 recovers the corresponding Monte Carlo-minus-value expression with compatible terminal/bootstrap treatment. [GAE](https://arxiv.org/abs/1506.02438)

**Worked example, not a measured result:** take three actions, rewards (0, 0, 1), gamma = 0.9, lambda = 0.8, values (0.4, 0.5, 0.6), and terminal value zero. The TD residuals are (0.05, 0.04, 0.4); backward recursion $`\widehat A_t=\delta_t+\gamma\lambda\widehat A_{t+1}`$ gives advantages (0.28616, 0.328, 0.4). Monte Carlo returns are (0.81, 0.9, 1). These are different objects, and none establishes that the third action alone caused success.

**LLM-specific consequences:** with gamma = 1 and only one terminal outcome reward, all response tokens initially share the same Monte Carlo return. Their score gradients still differ; a prefix-dependent value baseline, tokenwise KL penalties, or intermediate rewards can further differentiate advantages. Group-relative response rewards remove a critic, but do not automatically discover which reasoning step was wrong. See [agent credit assignment](agents.md#agent-rl) for tool transitions and [process supervision](reasoning.md#reasoning-overview) for intermediate labels.

**Termination versus truncation:** a token/time cap can either define failure in the actual task or merely cut a continuing trajectory for collection. Bootstrap only when the objective and available continuation state justify it; do not treat every timeout as a natural terminal or every truncation as safely continuable. Never propagate returns across unrelated packed responses. Track masks, EOS handling, value error, advantage variance, and action/token normalization.

<a id="rl-reward-hacking"></a>
#### [To be reviewed] Reward Hacking: Proxy Gains Versus Intended Outcomes

Reward hacking occurs when optimization exploits a mismatch between the implemented reward and the intended objective. It need not involve an explicit intent to deceive. Ordinary generalization error, reward-model noise, benchmark contamination, and reward exploitation can produce overlapping symptoms; rising training reward alone does not diagnose which mechanism occurred.

| Failure mechanism | Illustrative LLM / agent behavior | Evidence and intervention |
| --- | --- | --- |
| Learned-proxy exploitation | Verbose, confident, or agreeable answers win a judge despite worse factual accuracy | Compare blinded independent judgments and task checks; test length/style-controlled pairs and audit reward extrapolation |
| Incomplete verifier specification | A patch passes narrow tests while violating an untested nonmutation or edge-case requirement | Expand development checks from the task contract; use independent final tests and inspect the actual artifact |
| Process-reward exploitation | A response adds many superficially plausible intermediate steps to collect positive local scores | Check step correctness, redundancy, terminal success, and reward aggregation; compare matched-token alternatives |
| Reward-channel tampering | An agent changes a result file or evaluation component instead of solving the task | Grade the submitted artifact in an isolated evaluator; preserve access boundaries and immutable evaluation records |
| Misweighted objectives | Formatting or brevity bonuses outweigh required correctness or evidence | Inspect every reward component and their units; compare constrained/gated objectives with a weighted sum |

**Diagnosis example:** suppose training reward rises while hidden-test success falls. Freeze several checkpoints and rerun them with the same sampling budget, independent grader, and held-out inputs. Inspect disagreement cases: are tests incomplete, labels wrong, or outputs exploiting a style/format shortcut? Fixing the verifier and retraining is different from lowering the learning rate against the same defective objective. Keep final tests outside iterative reward development. The reward-overoptimization study by Gao et al. demonstrates proxy/gold divergence in a controlled setup where the “gold” signal is itself another model, not an infallible measurement of human utility. [Reward-model overoptimization](https://proceedings.mlr.press/v202/gao23h.html)

**Mitigations are complementary:** improve reward coverage, audit on-policy failures, use independent evaluation, isolate grading from mutable agent state, and limit unsupported distributional drift. A stronger reward model, KL regularization, ensembles, or early stopping may help, but none repairs an incorrect specification by definition. RLVR removes the need for a learned reward model in its pure form; it still optimizes acceptance by a finite checker. See [feedback-source limitations](#rl-feedback-comparison) and [grader validity](benchmarking-and-evaluation.md#eval-graders).

<a id="rl-reward-design"></a>
#### [To be reviewed] Outcome Rewards, Process Rewards, and Reward Shaping

**Outcome supervision** rewards the final answer or environment state; **process supervision** labels intermediate reasoning/actions. Outcome feedback can be sparse but comparatively easy to verify. Process feedback can provide earlier distinctions, while introducing annotation cost, local-label errors, and incentives to optimize the visible step format. Neither granularity guarantees a faithful reasoning trace or better end-to-end behavior. [Process supervision](https://arxiv.org/abs/2305.20050)

Specify whether process rewards are summed, averaged, minimized, or used to score whole trajectories. Summing positive step scores can reward unnecessary steps; averaging changes relative sequence weighting and does not remove judge exploitation. A per-token cost explicitly trades task performance against length, rather than being a neutral implementation trick. Keep correctness, format, latency, and resource costs separately logged even when the training objective combines them.

**Potential-based shaping** illustrates when an auxiliary signal can preserve a task's optimal policies under specific MDP assumptions. Add $`F(s_t,s_{t+1})=\gamma\Phi(s_{t+1})-\Phi(s_t)`$ to the original transition reward. Its discounted sum telescopes to $`\sum_{t=0}^{T-1}\gamma^tF(s_t,s_{t+1})=-\Phi(s_0)+\gamma^T\Phi(s_T)`$. If the boundary term is policy-independent, for example zero terminal potential in the episodic setting, this adds only a fixed offset for a fixed start. Infinite-horizon discounted results need the corresponding boundedness/vanishing-boundary conditions. A learned “good reasoning step” bonus is not automatically of this form. [Ng, Harada, and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)

**Example:** rewarding an agent every time it runs a test can favor repeatedly running the same test without fixing the bug. A potential tied to verified progress must still satisfy the shaping definition and boundary conditions to inherit that invariance argument. Under finite compute, function approximation, clipping, or imperfect optimization, policy invariance of the ideal objective is not a guarantee of an identical learned policy.

<a id="rl-exploration"></a>
#### [To be reviewed] Exploration, Sparse Rewards, and Critic-Free Estimators

Policy gradients need informative sampled behavior. When every attempted solution fails, a terminal binary reward gives little differentiation; when every solution succeeds, it gives little discrimination among those successes. With iid binary outcomes of probability p and group size G, the probability of a mixed-success group is $`1-p^G-(1-p)^G`$. At p = 0.1 and G = 4, it is 0.3438; “at least one success” is 0.3439. The distinction matters because all-success groups also have zero relative outcome advantage. Real samples can be correlated, so this calculation is an illustrative conditional model.

| Estimator / algorithm | Baseline or advantage source | Main trade-off |
| --- | --- | --- |
| REINFORCE | Sampled reward-to-go, optionally minus an action-independent baseline | Simple Monte Carlo estimator; variance and sparse exploration can be costly |
| RLOO | Other independent responses to the same prompt supply a leave-one-out reward baseline | Avoids a separately learned critic; needs multiple samples and valid baseline/independence assumptions |
| PPO actor–critic | Commonly learned values plus GAE, with a clipped actor surrogate | Supports prefix-dependent baselines; adds value fitting, memory, and update choices |
| GRPO | Group-relative feedback, commonly centered and standardized | Avoids a learned value model in the original recipe; normalization, zero-variance groups, and prompt weighting need care |

For a fixed-size group of iid responses and unnormalized rewards, including a sample's own reward in its group-mean baseline scales the expected REINFORCE signal by $`(G-1)/G`$ compared with the no-baseline signal. A leave-one-out mean avoids that particular self-inclusion effect. Dividing by a random group standard deviation adds another transformation; it is not merely baseline subtraction and need not preserve the same gradient target. [RLOO / REINFORCE-style optimization](https://arxiv.org/abs/2402.14740), [original GRPO](https://arxiv.org/abs/2402.03300)

**Concrete group:** rewards (0, 0, 1, 1) have mean 0.5 and population standard deviation 0.5, producing standardized advantages (-1, -1, 1, 1) before any epsilon adjustment. Four zero rewards produce no relative **outcome** signal; entropy/KL terms or other reward components may still produce updates. A larger group increases sampling cost and does not create missing capabilities by itself.

Improve exploration using a suitable initialization, difficulty curriculum, useful demonstrations, diversity-aware prompt/sample collection, or a better environment/action interface. Higher temperature and entropy regularization can expand coverage but can also produce invalid actions. Verify that changed sampling filters match the behavior probabilities used for correction. Report mixed-reward-group rate, success by difficulty, distinct valid solutions, rollout cost, and independent pass@1; avoid claiming a gain from selecting only productive groups without counting discarded rollouts. [Sampling/length optimization study](https://arxiv.org/abs/2503.14476)

<a id="rl-kl-control"></a>
#### [To be reviewed] KL Control, PPO Clipping, Entropy, and Length

**Two different anchors:** PPO's old policy generated the rollout; the reference policy defines a regularization target, often a fixed SFT checkpoint. Refreshing the old policy does not refresh the reference unless the recipe explicitly says so. Ratio clipping controls a surrogate relative to the former; a reference-KL term penalizes drift from the latter. Neither is a guarantee of preserving task quality. See [PPO update details](#to-be-reviewed-ppo-update-details).

For one fixed prompt, a fixed response reward, beta > 0, and a finite normalizer, optimizing expected reward minus $`\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})`$ over response distributions supported by the reference gives $`\pi^*(y\mid x)=\pi_{\mathrm{ref}}(y\mid x)\exp(R(x,y)/\beta)/Z(x)`$. This idealized distributional optimum is not a guarantee about a finite neural-network PPO run. It shows why reward scale and beta interact: multiplying an unnormalized reward by ten at fixed beta has the same ideal trade-off as dividing beta by ten. Reward whitening, clipping, and other objective changes require a separate analysis. [KL-regularized preference framework](https://arxiv.org/abs/2305.18290)

**Sequence versus token KL:** with matching autoregressive sample spaces and termination conventions, sequence KL equals the current-policy expectation of summed conditional token KLs. On current-policy samples, the log-ratio sum $`\sum_t\log[\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{ref}}(a_t\mid s_t)]`$ estimates the sequence divergence in expectation; an individual sample can be negative. A per-token mean is a different length-weighted quantity. Filtered/stale rollouts, missing EOS, and truncation can invalidate a naive interpretation of that logged estimate. The conditional-divergence and sampling distinctions are developed in [OPD](#distillation-and-opd).

**Entropy is another control.** An entropy bonus encourages a wider policy at visited states; it is not the same as staying near a reference distribution. More entropy can mean useful strategy diversity or merely more malformed tokens. Longer answers can reflect useful computation, a verbosity-biased reward, reward aggregation, or changed stopping behavior. Compare verified quality at matched output/tool budgets, monitor EOS/truncation, and inspect length by success/failure rather than assuming either longer or shorter is always better.

<a id="rl-policy-data"></a>
#### [To be reviewed] On-Policy, Off-Policy, and Offline Learning

**On-policy** describes compatibility between the collection policy and the update estimator. PPO collects from an old policy and intentionally reuses that batch for bounded updates; it is not “fresh data for every gradient step.” **Off-policy** methods learn from behavior generated by another policy with a justified estimator/objective. **Offline RL** uses a fixed interaction dataset without new collection; its major challenges include support/coverage and distribution shift. These are different axes from whether labels were supplied by humans, an AI judge, or a verifier. [Offline RL review](https://arxiv.org/abs/2005.01643)

For a fixed prompt, sequence importance sampling can use $`w(y)=\prod_t\pi_\theta(a_t\mid s_t)/\mu(a_t\mid s_t)`$ when the behavior policy mu has adequate support. Products can have extreme variance; clipping or self-normalization trades bias for stability. Per-token PPO ratios alone are not a general exact correction for arbitrary stale histories or changed environment distributions. Record actual behavior probabilities, rollout version/age, sampler, and environment state.

**Interview distinction:** a frozen set of successful rollouts trained by negative log-likelihood is rejection-sampling SFT. Basic offline DPO uses preference pairs and a reference-relative objective. Neither becomes online policy-gradient RL merely because an earlier model generated the examples. OPD identifies teacher feedback on student-visited states, not one particular reward source. Use the [SFT/RL/OPD comparison](#sft-rl-opd-comparison) instead of assigning a training method from data provenance alone.

<a id="rl-interview"></a>
#### [To be reviewed] Interview Discussion: Questions and Follow-ups

A useful answer states the objective, sampling assumptions, failure mechanism, and an experiment that could distinguish competing explanations. The following prompts connect the derivations above to concrete system decisions.

| Question | Core answer | Follow-up to prepare |
| --- | --- | --- |
| Why use RL after SFT? | SFT imitates provided targets; RL can optimize a specified outcome on policy-generated behavior. Gains depend on reward validity, exploration, and budget. | Compare with stronger demonstrations, rejection-sampling SFT, and inference-time selection at matched total cost. |
| Is LLM RL a contextual bandit or an MDP? | A whole single-turn response can be one bandit action, or be factorized into token decisions. Interactive agents add environment transitions and observations. | Which quantities does the policy control, and what changes when feedback arrives after several tool calls? |
| How does a final scalar reward train earlier tokens? | The likelihood-ratio gradient changes sampled action probabilities according to return/advantage. Baselines reduce variance, without proving which token caused success. | Derive reward-to-go, explain GAE, and handle observation tokens and trajectory boundaries. |
| Does removing the critic solve credit assignment? | No. GRPO changes the estimator and memory cost; a response-level advantage can still assign the same outcome signal to every token. | What happens with all-equal rewards, tiny groups, or unequal prompt difficulty? |
| Why can reward improve while answer quality falls? | The proxy or verifier may be exploitable, or evaluation/sampling may have changed. | Freeze checkpoints, compare independent graders, and inspect reward–correctness disagreements. |
| Does RLVR eliminate reward hacking? | It eliminates some learned-proxy issues, but a verifier still defines a limited acceptance specification. | Design hidden edge cases and isolate artifact grading from the agent's mutable workspace. |
| Why does PPO clipping not guarantee small KL? | The clipped surrogate is not a hard global constraint on distributions or all unseen actions. | Distinguish old-policy ratios, reference KL, actual KL monitoring, and early stopping. |
| Can larger groups or higher temperature fix sparse rewards? | They can increase exploration under a fixed capability distribution, at extra cost and possibly more invalid actions. | Estimate mixed-group probability, count discarded rollouts, and compare curriculum or SFT initialization. |
| Are process rewards always better? | They offer denser measurements but can be wrong, correlated, or vulnerable to step/length incentives. | Compare outcome-only, process-only, and combined rewards with independent terminal grading. |
| Can we reuse old rollouts indefinitely? | Only with an objective/estimator justified for the behavior distribution and sufficient support; variance and policy lag remain constraints. | Audit filtered behavior log-probabilities, importance weights, environment drift, and refresh cadence. |
| How do DAPO and GSPO differ from GRPO? | DAPO changes token-ratio clipping, group sampling, reduction, and length handling; GSPO changes the ratio and clipping unit to the response. | Derive the geometric-mean ratio and distinguish it from the raw sequence importance weight. |
| Does OPSD require a stronger external teacher? | The cited method uses privileged reference context with a self teacher; reported experiments freeze its initial checkpoint. | Separate new information from model size, and compare full-vocabulary gradients with sampled-token feedback. |
| What does lower actor loss establish? | Only progress on that sampled surrogate under its sign/reduction convention; changing advantages and clipping make cross-run loss values hard to compare. | Show independent correctness, actual KL, entropy, length, cost, and retained capabilities. |

<a id="rl-debugging"></a>
##### [To be reviewed] A Practical RL Debugging Sequence

1. **Validate the reward contract:** feed known valid/invalid outputs through the parser, verifier, and reward aggregator. Check units, format bonuses, empty outputs, exceptions, and reward provenance.
2. **Validate the estimator:** compare sampled log-probabilities with the actual behavior sampler; inspect masks, advantage signs, detached targets, reduction denominators, and terminal/truncation handling. Before an update, the current/old ratio should be one when weights and likelihood/sampler conventions truly match.
3. **Locate signal loss:** inspect all-equal groups, value error, reward/advantage variance, gradient norms, clipping fraction, and rollout age. Separate actor signal from KL/entropy contributions.
4. **Measure the intended outcome:** hold evaluation prompts, grading, and inference budget fixed across checkpoints. Track hidden-test/task success, reward components, length, diversity, retained capabilities, and total rollout/judge/tool cost.
5. **Run one controlled intervention:** change the verifier, curriculum, KL scale, rollout refresh, or estimator reduction individually. A joint recipe change may improve a system while leaving the mechanism unidentified. Use the [evaluation protocol](benchmarking-and-evaluation.md#eval-protocol) to preserve that distinction.


<a id="preference-optimization"></a>
### [To be reviewed] Preferences, Reward Models, and AI Feedback

Preference labels and reward signals are measurements with a specified population and rubric. Direct preference objectives and online reward optimization use them differently; the collection/judging process must be evaluated independently of the policy update.

#### [To be reviewed] Direct Preference Optimization (DPO)

- DPO directly optimizes preferred/rejected response pairs without fitting an explicit reward model or performing on-policy rollouts in its basic form. Let $`\Delta_\theta=\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}`$; its loss is $`-\log\sigma(\beta\Delta_\theta)`$ under the standard formulation.
- **Example:** the update changes relative likelihoods rather than maximizing the preferred response's likelihood in isolation. Sequence length, reference choice, label noise, and offline pair coverage affect behavior. DPO uses preference labels; it does not by itself improve how human feedback is collected. [DPO](https://arxiv.org/abs/2305.18290)

#### [To be reviewed] Feedback Collection and Reward-model Evaluation

- Collect comparisons on policy-relevant prompts with an explicit rubric and multiple annotators where ambiguity matters. Randomize presentation order, account for ties and disagreement, and inspect whether length, style, or answer position predicts labels independently of correctness.
- **Example:** a longer answer can win superficial preference comparisons while adding factual errors. Include adversarial counterexamples and independently validated correctness labels. Synthetic preferences can expand coverage but inherit the judge's limitations; periodically audit them against held-out human judgments.

- Evaluate held-out pairwise reward accuracy and calibration, then measure policy quality independently after optimization. Group splits by prompt family reduce leakage from near-duplicate comparisons. Reward accuracy averaged over easy pairs can hide poor discrimination among strong candidate responses.
- **Example:** a reward model can score 90% on obviously good/bad pairs while systematically favoring verbose hallucinations among near-ties. Track policy win rates, task correctness, response length, disagreement slices, and reward-versus-human divergence. [RewardBench](https://arxiv.org/abs/2403.13787)

#### [To be reviewed] RLAIF and Judge Bias

- RLAIF replaces or supplements human preference labels with feedback from an AI judge. Constitutional AI combines a principle-guided critique/revision stage with AI-generated preference comparisons for a later reinforcement-learning stage.
- **Example:** a judge scores candidate responses against a written rubric, and a policy learns from those scores or comparisons. Record the judge model, rubric, prompt, and sampling settings: “AI feedback” is a family of data-generation procedures, not one optimization algorithm. [Constitutional AI](https://arxiv.org/abs/2212.08073)

- Automated judging can scale annotation and improve consistency with an explicit rubric, but systematic judge errors can also scale. Assess position bias, preference for verbosity or its own style, and agreement on difficult subgroups.
- **Example:** swap candidate order and paraphrase equivalent answers; a large preference reversal suggests a measurement artifact. Keep independent human audits and task-specific verifiers for claims the judge cannot reliably establish. Strong agreement with one judge is not evidence of universal human preference.

#### [To be reviewed] Unresolved Terminology: KPO

- **Terminology to verify:** `KPO` is not uniquely identified by the surrounding outline. Keep the original label until a paper title or implementation establishes its intended expansion; do not treat it as a synonym for PPO, DPO, KTO, or GRPO.
- If **KTO** was intended, it uses a prospect-theory-inspired objective with desirable/undesirable examples rather than requiring paired preferences. If **GRPO** was intended, group-relative rewards provide an advantage baseline for sampled outputs. These are different methods with different data and rollout requirements. [KTO](https://arxiv.org/abs/2402.01306), [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300)

<a id="distillation-and-opd"></a>
### [To be reviewed] Distillation: Offline Traces and On-policy Distillation (OPD)

Distillation transfers supervision from a teacher, which need not always be larger: a stronger checkpoint, privileged-context model, ensemble, or task-specialist can supply a useful teacher signal. State what additional information or compute makes that teacher helpful.

| Procedure | Prefix distribution used in training | Supervision | Main limitation to test |
| --- | --- | --- | --- |
| Offline trace SFT / sequence distillation | Fixed teacher/reference responses | Teacher-produced target tokens or validated solutions | Student mistakes can lead to prefixes absent from the dataset |
| Offline logit distillation | Fixed dataset prefixes | Teacher conditional probabilities or logits | Better targets on the same fixed states do not remove state-distribution mismatch |
| OPD | Refreshed student-generated prefixes | Teacher probabilities, local divergences, or sampled-token feedback | Teacher reliability on student states, support/serialization mismatch, rollout cost |

OPD addresses exposure to **the student's own states**: generate a response using the student and evaluate the teacher at those exact prompt-plus-prefix states. This differs from asking the teacher to produce a separate ideal answer and applying SFT to it. Both can be useful and can be combined. [Generalized Knowledge Distillation](https://arxiv.org/abs/2306.13649)

#### [To be reviewed] OPD Objectives and the Gradient Distinction

At a fixed prefix $`s`$, let $`p_\theta(\cdot\mid s)`$ be the student distribution and $`p_T(\cdot\mid s)`$ a fixed teacher. **Forward KL**, $`D_{\mathrm{KL}}(p_T\|p_\theta)`$, gives soft-target cross-entropy up to a teacher-only constant. **Reverse KL**, $`D_{\mathrm{KL}}(p_\theta\|p_T)=\sum_v p_\theta(v\mid s)\log[p_\theta(v\mid s)/p_T(v\mid s)]`$, weights discrepancies by the student's probabilities. KL direction alone does not define whether the prefix data are on-policy. [GKD objective family](https://arxiv.org/abs/2306.13649)

An illustrative local objective is $`\mathcal{L}_{\mathrm{local}}(\theta;\theta_{\mathrm{old}})=\mathbb{E}_{s\sim d_{\pi_{\mathrm{old}}}}D_{\mathrm{KL}}(p_\theta(\cdot\mid s)\|p_T(\cdot\mid s))`$, where the sampled prefix distribution is held fixed during the update. Refreshing rollouts makes the state collection on-policy relative to the rollout student, subject to policy lag and the actual sampler.

**Do not equate the local update with the exact gradient of sequence-level reverse KL.** The chain rule for KL decomposes the sequence-level *value* into expected conditional KLs, but the current-policy prefix distribution also depends on $`\theta`$. Detaching sampled prefixes omits that occupancy-distribution gradient. Sequence-policy-gradient treatments include downstream effects through returns; local token-feedback recipes deliberately make a different credit-assignment/variance trade-off. This distinction matters when interpreting claims that two implementations “minimize reverse KL.” [OPD implementation case study](https://thinkingmachines.ai/blog/on-policy-distillation/), [2026 analysis of OPD bias and variance](https://arxiv.org/abs/2603.25562)

**Implementation trap.** Sampling discrete token IDs and naively differentiating only their mean `log p_student - log p_teacher` is not a complete reverse-KL gradient estimator. Use an explicitly derived policy-gradient surrogate with the appropriate detached feedback, or differentiate a full conditional-distribution KL at fixed states. State the estimator, masking, baseline, and normalization; these choices are not interchangeable.

**Illustrative state.** The student generates an arithmetic intermediate result that is already wrong. OPD queries the teacher's next-token distribution *conditioned on that erroneous prefix*. The teacher may suggest recovery, continue an incorrect pattern, or become unreliable off its typical support. Dense feedback helps only when that conditional guidance is useful; it does not certify a correct final answer.

#### [To be reviewed] OPD Implementation and Evaluation

1. Select prompts with source-separated development/test splits and freeze a versioned teacher. Document whether it has privileged solutions or additional context.
2. Generate student rollouts and record the actual behavior probabilities, temperature, top-p/top-k settings, EOS, and truncation masks. Sampling filters change the behavior distribution; raw-model likelihoods cannot silently stand in for filtered behavior probabilities when a correction requires the latter.
3. Score teacher conditionals on each selected student prefix with correct next-token alignment. Direct tokenwise comparison requires matching token semantics/IDs and serialization, not merely equal vocabulary sizes. Different tokenizers require an explicit alignment or sequence-level approach.
4. Choose a local full-distribution, truncated-support, or sampled-feedback objective. Teacher top-k responses are not a full vocabulary distribution: document residual mass and any renormalization, because truncation changes the objective and support. Mask padding and define how EOS/truncated responses contribute.
5. Update the student and refresh rollouts at a justified cadence. Measure teacher entropy/reliability, policy lag, response length, gradient statistics, retained capabilities, and **teacher-inclusive** GPU time or API cost. Keep teacher computation outside the student gradient graph unless joint training is explicitly intended.

**Evaluation example.** Compare verified teacher-trace SFT, offline soft-target distillation, OPD, and reward-based RL from the same student checkpoint. Match prompts, data access, target evaluation decoding, and a clearly defined compute budget; report both student-update and total rollout/teacher cost. Judge final correctness independently of the teacher and test new problem families. A smaller student can outperform its teacher on a particular benchmark after additional training or selection, but this is an empirical result, not a guarantee of distillation. Recent analyses document teacher-support and tokenizer/special-token failures. [OPD failure-mode study](https://arxiv.org/abs/2603.25562)

<a id="opd-opsd"></a>
#### [To be reviewed] OPSD: On-Policy Self-Distillation with Privileged Solutions

Here **OPSD** means *On-Policy Self-Distillation* in **Self-Distilled Reasoner** (2026). A model supplies both student and teacher distributions under different contexts: the student sees question $`x`$, while the teacher additionally sees reference solution $`y^\star`$. The student samples a response; the teacher scores continuations along **that student's exact prefixes**, using the reference as privileged training information. The reference is absent at student inference. This is supervised self-distillation, not learning from no new information. [Paper, v3](https://arxiv.org/html/2601.18734v3), [official implementation](https://github.com/siyan-zhao/OPSD)

**Framework versus experimental teacher.** The conceptual algorithm writes both branches as the same parameterized model with different conditioning, with gradients stopped through the teacher. The v3 experiments instead hold the teacher at the **initial checkpoint** for stability. A frozen initial teacher and a refreshed current-model teacher are different training targets; record the actual choice. Self-distillation does not require that the teacher track every student update. [Experimental setup, Section 4.1](https://arxiv.org/html/2601.18734v3#S4.SS1)

For a fixed teacher checkpoint $`\theta_0`$, write $`q_t=p_{\theta_0}(\cdot\mid x,y^\star,\hat y_{<t})`$ and $`p_t=p_\theta(\cdot\mid x,\hat y_{<t})`$. For rollout length $`T=|\hat y|`$, a local full-vocabulary update uses:

$`L_{\mathrm{OPSD}}(\theta)=\mathbb{E}_{(x,y^\star),\,\hat y\sim p_{\mathrm{old}}(\cdot\mid x)}\left[\frac{1}{T}\sum_{t=1}^{T}D\left(\operatorname{sg}[q_t],p_t\right)\right]`$.

Here $`p_{\mathrm{old}}`$ denotes the rollout snapshot refreshed during training; sampled prefixes and teacher targets are detached in this local update. Specify the divergence direction: forward KL is $`D_{\mathrm{KL}}(q_t\Vert p_t)`$, reverse KL is $`D_{\mathrm{KL}}(p_t\Vert q_t)`$; generalized JSD is another option. The paper also studies sampled-token policy-gradient feedback. Full-distribution matching, its sampled-action surrogate, and the exact gradient of sequence KL should not be conflated; see the [OPD gradient distinction](#to-be-reviewed-opd-objectives-and-the-gradient-distinction).

**Implementation detail worth checking:** v3 caps individual *vocabulary-entry divergence contributions* at each rollout position before summing, rather than clipping a response probability ratio or merely the scalar total KL. In its notation, $`\ell_{t,v}=q_t(v)f(p_t(v)/q_t(v))`$ and the clipped token mean is $`T^{-1}\sum_t\sum_v\min(\ell_{t,v},\tau)`$. This changes the divergence objective; the chosen generator, threshold, and support handling matter. The released code also reports chat-template fixes, so pin a revision and verify student/teacher response alignment. [Clipping definition](https://arxiv.org/html/2601.18734v3#S3.SS2), [code updates](https://github.com/siyan-zhao/OPSD#updates)

**Illustrative two-token vocabulary.** On the student's prefix, suppose $`p_t=(0.6,0.4)`$ and the solution-conditioned teacher gives $`q_t=(0.9,0.1)`$. Forward KL is approximately 0.2263 nats; reverse KL is approximately 0.3112 nats. Forward-KL gradients with respect to student logits are $`p_t-q_t=(-0.3,0.3)`$. These directions favor the teacher's first continuation even when the completed student response is incorrect; no all-correct/all-wrong group is required. This toy distribution does not establish that the teacher can recover from every wrong reasoning prefix.

| Recipe | States used for updates | What supplies supervision? |
| --- | --- | --- |
| Verified-trace SFT / rejection-sampling self-training | Fixed retained demonstration prefixes | Hard next-token targets in accepted traces |
| OPD with an external teacher | Refreshed student prefixes | A separately selected teacher's conditional distribution |
| OPSD in the cited paper | Refreshed student prefixes | Same-model-family teacher with a privileged reference solution; frozen initial teacher in reported experiments |
| Outcome-based GRPO/DAPO/GSPO | Sampled response groups | Verifier rewards and relative advantages |

**Evaluation design:** ablate reference access, teacher freezing, divergence/clipping, and rollout length separately. Keep verified references inside training splits, evaluate with question-only inputs on held-out problem families, and grade independently of teacher likelihood. Report teacher-prefill/scoring cost and checkpoint selection as well as sampled student tokens; sharing a model lineage does not make teacher inference free. Compare against SFT and RL at matched supervision access and clearly defined budgets, with multiple seeds and retained-capability checks. Dense guidance can inherit flawed solutions or reward stylistic agreement, so it complements rather than replaces [reward-hacking analysis](#rl-reward-hacking) and [independent evaluation](benchmarking-and-evaluation.md#benchmarking-and-evaluation).

### [To be reviewed] Synthetic Data, Rejection Sampling, and Iterative Recipes

Synthetic data describes **how examples are obtained**. Rejection sampling retains candidates using a specified verifier or judge, after which SFT may train on the retained responses. Online RL instead updates from rewarded policy samples using a chosen policy objective; OPD supplies teacher guidance on student states. A system can iterate all three.

**Example.** Generate eight solutions per source problem, execute a checker, retain valid solutions, and SFT the next checkpoint. This changes the training distribution toward solutions accepted by that checker; easy problems may dominate because they produce more accepted samples. Cap or reweight per-problem contributions, inspect diversity and contamination, and compare to training on the same accepted-token budget without another iteration. Verification reduces some label errors but cannot fix an incomplete specification. [Iterative rejection sampling and preference optimization in Llama 3](https://arxiv.org/abs/2407.21783)

Keep model-generated training content, on-policy rollouts, teacher/student roles, and feedback labels versioned. “Self-improvement” is a claim to test against a frozen checkpoint and independent evaluation, rather than an explanation of why a method works.

### [To be reviewed] Stage Gates and Reproducible Comparisons

At each transition, save a checkpoint and evaluate **target capability, general retention, calibration/abstention where relevant, data contamination, and deployment cost**. Use matched evaluation prompts and decoding. Track the full lineage: model/tokenizer revisions, mixture schedules, optimizer resets, trainable modules, sampling policy, teacher/judge/verifier versions, and tool/environment definitions.

**Ablation plan:** (a) original checkpoint, (b) additional tokens at the old mixture, (c) mid-training at the new mixture, (d) behavior-restoring SFT, (e) OPD, and (f) RL when justified by the task. Match the budgets appropriate to the causal question: fixed tokens for a data-mixture comparison, and total compute/cost for a practical pipeline comparison. An equal number of gradient steps is insufficient when sequence lengths, teacher inference, or rollout multiplicities differ.


<a id="llm-infra"></a>
<a id="llm-engineering"></a>
## [To be reviewed] Shared Training Systems

**LLM infra map:** training infrastructure covers precision, state sharding, parallelism, parameter updates, and checkpoint execution in this section. Runtime infrastructure is covered in [Inference and Serving](#llm-inference): [decoding policy](#llm-decoding), [KV cache management](#llm-cache-management), [speculative acceleration](#llm-speculative), and [serving systems](#llm-serving-systems). Transformer [cache mechanics](transformer.md#tf-cache) and [kernel/memory costs](transformer.md#tf-cost) provide the architectural foundations.

<a id="mixed-precision"></a>
### [To be reviewed] Mixed Precision

- **Numerics:** FP16 and BF16 both use 16 bits, but FP16 has a narrower exponent range and more fraction bits; BF16 has FP32-like exponent range with fewer fraction bits. FP16 is more prone to gradient underflow and overflow. Precision choices apply separately to weights, activations, accumulation, and optimizer states. [Mixed precision training](https://arxiv.org/abs/1710.03740)
- **Implementation:** automatic mixed precision selects operator dtypes; numerically sensitive reductions often retain higher precision. FP16 training commonly uses loss scaling, then unscales gradients before clipping and stepping. BF16 generally avoids the FP16-style need for loss scaling, but its larger range does not eliminate rounding error or instability. Check hardware support. [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- **Memory example:** 7 billion weights at two bytes each occupy about 14 GB in decimal units, before gradients, optimizer states, activations, or allocator overhead. Full training memory cannot be inferred from weight dtype alone. Validate a short run against a higher-precision baseline using losses, gradient norms, nonfinite counts, and throughput.

[[arXiv](https://arxiv.org/abs/1710.03740)]


<a id="distributed-training"></a>
### [To be reviewed] Distributed Training

[[Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)][[Multi-GPU Training](https://www.youtube.com/watch?v=gXDsVcY8TXQ)] [[ANLP-Parallelism and Scaling](https://www.youtube.com/watch?v=Mpg1YJfAEH0)] [[知乎](https://zhuanlan.zhihu.com/p/598714869)]
- Data parallel [[Distributed Data Parallel (DDP) using PyTorch](https://www.youtube.com/watch?v=-K3bZYHYHEA&list=PL_lsbAsL_o2CSuhUhJIiW0IkdT5C2wGWj&index=1)] [[Collective Operation](https://en.wikipedia.org/wiki/Collective_operation)]
    - [To be reviewed] `torch.nn.DataParallel` is a single-process replicated-module wrapper. PyTorch recommends DDP for multi-GPU training; avoid the unsupported claim that no one uses DP or that it cannot participate in any manually partitioned model.
        - [To be reviewed] DP scatters inputs, replicates the module, gathers outputs, and reduces replica gradients to the original module. The loss location depends on what is wrapped; a fixed “output gather → loss scatter” sequence is not its general definition. [DataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)
        - Drawbacks
            - Main GPU is a bottleneck.
            - Extra gather/reduction memory and communication on the configured primary device, commonly `device_ids[0]`.
            - Not scalable across machines

    - PyTorch DDP: `torch.nn.DistributedDataParallel`
        - [To be reviewed] DDP synchronizes gradients during backward, but can also communicate during construction and forward, for example parameter verification/initial synchronization and configured buffer broadcasts. SyncBatchNorm or custom distributed modules can add forward collectives. [DDP design](https://docs.pytorch.org/docs/stable/notes/ddp.html)
        - [To be reviewed] DDP reduces gradient buckets across its process group, normally averaging gradients. Identical initial parameters, synchronized updates, and compatible optimizer state are needed to keep replicas aligned. Unequal local valid-token counts require correct weighting for a global token-mean objective.
        - Gradient all-reduce can overlap bucketwise with backpropagation; it need not wait for the whole backward pass. Communication backend, topology, bucket sizes, and compute/communication balance determine scaling.

        ```python

        # [To be reviewed] Single-node Linux CUDA/NCCL example.
        import os
        import torch
        import torch.distributed as dist

        def ddp_setup(rank, world_size):
            """Here rank is also the local GPU index; use global/local ranks separately on multiple nodes."""
            if not dist.is_nccl_available() or not torch.cuda.is_available():
                raise RuntimeError("This example requires CUDA and an NCCL-enabled PyTorch build.")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "12355")
            torch.cuda.set_device(rank)
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

        ```

    - ZeRO (Zero Redundancy Optimizer) reduces replicated model-state storage by sharding across data-parallel workers. It can increase feasible model size, but activations, temporary gathers, and communication still determine peak memory. The stages define which persistent states are partitioned:
        - ZeRO-1 shards optimizer states. Model parameters and gradients remain replicated; implementations synchronize gradients and distribute updated parameter partitions using their chosen collectives.
        - [To be reviewed] ZeRO-2 partitions optimizer states and gradients. Implementations can use reduce-scatter or related collectives, then exchange updated parameter shards as required. The ownership/sharding semantics, not one mandatory “reduce then broadcast” sequence, define the stage.
        - [To be reviewed] ZeRO-3 additionally shards parameters and gathers the needed parameter groups around computation. Persistent states are sharded, but temporary gathered parameters and buffers still consume memory. This is parameter/state sharding in a data-parallel training scheme, distinct from tensor or layer partitioning. [ZeRO](https://arxiv.org/abs/1910.02054)
            - [To be reviewed] Reduce-scatter followed by all-gather can implement an all-reduce over the same participants and compatible partitioning. A reduce-to-one operation is a different collective.

        <div align="center">
            <img src="../figs/zero.png" width="90%">
        </div>

        - [To be reviewed] The ZeRO figure estimates persistent model-state bytes for its stated mixed-precision Adam assumptions. It excludes activations, temporarily gathered parameters, communication buffers, and allocator overhead; its numbers are not a peak-memory guarantee for a current implementation.

    - [To be reviewed] **FSDP** supplies PyTorch-native fully sharded data-parallel training with configurable resharding and precision behavior. Activation checkpointing and CPU offload are separate memory/computation choices; neither is equivalent to full state sharding. Profile their combined behavior under the actual configuration.

    - [To be reviewed] Ordinary replicated data parallelism processes different local data on each replica and synchronizes gradients; it should not be confused with the persistent-state layout of fully sharded training.

- Model parallel
    - [To be reviewed] Pipeline parallelism [[GPipe](https://arxiv.org/abs/1811.06965)]: split layers across multi-GPUs when model is too large.

        - Split consecutive layers into stages and send microbatch activations between them. A simple synchronous fill/drain schedule with $`p`$ balanced stages and $`m`$ microbatches has idealized utilization about $`m/(m+p-1)`$, ignoring communication and imbalance.
        - **Example:** four stages and eight microbatches yield about 8/11 utilization in that simplified model. More microbatches reduce the fractional bubble but can increase activation storage or reduce per-microbatch kernel efficiency. 1F1B schedules trade scheduling complexity and memory against idle time; they are not unique to one model family. [GPipe](https://arxiv.org/abs/1811.06965), [Megatron schedules](https://arxiv.org/abs/2104.04473)

        - [To be reviewed] Pipeline bubbles arise from schedule fill/drain, dependencies, and imbalance. 1F1B schedules predate DeepSeek and are not the defining novelty of DualPipe or another particular pipeline schedule.
    - [To be reviewed] Tensor parallelism

        - Shard individual linear transformations or attention heads across devices. Column- and row-parallel matrix multiplications place collectives at chosen points so the composed operation matches the unsharded mathematical computation.
        - **Example:** split an MLP expansion across four GPUs, then combine partial outputs in its contraction. Tensor parallelism saves per-device weight/activation memory but adds frequent communication, so fast interconnects are valuable. A dimension must satisfy the implementation's partitioning requirements. [Megatron-LM](https://arxiv.org/abs/1909.08053)

        - [To be reviewed] Sequence (activation) parallelism

            - In Megatron-style sequence parallelism, shard sequence-dimension activations for operations such as normalization and dropout around tensor-parallel layers, using appropriate collectives to transition layouts. This reduces replicated activation storage.
            - Distinguish this from **context parallelism**, which partitions long-context attention work and communicates K/V information across sequence partitions. **Example:** merely slicing a sequence and performing independent attention on each slice changes the model by deleting cross-slice dependencies. [Activation-memory analysis](https://arxiv.org/abs/2205.05198)


    - [To be reviewed] Expert (MoE) parallel

        - Place different MoE experts on different devices, route token representations to the selected experts, then return and combine their outputs. The all-to-all routing pattern and expert load balance are central systems costs.
        - **Example:** a top-2 router can select two experts per token while many other expert weights remain resident. Active parameter count describes selected computation, while total expert count still affects storage and communication. Track routing imbalance, capacity handling, and tokens dropped or rerouted. [Switch Transformers](https://jmlr.org/papers/v23/21-0998.html), [DeepSeek-V3](https://arxiv.org/abs/2412.19437)




- [To be reviewed] Hybrid parallel

    - Combine orthogonal partition axes. For a dense setup with independent data, tensor, and pipeline groups, a simple device-count decomposition is $`N=d\times t\times p`$. Context or expert axes require the implementation's specific grouping semantics.
    - **Example:** on 64 GPUs, $`d=8,t=4,p=2`$ is one feasible topology, not a performance recommendation. Match tensor-parallel communication to the fastest links, balance pipeline stages, and profile whether data-parallel communication overlaps computation. [Large-scale parallel training](https://arxiv.org/abs/2104.04473)

- [To be reviewed] Choosing the right parallelism strategy (memory, efficiency, complexity trade-offs)

    - Estimate memory by component: weights, gradients, optimizer states, activations, temporary buffers, and communication buffers. Then locate the bottleneck: capacity, memory bandwidth, compute, interconnect bandwidth, or pipeline imbalance.
    - **Example:** if optimizer state dominates, state sharding can help without introducing tensor collectives inside every layer. If one layer cannot fit on a GPU, layer-level partitioning or another memory strategy is needed. Compare tokens/second, step time, peak memory, communication fraction, and recovery cost at the same global batch and sequence lengths.



| Method                     | What is Split?                  | Type of Parallelism             |
|----------------------------|---------------------------------|---------------------------------|
| **Tensor Parallelism**     | Weights/tensors                 | Model parallelism               |
| **Pipeline Parallelism**   | Layers                          | Model parallelism               |
| **Expert Parallelism (MoE)** | Experts (modules)               | Model parallelism (MoE)         |
| **Data Parallelism**       | Input data batches              | **NOT** model parallelism       |
| **ZeRO Parallelism**       | Optimizer states, gradients, and/or parameters by stage | State/parameter sharding |

<a id="fine-tuning"></a>
<a id="update-parameterization"></a>
### [To be reviewed] Update Parameterization and PEFT

Choose an update parameterization independently of the stage/objective. The mechanisms below can support suitable SFT, RL, or distillation training; compare quality, trainable-state memory, activation cost, checkpoint management, and serving integration.

- Types
    - [To be reviewed] Full fine-tuning vs Parameter-efficient fine-tuning

        - Full fine-tuning updates the backbone; PEFT constrains adaptation to a smaller parameter set. The relevant trade-off includes optimization freedom, data scale, trainable-state memory, checkpoint management, and inference integration.
        - **Example:** a small domain dataset may favor a constrained adapter for regularization and maintainability, while a broad distribution change may justify full adaptation. Compare with matched hyperparameter-search effort and inspect both target performance and retained general capabilities.

    - Parameter-efficient fine-tuning (PEFT) [[paper](https://arxiv.org/pdf/2110.04366)]
        - [To be reviewed] Prompt/Prefix Tuning

            - Prompt tuning learns continuous input prompt embeddings with a frozen backbone. Prefix tuning learns additional continuous prefix states, commonly supplied to attention across layers. Their parameters are not ordinary human-readable prompt text.
            - **Example:** one shared backbone can switch tasks by loading different learned prefixes. Prefix length affects attention work and context usage, and effectiveness depends on model/task scale. [Prefix tuning](https://arxiv.org/abs/2101.00190), [prompt tuning](https://arxiv.org/abs/2104.08691)

        - [To be reviewed] Adapters

            - Insert trainable bottleneck modules inside a frozen model, often with a residual form $`h'=h+W_{\mathrm{up}}\,\sigma(W_{\mathrm{down}}h)`$. The bottleneck limits trainable parameters while permitting nonlinear adaptation.
            - **Example:** maintain separate adapters for legal and biomedical tagging over one encoder. Added modules can add runtime work unless the implementation can fuse or otherwise optimize them. Choose insertion locations and bottleneck size through held-out evaluation. [Adapter paper](https://arxiv.org/abs/1902.00751)

        - [To be reviewed] LoRA [[YouTube](https://www.youtube.com/watch?v=eC6Hd1hFvos&t=959s)]

            - Parameterize a frozen matrix's update as $`\Delta W=(\alpha/r)BA`$, with $`A\in\mathbb{R}^{r\times d_{\mathrm{in}}}`$ and $`B\in\mathbb{R}^{d_{\mathrm{out}}\times r}`$. This trains $`r(d_{\mathrm{in}}+d_{\mathrm{out}})`$ parameters rather than $`d_{\mathrm{in}}d_{\mathrm{out}}`$ for that matrix.
            - **Illustrative calculation:** a 4,096-by-4,096 matrix has 16,777,216 entries; rank 16 adds 131,072 entries, about 0.78%. Rank, scaling, target modules, initialization, and data all matter. A merge can remove adapter matmul overhead when precision and deployment constraints permit it. [LoRA](https://arxiv.org/abs/2106.09685)

        - [To be reviewed] QLoRA trains adapters through a frozen four-bit base representation; its original recipe includes NF4, double quantization, and paged optimizers to manage some memory spikes. Paging is not a guarantee against every OOM, and dequantized computation/activations still consume memory. [QLoRA](https://arxiv.org/abs/2305.14314)
        - [To be reviewed] BitFit tunes existing bias parameters in the studied architecture. A bias-free Transformer may expose few or none of those parameters; apply the method only after inspecting the actual parameterization. [BitFit](https://arxiv.org/abs/2106.10199)
- [To be reviewed] When and why to freeze layers

    - Freezing removes parameter updates and optimizer states for selected weights, but gradients may still need to pass through a frozen operation to reach an earlier trainable component. Activation savings depend on the resulting computation graph.
    - **Example:** training only a classifier on fixed encoder features can cache the features; training input-side prompts through a frozen encoder still requires backpropagation through that encoder. Decide based on adaptation needs, available data, and measured memory, rather than trainable-parameter count alone.

- [To be reviewed] Layer-wise learning rates & layer freezing strategies

    - Layerwise learning-rate decay assigns smaller steps to selected lower layers, while gradual unfreezing changes which layers train over time. These are separate controls over plasticity.
    - **Example:** begin with a task head, then unfreeze upper encoder blocks if the representation is insufficient. Evaluate against a uniform-learning-rate baseline; the assumption that lower layers should move less can fail under a large domain change. Restore optimizer/scheduler behavior correctly when adding newly trainable parameters.

<a id="llm-quantization"></a>
### [To be reviewed] Quantization and Storage Precision

- Quantization maps high-precision values into a lower-bit representation with scales and sometimes zero points. Distinguish weight-only from weight-and-activation quantization, post-training quantization from quantization-aware training, and per-tensor from finer-grained scaling.
- QLoRA stores a frozen base in a low-bit representation and trains higher-precision adapters; it does not directly optimize every four-bit base weight. Quantization metadata, adapters, activations, and caches add memory beyond the nominal bit count. **Example:** compare downstream accuracy and rare-token behavior after quantization, plus throughput on the actual kernels. [QLoRA](https://arxiv.org/abs/2305.14314)

### [To be reviewed] Training Toolchains

- TRL supplies training workflows for language-model post-training, including supervised and preference/RL objectives. Select a trainer based on the actual data and objective: demonstration records, preferred/rejected pairs, and online reward functions are not interchangeable inputs.
- **Reproducibility:** record library versions, model and tokenizer revisions, chat template, loss masks, reward code, and checkpoint-resume state. A supported trainer does not validate the correctness of the training data or reward. [TRL documentation](https://huggingface.co/docs/trl/index)

State/parameter sharding is described once in [Distributed Training](#distributed-training); it can be combined with full tuning or PEFT. Use the actual model/tokenizer revisions, loss masks, sampler, reward/verifier code, and framework versions to define a run.

<a id="optimization-and-efficiency"></a>
### [To be reviewed] Accumulation, Checkpointing, and Communication
- [To be reviewed] Gradient accumulation

    - Accumulate gradients across microbatches before an optimizer step. For equal-sized microbatches, divide each mean loss by the number of accumulation steps to recover the large-batch mean gradient. With variable valid-token counts, normalize by the total relevant token count instead.
    - **Example:** averaging a 10-token microbatch loss and a 100-token microbatch loss weights their tokens differently from a global token mean. Dropout, batch normalization, clipping frequency, and floating-point accumulation can also prevent exact equivalence to one physical large batch. [AMP accumulation example](https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation)

    - [To be reviewed] **Batch accounting (applies to accumulation, independently of checkpointing):** for equal microbatches, global batch size is $`B_{\mathrm{global}}=B_{\mathrm{micro}}\,A\,D`$, where $`A`$ is accumulation steps and $`D`$ is data-parallel replicas. Normalize gradients/loss by the intended total example or valid-token count. “Average gradients” needs that explicit weighting when microbatch lengths differ.

- [To be reviewed] Activation checkpointing
    - Activation checkpointing/ Activation recomputation: Recompute some activations during the backward pass. Store some activations during the forward pass as checkpoints. Discard other activations and recompute them during the backward pass.
        - Save memory but take longer to compute.

- [To be reviewed] Overlap + bucketing

    - Launch communication for ready gradient buckets while backward computation continues. Bucket size trades launch overhead against how early overlap begins. Some distributed implementations can suppress repeated synchronization during accumulation and synchronize on the final microbatch.
    - **Example:** if every microbatch synchronizes despite eight-step accumulation, communication cost can be much larger than expected. Verify traces and framework semantics; excessive communication buffers can also raise peak memory. [DDP design notes](https://docs.pytorch.org/docs/stable/notes/ddp.html)

<a id="llm-inference"></a>
## [To be reviewed] Inference and Serving

- [To be reviewed] Inference Framework

    - Separate **prefill**, which processes known prompt tokens, from **decode**, which appends generated tokens while reading cached keys/values. KV caching removes repeated projection/computation for prior tokens but its storage grows with active sequence length and concurrency.
    - For a conventional cache with $`L`$ layers, $`B`$ sequences, $`T`$ cached tokens, $`H_{\mathrm{KV}}`$ KV heads, head width $`d_h`$, and $`b`$ bytes per element, storage is approximately $`2LBTH_{\mathrm{KV}}d_hb`$. This excludes metadata and temporary buffers; compressed-cache architectures need their own model.
    - **Illustrative calculation:** 32 layers, 8 KV heads, width 128, two-byte cache, and 8,192 tokens consume 1 GiB per sequence. At 32 such concurrent sequences, that alone reaches 32 GiB. GQA reduces cache head count; paged allocation reduces fragmentation. [GQA](https://arxiv.org/abs/2305.13245), [PagedAttention](https://arxiv.org/abs/2309.06180)
    - Speculative decoding drafts several tokens and verifies them with a target model. Exact speculative-sampling algorithms preserve the target distribution under their acceptance/correction rules; accepting arbitrary drafts does not. Speedup depends on acceptance, draft cost, and hardware. [Speculative decoding](https://arxiv.org/abs/2211.17192)


- [To be reviewed] Model Serving

    - Continuous batching admits new requests as others finish, improving utilization for variable-length generation. Admission control should budget prompt work, active KV cache, and expected output work; a fixed request count does not characterize load.
    - **Example:** a few 100k-token prompts can monopolize prefill and harm short-request latency. Compare time to first token, inter-token latency, end-to-end latency, and throughput under specified prompt/output length distributions and arrival rates. Report p50/p95/p99 and the fraction meeting a latency objective, not only peak tokens/second.
    - Prefix caching requires exact compatibility of token prefix, model state, and cache semantics. Quantization and parallelism change both capacity and numerical behavior; benchmark quality as well as latency. Record model/tokenizer versions, sampling settings, cache policy, and error/timeout rates. [Serving system design](https://arxiv.org/abs/2309.06180)

<a id="llm-decoding"></a>
### [To be reviewed] Decoding Strategies: Search, Sampling, and Constraints

A decoding rule converts conditional logits into an actual output policy; it does not change the trained weights. For temperature tau > 0, $`p_\tau(v\mid s)=\exp(z_v/\tau)/\sum_u\exp(z_u/\tau)`$. Tau = 1 retains the original softmax; a zero-temperature setting conventionally selects greedily rather than dividing by zero. Temperature modifies relative probabilities, not a calibrated probability of factual correctness.

| Strategy | Operation | Main trade-off / failure |
| --- | --- | --- |
| Greedy | Choose the highest-scoring allowed token at each step | Cheap, locally optimal choices need not maximize full-sequence probability or task utility |
| Beam search | Keep a limited set of high-scoring partial sequences | Approximate sequence search, with extra state/cache work; width and length scoring matter, and likelihood need not track usefulness |
| Categorical sampling | Sample from the specified conditional distribution | Preserves diversity but can sample low-quality continuations |
| Top-k sampling | Retain k highest-probability tokens and renormalize | Fixed candidate count can be too restrictive or include a weak tail depending on context |
| Nucleus / top-p sampling | Retain the smallest sorted prefix whose cumulative probability reaches the threshold, under a stated tie convention | Adapts candidate count to concentration; still changes the model distribution |
| Constrained decoding | Restrict allowed tokens using a grammar/schema or task-specific state | Can enforce supported structural constraints, but valid JSON or an allowed ID does not establish semantic correctness |

**Example:** probabilities (0.55, 0.25, 0.15, 0.05) give top-k with k = 2 the renormalized distribution (0.6875, 0.3125, 0, 0). Top-p = 0.9 instead retains the first three entries, whose cumulative mass is 0.95. Temperature, penalties, constraints, and truncation filters can interact; record their order and implementation rather than assuming named options commute. [Generation strategies](https://huggingface.co/docs/transformers/generation_strategies), [nucleus sampling](https://arxiv.org/abs/1904.09751)

Specify EOS, stop strings, maximum new tokens, truncation, repetition/length penalties, and invalid-output handling. For beam search, a length-normalized score is a different objective from raw sequence log-likelihood; larger beams do not guarantee better open-ended generation. For reproducibility, record the random seed and full configuration; hardware, kernels, batching, and ties can still affect results. For RL/OPD, the filtered sampling distribution is the behavior policy whose probabilities matter for correction. See [on/off-policy data](#rl-policy-data).

<a id="llm-cache-management"></a>
### [To be reviewed] KV Cache Management and Attention Kernels

The [Transformer cache explanation](transformer.md#tf-cache) derives why unchanged causal-prefix K/V can be reused. The memory formula above counts cache payload; allocation and sharing determine how much device memory is actually required. For unequal sequence lengths, replace B times T by the sum of active cached lengths under the assumed architecture. Prefix/block sharing can reduce physical duplication, while reservation, metadata, and temporary buffers add overhead.

| Mechanism | What it changes | Limit to check |
| --- | --- | --- |
| Dynamic / static cache | Grow storage as needed versus reserve a configured capacity/layout | Static layouts can suit compilation, but unused capacity consumes memory or needs masking |
| Paged KV allocation / PagedAttention | Address logical token blocks through a block mapping instead of requiring one contiguous request allocation | Reduces fragmentation/redundant storage; does not remove the full-context payload or history reads |
| Prefix caching | Reuse compatible K/V for an identical processed prefix | Saves repeated prefix work; unrelated or changed prefixes miss, and continuation work remains |
| GQA / MQA | Share K/V heads structurally | Reduces cache entries at matched shapes, but changing an existing checkpoint requires compatible architecture/training |
| KV quantization / offload | Lower storage precision or move some state away from device memory | Metadata, dequantization, transfers, and quality changes can offset gains |
| Sliding-window cache | Retain only the attention history required by the model's window pattern | Truncating an arbitrary full-attention model changes its computation; hybrid layers need layer-specific policies |

Keep model/adapters, exact token prefix, positions, masks, and execution semantics compatible when reusing cache. Shared services also need appropriate cache isolation. Beam branches need correct cache sharing or reordering; stale beam-to-cache indices can produce plausible but wrong outputs. [Cache strategy reference](https://huggingface.co/docs/transformers/kv_cache), [prefix-caching implementation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

**FlashAttention and PagedAttention address different layers of the problem.** FlashAttention reduces attention-kernel memory traffic and avoids materializing the full score matrix; paged KV management organizes persistent decoding state. They can coexist. Neither is equivalent to changing the sampling rule. See [attention arithmetic and memory](transformer.md#tf-cost) for the kernel-level distinction.

<a id="llm-speculative"></a>
### [To be reviewed] Speculative Decoding: Acceleration Versus a New Output Policy

An exact speculative-sampling algorithm uses a cheap draft distribution q and a target distribution p, then verifies drafted tokens with the target. At a given matched prefix, accept a proposed token x with probability $`\min(1,p(x)/q(x))`$. On rejection, sample from the normalized positive residual $`[p-q]_+`$ when it has nonzero mass. The full algorithm also handles prefixes, accepted blocks, and the additional target sample after a fully accepted block. Simply accepting the target's top few draft matches is not this stochastic correction rule.

**One-step example:** p = (0.6, 0.4), q = (0.8, 0.2). Accepted contributions are (0.6, 0.2); the remaining rejection mass 0.2 is assigned to the second token, recovering p. Exactness concerns the target distribution under the specified algorithm and processing rules, not identical random outputs under arbitrary seeds or an approximate verifier. [Speculative sampling algorithm](https://arxiv.org/abs/2211.17192)

Draft cost, acceptance, verification shape, KV rollback, concurrency, and hardware determine speed. Report end-to-end latency/throughput, not only accepted draft tokens. Quantized or otherwise changed target logits define a potentially different target distribution; distribution preservation does not undo that numerical change.

<a id="llm-serving-systems"></a>
### [To be reviewed] Serving: Scheduling, Parallelism, and Performance Evidence

**Scheduling:** continuous/iteration-level batching admits and retires requests during generation. Chunked prefill divides long prompt processing into schedulable pieces so one prompt need not monopolize a scheduling interval. Prefill/decode disaggregation uses separate execution pools, requiring KV transfer and additional scheduling. These choices can improve a particular latency/throughput frontier, but communication, queueing, and workload mix can reverse the benefit. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), [DistServe](https://arxiv.org/abs/2401.09670)

**Parallelism and precision:** inference tensor/pipeline/expert partitioning reuse some ideas from [training parallelism](#distributed-training), but decode microbatches, communication frequency, and pipeline occupancy differ. Weight quantization and KV quantization save different memory components. Profile actual kernels and communication instead of choosing device count or dtype from parameter count alone.

**Measure the user-visible result:** separate queueing, prefill, and generated-token time; report time to first token (TTFT), inter-token latency, end-to-end latency, throughput, and success under stated latency objectives. Prefill is often compute-heavy and small-batch decode often bandwidth-sensitive, but sequence length, batching, architecture, and hardware determine the actual bottleneck. A token/s number without prompt/output lengths and arrival rates is not a comparable service result.

**Example investigation:** a service OOMs at high concurrency while short requests miss their TTFT target. First account for active/reserved KV bytes and output-length tails; separately inspect prefill queueing and scheduling. Test a cache-allocation change, prefix reuse, and a prefill scheduling change individually, holding model, quality checks, request trace, and output policy fixed. Count rejected requests, failures, preemptions/recomputation, and draft/KV-transfer costs. A successful run at low offered load does not establish peak reliable capacity.


<a id="evaluation-and-monitoring"></a>
## [To be reviewed] Evaluation and Monitoring

For benchmark families, static/dynamic/interactive designs, saturation, contamination, and statistical comparisons, see [Benchmarking & Evaluation](benchmarking-and-evaluation.md). The [reproducibility protocol](benchmarking-and-evaluation.md#eval-protocol) connects these metrics to a complete experimental setup.

- Evaluation metrics
    - [To be reviewed] Perplexity

        - Perplexity is $`\exp[-N^{-1}\sum_{t=1}^N\log p_\theta(x_t\mid x_{<t})]`$ for a specified tokenization and context protocol. Use valid target tokens only; masking, document boundaries, and truncation change the quantity being measured.
        - **Example:** lower held-out perplexity may coexist with worse instruction following. Compare like-for-like tokenizers and corpora, and pair perplexity with task metrics. If tokenizers differ, report a common text-normalized likelihood measure with explicit conventions.

    - [To be reviewed] Human-evaluated metrics (helpfulness, harmlessness, alignment)

        - Define rubrics for correctness, groundedness, instruction adherence, usefulness, and relevant harm categories. Blind and randomize model identity and response order; allow ties and report annotator disagreement.
        - **Example:** evaluate whether a cited answer is supported by the supplied source separately from whether its claim happens to be true. Use paired uncertainty estimates clustered by prompt when multiple samples share a question. Preference scores depend on the judge population and rubric. [Evaluation practices](https://arxiv.org/abs/2211.09110)

    - [To be reviewed] Automated evaluation metrics (ROUGE, BLEU, BERTScore)

        - BLEU measures modified n-gram precision with a brevity penalty; ROUGE variants emphasize lexical overlap or subsequences; BERTScore aligns contextual token embeddings. Their scoring conventions differ, and they are not direct measures of factual correctness.
        - **Example:** “the treatment increased mortality” and “the treatment decreased mortality” share most words. A high overlap score can miss the consequential error. For verifiable tasks, add execution, exact symbolic checks, or evidence-based factual assessment, and correlate any automated judge with human audits. [BLEU](https://aclanthology.org/P02-1040/), [ROUGE](https://aclanthology.org/W04-1013/), [BERTScore](https://arxiv.org/abs/1904.09675)

- Monitoring and Logging
    - [To be reviewed] wandb

        - Track data/model/configuration versions, optimizer and scheduler state, loss by task and length, gradient norms, nonfinite events, throughput, and memory. A run dashboard is useful only if its logged quantities have stable definitions.
        - **Example:** a sudden loss jump can come from a changed data mixture rather than optimizer instability. Log consumed-token distributions and checkpoints around transitions. Avoid putting private training examples into a shared dashboard unless the data policy permits it. [W&B experiment tracking](https://docs.wandb.ai/models/track)

    - [To be reviewed] TensorBoard

        - Use scalar series for losses and learning rates, histograms for diagnostic distributions, and a profiler for operator, memory, and communication behavior. Tag experiments consistently and preserve raw configuration alongside plots.
        - **Example:** two runs can have the same training loss but very different useful tokens/second because one spends time on padding or synchronization. A wall-clock and profiler comparison distinguishes the bottleneck. [TensorBoard guide](https://www.tensorflow.org/tensorboard/get_started)

<a id="recent-interesting-papers"></a>
## [To be reviewed] Recent interesting papers

This is a dated reading list selected on **2026-10-26**. Years indicate first public report dates. Reported results belong to each paper's model, data, hardware, and evaluation protocol; the questions below are suggested checks and experiments, not additional findings claimed by the papers.

| Paper | Main contribution to inspect | Research question / replication check |
| --- | --- | --- |
| [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024) | A large MoE training system combining Multi-head Latent Attention, expert routing/load balancing, and multi-token prediction. | Separate active compute from total parameter storage. Which quality/throughput gains survive when data, training tokens, hardware topology, and inference budget are matched? Check what costs the reported training-compute accounting includes. |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025) | Studies reinforcement-learning-driven reasoning, an RL-only R1-Zero setup, and an R1 pipeline incorporating cold-start and other training stages. | Separate R1-Zero observations from the complete R1 recipe. Reproduce verifier definitions, sampling settings, output-length distributions, contamination checks, and gains at matched test-time compute. |
| [DAPO](https://arxiv.org/abs/2503.14476) (2025) | An openly described large-scale RL system emphasizing optimization and sampling details needed for reasoning-model training. | Ablate clipping, sampling, and length-handling changes under a fixed rollout-token budget. Does improved reward correspond to independently checked solution accuracy and transfer beyond the training-style benchmark? |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (2025) | Dense and MoE model families with integrated thinking/non-thinking behavior and inference-budget control. | Compare modes on the same questions and account for generated tokens, latency, and quality. Does a budget policy allocate effort to difficult examples, or simply produce longer outputs? |
| [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) (2026) | Joint text/vision training and an Agent Swarm framework for parallel task execution. | Separate modality-training gains from orchestration gains. Reevaluate with matched total tokens, tool calls, task permissions, and concurrency; distinguish reduced wall-clock latency from reduced total compute. |

For each paper, record **claim → assumptions → comparison baseline → ablation → uncertainty → failure cases**. Prefer the released implementation and the specific report revision used for reproduction; benchmark rank alone is insufficient to identify which mechanism caused an improvement.

## Related chapters

- [Agents](agents.md#agents)
- [RSI & Harness](rsi-and-harness.md#rsi-and-harness)
- [Benchmarking & Evaluation](benchmarking-and-evaluation.md#benchmarking-and-evaluation)

<a id="chunking-and-vectorization"></a>
<a id="embedding-models"></a>
<a id="generation"></a>
<a id="rag"></a>
<a id="re-ranking"></a>
<a id="retrieval"></a>
<a id="to-be-reviewed-chunking-and-vectorization"></a>
<a id="to-be-reviewed-embedding-models"></a>
<a id="to-be-reviewed-generation"></a>
<a id="to-be-reviewed-re-ranking"></a>
<a id="to-be-reviewed-retrieval"></a>

- [RAG](rag.md#rag)

<a id="components-of-qa-systems"></a>
<a id="evaluation"></a>
<a id="qa-system"></a>
<a id="to-be-reviewed-components-of-qa-systems"></a>
<a id="to-be-reviewed-evaluation"></a>
<a id="to-be-reviewed-types-of-qa-systems"></a>
<a id="types-of-qa-systems"></a>

- [QA systems](qa.md#qa-system)

<a id="reasoning"></a>
<a id="to-be-reviewed-reasoning"></a>

- [Reasoning](reasoning.md#reasoning)
