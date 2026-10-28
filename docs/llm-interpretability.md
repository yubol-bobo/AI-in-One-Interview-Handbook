<a id="llm-interpretability"></a>
# [To be reviewed] LLM Interpretability: Features, Causal Interventions, and Control

[Handbook contents](../README.md#table-of-contents) | [Previous: Large Language Models](large-language-models.md) | [Next: Agents](agents.md)

## Contents

- [What Kind of Explanation Are We Seeking?](#interp-scope)
- [Features, Neurons, Directions, and Superposition](#interp-features)
- [Probes, Logit Lens, and Observational Evidence](#interp-probes)
- [Activation Patching: A Controlled Internal Intervention](#interp-patching)
    - [Metrics, Corruptions, and a Worked Example](#interp-patching-metrics)
    - [Necessity, Sufficiency, and Redundant Paths](#interp-patching-limits)
- [Attribution Patching and Path Patching](#interp-paths)
- [Activation Steering: Directions, Coefficients, and Side Effects](#interp-steering)
- [Sparse Autoencoders: Objective and Dictionary Geometry](#interp-sae)
- [Evaluating SAE Features and Failure Modes](#interp-sae-evaluation)
- [SAE Feature Interventions and Reconstruction Error](#interp-feature-editing)
- [Circuits, Transcoders, and Attribution Graphs](#interp-circuits)
- [An End-to-End Research Experiment](#interp-experiment)
- [Implementation and Cache Correctness](#interp-implementation)
- [Frequent Interview Questions](#interp-interview)
- [Tools and Primary Reading](#interp-reading)

<a id="interp-scope"></a>
## [To be reviewed] What Kind of Explanation Are We Seeking?

LLM interpretability asks which representations and computations produce a specified behavior. **Mechanistic interpretability** seeks an account in terms of internal components and their interactions, supported by interventions. Behavioral evaluation asks what the system does; input attribution assigns influence under a perturbation rule; a generated rationale is another model output. These establish different claims. See [general XAI](machine-learning.md#xai), [Transformer computation](transformer.md#transformer), and [evaluation](benchmarking-and-evaluation.md#benchmarking-and-evaluation).

State the **explanandum** first: “Why does this checkpoint prefer Mary to John at the final position of these indirect-object prompts?” is more testable than “Where is reasoning?” Scope the explanation to a model, activation site, input distribution, metric, and intervention family. A small-task explanation is not automatically an explanation of the whole model.

| Method | Main question | Evidence supplied | What it does not establish alone |
| --- | --- | --- | --- |
| Probe / visualization | What is decodable or correlated with behavior? | Association under a stated readout | That the model uses that information |
| Activation patching / ablation | What changes after replacing this state? | A model-internal causal effect under the chosen intervention | A unique, minimal mechanism or real-world causal effect |
| Attribution patching | Which finite patches might matter? | A local gradient approximation for screening | The effect of every finite replacement |
| Steering | Can changing a direction alter behavior? | Controllability and measured side effects | A single concept or the natural cause of behavior |
| SAE dictionary learning | Can activations be decomposed into sparse components? | A candidate feature decomposition | A complete, uniquely correct concept ontology |
| Circuit analysis | How do components combine to implement behavior? | Tested hypotheses about computations and paths | Global completeness outside the tested setting |

**Generated reasoning is a distinct observable.** A coherent rationale need not report a bias that influenced the answer. A useful reasoning trace can improve performance without being an exhaustive internal transcript. Test faithfulness rather than assuming universal transparency or universal unfaithfulness. [CoT faithfulness experiments](https://arxiv.org/abs/2305.04388), [reasoning chapter](reasoning.md#reasoning).

<a id="interp-features"></a>
## [To be reviewed] Features, Neurons, Directions, and Superposition

A **neuron** is an architectural coordinate; an **activation** is its value, or a vector/tensor of values, on an input. A **feature** can mean a task property, a represented variable, or a dictionary latent. Specify the meaning. A direction is a vector in activation space; a feature activation is a scalar coefficient varying across tokens and contexts. An attention head is a computation with Q/K/V projections, not a single semantic feature.

**Polysemanticity** means a neuron responds to multiple apparently unrelated patterns. **Superposition** is an account in which more features than dimensions are encoded using non-orthogonal directions, with interference. Sparse activity can make this useful despite collisions. Toy-model evidence motivates looking beyond individual neurons; it does not prove that every concept in every LLM has a stable linear direction. [Toy Models of Superposition](https://arxiv.org/abs/2209.10652)

**Geometric example.** In two dimensions, choose three unit directions $`d_1=(1,0)`$, $`d_2=(0,1)`$, and $`d_3=(1,1)/\sqrt{2}`$. The three one-feature states are distinguishable, but $`d_1+d_2=\sqrt{2}d_3`$. Arbitrary combinations are ambiguous. Sparsity and distributional assumptions help inference; an overcomplete dictionary is not an invertible independent basis.

A “French-language feature” label may summarize strong activations while also responding to quotations, names, or formatting. Its name is a hypothesis, not its exact mathematical support. Localized and distributed representations can coexist. The [SAE section](#interp-sae) separates the detector from the decoded direction.

<a id="interp-probes"></a>
## [To be reviewed] Probes, Logit Lens, and Observational Evidence

A linear probe might predict a label using $`\hat p(y=1\mid h)=\sigma(w^Th+b)`$. High test accuracy shows accessibility to that readout on that distribution. Lexical shortcuts, memorization, and correlated labels can explain it. Hold out entities/templates, compare readout capacities and appropriate control tasks, and separate probe fitting from intervention evaluation. A random-representation baseline and a learned-label control task ask different questions. [Probe controls](https://arxiv.org/abs/1909.03368)

A **logit lens** applies an output-style normalization/readout to intermediate residual states. The final unembedding was trained for final-layer representations; earlier layers need not share exactly that geometry. A **tuned lens** learns layerwise affine translators, improving the readout in its studied settings while introducing an additional fitted model. These views do not mean a layer has committed to a future token. [Tuned Lens](https://arxiv.org/abs/2303.08112)

**Example:** a probe decodes nationality from an early state, but one patch does not change the answer. Downstream non-use, redundant information, an unsuitable intervention, or an insensitive metric could explain the result. Probe success plus one null patch cannot distinguish them. A direct unembedding contribution also omits later computation and changing normalization; it is not automatically a total causal effect.

<a id="interp-patching"></a>
## [To be reviewed] Activation Patching: A Controlled Internal Intervention

Run the frozen model on **clean** and **corrupted** inputs, cache activations, replace selected corrupted-run activations with clean counterparts, and measure the output change after downstream recomputation. Sites include residual vectors, MLP outputs, head outputs, or specified subspaces at particular layers/positions. “Layer 12” is insufficient: pre/post normalization and pre/post residual addition are different sites. [Patching introduction](https://arxiv.org/html/2404.15255v1)

A denoising patch sets $`h_{\ell,t}^{\mathrm{patched}}\leftarrow h_{\ell,t}(x_{\mathrm{clean}})`$ during the corrupted run. Record all other interventions and execution settings. This manipulates the computational graph; it does not establish that a corresponding real-world variable was randomized.

**Concrete task:** compare “When John and Mary went to the store, John gave a bottle of milk to” with a version having Mary as the giver. Measure preference for Mary over John at the next position, after checking the tokenizer, candidate tokens, and baseline behavior. Swap names, roles, and templates across examples. Align by semantic role and actual tokenization rather than assuming every name is one token.

Screen layer/position sites, narrow to head/MLP contributions, then test a computational hypothesis on held-out pairs. Transplanting the whole final state can almost directly transplant an answer; that is a useful implementation control, not localization of the earlier computation that produced it. [Method and metric sensitivity](https://arxiv.org/html/2309.16042v2)

<a id="interp-patching-metrics"></a>
### [To be reviewed] Metrics, Corruptions, and a Worked Example

For single-token target A and foil B at the same position, use logit difference $`m=z_A-z_B`$. Under one softmax this equals $`\log p(A)-\log p(B)`$, even when other vocabulary mass changes. Target probability can be affected by that competing mass; accuracy can miss sub-threshold changes. For multi-token answers, score full conditional sequences under a declared protocol, rather than one arbitrary subtoken.

A common restoration score is $`R=(m_{\mathrm{patch}}-m_{\mathrm{corrupt}})/(m_{\mathrm{clean}}-m_{\mathrm{corrupt}})`$. Clean = 2, corrupt = −2, patched = 1 gives R = 0.75. This is an illustrative calculation, not a measured checkpoint result. R can be negative or exceed 1; it is not a calibrated “percentage of the reasoning circuit.” Small denominators are unstable. A ratio of average effects and an average of pairwise ratios weight examples differently.

Choose corruptions that change the intended variable while controlling nuisance structure. Natural resampling and Gaussian noise are different interventions; neither guarantees an in-distribution mixed state. Repeat across corruption families and record baseline gaps. Use discovery/confirmation splits and uncertainty over independent examples when screening many sites. [Metric/corruption study](https://arxiv.org/html/2309.16042v2), [interpretation guidance](https://arxiv.org/html/2404.15255v1)

<a id="interp-patching-limits"></a>
### [To be reviewed] Necessity, Sufficiency, and Redundant Paths

**Denoising** inserts clean state into a corrupted background to test restoration. **Noising** inserts corrupted state into a clean background to test damage. Zero, mean, and resample ablations use other baselines; none universally represents absence of information.

Restoration suggests sufficiency **in that background**; damage supports conditional importance under that noising rule. Neither gives universal logical necessity/sufficiency. Deleting one of two OR-like redundant routes can do little; restoring one of two missing AND-like inputs can also do little. Sites interact, and positive/negative contributions can cancel. Test joint interventions and backup behavior before declaring a component irrelevant. [Redundancy examples](https://arxiv.org/html/2404.15255v1)

**Localization is not an algorithmic explanation.** A site may relay earlier computation; an ablation may trigger recovery. Predict responses to controlled input changes, identify dependencies, and test those predictions. Report nulls as conditional nulls, rather than proof that a feature is absent.

<a id="interp-paths"></a>
## [To be reviewed] Attribution Patching and Path Patching

**Attribution patching** approximates a finite replacement by a first-order expansion at a specified background: $`\Delta m\approx(\nabla_hm|_{h_{\mathrm{corrupt}}})^T(h_{\mathrm{clean}}-h_{\mathrm{corrupt}})`$. Gradient passes can rank many sites, with costs for graph and activation storage. This differs from directly measuring every patch. [AtP*](https://arxiv.org/abs/2403.00745)

For $`m(h)=h^2`$, patching 0 to 1 changes the metric by 1, while the first-order estimate at zero is 0. Saturation, interactions, and cancellation can produce false negatives. Confirm important candidates and a sample of rejected candidates with exact finite patches; improved screening variants do not remove all approximation error.

**Path patching** restricts routes an intervention can travel along, distinguishing “A affects the output” from “A affects it through B.” Define the graph, sender/receiver sites, frozen values, and downstream recomputation. A node edit usually affects all outgoing routes; an edge/path experiment asks a different counterfactual. Automated discovery searches task-conditioned graphs whose results depend on units, objectives, and pruning thresholds. [Automated Circuit Discovery](https://arxiv.org/abs/2304.14997)

<a id="interp-steering"></a>
## [To be reviewed] Activation Steering: Directions, Coefficients, and Side Effects

Steering changes runtime activations to influence behavior, often without changing weights. A contrastive mean-difference recipe uses matched positive/negative examples and an explicit extraction site:

$`v_\ell=\frac{1}{N}\sum_{i=1}^{N}(h_\ell(x_i^+)-h_\ell(x_i^-)),\qquad h'_{\ell,t}=h_{\ell,t}+\alpha v_\ell`$.

Contrastive Activation Addition (CAA) studies such differences and adds/subtracts vectors during generation. Extraction/injection positions, prompt templates, and coefficient conventions are part of the recipe; this formula does not reproduce every steering method. [CAA](https://aclanthology.org/2024.acl-long.828/), [Activation Addition](https://arxiv.org/abs/2308.10248)

**Matched example:** contrast comparable responses that do or do not express uncertainty appropriately. If all positives share one topic, length, or answer label, the direction can encode those shortcuts. A useful behavioral handle need not identify a unique uncertainty concept.

Adding $`\alpha v`$ and $`\alpha v/\|v\|_2`$ uses different units. Record vector norm, residual scale, layer, token mask, and centering/normalization. Sweep positive, negative, and zero coefficients; compare norm-matched random directions, shuffled contrast labels, and prompt baselines. Measure correctness, fluency, calibration, unrelated skills, and held-out topic transfer alongside the intended effect. Repeating a judge-preferred catchphrase is weak control evidence.

Probe weights, steering vectors, and SAE decoder directions solve different problems. Probe weights read from a space; decoder directions write into it. Large perturbations changing behavior do not establish that natural examples use exactly that route or that the effect is selective at small doses.

Steering edits runtime state; [SFT/RL](large-language-models.md#sft-rl-opd-comparison) update parameters. They can be combined. A vector can still require fitting data and model-specific access; no weight update does not mean zero fitting/deployment cost. Hidden-state steering also differs from output-logit edits or grammar constraints.

<a id="interp-sae"></a>
## [To be reviewed] Sparse Autoencoders: Objective and Dictionary Geometry

An **SAE (Sparse Autoencoder)** is an auxiliary model trained on selected activations from a usually frozen language model. It maps a d-dimensional activation to a sparse code of n candidate features, often with $`n>d`$, then reconstructs the activation. Overcompleteness and few active latents coexist. This is different from sparse model weights, sparse attention such as [NSA](transformer.md#tf-nsa), and [MLA's trained KV compression](transformer.md#tf-mla).

A representative centered ReLU SAE uses column-vector conventions:

$`f(h)=\operatorname{ReLU}(W_{\mathrm{enc}}(h-b_{\mathrm{dec}})+b_{\mathrm{enc}}),\qquad \hat h=b_{\mathrm{dec}}+W_{\mathrm{dec}}f(h)`$,

$`\mathcal L=\mathbb E_h[\|h-\hat h\|_2^2+\lambda\|f(h)\|_1]`$.

Here $`W_{\mathrm{enc}}\in\mathbb R^{n\times d}`$, $`W_{\mathrm{dec}}\in\mathbb R^{d\times n}`$, and $`f(h)\in\mathbb R^n`$. Decoder column $`d_j=W_{\mathrm{dec}}[:,j]`$ is a write direction; encoder row j plus bias is its detector. Encoder and decoder need not be transposes or inverses. Constrain decoder norms or use a norm-aware penalty: otherwise shrinking activations while enlarging decoder columns can evade L1 without changing reconstruction. [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html), [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

A **TopK SAE** retains a specified number of latent preactivations rather than relying only on L1 for sparsity; gated/thresholded variants make other trade-offs. Specify the activation function, normalization, reduction, and dead-latent treatment. TopK constrains sparsity, not semantic quality. [Scaling and evaluating SAEs](https://arxiv.org/html/2406.04093v1)

**Toy decomposition:** with columns $`d_1=(1,0)`$, $`d_2=(0,1)`$, and $`d_3=(1,1)/\sqrt2`$, code (2,0,0) reconstructs (2,0). Both codes $`(0,0,\sqrt2)`$ and (1,1,0) reconstruct (1,1), but their L1 norms are $`\sqrt2`$ and 2. The objective can prefer a combined-pattern latent over a compositional pair. Sparse reconstruction need not discover the researcher's preferred concepts.

<a id="interp-sae-evaluation"></a>
## [To be reviewed] Evaluating SAE Features and Failure Modes

Evaluate several axes rather than relying on attractive top-activation examples.

| Axis | What to measure | Failure to investigate |
| --- | --- | --- |
| Reconstruction | Held-out error under fixed scaling and relevant baselines | Average error hides behaviorally important directions |
| Downstream fidelity | Substitute reconstructions; measure LM loss and target behavior | Frequent variance is reconstructed while crucial computation changes |
| Sparsity/utilization | Active-count distribution $`\lVert f(h)\rVert_0`$, firing rates, dead latents | Mean sparsity hides tails; rare/dead features distort coverage |
| Semantic selectivity | Held-out positives, hard negatives, activation prediction | A vivid label fits a cherry-picked tail |
| Causal selectivity | Feature edits, dose response, unrelated-capability controls | Plausible detectors have broad write effects |
| Stability/coverage | Vary seed, width, sparsity, domain | Feature IDs, granularity, or coverage are unstable |

The reconstruction–sparsity frontier is useful engineering evidence, not a ranking of mechanistic truth. Features can split into narrower cases as the dictionary changes. **Absorption** can make a general feature stop firing on examples represented by specific child features; absence of its activation then does not establish absence of the concept. [SAE evaluation](https://arxiv.org/html/2406.04093v1), [splitting and absorption, v6](https://arxiv.org/abs/2409.14507v6)

**Example:** a “starts with S” latent may fire for many S-initial words but miss a subset handled by specific lexical features. Best-positive lists hide those false negatives. Evaluate controlled vocabulary, changed contexts, and the full activation distribution. Labels such as deception or confidence have harder ground-truth problems; judge agreement remains a measurement under that judge.

Reconstruction residuals can retain important signals. A larger SAE can expose structure while changing granularity; dictionary width alone does not establish “all features found.” Unsupervised training avoids labeling every activation, but human/automated feature explanations introduce assumptions. Separate explanation-generation examples from semantic validation.

<a id="interp-feature-editing"></a>
## [To be reviewed] SAE Feature Interventions and Reconstruction Error

Write $`h=b_{\mathrm{dec}}+\sum_j f_j(h)d_j+e`$, with reconstruction residual $`e=h-\hat h`$. A coordinate edit keeping the original other coefficients and residual is:

$`h'=h+(a-f_j(h))d_j`$.

Setting a = 0 subtracts that decoded contribution; larger a amplifies it. Replacing h with an edited reconstruction drops e unless it is added back. These are different interventions. Preserve the residual when the experiment is intended to change only one decoded contribution, and separately test reconstruction-only replacement.

**Numerical example:** $`f_1=2`$, $`d_1=(1,0)`$, other decoded contributions (0,1), and $`e=(0.1,-0.2)`$ give h = (2.1,0.8). Ablating feature 1 with e retained gives (0.1,0.8); dropping e gives (0,1). Their output difference cannot be assigned solely to feature 1.

Non-orthogonal columns mean re-encoding $`h'`$ can change multiple detector outputs. Editing one coordinate in the original decomposition does not hold every re-encoded coordinate fixed. Large/negative edits can also leave the training distribution. Compare matched random/neighboring directions and inspect collateral behavior. The scaling study illustrates useful control and broad side effects. [Feature manipulation and limits](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

<a id="interp-circuits"></a>
## [To be reviewed] Circuits, Transcoders, and Attribution Graphs

A circuit is a proposed computation implemented by interacting components, not a ranked list. In an indirect-object task, a hypothesis might involve tracking names, suppressing the giver, and moving the remaining candidate toward the output. Stronger evidence predicts interventions and transfers across controlled variations, rather than assigning roles from a heatmap.

An SAE reconstructs activations at a chosen site. A **transcoder** approximates a module's input-to-output computation using sparse features; cross-layer variants can connect reads to later writes. Such replacement models make interactions easier to analyze but introduce approximation error. [Circuit Tracing methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)

The cited 2025 attribution-graph method uses cross-layer transcoders, freezes attention patterns and normalization denominators for input-local linear attribution, and includes error nodes for unexplained MLP outputs. Edges are attributions in that replacement, not automatically exact causal edges for arbitrary changes to the original model. Validate graph hypotheses with original-model interventions. Work modeling attention interactions addresses an additional part of the problem. [Attention feature interactions](https://transformer-circuits.pub/2025/attention-qk/index.html)

Report original versus replacement behavior, error contributions, pruning thresholds, held-out performance, contradictory interventions, and backup circuits. A small graph can suffice on a narrow dataset while omitting alternative strategies. Model-internal causal evidence, semantic correctness of labels, and circuit generalization are separate questions.

<a id="interp-experiment"></a>
## [To be reviewed] An End-to-End Research Experiment

**Question:** does the model use an uncertainty-related representation to abstain when a passage lacks the answer? Start with an inspectable checkpoint. This is a proposed design, not a claim that a universal abstention feature exists.

1. **Define task and splits.** Pair answerable/unanswerable cases while controlling topic, length, wording, and leakage. Hold out passage families/entities. Report correctness, useful abstention, and over-abstention separately; refusing everything is not success.
2. **Observe.** Fit simple probes or inspect SAE candidates at declared sites. Distinguish missing passage evidence, missing parametric knowledge, and generic refusal. A match to “unknown” is not automatically semantic uncertainty.
3. **Patch.** Change evidence availability with minimal unrelated variation. Test denoising/noising and confirm sites on new prompts. Check whether a patch transfers answer content rather than only evidence-status information.
4. **Test feature labels.** Include hard negatives: quoted uncertainty, a certain answer followed by “I am unsure,” known answers absent from the passage, and unavailable answers with confident wording. Compare decoder edits with full-state patches and reconstruction controls.
5. **Steer.** Sweep signed coefficients and token scopes; compare prompt instructions, random directions, shuffled labels, and no intervention. Use free generation and fixed-continuation scoring. Measure correctness, calibration, repetition, and unrelated regressions at matched decoding budgets.
6. **Trace and falsify.** Predict upstream/downstream routes, test paths and joint ablations, and seek counterexamples. Bootstrap independent passage/question families, report run/SAE-seed variability, and reserve confirmation data until choices are fixed.

A detector without causal effect is a useful negative result. A steering handle with an unclear natural mechanism is a control result. Predicting held-out interventions supports a stronger explanation. Keep these conclusions separate. [RAG evidence](rag.md#rag), [grader validity](benchmarking-and-evaluation.md#eval-graders), [statistics](statistics.md#statistics).

<a id="interp-implementation"></a>
## [To be reviewed] Implementation and Cache Correctness

Record model/tokenizer/adapter revisions, prompt serialization, hook shape, layer/position/normalization boundary, candidate IDs, metric, coefficient/dtype, SAE weights/scaling, decoding settings, and split IDs. Before versus after an MLP projection edits different spaces. Name axes, especially with grouped heads or sharding.

Distinguish [prefill and decode](large-language-models.md#llm-prefill). Prompt-wide steering differs from steering newly generated positions only. An activation change can alter downstream layers' K/V. Reusing downstream state from an unsteered prompt after steering that prompt mixes histories. Decode-only steering can intentionally retain an unsteered prefill cache; document the policy. Editing an [MLA latent](transformer.md#tf-mla) also differs from editing the full residual stream.

**Implementation checks:** zero-coefficient and same-activation patches should reproduce baseline within numerical tolerance. Restrict hooks to intended positions/examples, avoid unintended mutation of shared cached tensors, and remove temporary hooks. Compare cached/uncached execution with identical intervention scope, positions, and dropout settings. Once free generations diverge, equal token indices no longer mean equal histories; fixed-continuation scoring answers a complementary question.

Shape compatibility alone does not validate an SAE for another model revision or normalization. Fused kernels can hide sites exposed by eager references, and instrumentation can change precision/cost. Verify the intervention actually operates where intended before interpreting null results or plausible outputs.

<a id="interp-interview"></a>
## [To be reviewed] Frequent Interview Questions

| Question | Strong answer / experiment to prepare |
| --- | --- |
| Is a feature the same as a neuron? | Define detector/coefficient/direction; superposition is a motivated account, not a universal theorem. |
| Why does a good probe not establish causal use? | It is an auxiliary readout. Test interventions, redundancy, and transfer while controlling capacity/shortcuts. |
| Does restoration 0.9 mean 90% of the mechanism was found? | It normalizes one metric/corruption. Inspect raw gaps, denominators, interactions, and held-out effects. |
| Why can denoising work while noising does little? | Different backgrounds and redundant routes. Construct an OR-like example and test joint edits. |
| Activation versus attribution patching? | Finite replacement versus a gradient approximation. Give a nonlinear counterexample and confirmation plan. |
| How can SAE latents outnumber hidden dimensions? | It is an overcomplete sparse dictionary, not an independent basis. Explain collisions and non-unique semantic recovery. |
| Which SAE vector writes a feature? | The scaled decoder direction; the encoder row detects it. Preserve reconstruction error and test re-encoding effects. |
| Why can strong steering weaken explanatory evidence? | Large off-distribution perturbations may act nonspecifically. Test signed dose response and matched controls. |
| Are attribution graphs literal model thoughts? | They have local replacement/attribution assumptions. Validate predictions in the original model. |
| Can steering reuse all old KV? | Only where state stays consistent with the intervention; prompt-wide changes can invalidate downstream cache. |

<a id="interp-reading"></a>
## [To be reviewed] Tools and Primary Reading

[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) supports internal-model analysis; [nnsight](https://nnsight.net/) provides tracing/interventions; [SAELens](https://decoderesearch.github.io/SAELens/latest/) supports SAE training/analysis. Pin compatible versions and actual model/hook support. Tools and dictionaries enable experiments; they do not validate semantic claims. This chapter does not claim to reproduce model-training results.

A focused reading sequence is **superposition → patching/metrics → contrastive steering → SAE evaluation → circuits**. Original studies define particular recipes; compare them using their equations and controls. For parameter updates return to [post-training](large-language-models.md#post-training); for tool trajectories continue to [Agents](agents.md#agents).
