<a id="benchmarking-and-evaluation"></a>
# [To be reviewed] Benchmarking & Evaluation

[Handbook contents](../README.md#table-of-contents) | [Previous: Large Language Models](large-language-models.md) | [Next: RAG](rag.md)

## Contents

- [What is being evaluated?](#eval-unit)
- [Static, dynamic, and interactive benchmarks](#eval-taxonomy)
- [Representative LLM benchmarks](#eval-llm-benchmarks)
- [Representative agent benchmarks](#eval-agent-benchmarks)
- [Version and maintenance changes](#eval-version-changes)
- [Benchmark saturation](#eval-saturation)
- [Metrics, reliability, and cost](#eval-metrics)
- [Judges, graders, and construct validity](#eval-graders)
- [Contamination and evaluation integrity](#eval-contamination)
- [Uncertainty and fair comparisons](#eval-statistics)
- [Reproducibility protocol](#eval-protocol)
- [A concrete evaluation portfolio](#eval-portfolio)
- [Reading a leaderboard as a researcher](#eval-reading)

This chapter treats evaluation as a measurement problem: define a capability or deployment outcome, choose tasks that represent it, and justify the inference drawn from the scores. The catalog includes widely used and historically influential LLM/agent evaluations alongside newer extensions; it is a representative map, not a measured ranking of benchmark adoption. **Sources and version-sensitive notes were checked on 2026-10-27.** Pin the actual release used in an experiment; benchmark names and live leaderboards alone are insufficient specifications.

For architectures and training, see [LLMs](large-language-models.md). For task-specific mechanisms and metrics, see [RAG](rag.md), [QA](qa.md#evaluation), [Reasoning](reasoning.md), and [recommendation evaluation](recommendation-systems.md#rec-evaluation). The [Statistics](statistics.md) chapter supplies the distinction between a hypothesis test, an effect, and uncertainty.

<a id="eval-unit"></a>
## [To be reviewed] 1. Define the Evaluation Unit and the Claim

Distinguish a **benchmark dataset**, an **evaluation protocol**, a **harness**, and a **leaderboard**. A dataset supplies tasks; a protocol specifies prompts, access, budgets, and scoring; a harness implements the protocol; a leaderboard aggregates selected submissions. Two papers using the same dataset can be evaluating different systems or different estimands.

| Evaluation unit | What is held fixed? | What the result can support |
| --- | --- | --- |
| Base model / checkpoint | Tokenizer, task adapter, decoding, tools disabled or explicitly controlled | Performance under that elicitation protocol |
| Instruction-following assistant | System prompt, chat template, generation budget, allowed context | Quality of the assistant configuration on the selected tasks |
| Tool-using agent | Model plus scaffold, tools, memory, retries, environment, permissions | End-to-end success of that complete system |
| Product workflow | User population, assistance policy, integration, human review, outcome horizon | Incremental utility under the experiment's deployment conditions |

**Example:** replacing both the model and the repository-navigation tool can improve software-task completion. That comparison supports a system improvement. To attribute the gain to the model, additionally hold the scaffold/tool interface fixed or run a factorial ablation. A better tool-use interface is a real contribution, but it changes the causal question.

Use several measurements when the construct is broad: correctness, robustness, calibration/abstention, safety, efficiency, and relevant subgroup performance. An aggregate can conceal opposite changes in these dimensions. [HELM](https://arxiv.org/abs/2211.09110) is a foundational example of evaluating multiple scenarios and metrics with explicit coverage and standardization goals.

<a id="eval-taxonomy"></a>
## [To be reviewed] 2. Static, Dynamic, and Interactive: Separate Axes

**Static versus dynamic concerns how tasks, distributions, or environments change. Interactive concerns whether actions receive observations that influence later actions.** A fixed task collection can require extensive interaction. A periodically refreshed coding-question collection can still be scored as independent program-generation problems.

| Design | What changes or stays fixed? | Interaction during an attempt? | Example and interpretation |
| --- | --- | --- | --- |
| Static question set | A specified set of questions/answers is frozen | Usually unnecessary for closed-book QA; a protocol can permit tools | MMLU or a fixed MATH release; useful for controlled historical comparisons |
| Refreshed / temporal benchmark | New timestamped tasks are periodically added or replaced | Depends on the task | LiveCodeBench cohorts; compare systems on the same release/date window |
| Dynamic adversarial collection | Annotators or generators create examples against selected models | Interaction may occur during **dataset construction**, not evaluation | Dynabench; report the target model and collection round |
| Fixed interactive environment suite | Task instances and environment snapshots can be frozen | Yes: actions change state and reveal observations | WebArena, OSWorld, or repository-repair tasks |
| Simulated user interaction | User utterances/actions depend on the conversation and simulator | Yes, potentially stochastic and partially observable | Tau-bench / tau2-bench; simulator identity and seeds are part of the setup |
| Live external-world evaluation | Websites, search indexes, services, or facts may drift | Often yes, but freshness and interactivity remain distinct | Browsing-agent evaluation; timestamp and archive the accessible evidence |

The refreshed-data idea is illustrated by [LiveCodeBench](https://arxiv.org/abs/2403.07974); human/model-in-the-loop construction by [Dynabench](https://arxiv.org/abs/2104.14337); controlled interactive execution by [WebArena](https://arxiv.org/abs/2307.13854). “Dynamic” should always be expanded into the concrete mechanism being changed.

Other independent axes include **public versus private**, **natural versus synthetic**, **answer versus trajectory/state grading**, and **offline versus online user evaluation**. A hidden static test can leak through repeated submissions; a fresh synthetic test can repeat a known template. Interactivity alone establishes neither distributional realism nor long-horizon difficulty.

**Design trade-off:** freeze environments and graders for reproducibility, then maintain a separately versioned fresh challenge set for robustness to new tasks. Keep an anchor set when tracking trends across releases. When the mixture changes, distinguish score movement caused by the model from movement caused by task difficulty or weighting.

<a id="eval-llm-benchmarks"></a>
## [To be reviewed] 3. Representative LLM Benchmarks: What Each Measures

The rows identify task families and common evaluation forms. A family's historical importance does not establish that its original test still separates current frontier systems. Report the exact subset, language, modality, elicitation, and tool policy.

| Capability / family | Benchmark examples | Typical evidence / score | Important interpretation limit |
| --- | --- | --- | --- |
| Broad academic knowledge | [MMLU](https://arxiv.org/abs/2009.03300), [MMLU-Pro](https://arxiv.org/abs/2406.01574) | Multiple-choice accuracy across subjects; Pro increases reasoning emphasis and answer choices | Prompt/choice scoring and subject weighting matter; broad exam coverage is not general competence |
| Graduate-level science | [GPQA](https://arxiv.org/abs/2311.12022), including the Diamond subset | Expert-authored biology/chemistry/physics questions | Name the subset; “Google-proof” describes the original validation setup, not a guarantee against memorization or future tools |
| Diverse difficult reasoning | [BIG-Bench Hard / BBH](https://arxiv.org/abs/2210.09261) | Selected challenging reasoning tasks under specified prompting | Historical task selection and chain-of-thought examples define the measurement |
| Novel visual abstraction | [ARC-AGI-2](https://arcprize.org/arc-agi/2) | Infer transformations from small input/output grid demonstrations | Pin task split, exact-output rules, test-time adaptation, attempts, and compute; the benchmark operationalizes a particular generalization construct |
| School / competition mathematics | [GSM8K](https://arxiv.org/abs/2110.14168), [MATH](https://arxiv.org/abs/2103.03874), year-specific [AIME](https://maa.org/maa-invitational-competitions/) problem sets | Final-answer accuracy with an explicit parser or mathematical checker | AIME is a competition source, not one invariant benchmark implementation; pin year, questions, sampling, and tool policy |
| Advanced academic / mathematical challenge | [Humanity's Last Exam](https://arxiv.org/abs/2501.14249), [FrontierMath](https://arxiv.org/abs/2411.04872) | Expert-authored hard questions; versions differ in subject/modal coverage and answer verification | Distinguish public/private sets, tiers, tools, and revisions; academic difficulty is a different construct from autonomous research |
| Short program synthesis | [HumanEval](https://arxiv.org/abs/2107.03374), [MBPP](https://arxiv.org/abs/2108.07732), [EvalPlus / HumanEval+](https://arxiv.org/abs/2305.01210) | Functional correctness on tests; pass@k under a declared sampling protocol | Passing a finite test suite is evidence about that suite; EvalPlus demonstrates why stronger tests can change conclusions |
| Fresh coding tasks | [LiveCodeBench](https://arxiv.org/abs/2403.07974) | Timestamped competition tasks and code-related scenarios | Specify the date cohort and scenario; newer tasks reduce some contamination routes without proving zero exposure |
| Checkable instruction following | [IFEval](https://arxiv.org/abs/2311.07911) | Programmatic instruction-level/prompt-level checks | Strict/loose variants and checker rules differ; satisfying format constraints does not establish truthfulness or usefulness |
| Assistant preferences / conversation | [MT-Bench](https://arxiv.org/abs/2306.05685), [Chatbot Arena](https://arxiv.org/abs/2403.04132) | Model-judge scores or human pairwise preferences, respectively | Judge/user population, response order, verbosity, and sampling differ; preference is not a correctness certificate |
| Long-context use | [LongBench](https://arxiv.org/abs/2308.14508), [RULER](https://arxiv.org/abs/2404.06654) | Long-context task performance or controlled synthetic length sweeps | Pin the version and length protocol; a single needle lookup does not establish long-document reasoning |
| Chinese-language breadth | [CMMLU](https://arxiv.org/abs/2306.09212) | Chinese subject-level multiple-choice evaluation | Chinese academic knowledge does not cover every dialect, locale, or multilingual interaction |
| Multimodal academic reasoning | [MMMU](https://arxiv.org/abs/2311.16502) | Questions requiring images and subject knowledge | Image resolution, preprocessing, text-only subsets, and tools change comparability |
| Refusal / adversarial robustness | [HarmBench](https://arxiv.org/abs/2402.04249) | Behavior-specific evaluation under a declared red-team and grading setup | Report attack budget and benign-task utility/over-refusal separately; one attack suite is not comprehensive safety evidence |

For an accuracy number, check how an answer is extracted. Scoring a multiple-choice label by token log-likelihood, normalizing option-string likelihood by length, and parsing a generated final answer can give different results. Similarly, mathematical equivalence, exact normalized strings, and a model judge do not define the same correctness relation. Inspect the task adapter, not only the benchmark name. [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) provides inspectable implementations; pin a commit and task configuration.

<a id="eval-agent-benchmarks"></a>
## [To be reviewed] 4. Representative Agent and Workflow Benchmarks

Agent results depend on both the policy and the environment/tool interface. Record observation modality, allowed actions, persistence, network access, user simulation, execution budget, and graders. A fixed repository issue or web task can remain fully interactive during execution.

| Family | Representative benchmark | What the agent must accomplish | Main scoring / reproducibility concern |
| --- | --- | --- | --- |
| Repository software engineering | [SWE-bench family](https://www.swebench.com/), including Verified; [SWE-bench Pro](https://arxiv.org/abs/2509.16941) | Resolve an issue in a repository and produce a patch | Test adequacy, repository/image versions, solution leakage, patch isolation, and scaffold; variants are not interchangeable |
| Terminal work | [Terminal-Bench](https://www.tbench.ai/) | Complete executable tasks through a terminal/tool environment | Pin suite release, agent/harness, environment image, verifier, timeout, and resource limits |
| General assistant tasks | [GAIA](https://arxiv.org/abs/2311.12983) | Combine reasoning, browsing, files, tools, and sometimes multimodal inputs | Short final-answer grading can hide different trajectories; public questions and changing web evidence require care |
| Web interaction | [WebArena](https://arxiv.org/abs/2307.13854) | Perform tasks on realistic controlled websites | Check final application state; site snapshots, resets, observation/accessibility interface, and permissions matter |
| Desktop interaction | [OSWorld](https://arxiv.org/abs/2404.07972) | Carry out tasks in real computer applications | Execution-based state evaluation, OS/application versions, display configuration, and observation/action interfaces |
| Interactive exploration and adaptation | [ARC-AGI-3](https://arcprize.org/arc-agi/3) | Explore unfamiliar game-like environments and learn how to achieve goals through feedback | Action efficiency, environment/level versions, state resets, observation format, and within-attempt learning; different from ARC-AGI-2's static grids |
| Function / tool calling | [Berkeley Function Calling Leaderboard / BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) | Select and call tools correctly; later versions add multi-turn/agentic settings | Name the category and version: syntax/AST matching, execution, and multi-turn task success are different tests |
| Conversational service agents | [Tau-bench](https://arxiv.org/abs/2406.12045), [tau2-bench](https://arxiv.org/abs/2506.07982) | Interact with a simulated user, use APIs, and obey domain policies | User simulator, policy version, database state, and repeated-run reliability; tau2 adds user-side actions in a shared environment |
| Difficult web research | [BrowseComp](https://openai.com/index/browsecomp/) | Find hard-to-locate information and return a checkable answer | Search provider, website drift, accessible evidence, task contamination, and final-answer grader |
| Professional deliverables | [GDPval](https://openai.com/index/gdpval/) | Produce work artifacts using supplied context/reference files | Expert rubric and comparison protocol; the original one-shot design does not test every iterative professional workflow |
| AI research replication | [PaperBench](https://openai.com/index/paperbench/) | Reimplement research contributions and conduct experiments against hierarchical replication rubrics | Separate artifact/rubric credit from independently verified execution; pin compute, data access, time budget, and judge |
| Robustness to untrusted tool data | [AgentDojo](https://arxiv.org/abs/2406.13352) | Complete legitimate tool tasks while resisting adversarial instructions in external data | Measure both attack success and useful task completion under a fixed threat model |

**Workflow realism has several dimensions.** GDPval's original release spans professional deliverables and uses expert comparisons, while its authors explicitly distinguish that one-shot setup from future iterative work. BrowseComp's short verifiable answers make scoring practical, but the authors note the limited correspondence to open-ended user requests. Task realism, interaction, and grading convenience should be assessed separately. [GDPval limitations](https://openai.com/index/gdpval/), [BrowseComp design](https://openai.com/index/browsecomp/)

**Task duration is another measurement axis.** METR's time-horizon methodology fits success against task length measured by human completion time at a specified reliability level. A 50% horizon is not an agent's wall-clock runtime, a guarantee of reliable job automation, or a deadline for future capabilities. Human expertise/familiarization, task distribution, and scaffold influence the result. [METR methodology](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)

<a id="eval-version-changes"></a>
## [To be reviewed] 5. Dated Maintenance Notes: Names Are Not Stable Protocols

These examples were checked on **2026-10-27**. They illustrate why a benchmark registry needs versions and maintenance history rather than permanent claims that one suite is the definitive frontier test.

- **SWE-bench Verified:** a historically important curated subset, still listed by the official SWE-bench project. OpenAI's February 23, 2026 audit reports residual test/task-definition issues and contamination and recommends SWE-bench Pro for its frontier coding evaluation. This is an attributed evaluation decision; it does not imply that every Verified task is invalid or that every other organization has stopped using it. [Official family](https://www.swebench.com/), [audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)
- **SWE-bench Pro V2:** the October 2026 report describes repaired/retired tasks and a hardened evaluation protocol. Results from the original public split and V2 do not share an identical task distribution or harness. Record which revision is used rather than treating “Pro” as one immutable number. [V2 report](https://arxiv.org/abs/2610.24206)
- **Terminal-Bench:** the official homepage currently labels its main suite **4.0**. Historical reports for 1.0/2.0 remain results for those releases. Pin task definitions, harness, environment, and grader rather than copying an older setup under the current suite label. [Official site](https://www.tbench.ai/)
- **LiveBench:** the original paper describes refreshed, objectively scored tasks and contamination mitigation; the current site advertises **LiveBench 2.0**. Do not transfer the original suite's categories or grader assumptions to a new release without inspecting its methodology. The paper's revised title uses “contamination-limited”; no freshness label by itself proves absence of contamination. [Paper](https://arxiv.org/abs/2406.19314), [versioned site](https://livebench.ai/)
- **BFCL:** the official page identifies V4 and distinguishes earlier AST-oriented, expanded-data, multi-turn, and agentic evaluation revisions. A single-call score from one release is not an agentic overall score from another. [BFCL methodology links](https://gorilla.cs.berkeley.edu/leaderboard.html)

<a id="eval-saturation"></a>
## [To be reviewed] 6. Saturation: When a Score Stops Being Informative

Benchmark saturation means the evaluation supplies insufficient discrimination for the capability range or decision of interest. High average accuracy is one symptom; insufficient statistical power, coarse grading, narrow task support, and a defective ceiling can also reduce information. Difficulty is relative to the tested systems and permitted compute. A historical hard benchmark need not remain hard indefinitely.

Separate at least four diagnoses:

| Symptom | Possible cause | Useful follow-up |
| --- | --- | --- |
| Scores cluster near the upper bound | Tasks are mostly easy under the current protocol | Retain the anchor set; add independently validated harder or broader task strata |
| Systems tie despite visibly different outputs | Grader misses relevant distinctions or counts only final strings | Audit labels/tests and add well-defined partial, process, or outcome measures |
| Gains are tiny relative to uncertainty | Too few independent tasks or repeated near-duplicates | Increase independent coverage and use paired uncertainty estimates |
| Public score is strong but fresh transfer fails | Contamination, benchmark-specific adaptation, or distribution mismatch | Audit exposure and run sealed, temporal, or different-source transfer evaluations |

Saturation is not evidence that a model has mastered the whole domain. MMLU-Pro explicitly motivates a more discriminative successor to the original MMLU; HLE targets harder academic coverage. Those design motivations do not make either suite permanently unsaturable. [MMLU-Pro](https://arxiv.org/abs/2406.01574), [HLE](https://arxiv.org/abs/2501.14249)

**Statistical example:** on 500 paired binary tasks, suppose A beats B on three tasks and loses on none; all others tie. The accuracy gap is 0.6 percentage points. Under an illustrative independent-discordance null, the two-sided exact sign-test p-value is 0.25, despite A winning every observed disagreement. A deterministic leaderboard order can suggest more certainty than the comparison supports. Task clustering and adaptive model selection require additional treatment.

**Do not solve saturation merely by adding opacity.** Harder tasks need defensible answers, adequate graders, and a deployment-relevant construct. Otherwise measured failure can reflect ambiguity or environment defects. Preserve public anchors for longitudinal comparison while maintaining audited fresh challenges; report both, with explicit weights. Changing task mixture, allowed inference compute, or judge simultaneously prevents simple historical trend interpretation.

<a id="eval-metrics"></a>
## [To be reviewed] 7. Metrics: Accuracy, Search, Reliability, and Cost

**Choose the denominator.** Per-question accuracy, macro-subject accuracy, per-action correctness, and end-to-end task success weight different units. Log invalid outputs, truncations, timeouts, and infrastructure failures separately. Define an inclusion/adjudication rule in advance; silently dropping failed runs can inflate success. A tool-call syntax score and a correct final database state answer different questions.

For binary outcomes, an illustrative weighted task score is $`\widehat S=\sum_{i=1}^{n}w_i y_i/\sum_{i=1}^{n}w_i`$, with $`y_i\in\{0,1\}`$, nonnegative weights, and positive total weight. Choose weights before looking at results. Specify whether they represent deployment prevalence, equal task-family importance, or another target distribution. An overall mean can improve while a critical slice degrades.

### [To be reviewed] pass@k Versus pass^k

For n sampled attempts with c successes on a task, the common without-replacement estimator for finding at least one success among k attempts is $`\widehat{\operatorname{pass@k}}=1-\binom{n-c}{k}/\binom{n}{k}`$, for n at least k. Define an impossible numerator combination as zero. This is an oracle-success measure: a deployed selector may not know which sample is correct. Best-of-N with a fallible verifier is a separate system. [HumanEval estimator](https://arxiv.org/abs/2107.03374), [reasoning evaluation](reasoning.md)

Tau-bench instead emphasizes **pass^k**, successful completion on **all** k repeated trials. The corresponding all-success subset estimator is $`\widehat{\operatorname{pass^k}}=\binom{c}{k}/\binom{n}{k}`$, taking the numerator as zero when c is below k. Under independent identically distributed attempts with per-task probability p, the two population quantities are $`1-(1-p)^k`$ and $`p^k`$, respectively. At p = 0.8 and k = 5, these are 0.99968 and 0.32768. Search opportunity and repeatable reliability can move in very different directions. [Tau-bench reliability definition](https://arxiv.org/html/2406.12045v1)

For heterogeneous tasks, average the appropriate per-task quantity; generally $`\mathbb E[p_i^k]\ne(\mathbb E[p_i])^k`$. The iid assumption also fails with shared state, persistent memory, correlated seeds, or dependent simulator behavior. Reset environments and state what is resampled. The binomial estimators estimate subset statistics from repeated runs; they do not remove dependencies or selection bias.

### [To be reviewed] Accuracy–Cost and Reliability–Latency Frontiers

Report single-attempt success and success under a declared retry/search policy, with generated tokens, tool calls, wall time, monetary cost, and human intervention. Compare at matched budgets where the scientific question permits, and also report a useful frontier across budgets. Include failures in average cost; with a nonzero success count, dividing total evaluation expenditure by successes can summarize cost per success, but hides difficulty mix and variance.

Measure latency distributions and deadline-conditioned success rather than only mean throughput. A two-hour agent and a two-minute agent can have the same unconstrained success rate but very different usefulness. A higher budget is a valid experimental factor when explicitly measured; it is a confound when hidden inside a “model” comparison.

<a id="eval-graders"></a>
## [To be reviewed] 8. Graders, Judges, and Measurement Validity

| Grading mechanism | Useful evidence | Failure mode to audit |
| --- | --- | --- |
| Exact / normalized answer matching | Reproducible correctness for well-specified short answers | Valid aliases, units, formatting, ambiguity, and answer leakage |
| Executable tests / formal checks | Functional behavior or specified invariants | Missing edge cases, overly implementation-specific tests, reward exploits |
| Environment-state verifier | Task completion in an application or backend | A narrow state predicate can miss unauthorized side effects or communication requirements |
| Human expert rubric | Contextual correctness and artifact utility | Expertise, rubric ambiguity, identity/order effects, disagreement, and cost |
| LLM-as-a-judge | Scalable rubric application or pairwise comparison | Position/verbosity/self-preference bias, weak subject knowledge, and prompt injection through evaluated text |

**Example:** a patch passes the added regression test while deleting another feature; a customer-service agent changes a booking correctly but violates the required confirmation policy. Score the properties required by the task, and keep grader tests separate from information available to the agent when the protocol requires that separation. Test adequacy is an empirical property, not a consequence of using code to grade. [EvalPlus](https://arxiv.org/abs/2305.01210), [SWE-bench audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/)

For model judges, fix the judge revision, rubric, prompt, reference access, sampling, and tie policy. Randomize or swap response order, blind model identity, and calibrate on independently expert-labeled difficult examples. Measure disagreement and ranking sensitivity when changing judges. A judge's confident explanation is not a correctness label. [MT-Bench / judge-bias study](https://arxiv.org/abs/2306.05685)

Retain evaluated material as data within the grading boundary. Separate task success from attempts to manipulate a judge or exploit a verifier, and keep useful-task completion visible when evaluating defenses. [AgentDojo](https://arxiv.org/abs/2406.13352) provides a concrete setting for evaluating agents exposed to adversarial tool outputs.

<a id="eval-contamination"></a>
## [To be reviewed] 9. Contamination, Benchmark Adaptation, and Integrity

Distinguish **training exposure** to questions/solutions, **development overfitting** from repeated benchmark-driven choices, and **evaluation-time leakage** through tools, files, search, or scorer access. Also distinguish legitimate task knowledge from access to a benchmark's withheld answer or patch. The boundary depends on the declared task and tool policy.

**A date split is useful but incomplete.** Verify the task's publication time against a defensible model/data cutoff, and record later fine-tuning, synthetic-data generation, and evaluation access. A newly packaged dataset can contain old source questions or solutions. A private answer key does not make already-public prompts unobserved. Fresh code challenges and a maintained evaluation policy reduce particular risks; the label “contamination-free” is not a proof. [LiveCodeBench](https://arxiv.org/abs/2403.07974), [LiveBench](https://arxiv.org/abs/2406.19314)

For agent execution, preserve allowed network routes, repository snapshots, visible files, and tool schemas. Regrade the submitted artifact in a clean verifier environment when the protocol calls for patch-only evaluation. Otherwise agent edits outside the artifact or retrieval of upstream solutions can change what the score measures. [SWE-bench Pro V2 protocol](https://arxiv.org/abs/2610.24206)

Treat overlap detection, unusual string recall, paraphrase sensitivity, and fresh-source performance as pieces of evidence. None alone establishes the causal amount of benchmark inflation. Do not infer contamination solely from a strong score, or infer its absence from a vendor's unverified cutoff. Report uncertainty about training exposure and include different-source generalization tests.

Maintain separate development and final evaluation access. Log submissions, preserve task hashes and access histories, rotate audited challenge sets when justified, and review task corrections transparently. Compare old and revised task versions on an overlap set when making trend claims. A continuously reused private leaderboard can become an optimization target even without publicly released labels.

<a id="eval-statistics"></a>
## [To be reviewed] 10. Uncertainty, Paired Comparisons, and Aggregation

For independent equally weighted Bernoulli tasks, $`\operatorname{SE}(\widehat p)\approx\sqrt{\widehat p(1-\widehat p)/n}`$ is a simple descriptive approximation. It is unreliable at very small counts or extreme proportions and does not account for dependent questions, repeated seeds, judge variation, or benchmark selection. Use an interval appropriate to the design and clearly name the sampling population.

Compare systems on the same tasks with **paired** estimates. Bootstrap at the appropriate independent unit, often the problem/source/repository/task family, carrying all associated attempts together. For agent suites, separate variability across tasks, seeds, user simulations, and infrastructure. Resampling individual tokens or tool steps usually understates uncertainty in end-to-end performance.

Report effect sizes and intervals, not only p-values or rank. Predeclare primary outcomes when testing many suites, and account for multiple comparisons or exploratory selection. Repeatedly choosing prompts/checkpoints from public test scores changes the statistical problem. Confidence conditional on a fixed task suite is different from uncertainty about deployment transfer.

**Aggregation example:** model A may improve coding by ten points and worsen multilingual performance by five. Whether its overall number rises depends on the category weights. Publish per-suite/slice results, weights, exclusions, and overlap. Arithmetic means of unrelated percentages or preference ratings do not automatically define a meaningful universal intelligence scale.

Keep score direction, normalization, and selection policy explicit. A calibration metric can improve without changing argmax accuracy; abstention can improve precision while reducing coverage. Evaluate such trade-offs rather than hiding them in one composite. See [Statistics](statistics.md) and [ML evaluation](machine-learning.md#model-evaluation-and-selection) for the supporting concepts.

<a id="eval-protocol"></a>
## [To be reviewed] 11. A Reproducible Evaluation Protocol

The following is a proposed experimental template, not a claim that every benchmark supports every control.

| Record | Minimum useful specification |
| --- | --- |
| Claim and population | Intended capability, target task/user distribution, model-only versus system comparison |
| Task artifact | Dataset/release/split, task IDs and hashes, collection dates, exclusions, public/private access |
| Model and elicitation | Checkpoint/API snapshot, tokenizer/template, system prompt, few-shot examples, parsing, context policy |
| Agent scaffold | Code revision, tools, memory, retries, delegation, permissions, human intervention |
| Environment | Container/VM and dependency versions, reset policy, network/search access, files, simulator configuration |
| Budget | Output and total tokens, attempts, tool calls, timeouts, concurrency, cost accounting |
| Grader | Version, tests/rubric, references, judge/simulator identity, randomization, tie handling |
| Analysis | Seeds, repetitions, paired comparison, clustering, primary metrics, confidence intervals, weighting |
| Audit artifacts | Prompts/completions where permitted, actions, observations, artifacts, verifier logs, timestamps and run IDs |

Run a small harness-validation set before measuring models. Confirm that known valid outputs pass, intentional invalid outputs fail, environment resets work, and artifacts are graded in the intended state. Keep these checks separate from model-prompt tuning on the final test set. Validate any changed grader or dependency before comparing against historical results.

Use an inspectable harness when suitable, such as the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), but also pin its task adapters and configuration. Framework defaults, chat templates, and backend behavior can change. Save machine-readable per-task results in addition to a leaderboard screenshot, and distinguish an interrupted run from a completed negative result.

<a id="eval-portfolio"></a>
## [To be reviewed] 12. Concrete Portfolio: Evaluating a Research-and-Code Assistant

This is an illustrative design for an assistant that answers technical questions, searches evidence, edits code, and produces a deliverable. It is not a universal recommended suite or a reported experimental result.

| Question | Candidate evidence | Necessary control / ablation |
| --- | --- | --- |
| Does the checkpoint handle scientific knowledge and explicit instructions? | MMLU-Pro / GPQA plus IFEval | Fixed prompts/tool policy; inspect correctness and instruction compliance separately |
| Can it solve fresh coding problems? | A declared LiveCodeBench date cohort | Fixed interpreter, tests, sampling, and answer extraction; report pass@1 and a search budget |
| Can the complete agent modify a repository? | A declared SWE-bench-family revision and terminal tasks | Freeze scaffold/images; audit patch-only grading and solution access |
| Can it find and use evidence? | GAIA / BrowseComp plus a private source-separated research set | Search snapshot/logs, independently checked support, oracle-evidence ablation |
| Can it act correctly through interfaces? | WebArena / OSWorld tasks appropriate to the product | Observation/action modality, reset state, deadline, final-state checks |
| Is repeated execution reliable? | Repeated independent runs, or an appropriate tau-bench setting | Explicit simulator/state resets; report success and all-trials reliability |
| Are outputs useful to experts? | A blinded domain rubric or GDPval-style work artifacts | Expert review, source support, artifact validity, and separate cost/latency |
| Does it remain useful under untrusted inputs? | A scoped AgentDojo-style robustness test and benign controls | Fix threat model; report both attack resistance and legitimate task success |

Start with a model-only baseline, then add tools, retrieval, and a scaffold in controlled stages. Compare oracle evidence, ordinary retrieval, and no retrieval to localize failures, as in [RAG evaluation](rag.md#generation) and [QA evaluation](qa.md#evaluation). Match the budget appropriate to the causal question and separately measure the best practical system under a deployment budget.

Retain a frozen anchor suite for regression detection, a versioned fresh suite for generalization, and an interactive workflow suite for end-to-end behavior. These serve different purposes; their weighted aggregate needs an explicit interpretation. Check that any apparent reasoning improvement also improves independently verified outcomes at matched compute. [Reasoning evaluation](reasoning.md#reasoning-overview)

<a id="eval-reading"></a>
## [To be reviewed] 13. Reading a Leaderboard as a Researcher

Before interpreting a score, reconstruct **task release → model/scaffold → allowed information → inference budget → grader → aggregation → uncertainty**. Then ask which ablation supports the claimed mechanism and which plausible failure mode remains unmeasured.

A credible result names its setup, makes invalid/failed attempts visible, and supports a bounded claim. “Best on one frozen suite,” “preferred by this judge population,” and “more useful in this deployment experiment” are different findings. Benchmark maintenance, saturation, and external validity remain ongoing scientific work rather than properties settled by a benchmark's title.
