# Chapter Review — 2026-10-27

[Handbook contents](../README.md#table-of-contents)

## Scope and review status

Reviewed the README and all 13 chapter documents against baseline commit `f4e992b`: factual statements and assumptions, duplicate explanations, mathematical notation, diagrams, terminology, and local/external references. This is an editorial and technical review, not independent reproduction of every cited experiment. Existing **[To be reviewed]** markers remain in place; corrections do not imply author approval of the marked material.

## Chapter coverage and corrections

| Chapter | Review result / action |
| --- | --- |
| [Statistics](statistics.md) | Specified the null model and extremeness rule for p-values; removed the continuous-density/probability conflation in MLE and linked the finite-MLE counterexample. |
| [Machine Learning](machine-learning.md) | Fixed malformed TOC markup; qualified metric/cost relationships, regularization, feature selection, ENN, stratification, kernels, stacking, and silhouette interpretation. Removed repeated definitions and added direct links from the taxonomy to detailed algorithms. |
| [Deep Learning](deep-learning.md) | Corrected chain-rule notation, GLU multiplication notation, initialization assumptions, ReLU inactivity, and recurrent-gradient claims. Replaced the erroneous activation illustration and oversimplified initialization graphic. |
| [Transformer](transformer.md) | Checked shapes, RoPE, masking, attention/cache behavior, normalization, parameter/MAC/cache arithmetic, and the architecture diagram. Updated the diagram's reference to its independent chapter. |
| [NLP](nlp.md) | Fixed the obsolete “section above” reference, clarified TF-IDF implementation conventions, and stopped classifying retrieval as necessarily unsupervised. |
| [Large Language Models](large-language-models.md) | Clarified division in gradient accumulation and bucketwise DDP overlap; moved batch accounting under accumulation and made communication its own entry. Removed unconditional ZeRO/OOM language and ambiguous horizontal/vertical parallelism labels. Checked SFT/RL/DPO/OPD and feedback-source distinctions. |
| [Agents](agents.md) | Reviewed tool contracts, planning, memory, coordination, trajectory supervision, RL, and execution/training boundaries. No additional substantive correction identified in this pass. |
| [RSI & Harness](rsi-and-harness.md) | Reviewed recursion criteria, state/effect recovery, evaluator separation, resource accounting, and selection-bias examples. No additional substantive correction identified in this pass. |
| [Benchmarking & Evaluation](benchmarking-and-evaluation.md) | Reviewed benchmark taxonomy, saturation, grader validity, contamination, paired comparisons, and pass@k versus pass^k. Rechecked version-sensitive examples against primary sources. |
| [RAG](rag.md) | Removed obsolete outline bullets after the detailed sections, including the conflicting classification of score normalization/query expansion as learned rerankers. Clarified the nonempty relevant-set condition for recall. |
| [QA](qa.md) | Removed duplicate outlines, including the incorrect equivalence of factoid with extractive and non-factoid with abstractive. Preserved the independent taxonomy axes and task-specific evaluation explanation. |
| [Reasoning](reasoning.md) | Checked verifier/search distinctions, pass@k, and links to training, agent execution, and evaluation. No additional substantive correction identified in this pass. |
| [Recommendation Systems](recommendation-systems.md) | Checked retrieval/ranking/slate distinctions, negative sampling versus exposure correction, semantic-ID search examples, generative families, and time-valid evaluation. Unified the displayed reranker spelling. |

The old activation graphic plotted tanh with an incorrect vertical scale. The new [PNG](../figs/activation-functions.png) and [SVG](../figs/activation-functions.svg) are computed directly from the defining functions, with parameters shown. The initialization table now states variance, gain, and applicability explicitly. Historical raster files remain in the repository for provenance but are no longer embedded in the chapter.

## Canonical terminology and cross-chapter references

Use the displayed names below. Stable older fragments remain valid, including `re-ranking`; terminology cleanup does not require renaming bookmarked URLs.

| Displayed term / distinction | Canonical explanation |
| --- | --- |
| Pre-training → mid-training → post-training | [LLM lifecycle](large-language-models.md#stages); name boundaries are conventions |
| SFT / online RL / OPD; offline preference optimization | [Method comparison](large-language-models.md#sft-rl-opd-comparison), [DPO](large-language-models.md#preference-optimization), [OPD estimators](large-language-models.md#distillation-and-opd) |
| RLHF / RLAIF / RLVR / RLCD | [Feedback construction](large-language-models.md#rl-feedback-comparison), separate from PPO/GRPO update algorithms |
| LoRA, QLoRA, PEFT | [Update parameterization](large-language-models.md#update-parameterization), separate from the training stage |
| AdaGrad, RMSprop, Adam, AdamW, AdaBoost, t-SNE | [ML mechanisms](machine-learning.md); preserve method-specific capitalization |
| Reranking / reranker | [Query–passage rescoring](rag.md#re-ranking) versus [slate reranking](recommendation-systems.md#rec-reranking) |
| Token embedding / contextual state / retrieval vector | [Representation distinctions](large-language-models.md#embedding) |
| Linear Discriminant Analysis / Latent Dirichlet Allocation | [Supervised classifier](machine-learning.md#supervised-learning) versus [topic model](nlp.md#statistical--ml-era); expand the ambiguous acronym LDA in context |
| Runtime harness / evaluation harness | [Responsibilities and boundaries](rsi-and-harness.md#harness-runtime) |
| pass@k / pass^k | [Search opportunity versus repeated-run reliability](benchmarking-and-evaluation.md#eval-metrics) |

The unresolved label **KPO** remains explicitly marked for review in [LLM terminology](large-language-models.md#to-be-reviewed-unresolved-terminology-kpo). Its intended paper/method cannot be established from the original outline; no expansion was invented.

## Validation

The figures in this section describe the review snapshot committed as `f28d16b`. Subsequent RL additions are recorded separately below.

- All **353 baseline review markers** are retained; the reviewed documents and this report contain **367 markers** in total.
- **441 formulas** survive GitHub Markdown rendering intact and compile with MathJax without errors.
- **425 local links**, **11 local image references**, **40 tables**, and all baseline heading/alias fragments pass structural checks across **15 Markdown files** (README, 13 chapters, and this report).
- All **3 Mermaid diagrams** parse. The corrected activation plot and updated Transformer diagram were visually inspected.
- All **9 Python examples** and **2 JSON examples** parse; all **7 notebooks** pass code-cell syntax inspection.
- Numerical checks pass for finite-difference backpropagation, stable binary cross-entropy, causal label shifts, attention outputs/gradients versus PyTorch SDPA, full versus cached attention, causal masks, RoPE identities, pass@k/pass^k estimators, and the relevant arithmetic examples.

Validation checks all chapter navigation, TOC links, local image references, and preserved baseline fragments. Formulas are checked both through GitHub's Markdown renderer and a MathJax parser. Python/JSON examples are syntax checked; focused numerical checks cover backpropagation, attention, cache equivalence, causal masks, and the chapter's arithmetic examples. This does not claim execution of full distributed training, hosted fine-tuning, every notebook, or every cited system.

## External links and access limits

The initial sweep checked **323 unique external URLs**: **307 returned HTTP 200**, **11 returned HTTP 403**, and **5 had connection failures**. YouTube resources were checked through oEmbed. No HTTP 404/410 was confirmed among the original URLs. A successful HTTP response is availability evidence, not proof of every linked claim or durable future availability.

After replacements and additions, the **326 URLs currently referenced** in the chapters, README, and this report comprise **316 successful HTTP checks**, **8 direct-checker 403 responses**, and **2 connection failures**. The two remaining direct-checker failures are Hugging Face pages that opened successfully through independent web retrieval. Access restrictions are recorded below rather than mislabeled as confirmed broken links.

Follow-up actions:

- Replaced the timing-out K-means++ PDF link with the [Google Research publication record](https://research.google/pubs/k-means-the-advantages-of-careful-seeding/).
- Replaced the timing-out Stanford BM25 page with the [author-hosted Robertson/Zaragoza paper](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf).
- Replaced the RRF publisher link that rejected automated access with the [author-hosted paper](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
- Replaced the blocked secondary normalization article with [primary framework documentation](https://docs.pytorch.org/docs/stable/nn.html#normalization-layers).
- Changed the Star History embed to a direct interactive-chart link, avoiding dependence on an image endpoint that rejected access.
- Independently opened the three initially disconnected Hugging Face resources: [SFT Trainer](https://huggingface.co/docs/trl/sft_trainer), [TRL](https://huggingface.co/docs/trl/index), and the [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook).
- Cross-checked the identities/titles of 165 distinct cited arXiv papers against the arXiv metadata API. Titles can change across revisions; pin a revision for reproduction.

The following retained sources rejected the direct checker with 403. They are **access-restricted/unverified by that checker**, not established dead links. Primary-source pages were cross-checked through web retrieval where available; secondary Zhihu references remain supplemental.

| Source | Affected chapter / interpretation |
| --- | --- |
| [MAA competition page](https://maa.org/maa-invitational-competitions/) | Benchmarking: competition source; select the exact year/problem release |
| [BrowseComp](https://openai.com/index/browsecomp/) | Benchmarking: primary benchmark announcement |
| [GDPval](https://openai.com/index/gdpval/) | Benchmarking: primary release and limitations |
| [PaperBench](https://openai.com/index/paperbench/) | Benchmarking: primary research-replication benchmark |
| [SWE-bench Verified audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | Benchmarking: attributed maintenance decision |
| [Distributed-training note](https://zhuanlan.zhihu.com/p/598714869) | LLM: supplemental; primary papers/framework references are also supplied |
| [Word2Vec note](https://zhuanlan.zhihu.com/p/61635013) | NLP: supplemental; the original Word2Vec paper is also supplied |
| [ASA statement DOI](https://doi.org/10.1080/00031305.2016.1154108) | Statistics: publisher rejects the direct checker; bibliographic source retained |

Version-sensitive checks included the [SWE-bench Pro V2 report](https://arxiv.org/abs/2610.24206), [Terminal-Bench homepage](https://www.tbench.ai/), [LiveBench site](https://livebench.ai/), and the [dated provider SFT guide](https://developers.openai.com/api/docs/guides/supervised-fine-tuning). Their availability/version statements are dated observations, not permanent training or evaluation rules.

## RL follow-up — 2026-10-27

Expanded the [LLM RL discussion](large-language-models.md#online-rl) with credit assignment, reward-to-go, baselines/GAE, reward hacking, process/outcome rewards, potential-based shaping, sparse exploration, RLOO/PPO/GRPO distinctions, KL/entropy/length effects, on/off-policy data, interview questions, and a debugging sequence. New curricular sections and cross-chapter explanations carry review markers. Existing content and review markers from `f28d16b` are preserved.

The examples explicitly separate estimator credit from causal attribution, verified acceptance from full correctness, and surrogate-loss improvement from independent task improvement. The worked GAE, shaping, mixed-group probability, and self-inclusion calculations are checked numerically; GitHub/MathJax rendering, local references, and newly cited source availability are checked for the extension.

## Statistics foundations follow-up — 2026-10-27

Expanded [Statistics](statistics.md) with a compact foundation for ML/DL: conditional probability and common distributions, moments/dependence, sampling and uncertainty, testing/power/multiplicity, Bayesian/MAP inference, experimental design/causality, and information/calibration. Original hypothesis-test and MLE material and review markers remain. New curricular headings carry review markers, and the README/ML chapter link to the foundations.

Checked the base-rate, mean/variance, Bernoulli-interval, and Beta–Bernoulli examples numerically, plus formula rendering and local/source links. The prior audit's validation counts remain a historical snapshot rather than totals for these later additions.

## Attention background follow-up — 2026-10-27

Added a [historical/conceptual introduction](transformer.md#tf-background): distributional semantics and static/contextual representations, the recurrent seq2seq bottleneck and soft alignment, additive/dot/bilinear/scaled scores, and the architecture-level move to Transformer self-attention. The representative NMT path is distinguished from a claim that distributional semantics mathematically implies attention or that the score families replaced one another universally.

All five new background headings carry review markers; the 14 numbered Transformer modules and diagram numbering are preserved. The README and NLP chapter point to the canonical explanation. Checked score dimensions, additive/concat equivalence, softmax/context calculations and scaling examples, plus formula rendering and source/local links.
