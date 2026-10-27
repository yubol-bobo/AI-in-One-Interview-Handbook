# Q&A System

[Handbook contents](../README.md#table-of-contents) | [Previous: RAG](rag.md) | [Next: Reasoning](reasoning.md)

## Contents

- [Types of QA Systems](#types-of-qa-systems)
- [Components of QA Systems](#components-of-qa-systems)
- [Evaluation](#evaluation)

<a id="types-of-qa-systems"></a>
## [To be reviewed] Types of QA Systems

- **Factoid versus explanatory** describes answer form; **extractive versus generative** describes how the answer is produced. These axes are not equivalent: a generated answer may be a one-word factoid, and an extracted passage may contain an explanation.
- **Open-domain versus closed-domain** describes the scope of questions or knowledge sources. RAG is a system architecture that retrieves evidence before generation, rather than a mutually exclusive alternative to an LLM-based system.
- **Example:** “What is the release date?” can be answered by an extractive span reader or a generative RAG model. “Why did the launch move?” may require synthesizing several dated documents. A corpus-bounded support assistant should be evaluated on whether its permitted sources support the answer, even when the base model remembers a plausible answer from elsewhere. [SQuAD task definition](https://arxiv.org/abs/1606.05250), [RAG architecture](https://arxiv.org/abs/2005.11401)

- Factoid (simple answer, extractive)
- Non-factoid (explanatory, abstractive)
- Open-domain vs Closed-domain Q&A
- Traditional Q&A vs RAG vs LLM-based Q&A

<a id="components-of-qa-systems"></a>
## [To be reviewed] Components of QA Systems

- **Question processing:** resolve conversational references, classify required answer type, extract entities and constraints, and decide whether decomposition or clarification is needed. Preserve negation, units, dates, and access scope when rewriting a query.
- **Retrieval:** retrieve candidates, enforce corpus and permission constraints, rerank, and build a provenance-preserving evidence set. **Reading/generation:** either predict answer spans with a calibrated no-answer option or synthesize an answer constrained by the evidence. [DPR](https://arxiv.org/abs/2004.04906), [SQuAD 2.0](https://arxiv.org/abs/1806.03822)
- **Post-processing:** validate structured output, normalize units where justified, check citation support, merge duplicate claims, and decide whether confidence meets the task's threshold. Token probability is not automatically a calibrated probability that an entire answer is correct.
- **Example:** for “How much did revenue change from 2023 to 2024?”, retrieve both comparable figures, confirm currency and fiscal periods, compute $`(R_{2024}-R_{2023})/R_{2023}`$, and cite both values. A correct subtraction using mismatched fiscal scopes is still a wrong answer. Use a calculator or code for arithmetic and preserve the supporting evidence.
- Evaluate each interface separately: query-rewrite fidelity, evidence recall, reader accuracy given gold evidence, and complete-system answer quality. A single end-to-end score does not identify which component needs improvement.

- Question Processing
    - Question type classification (factoid, procedural, descriptive)
    - Query parsing and expansion

- Document/Passage Retrieval
    - Dense retrieval (embedding-based retrieval)
    - Sparse retrieval (BM25, TF-IDF)
    - Hybrid approaches
    - Re-ranking of retrieved passages

- Answer Extraction/Generation
    - Extractive Q&A (span-based)
    - Generative Q&A (abstractive, using LLMs)

- Answer Post-processing
    - Summarization, filtering irrelevant answers
    - Confidence scoring and answer ranking


<a id="evaluation"></a>
## [To be reviewed] Evaluation

- **Extractive QA:** normalized exact match checks answer-string agreement; token F1 measures overlap with reference answer tokens under a specified normalization. Multiple valid reference spans may be needed. The evaluator's tokenizer and treatment of articles, punctuation, and aliases are part of the metric.
- **Example:** “NYC” and “New York City” can be semantically equivalent but score differently without alias handling. For unanswerable questions, assess both answer accuracy and the error/coverage trade-off of abstention. Select the abstention threshold on development data. [SQuAD evaluation](https://arxiv.org/abs/1606.05250), [unanswerable QA](https://arxiv.org/abs/1806.03822)
- **Generative QA:** report factual/task correctness, completeness of required claims, source support, citation quality, and usefulness. BLEU, ROUGE, and METEOR compare reference-text similarity; METEOR can align variants such as stems and synonyms under its resources. Similarity remains an imperfect proxy for supported correctness. [METEOR](https://aclanthology.org/W05-0909/)
- **Failure localization:** compare retrieved evidence with gold evidence, and the reader with oracle evidence. Group errors into missing evidence, wrong interpretation, unsupported generation, and inappropriate answer/abstention. Report uncertainty clustered by source or question family when examples are dependent.
- Keep a temporal or source-separated test set, include conflicting/outdated sources, and compare against a simple extractive or retrieval baseline. Human review should use a written rubric and blinded model identity.

- Extractive Q&A Evaluation
    - Exact Match
    - F1-score
- Generative Q&A Evalution
    - ROUGE
    - BLEU
    - METEOR score
    - Human evaluation
