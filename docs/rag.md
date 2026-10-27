# RAG

[Handbook contents](../README.md#table-of-contents) | [Previous: Large Language Models](large-language-models.md) | [Next: QA Systems](qa.md)

## Contents

- [Retrieval](#retrieval)
- [Embedding models](#embedding-models)
- [Chunking and Vectorization](#chunking-and-vectorization)
- [Re-ranking](#re-ranking)
- [Generation](#generation)


[[ANLP](https://www.youtube.com/watch?v=KfQaYk4k9eM)]
<a id="retrieval"></a>
## [To be reviewed] Retrieval

- **Sparse retrieval:** BM25 uses exact term matches with term-frequency saturation and length normalization. Preserve identifiers and important terminology during tokenization. **Dense retrieval:** a bi-encoder scores separately encoded queries and passages, enabling precomputed passage vectors and approximate nearest-neighbor search. Dense models can recover paraphrases but can miss exact constraints. [BM25 textbook](https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html), [DPR](https://arxiv.org/abs/2004.04906)
- **Hybrid retrieval:** take the union of sparse and dense candidates, deduplicate, and fuse rankings. Reciprocal rank fusion uses $`\operatorname{RRF}(d)=\sum_r 1/(c+\operatorname{rank}_r(d))`$ over rankings containing the document; $`c`$ is a chosen damping constant. Rank fusion avoids directly averaging incompatible BM25 and cosine score scales. [Original RRF paper](https://doi.org/10.1145/1571941.1572114)
- **Example:** for “E137 after firmware 2.1,” sparse matching preserves the exact code/version while dense search finds a paraphrased troubleshooting description. Filter by the applicable product/version and access permissions. A relevant passage for a different version is a retrieval error even if its semantic score is high.
- **Diagnosis:** separately measure representation quality, ANN search loss against exact vector search, filtering correctness, and candidate recall. A reranker cannot recover a passage absent from its candidate pool. For evidence-required QA, candidate evidence recall constrains grounded answer coverage. Evaluate no-retrieval, sparse, dense, and hybrid baselines on the same corpus snapshot.

- Sparse retrieval (BM25, TF-IDF based retrival)
- Dense retrieval (Embedding-based retrieval)
- Hybrid retrieval

<a id="embedding-models"></a>
## [To be reviewed] Embedding models

- A retriever embedding is trained for a scoring geometry, often contrastive similarity between queries and relevant documents. Model labels such as “SentenceTransformers,” E5, and GTE describe families or tooling; actual checkpoint, training data, task instructions, normalization, and maximum length matter.
- **Selection protocol:** create domain-specific query/relevance pairs; evaluate multilingual slices, rare entities, negation, exact constraints, and hard negatives. Sentence similarity and document retrieval test different capabilities. Choose vector dimension and quantization jointly with recall, memory, indexing time, and query latency. [Sentence-BERT](https://aclanthology.org/D19-1410/), [BEIR](https://arxiv.org/abs/2104.08663)
- **Example:** a query asking for an exception to a policy should not retrieve only the general rule because the two are semantically similar. Include the general-rule passage as a hard negative when it does not answer the exception question; check for alternative valid positives before labeling negatives.
- If the encoder, tokenizer, instructions, pooling, or normalization changes, treat the embedding index as versioned derived data and assess whether re-embedding is required. Compare providers using measured retrieval quality and operational constraints; avoid assuming a product's current model name or dimensionality from a historical note.
- **Metric definition:** for a query with a known relevant set $`R_q`$, $`\operatorname{Recall@k}=|\operatorname{TopK}(q)\cap R_q|/|R_q|`$. Some QA benchmarks instead report the fraction of queries with at least one answer-bearing passage in the top k; name that hit-rate definition explicitly. Incomplete judgments limit both interpretations.

- Open-source models (SentenceTransformers, E5, GTE)
- Commericial APIs (OpenAI Embedding, Cohere embedding)
- How to choose embedding models
    - Semantic vs lexical relevance
    - Embedding dimension, speed, accuracy trade-offs
- Evaluation
    - Retrieval accuracy metrics(Recall@k, MRR)
    - Semantic similarity tasks (STS benchmarks)

<a id="chunking-and-vectorization"></a>
## [To be reviewed] Chunking and Vectorization

- Chunking defines the retrieval unit, not merely a preprocessing convenience. Small chunks can sharpen matching but omit definitions and qualifiers; large chunks preserve context but dilute relevance and consume the generator's budget. Overlap reduces boundary loss while increasing duplication and index size.
- Start with structure-aware boundaries such as headings, paragraphs, code blocks, and table rows, then enforce token limits. Fixed-size, recursive, and semantic splitters should be compared against the same evidence set; semantic boundaries can also be wrong. Keep parent-document identity, offsets, dates, and section paths. [Document-structure partitioning reference](https://docs.unstructured.io/open-source/core-functionality/partitioning)
- **Example:** a policy says “coverage lasts 90 days” in one paragraph and “except refurbished units: 30 days” in the next. Splitting between them can retrieve a fluent but incorrect answer. Retrieve a small child chunk for matching, then expand to a bounded parent span for generation when appropriate.
- **Experiment:** sweep chunk length and overlap under a fixed total context budget, deduplicate overlapping hits, and measure evidence coverage plus grounded answer accuracy. Also measure index size and latency. There is no universal optimal chunk size or overlap ratio. [Retrieval-augmented generation](https://arxiv.org/abs/2005.11401)
- Do not allow chunks from the same source document to leak across development and test splits. Preserve tables and equations as coherent units where possible; a token budget that truncates a table's header can change its meaning.

- Importance of chunking
    - Optimal chunk size
    - Overlapping vs non-overlapping chunking
- Chunking strategies
    - Semantic chunking vs fixed-size chunking
    - Recursive chunking methods
- Impact of chunking strategy on retrieval quality and generation accuracy

<a id="re-ranking"></a>
## [To be reviewed] Re-ranking

- A cross-encoder jointly processes a query and candidate passage to estimate relevance; it captures token interactions that independent single-vector encoding compresses away. Its pairwise computation makes it suitable for a shortlist rather than an entire large corpus. Late-interaction retrievers such as ColBERT offer another trade-off by retaining token-level representations. [Cross-encoder reranking](https://arxiv.org/abs/1901.04085), [ColBERT](https://arxiv.org/abs/2004.12832)
- Score normalization and rank fusion combine retrieval systems; query expansion changes the query or candidate set. They are not automatically learned rerankers. Define the stage and candidate pool precisely when reporting an improvement.
- **Example:** rerank the top 100 retrieved passages into 10 context candidates, then remove redundant overlapping spans. A passage with the correct entity but wrong effective date should be a difficult negative. Train and evaluate with negatives representative of first-stage retrieval, not only random unrelated passages.
- **Metrics:** reciprocal rank is $`1/r`$ for the first relevant result, with zero if none appears; average precision averages precision at relevant ranks and divides by the number of known relevant items under the chosen protocol. Precision@k emphasizes top-k cleanliness; graded NDCG can reward strongly relevant evidence more than marginally useful passages. State truncation and judgment conventions.
- **Toy example:** relevance sequence [1, 0, 1] with exactly two relevant items gives AP $`(1+2/3)/2=5/6`$ and reciprocal rank 1. These metrics capture different properties. Report evidence recall before reranking and final answer performance to locate the source of any gain.

- Importance
- Methods of re-ranking
    - Cross-encoders (dense re-rankers)
    - Traditional re-ranking (score normalization ,query expansion)
    - Learned re-ranking (supervised methods)
- Evaluation
    - MAP
    - MRR
    - Precision @k

<a id="generation"></a>
## [To be reviewed] Generation

- Construct an evidence package with clear document boundaries, source IDs, relevant metadata, and an explicit task. Require citations to refer to supplied source IDs and assess support at the claim level. Retrieved text is untrusted data; instructions embedded in documents should not override the application's task or permission boundaries.
- **Example:** when sources disagree about a warranty duration, compare product scope and effective date before answering. If the evidence cannot resolve the conflict, return the bounded uncertainty rather than silently selecting the more convenient passage.
- **Context budget:** reserve space for the system/task instructions, conversation, evidence, and output. Deduplicate and choose complementary passages. More context is not automatically better; measure sensitivity to distractors and evidence position. [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- **Fusion-in-Decoder:** the published FiD architecture encodes question-passage pairs separately and lets a decoder attend to the combined encoder representations. This differs from concatenating all passages into a single decoder-only prompt. The former reduces cross-passage interaction in the encoder while retaining cross-passage aggregation in the decoder. [FiD](https://arxiv.org/abs/2007.01282)
- **Evaluation:** independently test retrieval relevance, evidence coverage, answer correctness, citation precision/coverage, and abstention on unanswerable queries. Use oracle-evidence and deliberately irrelevant-evidence ablations to separate retrieval failures from generation failures. Automated faithfulness judges need calibration against human-checked examples. [RAG evaluation framework](https://arxiv.org/abs/2309.15217)
- Retrieval makes knowledge easier to refresh and inspect; it cannot guarantee factuality when documents are false, stale, irrelevant, or misread by the generator.

- Incorporating retrieved context into generation
    - prompt engineering for RAG
    - context window management
- Handling large contexts effectively
    - Fusion-in-Decoder
    - Long-context Transformers
- Mitigating hallucinations through retrieval context management
