# NLP (Pre-LLM Era)

[Handbook contents](../README.md#table-of-contents) | [Previous: Transformer](transformer.md) | [Next: Large Language Models](large-language-models.md)

## Contents

- [Rule-Based Era (1960s-1980s)](#rule-based-era-1960s-1980s)
- [Statistical & ML Era](#statistical--ml-era)
- [Embedding & Deep‑Learning Era](#embedding--deeplearning-era)


Natural Language Processing (NLP) encompasses the interaction between computers and human languages, enabling machines to understand, interpret, and produce human text and speech. From basic tasks such as text parsing to more advanced ones like sentiment analysis and translation, NLP is integral to various applications, including virtual assistants, content classification, and information retrieval systems.

## Rule-Based Era (1960s-1980s)
- Key Innovation: Handcrafted linguistic rules.
- Core Techniques
    - Pattern matching & templates. Ex. ELIZA (1966): Pattern‑matching “therapist” bot.
    - Heuristic grammars (e.g. keyword spotting + script rules)
- Strengths & Limitations
    - ✅ Transparent, easy to debug.
    - ❌ Brittle, no real “understanding,” poor scalability.

## Statistical & ML Era

Statistical models learned parameters from data—first n‑grams, then classical ML.

- Text Pre‑processing
    - Tokenization is the process of breaking longer text into discrete units, or tokens, which could be words, n-grams, or characters. Common tokenization strategies include splitting by whitespace, punctuation, or specific vocabularies. Tokenization is a foundational step for various text processing tasks and is crucial for tasks like neural network for NLP.
    - Stemming vs. Lemmatization: Both lemmatization and stemming are methods for reducing inflected words to their root forms.
        - **Stemming** uses an algorithmic, rule-based approach to cut off word endings, producing the stem. This process can sometimes result in non-real words, known as "raw" stems. Example: The stem of "running" is "run."
            - Code:
                ```Python
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                stem = stemmer.stem("running")
                ```
        - [To be reviewed] **Lemmatization** maps inflected forms to dictionary lemmas using morphological information; handling ambiguity may require POS and context. NLTK WordNetLemmatizer with a supplied POS does not itself infer the full sentence context.
            - Code:
                ```Python
                from nltk.stem import WordNetLemmatizer
                # Requires WordNet data: nltk.download("wordnet") once after importing nltk.
                lemmatizer = WordNetLemmatizer()
                lemma = lemmatizer.lemmatize("running", pos="v")  # Need to specify part of speech (pos)
                ```
        - When to choose?
            - Stemming: Useful for tasks like text classification when speed is a priority. The technique is simpler and quicker than lemmatization, but it may sacrifice precision and produce non-real words.
            - [To be reviewed] Lemmatization can help a sparse lexical pipeline when preserving dictionary forms improves retrieval or interpretability, but its benefit is task/language dependent. Evaluate it against unchanged text.
            [To be reviewed] Modern pretrained neural tokenizers generally expect text processed according to their original normalization rules. Adding stemming or lemmatization can remove tense, negation-related structure, or entity distinctions and induce distribution shift.


    - [To be reviewed] Stopword removal can reduce sparse-feature dimensionality, but words such as negation and function words can carry crucial task information. Preserve them when needed and compare on validation data. Do not automatically apply a bag-of-words stopword policy to a pretrained language model.

- Shallow Text Representations
    - Bag‑of‑Words (BoW): The Bag of Words model, or BoW, is a fundamental technique in Natural Language Processing (NLP). This model disregards the word order and syntax within a text, focusing instead on the presence and frequency of words.
        - Working Mechanism
            - Text Collection: Gather a set of documents or a corpus.
            - Tokenization: Split the text into individual words, known as tokens.
            - Vocabulary Building: Identify unique tokens, constituting the vocabulary.
            - Vectorization: Represent each document as a numerical vector, where each element reflects word presence or frequency in the vocabulary.
        - Code
            ```python
            from sklearn.feature_extraction.text import CountVectorizer

            # Sample data
            corpus = [
                'This is the first document.',
                'This document is the second document.',
                'And this is the third one.',
            ]

            # Create BoW model
            vectorizer = CountVectorizer()
            X = vectorizer.fit_transform(corpus)

            # Visualize outputs
            print(vectorizer.get_feature_names_out())
            print(X.toarray())
            ```
            The output shows the unique feature names (vocabulary) and the BoW representations of each document.

        - Limitations
            - Loss of Word Order: Disregarding word dependencies and contextual meanings can hinder performance.
            - Loss of word meaning: Fails to distinguish meanings of polysemous words (words with multiple meanings) or homonyms.
            - Dimensionality: The vector's length equals the vocabulary size, getting unwieldy with large corpora.
            - Out-of-Vocabulary Words: Struggles to handle new words or spelling variations.

    - TF–IDF [[YouTube](https://www.youtube.com/watch?v=zLMEnNbdh4Q)] (Term Frequency-Inverse Document Frequency) is a numerical statistic used in natural language processing and information retrieval to indicate the importance of a word in a document relative to a collection of documents (corpus).
        - Term Frequency (TF): The term frequency measures how often a term appears in a document. $`\text{TF}(t, d)=\frac{f_{t, d}}{\sum_{t^{\prime} \in d} f_{t^{\prime}, d}}`$, $`f_{t, d}`$ represents the raw count of term $`t`$ in document $`d`$. The denominator sums the raw counts for all terms in document $`d`$ to normalize the frequency.
        - [To be reviewed] One common smoothed IDF convention is $`\operatorname{idf}(t)=\log[(1+N)/(1+\operatorname{df}(t))]+1`$, as used by scikit-learn when smoothing is enabled. Other systems use different formulas. State the convention, TF weighting, and vector normalization rather than mixing formulas across implementations. [TF-IDF reference](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
        - $`\text{TF}-\text{IDF}(t, d, D)=\text{TF}(t, d) \times \text{IDF}(t, D)`$. The product rewards within-document frequency and corpus rarity under the stated TF/IDF convention. For example, scikit-learn TfidfVectorizer defaults use raw-count TF, smoothed IDF, and L2 vector normalization. The formula above instead divides TF by document length; with the same IDF and subsequent L2 normalization, this positive per-document scale cancels, yielding the same nonzero vector. Other TF transforms, such as logarithmic TF, need not be equivalent.

- Language Modeling
    - N-grams are sequential word or character sets, with "n" indicating the number of elements in a particular set. They play a crucial role in understanding context and text prediction, especially in statistical language models.
        - Types: Unigrams: Single words $`w_i`$; Bigrams: Pairs of words $`w_{i-1}, w_i`$; Trigrams: Three-word sequences $`w_{i-2}, w_{i-1}, w_i`$.
        - Code
        ```python
        # In a notebook, first install NLTK with: %pip install nltk
        import nltk
        nltk.download('punkt_tab')

        from nltk import ngrams, word_tokenize
        # Define input text
        text = "This is a simple example for generating n-grams using NLTK."

        # Tokenize the text into words
        tokenized_text = word_tokenize(text.lower())

        # Generate different types of N-grams
        unigrams = ngrams(tokenized_text, 1)
        bigrams = ngrams(tokenized_text, 2)
        trigrams = ngrams(tokenized_text, 3)

        # Print the generated n-grams
        print("Unigrams:", [gram for gram in unigrams])
        print("Bigrams:", [gram for gram in bigrams])
        print("Trigrams:", [gram for gram in trigrams])
        ```

    - Smoothing: In n‑gram language models, maximum likelihood estimation often assigns a zero probability to n‑grams that do not occur in the training data. Smoothing techniques redistribute some probability mass from frequent (observed) n‑grams to those unseen in order to: Prevent zero probabilities (which can cause issues in probability chains or likelihood computations); Improve the model’s generalizability when encountering new text.
        - Laplace (Add‑One) Smoothing is one of the simplest methods available. It adds one to every count in the frequency table, ensuring no event has a zero probability. Despite its simplicity, this method can be overly naive—especially when the vocabulary size is large, since it tends to overestimate the probability of rare (or unseen) events.
            - For an n-gram $`w_1, \ldots, w_n`$ with count $`c\left(w_1, \ldots, w_n\right)`$, the smoothed probability is: $`P\left(w_n \mid w_1, \ldots, w_{n-1}\right)=\frac{c\left(w_1, \ldots, w_n\right)+1}{c\left(w_1, \ldots, w_{n-1}\right)+V}`$
                where:
                - $`c\left(w_1, \ldots, w_n\right)`$ is the count of the n-gram.
                - $`c\left(w_1, \ldots, w_{n-1}\right)`$ is the count of the ( $`\mathrm{n}-1`$ )-gram context.
                - $`V`$ is the vocabulary size.
            - Laplace smoothing is computationally simple but may not be optimal for larger vocabularies where its additive nature overly penalizes higher-frequency events.
        - Good–Turing Smoothing focuses on adjusting the probability estimates based on the frequency of frequencies—or counts‑of‑counts. The key observation is that the probability mass “lost” to unseen events can be estimated by how many events have been seen once, twice, etc.
        - Kneser–Ney smoothing is regarded as one of the most effective and sophisticated smoothing methods for n‑gram models. Its power lies in two key aspects:
            - Absolute Discounting: Subtract a constant discount $`D`$ from each nonzero n-gram count.
            - Continuation Probability: Instead of backing off to the raw frequency of lower-order n-grams, Kneser-Ney considers how likely it is that a word appears as a novel continuation. This is captured by the "continuation probability," which is the proportion of contexts in which a word appears.


- Supervised Learning for NLP
    - Text Classification (Multinomial Naïve Bayes, Logistic Regression on TF–IDF, SVM).
    - Sequence Labeling & NER (Hidden Markov Models (HMM), Conditional Random Fields (CRF), Rule‑based de‑identification).
        - Output: BIO tagging scheme.
- Retrieval, Similarity, and Unsupervised Topic Models
    - [To be reviewed] Similarity Metrics: Cosine, Jaccard

        - Cosine similarity measures vector direction and is insensitive to positive rescaling; Jaccard measures set overlap as intersection size divided by union size. Neither is a universal notion of semantic equivalence.
        - **Example:** two documents with the same set of words have Jaccard similarity one even if negation and ordering reverse the meaning. State the tokenization, normalization, and weighting used to define the representation.

    - [To be reviewed] Document Retrieval (vector space, BM25)

        - Rank documents with a sparse vector-space score or BM25, which combines query-term matches with term-frequency saturation and document-length normalization. Sparse matching is especially useful for exact identifiers, rare entities, and terminology.
        - **Example:** the exact error code `E137` can be decisive even if a semantic encoder treats it as an opaque string. Evaluate retrieval with relevance judgments rather than assuming lexical overlap implies relevance. [Robertson and Zaragoza: BM25](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)

    - Topic Modeling
        - [To be reviewed] LSA

            - Latent Semantic Analysis applies a truncated SVD to a term-document matrix, often TF-IDF weighted, to obtain a low-rank approximation. Nearby coordinates reflect shared corpus co-occurrence structure rather than explicit symbolic topics.
            - **Example:** “car” and “automobile” can become similar when they occur in similar documents despite little direct lexical overlap. Component signs and rotations within degenerate subspaces have no intrinsic semantic interpretation. [Truncated SVD](https://scikit-learn.org/stable/modules/decomposition.html#lsa)

        - LDA
            - [To be reviewed] Generative intuition

                - Here LDA means **Latent Dirichlet Allocation**, distinct from Linear Discriminant Analysis. Each document draws topic proportions; each token draws a topic and then a word from that topic's distribution. Dirichlet priors control document-topic and topic-word concentration.
                - **Example:** a document about AI in hospitals can mix machine-learning and clinical topics. The bag-of-words assumption discards word order; “topic” labels are analyst interpretations of statistical word distributions. [Original LDA paper](https://jmlr.org/papers/v3/blei03a.html)

            - [To be reviewed] Gibbs sampling overview

                - Collapsed Gibbs sampling integrates out topic proportions and word distributions, then repeatedly resamples each token's topic conditional on the other assignments. The conditional combines a smoothed document-topic count with a smoothed topic-word probability.
                - **Practice:** assess convergence across chains and initialization choices; a plausible top-word list is not a convergence diagnostic. Compare held-out predictive fit and topic stability, and keep vocabulary construction within the training protocol.



## Embedding & Deep‑Learning Era

Neural nets brought continuous representations and end‑to‑end learning.

[To be reviewed] The [distributional-hypothesis background](transformer.md#tf-distributional) connects context statistics to learned word representations and explains the move from static lexical vectors to input-dependent states. It is an intellectual motivation for distributional semantics, not a theorem requiring every contextual model to use attention.

- [To be reviewed] Word embeddings represent lexical items by dense vectors learned from a specified objective and corpus. Similarity depends on training, context definitions, and geometry; nearby words may be related or even antonyms rather than substitutable synonyms.
    - Word2Vec (2013) [[知乎](https://zhuanlan.zhihu.com/p/61635013)]
        - [To be reviewed] Word2Vec demonstrated useful distributional and analogy regularities in particular trained embeddings. Vector offsets such as king − man + woman can retrieve queen in some models, but success depends on corpus, vocabulary, scoring, and relations; it is not a guaranteed algebra of meaning.

        [To be reviewed] CBOW and skip-gram simplify the hidden computation used in earlier neural language models to make word-representation training efficient at large corpus scale. Their particular prediction objectives and approximation schemes determine what similarities the learned vectors capture. [Word2Vec](https://arxiv.org/abs/1301.3781)
        - CBOW: predicts the current word based on the context.

        - Skip-gram predicts surrounding words given the current word.
            - Objective function: Maximize the probability of any context word given the current center word.

        - [To be reviewed] Hierarchical softmax computes a word's probability along a tree path. A balanced tree gives $`O(\log V)`$ path length; a frequency-based Huffman tree makes common words cheap on average, but does not guarantee logarithmic worst-case depth for every word. Include embedding width when comparing arithmetic cost.
        - In the follow-up work, negative sampling is proposed.


    - [To be reviewed] GloVe

        - GloVe learns word vectors from global word-word co-occurrence statistics using a weighted regression objective on log counts. A common fitted relation is $`w_i^T\tilde w_j+b_i+\tilde b_j\approx\log X_{ij}`$ for observed co-occurrences.
        - **Example:** related context profiles can make “ice” and “snow” similar. A single static vector still merges different senses of “bank.” Analogy arithmetic is an empirical phenomenon, not a general algebra of meaning. [GloVe paper](https://aclanthology.org/D14-1162/)

    - [To be reviewed] fastText

        - This entry refers to **fastText**: represent a word using learned character n-gram vectors in addition to word-level information. Shared subword structure improves estimates for rare forms and can construct vectors for unseen strings with supported character n-grams.
        - **Example:** “walk,” “walked,” and “walking” share subword features. Morphological similarity is not guaranteed semantic equivalence; spelling errors and domain-specific identifiers still require evaluation. [fastText paper](https://arxiv.org/abs/1607.04606)

    - [To be reviewed] Evaluation: analogy tests, semantic similarity

        - Intrinsic evaluation tests analogy completion or agreement with human similarity judgments. Control vocabulary coverage, frequency, and relation type; cosine similarity may be dominated by anisotropy or frequent shared contexts.
        - **Example:** good performance on geographic analogies need not predict medical entity retrieval. Use a downstream task with frozen embeddings and matched training budgets to test utility, and inspect performance across relevant demographic or domain slices.

    - [To be reviewed] Contextual Embeddings (early): ELMo

        - ELMo forms context-dependent word representations from learned combinations of layers in a bidirectional language model. Its forward and backward recurrent language models are trained with directional objectives; downstream features can combine both directions.
        - **Example:** “bank” near “river” and “bank” near “loan” receive different representations. This addresses static word-vector polysemy but does not remove corpus bias or guarantee that all senses are separated. [ELMo](https://aclanthology.org/N18-1202/)

- Neural Architectures
    - [To be reviewed] Recurrent Nets: RNN → LSTM → GRU.

        - Recurrent encoders update a state sequentially. LSTM and GRU gates create controlled paths for retaining and updating information, improving long-range credit assignment relative to simple recurrent cells.
        - **Trade-off:** sequential time dependence constrains parallelism. Bidirectional encoders can use future input positions for tagging, but a deployed causal predictor must avoid that information leakage.

    - [To be reviewed] Encoder–Decoder & Seq2Seq (2014).

        - An encoder maps a variable-length source into a representation; an autoregressive decoder models a conditional output sequence. The original fixed-vector bottleneck can lose information on long inputs, motivating attention over encoder states.
        - **Example:** translation trains with ground-truth output prefixes but generates with model-produced prefixes, creating a train-generation mismatch often called exposure bias. Evaluate complete generated sequences. [Seq2Seq](https://arxiv.org/abs/1409.3215)

    - [To be reviewed] Attention Mechanism (2014 preprint / ICLR 2015).

        - Additive attention scores each encoder state against the current decoder state, normalizes the scores, and forms a weighted context. It makes relevant source information available at each output step instead of compressing everything into one vector.
        - **Example:** successive target words can attend to different source spans. Attention weights alone do not establish causal explanations of the final output. [Neural translation with alignment](https://arxiv.org/abs/1409.0473)
        - [To be reviewed] Continue with the [seq2seq-to-attention path](transformer.md#tf-seq2seq-attention) and [Bahdanau/Luong/Transformer score comparison](transformer.md#tf-score-history). Scoring functions, self/cross-attention sources, and visibility masks are separate design choices.

    - [To be reviewed] Transformer (2017)

        - Replace recurrence with self-attention and positionwise transformations, permitting parallel computation across known token positions during training. Positional information and masks specify ordering and visibility.
        - **Complexity:** dense self-attention compares all token pairs; training parallelism does not mean autoregressive decoding becomes fully parallel. See [Transformer attention](transformer.md#tf-attention) and [computation/memory costs](transformer.md#tf-cost) for the canonical explanations.

- Subword Tokenization
    - [To be reviewed] Byte‑Pair Encoding (BPE)

        - BPE iteratively merges frequent adjacent symbol pairs to construct a subword vocabulary. It balances reusable small units against shorter token sequences.
        - **Example:** a rare inflected form can be represented by several known pieces rather than one unknown-word symbol. Byte-level coverage requires a byte-level design; it is not guaranteed by every character-based BPE tokenizer. [Subword BPE](https://aclanthology.org/P16-1162/)

    - [To be reviewed] WordPiece / SentencePiece

        - WordPiece is a subword vocabulary/segmentation approach associated with models such as BERT. SentencePiece is a tokenizer implementation supporting algorithms including BPE and unigram language-model tokenization, trained directly from raw text with explicit normalization behavior.
        - **Example:** two tokenizers with equal vocabulary size may split Chinese, source code, and whitespace very differently. Record the tokenizer artifact and normalization rules along with model weights; token IDs are meaningful only relative to that vocabulary. [SentencePiece](https://aclanthology.org/D18-2012/)
