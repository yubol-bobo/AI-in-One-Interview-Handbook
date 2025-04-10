# PhD-ML-DL-LLM-Interview-Handbook

## Table of Contents

- [Machine Learning](#Machine-Learning)
    - [ML Basic](#ML-Basic)
    - [ML Algorithms](#ML-Algorithms)
        - [Algorithm Categories](#Algorithm-Categories)  
        - [Supervised Learning](#supervised-Learning)  
        - [Unsupervised Learning](#Unsupervised-Learning)  

- [Deep Learning](#Deep-Learning-Breadth)
    - [DL Basic](#dl-basic)
    - [DL Algorithms](#dl-algorithm)
- [natural Language Processing](#nlp)
- [Large Language Models](#Large-Language-Models)


 




## Machine Learning

### ML Basic
#### Gradient descent
- (Batch) Gradient descent [[ref](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)] [[B站](https://www.bilibili.com/video/BV1jh4y1q7ua/?spm_id_from=333.337.search-card.all.click&vd_source=c86f14ec33e79f08f7e2278747a071e8)]
    - The gradient $\left(\nabla_\theta J(\theta)\right)$ represents the direction of steepest ascent-the direction where the function increases the fastest. To minimize the function, you want to move in the opposite direction, i.e., the direction of steepest descent.
    - Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).
    - formula: $$\theta = \theta - \alpha \nabla_\theta J(\theta)$$, where $\theta:$ Parameters to be optimized. $\alpha$ : Learning rate (step size). $\nabla_\theta J(\theta)$ : Gradient of the cost function w.r.t. parameters.

- GD Variants
    - (Batch) GD: All data points at once
    - SGD: one data point per iteration
        - Updates parameters using only one randomly chosen training example at each step.
        - Formula: $$\theta=\theta-\alpha \nabla_\theta J\left(\theta ; x^{(i)}, y^{(i)}\right)$$.
    - Mini-batch GD: Small subset per iteration.
    - Cons:
        - Sensitivity to learning rate
        - Risk of getting stuck at saddle points
        - Premature convergence in flat regions: local minimum.

All following advanced optimization algorithms improve parameter updates by adjusting both the **gradient calculation (gc)** and the **learning rate strategy (lr)**.

- **Momentum (gc)** enhances standard gradient descent by adding a velocity term that accumulates past gradients. This velocity term smooths the updates, reducing oscillations and accelerating convergence, particularly in scenarios with noisy gradients or ill-conditioned optimization surfaces.
    - Pros: faster, jump out of local minimum, stable training.

- **Adagrad (lr)** uses adaptive learning rates for each parameter, automatically adjusting LR during training. Larger learning rates for infrequent parameters, smaller rates for frequent parameters.
    - Formula:  $$G_t=G_{t-1}+\left(\nabla_\theta J(\theta)\right)^2, \quad \theta=\theta-\frac{\alpha}{\sqrt{G_t+\epsilon}} \nabla_\theta J(\theta)$$

- **RMSprop (lr)**: Adagrad adapts the learning rate individually for each parameter, but its main drawback is the continual accumulation of squared gradients. Over time, this causes the learning rate to shrink excessively, sometimes stopping learning prematurely. RMSprop addresses this limitation by introducing an exponential decay factor. Instead of continuously accumulating all past squared gradients, it maintains a moving average of recent squared gradients.

Adagrad (accumulates indefinitely):

$$G_t=G_{t-1}+\left(\nabla_\theta J(\theta)\right)^2, \quad \theta=\theta-\frac{\alpha}{\sqrt{G_t+\epsilon}} \nabla_\theta J(\theta)$$


RMSprop (moving average):

$$\begin{aligned} & E\left[g^2\right]_t=\beta E\left[g^2\right]_{t-1}+(1-\beta)\left(\nabla_\theta J(\theta)\right)^2 \\ & \theta=\theta-\frac{\alpha}{\sqrt{E\left[g^2\right]_t+\epsilon}} \nabla_\theta J(\theta) \end{aligned}$$

- $\beta \approx 0.9$ : Exponential decay rate controlling the moving average.

- Adam
- AdamW
- Muon (Recent)

| Method | Adaptive LR | Momentum | Memory | Stability | Typical Use |
|--------|-------------|----------|--------|-----------|-------------|
| SGD | ❌ No | ❌ No | Low | Medium | Fast updates |
| Momentum | ❌ No | ✅ Yes | Low | Medium | Reduces oscillations |
| RMSprop | ✅ Yes | ❌ No | Medium | High | Adaptive LR |
| Adam | ✅ Yes | ✅ Yes | Medium | High | Most popular default |
| AdamW | ✅ Yes | ✅ Yes | Medium | High | Improved regularization |
| Muon | ✅ Yes | ✅ Yes | Low | High | Recent advancement |

      
#### Loss functions
- Logistic loss function
- Cross Entropy
- Hinge loss (SVM)

#### Model evaluation and selection
- Evaluation
    - TP, FP, TN, FN

    - Metrics:
        - Accuracy: Overall correctness
        - Precision ($\frac{T P}{T P+F P}$): Out of predicted positives, how many were actually positive?
        - Recall/sensitivity/TPR ($\frac{T P}{T P+F N}$): 	Out of actual positives, how many were predicted correctly?
        - Specificity/TNR ($\frac{T N}{T N+F P}$): Out of actual negatives, how many were correctly predicted negative?
        - F1-score ($2 \times \frac{\text { Precision } \times \text { Recall }}{\text { Precision }+ \text { Recall }}$): Harmonic mean balancing precision and recall. For multi-class problems, use macro or weighted averages. 
        - **How to choose**: Accuracy is misleading with imbalanced data. Precision, if avoiding false positives is crucial (e.g., spam detection). Recall (Sensitivity), if missing a positive case is costly (e.g., cancer detection, fraud detection). F1-score, if both precision and recall matter equally. Specificity, if correctly identifying negative cases is essential (medical tests).
        - Precision vs Recall: Precision clearly matters when the cost of a FP is high. (e.g., classifying email as spam—high precision avoids wrongly marking important emails). Recall (TPR) clearly matters when the cost of missing positives (FN) is very high (e.g., disease diagnosis—high recall ensures positive cases aren't missed).

    - ROC (Receiver Operating Characteristic) curve (TPR vs FPR, threshold selection)
        - ROC curve plots TPR (Recall) vs FPR (1 - Specificity) at various threshold levels.Y-axis (TPR/Recall): Correctly identified positives; X-axis (FPR): Incorrectly identified positives (False Alarms). Choose threshold clearly to maximize TPR and minimize FPR.
    - AUC (model comparison)
        - AUC clearly measures the model’s overall capability to distinguish classes irrespective of threshold. Model with higher AUC is generally better at distinguishing classes clearly.
        - Range [0.5, 1]
    - Confusion matrix: Confusion matrix clearly visualizes all predictions vs actual classes:

    <div align="center">
        <img src="figs/confusion_matrix.png" width="50%">
    </div>

- Bias/Variance [[ref](https://traintestsplit.com/bias-vs-variance-in-machine-learning/)]
    - Bias: The bias is the simplifying assumption made by the model to make the target function easy to learn. Low bias suggests fewer assumptions made about the form of the target function. High bias suggests more assumptions made about the form of the target data. The smaller the bias error the better the model is. If, however, it is high, this means that the model is **underfitting** the training data. 
    - Variance: Variance is the amount that the estimate of the target function will change if different training data was used. The target function is estimated from the training data, so we should expect the algorithm to have some variance. Ideally, it should not change too much from one training dataset to the next. This means that the algorithm is good at picking out the hidden underlying mapping between the inputs and the output variables. If the variance error is high this indicates that the model **overfits** the training data.
    - If our model is too simple and has very few parameters then it may have high bias and low variance. On the other hand if our model has large number of parameters then it’s going to have high variance and low bias. So we need to find the right/good balance without overfitting and underfitting the data. 

    <div align="center">
        <img src="figs/bias-vs-variance.png" width="60%">
    </div>

    - Underfitting/overfitting
        - Underfitting occurs when the model is too simple to capture underlying data trends. Solution:Increase model complexity (e.g., add features, use more complex models, reduce regularization).
        - Overfitting occurs when the model captures noise or random fluctuations (memorize every single detail) in training data. Solution: Reduce model complexity (e.g., remove features, apply regularization, use simpler models); Increase training data size; Early stopping.
    - Regularization: Regularization, such as L1 (LASSO) or L2 (Ridge), can be used to control model complexity to prevent overfitting and tackle high variance. They work by adding a penalty term to the magnitude of the coefficients.
        General form of regularized loss function:

        ![Regularized Loss Formula](https://latex.codecogs.com/svg.latex?\operatorname{Loss}(w)=\operatorname{Loss}_{\text{original}}(w)+\lambda\cdot{R}(w))


        - L0 is a technique in ML to encourage sparsity in a model's parameters. It penalizes the number of **non-zero parameters** in a model.

            ![L0 Formula](https://latex.codecogs.com/svg.latex?\|w\|_0=\sum_{i=1}^n\mathbb{I}(w_i\neq0))

            Use case: Feature Selection: L0 regularization is particularly useful in scenarios where the number of features is large, and only a small subset is expected to be relevant. It helps to automatically select a subset of features that contribute significantly to the model's performance.

        - L1 penalizes **absolute magnitude** of weight: 

            ![L1 Formula](https://latex.codecogs.com/svg.latex?R%28w%29%3D%5C%7Cw%5C%7C_2%5E2%3D%5Csum_%7Bi%3D1%7D%5En%20w_i%5E2)

            Encourages sparsity through geometry.
        
        - L2 penalizes **square magnitude** of weights:

            ![L2 Formula](https://latex.codecogs.com/svg.latex?R%28w%29%3D%5C%7Cw%5C%7C_2%5E2%3D%5Csum_%7Bi%3D1%7D%5En%20w_i%5E2)

            Penalizes large weights heavily, encourges smaller, diffuse weigths. Handles correlated features better.

        - L-infinity penalizes the **largest absolute weight** (maximum norm):

            ![L-infinity Formula](https://latex.codecogs.com/svg.latex?\|w\|_\infty=\max_{i}|w_i|)

            Constrains the maximum coefficient magnitude, balancing all weights' magnitudes uniformly

        - When prefer L1 over L2? For explicit feature selection or interpretability
        - When prefer L2 over L1? L2 (Ridge) shrinks correlated features toward each other, effectively managing correlated features.
        - What happens as $\lambda \rightarrow \infty$ for L1 and L2?  For both: parameters approach zero; L1 produces exact zeros faster.
        - Is L0 convex? Why does it matter? No, it's non-convex. This matters due to optimization difficulty and computational complexity.




    - Feature selection: Proper Feature Selection removes irrelevant features thereby reducing both bias and variance. It can be done through various methods like backward elimination, forward selection, and recursive feature elimination.
        - Backward Elimination: Start with all features, then iteratively remove the least significant feature.
        - Forward Selection: Start with no features, then teratively add the most significant feature until no improvement.
        - Recursive Feature Elimination (RFE): Iteratively fits a model and removes least important features based on coefficients or importance metrics.
        - Embedded Methods (L1 Regularization / Lasso):
            - Feature selection happens simultaneously with model fitting

- Data
    - Missing data
        - MICE (Multiple Imputation by Chained Equations): It's an iterative method of imputing missing data where each missing feature is repeatedly modeled and filled based on other features. It creates multiple datasets to capture uncertainty properly.
    - Imbalanced data: Imbalanced data occurs when one class significantly outnumbers the other(s), causing biased models toward the majority class.
        - Oversampling: Duplicate minority examples 
            - SMOTE (Synthetic Minority Oversampling Technique): 1. Chooses a random minority instance; 2. Finds k nearest minority neighbors; 3. Creates synthetic points along the line segments joining neighbors.
            - ADASYN (Adaptive Synthetic Sampling): Generates more synthetic data for harder-to-classify minority instances (closer to majority class boundary).
        - Undersampling: Remove majority examples strategically.
            - ENN (Edited Nearest Neighbor): Removes majority samples misclassified by a k-NN classifier (noisy or ambiguous instances).

    - Distribution shifts: A distribution shift happens when the data seen during training differs from the data encountered during deployment or testing. It breaks the core assumption that train and test data come from the same distribution.
        - Covariate Shift (X changed): This occurs when the input distribution $P(x)$ changes, but the conditional distribution $P(y \mid x)$ stays the same. Example: A spam detection model trained on emails from last year might see different phrasing this year, even if what qualifies as spam hasn’t changed.

        - Label Shift (Y changed): Here, the label distribution $P(y)$ changes, but the distribution of features given the label $P(x \mid y)$ remains the same. Example: In medical data, the proportion of patients with a certain condition may increase in a new population.
        - Concept Drift (f changed): The conditional distribution $P(y \mid x)$ itself changes. Example: A recommendation model may degrade as user preferences evolve.

- Sampling
    - Uniform sampling selects items from a dataset with equal probability for all elements.  Code: random.sample(data,k).  Time Complexity: $O(k)$
    - Negative sampling primarily used in recommendation systems and NLP (Word2Vec) to train efficiently on sparse data. It reduce computational cost by sampling a subset of negative examples (items not interacted with or words not co-occurring). (Word2Vec)  Time Complexity: Reduces computation from $O(N)$ (all negatives) to $O(k)$ (sampled negatives).
    - Reservoir sampling: Technique to uniformly sample $k$ items from a stream or large unknown-size dataset in one pass.
        - Use Cases: Sampling from streaming data or very large files/databases.
        - Algorithm: Maintain a reservoir of size $k$. For first $k$ items: directly fill reservoir. For each subsequent item $i>k$ : With probability $k / i$, randomly replace an item in the reservoir.
        - Time: $O(N)$, single pass.
        - Memory: $O(k)$, fixed space.
    - Stratified sampling: Technique where the dataset is divided into distinct subgroups (strata) based on certain attributes (e.g., age, gender, class), and samples are drawn from each subgroup proportionally or with specific representation.
        - Code: df.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=frac))
    - Questions
        - When would you prefer reservoir sampling over uniform random sampling? If you're sampling tweets from a live Twitter stream (which is continuously updating and has unknown total size), reservoir sampling is the ideal method to guarantee uniformity while using fixed memory.
        - How does stratified sampling reduce variance? If you're conducting a survey on voting preferences and you stratify by age groups (young, middle-aged, elderly), each subgroup's responses tend to be more similar internally than across the entire population. This homogeneity decreases the variance of estimates significantly compared to simple random sampling.




- Model Selection
    - K-fold Cross-validation is a resampling procedure used to evaluate machine learning models and tune hyperparameters.
    - $K=5$ or $K=10$ are most common.
    - Trade-off (Bias vs. Variance):
        - Large $K$ :
            - Pros: Low bias, more accurate estimates.
            - Cons: Higher variance, computationally expensive.
        - Small $K$ :
            - Pros: Faster computation, lower variance in estimates.
            - Cons: Higher bias, less stable estimates.
    - How to handle imbalanced datasets in K-fold CV? Use Stratified K-fold, ensuring each fold maintains the proportion of classes from the original dataset.
    - Does K-fold cross-validation prevent overfitting? No, it helps evaluate and select models accurately but doesn't inherently prevent overfitting. Use regularization, early stopping, or simpler models to control overfitting.
    - Whey k-fold CV might not be ideal? Cross-validation isn’t always ideal because it can become computationally expensive, particularly with complex models or very large datasets. It's also unsuitable for time-series data due to temporal dependencies that violate the assumption of independence.

- Hyper-Parameter Tuning
    - Grid search: Exhaustively evaluates all possible combinations of hyper-parameters provided in a predefined grid.
    - Random search: Randomly samples hyper-parameter values within predefined ranges.
    - Bayesian optimization: Uses past evaluation results to model the hyper-parameter space.
        - Popular libraries: `Hyperopt`, `Optuna`, `BayesSearchCV`. 
    - Other autoML packages: `Auto-skearn`, `h2o`.






### ML Algorithms
#### Algorithm Categories
- Types by (labeled) data
    - **Supervised Learning** learn from labeled training data. Each training example consists of input features and the output label.
        - Classification: Predicts categorical outputs (e.g., spam detection, image recognition). Algorithms: Logistic regression, Decision tree, RF,KNN, SVM.
        - Regression: Predicts continuous numerial outputs. (e.g., house price prediction, stock market forecasting). Algorithms: Linear regression, RF.
    - **Unsupervised Learning** learn patterns from unlabeled data. They explore structures, clusters, or features without labels. (e.g., Customer segmentation, dimensionality reduction)
        - Clustering: K-means, DBSCAN, Hierarchical clustering.
        - Dimensionality  Reduction: PCA, t-SNE.
    - **Semi-supervised Learning** combines small amounts of labeled data with large amounts of unlabeled data. Useful when labeling data is expensive or difficult. The model leverages labeled data to guide learning, while also making use of unlabeled data to generalize better and improve performance.
    - **Reinforcement Learning** learn through trial-and-error, reward-based systems. Gaining feedback from interactive environment instead of given data.
    - **Self-supervised Learning** involves creating "pseudo-labels" from the unlabeled data itself. The model learns a meaningful representation of the data by predicting parts of the input or generating transformations of the input. (BERT)

- Parametric vs non-parametric algorithms (if we assume a fixed functional form or not) 

(Functional form = Mathematical shape or structure that the model assumes to represent how the input relates to the output.)
    - **Parametric Algorithms**(e.g., Logistic Regression, Naive Bayes, Neural Networks) assume a fixed functional form with a finite set of parameters. They are computationally efficient and often easier to interpret, but less flexible.
    - **Non-parametric methods** (e.g., KNN, Decision Trees, Random Forests) make fewer assumptions, allowing more flexibility. Their complexity grows with the data, making them suitable for capturing complex patterns.

- Linear vs Nonlinear algorithms
    - Linear algorithms (Linear Regression, Logistic Regression, Linear SVM) assume a linear relationship and provide simplicity and interpretability.
    - Nonlinear algorithms (Decision Trees, Random Forest, Neural Networks, Kernel SVM) capture more complex patterns and relationships, offering flexibility but potentially at the cost of interpretability and efficiency.


#### Supervised Learning
- Linear Algorithms
    - K-Nearest Neighbors (KNN)
        - distance 
    - Linear Regression
        - Least squares, residuals, linear vs multivariate regression 
    - Logistic Regression
        - Cost function(equation, code), sigmoid function, cross entropy
    - Support Vector Machines
    - Naive Bayes
    - Linear discriminant analysis
      
- Decision Trees
    - Logits
    - Leaves
    - Training algorithm+stop criteria
    - Inference
    - Pruning

- Ensemble methods
    - Bagging vs Boosting
    - Random Forest
    - Boosting
        - Adaboost
        - GBM
        - XGBoost

          
   

#### Unsupervised Learning
- Clustering
    - Centroid models: k-means clustering
    - Connectivity models: Hierarchical clustering
    - Density models: DBSCAN
- Gaussian mixture models
- Latent Mixture Models
- Hidden Markov Models(HMMs)
    - Markov processes
    - Transition probability and emission probability
    - Viterbi algorithm
- Dimension reduction techniques
    - PCA
    - Independent Component Analysis (ICA)
    - T-SNE   
    - UMAP    










## Deep Learning 

### DL Basic
- Loss functions in DL
    - Cross entropy
    - Mean Squared Error
- Feedforward NNs
- Backpropagation
- Dropout
    - How to apply dropout to LSTM
- Vanishing/exploding gradient problem
- Activation functions
- Regularization/ Normalization
    - Batch normalization
    - Layer normalization
    - Early stopping
- Learning Rate
    - step decay
    - exponential decay
    - consine annealing

  

### DL Algorithm
#### CNN
#### RNN
#### LSTM
- Bi-LSTM
- GRU vs LSTM
#### GAN & Autoencoders
- Generative adversarial networks details
    - Generator vs discriminator
    - Common issues (model collapse, vanishing gradients)
- Autoencoders
    - Basic
    - Variational autoencoders (VAE)
    - Reconstruction loss, KL-divergence
#### VAE
#### Transformer
- Attention
    - details
    - Self-attention
    - Cross-attention
- BERT
- RoBERTa
- GPT-2,GPT-3, GPT-4
- T5, XLNet



## NLP

### NLP Basic
#### Text pre-processing
- Tokenization
- Stemming vs lemmatization
- Stop-word
- Punctuation and noise removal
- Text normalization(lowercasing, numbers, dates)

####Text representation
- Bag-of-Words(BoW)
- TF-IDF weighting


### NLP Tasks and Algorithms
#### Language modeling
- n-gram models
- Smoothing methods (Laplace, Good-turing)

#### Text Classification
- Naive Bayes (multinomial NB)
- Logistic regression with TF-IDF
- SVM for text classification

#### Text Similarity and Information Retrieval
- Cosine similarity
- Jaccard similarity
- Document retrieval

#### NER
- CRF (Conditional Random Fields)
- HMM
- De-identification


### Topic Modeling
#### Latent Semantic Analysis (LSA)
#### Latent Dirichlet Allocation (LDA)
- Generative process intuition
- Gibbs sampling (basic intuition)


### Word Embedding (Pre-LM Era)
#### Word2Vec (CBOW and Ski-gram architectures)
#### GloVe
#### FastText (Subword embeddings)
#### Evaluation of embeddings (semantic similarity, analogy tasks)
#### Contextual embedding optimization




## Large Language Models

### LLM Basic
#### Embedding
- how to train    
- word embedding (Post-LM Era)
    - Contextual embedding (BERT, GPT embeddings)
- positional embedding
    - absolute positinal embedding
    - Relative positional embedding
    - RoPE
    - when it's not that important?


#### Tokenization
- Types
    - Byte Pair Encoding (BPE)
    - WordPiece, SentencePiece, Unigram LM tokenizer
- Vocabulary choice implications
- Tokenization impact on model perfromance (OOV handling, multilingual scenarios)



#### Architecture
- Encoder-Decoder (e.g., original Transformer, T5, BART)
- Encoder-only (e.g., BERT, RoBERTa)
- Decoder-only (e.g., GPT series)
- Comparison & use-cases for each architecture type


#### Stages
- Pre training
    - Masked Language Modeling (MLM)
    - Causal Language Modeling (CLM)
- Post training
- Data Quality and Selection
    - High-quality data selection strategies (deduplication, filtering, diversity)
    - Data cleaning techniques for LLMs
- Fine-Tuning 
    - SFT
        - Instruction Fine-Tuning (IFT)
        - SFT vs IFT
        - Dataset curation for SFT and IFT

- Alignment & Human Feedback
    - RLHF
        - Motivation behind combining SFT and RLHF
        - Human feedback collection & augmentation techniques
        - Related metrics for RLHF (reward model accuracy, preference modeling)
        - Algorithms
            - PPO
            - DPO (How DPO augments human feedback effectively)
            - KPO

    - RLAIF
        - Differences from RLHF
        - Advantages and potential risks compared to RLHF





### LLM Engineering

#### Distributed training
- Data parallel
- Pipeline parallel
- Tensor parallel
- Sequence(Activation) parallel
- ZeRO (Zero Redundancy Optimizer) parallel
- Expert (MoE) parallel
- Hybrid parallel
- Choosing the right parallelism strategy (memory, efficiency, complexity trade-offs)


| Method                     | What is Split?                  | Type of Parallelism             |
|----------------------------|---------------------------------|---------------------------------|
| **Tensor Parallelism**     | Weights/tensors                 | Model parallelism               |
| **Pipeline Parallelism**   | Layers                          | Model parallelism               |
| **Expert Parallelism (MoE)** | Experts (modules)               | Model parallelism (MoE)         |
| **Data Parallelism**       | Input data batches              | **NOT** model parallelism       |
| **ZeRO Parallelism**       | Parameters, optimizer states    | Parameter sharding (special case)|






#### Fine-Tuning
- Full fine-tuning vs Parameter-efficient fine-tuning
- When and why to freeze layers
- Layer-wise learning rates & layer freezing strategies
- SFT Data Construction 

- Catastrophic forgetting mitigation
    - add general domain data as well 1:5 - 1:10
- LoRA/QLoRA
- GPT Fine-tuning


#### Optimization and Efficiency
- Mixed precision training (FP16, BF16)
- Gradient accumulation
- Memory optimization technique (activation checkpointing, grading checkpointing)


#### Evaluation and Monitoring
- Evaluation metrics
    - Perplexity
    - Human-evaluated metrics (helpfulness, harmlessness, alignment)
    - Automated evaluation metrics (ROUGE, BLEU, BERTScore)
- Monitoring and Logging 
    - wandb
    - TensorBoard






### RAG
#### Retrieval
- Sparse retrieval (BM25, TF-IDF based retrival)
- Dense retrieval (Embedding-based retrieval)
- Hybrid retrieval

#### Embedding models
- Open-source models (SentenceTransformers, E5, GTE)
- Commericial APIs (OpenAI Embedding, Cohere embedding)
- How to choose embedding models
    - Semantic vs lexical relevance
    - Embedding dimension, speed, accuracy trade-offs
- Evaluation 
    - Retrieval accuracy metrics(Recall@k, MRR)
    - Semantic similarity tasks (STS benchmarks)

#### Chunking and Vectorization
- Importance of chunking
    - Optimal chunk size
    - Overlapping vs non-overlapping chunking
- Chunking strategies
    - Semantic chunking vs fixed-size chunking
    - Recursive chunking methods
- Impact of chunking strategy on retrieval quality and generation accuracy

#### Re-ranking
- Importance
- Methods of re-ranking
    - Cross-encoders (dense re-rankers)
    - Traditional re-ranking (score normalization ,query expansion)
    - Learned re-ranking (supervised methods)
- Evaluation
    - MAP
    - MRR
    - Precision @k

#### Generation
- Incorporating retrieved context into generation
    - prompt engineering for RAG
    - context window management
- Handling large contexts effectively
    - Fusion-in-Decoder
    - Long-context Transformers
- Mitigating hallucinations through retrieval context management




### Q&A System 
#### Types of QA Systems
- Factoid (simple answer, extractive)
- Non-factoid (explanatory, abstractive)
- Open-domain vs Closed-domain Q&A
- Traditional Q&A vs RAG vs LLM-based Q&A

#### Components of QA Systems
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


#### Evaluation
- Extractive Q&A Evaluation
    - Exact Match
    - F1-score
- Generative Q&A Evalution
    - ROUGE
    - BLEU
    - METEOR score
    - Human evaluation


### Reasoning
- Information seeking

### Personalized Rec-system

### Recent interesting papers





## Reference
1. [Machine Learning/Data Science Interview Cheat sheets by Aqeel Anwar](https://sites.google.com/view/datascience-cheat-sheets#h.h40dwqqwv30w)
























