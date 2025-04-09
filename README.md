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
- Optimization
    - Gradient descent
        - concept
        - formula
        - code
    - SGD
    - Momentum
    - RMSprop
    - Adam
    - AdamW
    - Muon (Recent)

      
- Loss functions
    - Logistic loss function
    - Cross Entropy
    - Hinge loss (SVM)

- Model evaluation and selection
    - Evaluation
        - TP, FP, TN, FN
        - Accuracy, precision, recall/sensitivity, specificity, F1-score
            - how do you choose among these? (imbalanced data)
            - precision vs TPR (why precision)
        - ROC curve (TPR vs FPR, threshold selection)
        - AUC (model comparison)
        - Extension of the above to multi-class classification
        - Confusion matrix
 
    - Bias/Variance
        - Concept
        - Underfitting/overfitting
        - Regularization
        - L0,L1,L2,L_infinity    
    - Feature selection
    - Data
        - Missing data
        - Imbalanced data
        - Distribution shifts
    - Sampling
        - Uniform sampling
        - Reservoir sampling
        - Stratified sampling
    - Model selection
        - K-fold cross validation (good k?)

### ML Algorithms
#### Algorithm Categories
- Supervised, unsupervised, and semi-supervised learning (with examples)
    - Classification vs regression vs clustering
- Parametric vs non-parametric algorithms
- Linear vs Nonlinear algorithms


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
- Loss functions in dl
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
        - Related metrics

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
- Catastrophic forgetting mitigation





### RAG


### Q&A System 

### Reasoning
- Information seeking

### Personalized Rec-system

### Recent interesting papers





## Reference
1. [Machine Learning/Data Science Interview Cheat sheets by Aqeel Anwar](https://sites.google.com/view/datascience-cheat-sheets#h.h40dwqqwv30w)
























