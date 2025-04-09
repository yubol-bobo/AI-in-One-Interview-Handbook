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
    - Linear Regression
        - Least squares, residuals, linear vs multivariate regression 
    - Logistic Regression
        - Cost function(equation, code), sigmoid function, cross entropy
    - Support Vector Machines
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










## Deep Learning 

### DL Basic
- Feedforward NNs
- Backpropagation
- Dropout
    - How to apply dropout to LSTM
- Vanishing/exploding gradient problem
- Activation functions

  

### DL Algorithm
- CNN
- RNN
- LSTM
- GAN
- Transformer
    - Attention
        - details
        - Self-attention
        - Cross-attention
    - BERT
    - RoBERTa



## Large Language Models

### NLP Basic


### Large Language Model Basic
- Embedding
    - how to train    
    - word embedding: Word2Vec/Glove
    - positional embedding
        - when it's not that important
    - Tokenization
- Architecture
    - Encoder-decoder vs encoder-only vs decoder-only
    - 123
- Stages
    - Pre training
    - post training
    - select high quality data
    - LLM training stages and 作用
    - SFT vs IFT, data selection
    - RLHF
        - Why RLHF with SFT existed?
        - Related metrics
        - PPO
        - DPO
            - How DPO augment human feedbacks
        - KPO
    - RLAIF
        - RLHF vs RLAIF

### LLM Engineering

- Distributed training

### Quick Summary of Key Terms:

| Method                     | What is Split?                  | Type of Parallelism             |
|----------------------------|---------------------------------|---------------------------------|
| **Tensor Parallelism**     | Weights/tensors                 | Model parallelism               |
| **Pipeline Parallelism**   | Layers                          | Model parallelism               |
| **Expert Parallelism (MoE)** | Experts (modules)               | Model parallelism (MoE)         |
| **Data Parallelism**       | Input data batches              | **NOT** model parallelism       |
| **ZeRO Parallelism**       | Parameters, optimizer states    | Parameter sharding (special case)|


---
    - Data parallel
    - Pipeline parallel
    - Tensor parallel
    - Sequence(Activation) parallel
    - ZeRO (Zero Redundancy Optimizer) parallel
    - Expert (MoE) parallel
    - Hybrid parallel
 

- Fine-Tuning






### RAG


### Q&A System 

### Reasoning
- Information seeking

### Personalized Rec-system

### Recent interesting papers
























