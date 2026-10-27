# Machine Learning

[Handbook contents](../README.md#table-of-contents) | [Previous: Statistics](statistics.md) | [Next: Deep Learning](deep-learning.md)

## Contents

- [ML Basic](#ml-basic)
    - [Gradient descent](#gradient-descent)
    - [Black-Box Optimization](#black-box-optimization)
    - [Model evaluation and selection](#model-evaluation-and-selection)
- [ML Algorithms](#ml-algorithms)
    - [Algorithm Categories [Super Quick Overview](#algorithm-categories)
    - [Supervised Learning](#supervised-learning)
    - [Unsupervised Learning](#unsupervised-learning)
- [XAI](#xai)
    - [Feature importance](#feature-importance)
    - [SHAP](#shap)
    - [Attention-based](#attention-based)


## ML Basic
### Gradient descent
- (Batch) Gradient descent [[ref](https://machinelearningmastery.com/gradient-descent-for-machine-learning/)] [[B站](https://www.bilibili.com/video/BV1jh4y1q7ua/?spm_id_from=333.337.search-card.all.click&vd_source=c86f14ec33e79f08f7e2278747a071e8)]
    - The gradient $`\left(\nabla_\theta J(\theta)\right)`$ represents the direction of steepest ascent-the direction where the function increases the fastest. Gradient gives the direction of steepest ascend. So to minimize the function, take steps in the opposite direction.
    - Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost).
    - formula: $`\theta = \theta - \alpha \nabla_\theta J(\theta)`$, where $`\theta:`$ Parameters to be optimized. $`\alpha`$ : Learning rate (step size). $`\nabla_\theta J(\theta)`$ : Gradient of the cost function w.r.t. parameters.

- GD Variants
    - (Batch) GD: All data points at once
    - SGD: one data point per iteration
        - Updates parameters using only one randomly chosen training example at each step.
        - [To be reviewed] Formula: $`\theta_{t+1}=\theta_t-\alpha_t\nabla_\theta\ell(\theta_t;x_{i_t},y_{i_t})`$ for a sampled observation. A minibatch uses the appropriately normalized sum of sample gradients. The randomness is in the sampled data; the gradient is not automatically that of the full empirical objective at each step.
    - Mini-batch GD: Small subset per iteration.
    - Cons:
        - Sensitivity to learning rate
        - Risk of getting stuck at saddle points
        - [To be reviewed] Slow progress in nearly flat or ill-conditioned regions; a flat region is not necessarily a local minimum. Saddle points, local minima, and inadequate learning-rate schedules are distinct optimization issues.

[To be reviewed] These optimizers modify the update direction, its history, and/or its coordinatewise scale. Momentum is not itself an adaptive per-coordinate learning-rate method, and no listed method guarantees improved convergence or generalization on every task.

- **Momentum (gc)** enhances standard gradient descent by adding a velocity term that accumulates past gradients. This velocity term smooths the updates, reducing oscillations and accelerating convergence, particularly in scenarios with noisy gradients or ill-conditioned optimization surfaces.
    - $`v_t=\beta v_{t-1}+(1-\beta) \nabla J\left(\theta_t\right)`$
    - $`\theta_{t+1}=\theta_t-\alpha v_t`$
    - [To be reviewed] Momentum can reduce oscillation and accelerate progress on suitable objectives, but it does not guarantee escape from a local minimum or stable training for arbitrary hyperparameters.

- Nesterov Accelerated Gradient
    - Update rule
        - [To be reviewed] $`\tilde{\theta}_t=\theta_t-\beta v_{t-1}`$ (look ahead using the subtractive-velocity convention below). The sign must agree with $`\theta_{t+1}=\theta_t-v_t`$; an alternative convention stores a negative velocity and uses addition throughout.
        - $`v_{t}=\beta v_{t-1} + \eta \nabla_\theta \mathcal{L}\left(\tilde{\theta}_t\right)`$
        - $`\theta_{t+1}=\theta_t-v_{t}`$
    - Intuition
        - First take a "momentum step" to peek ahead at where you'd be $`\left(\tilde{\theta}_t\right)`$, then compute the gradient there.
        - This correction reduces lag in the momentum term, leading to more accurate, often faster convergence.


- **Adagrad (lr)** uses adaptive learning rates for each parameter, automatically adjusting LR during training. Larger learning rates for infrequent parameters, smaller rates for frequent parameters.
    - Formula:

      $`G_t=G_{t-1}+\left(\nabla_\theta J(\theta)\right)^2, \quad \theta=\theta-\frac{\alpha}{\sqrt{G_t+\epsilon}} \nabla_\theta J(\theta)`$

- [To be reviewed] **RMSprop (lr)** replaces Adagrad's cumulative squared-gradient sum with an exponential moving average. This avoids the particular monotonic denominator growth of Adagrad, but effective steps can still become too small or large; numerical stability and convergence depend on the objective and hyperparameters.


- **Adam**: Adam combines the advantages of (1)Momentum (first-order moment): Helps smooth updates, and (2) RMSprop (second-order moment): Provides adaptive per-parameter learning rates.
    Step 1: Calculate biased moments
    - First moment estimate (mean of gradients):

        $`m_t=\beta_1 m_{t-1}+\left(1-\beta_1\right) \nabla_\theta J(\theta)`$

    - Second moment estimate (mean of squared gradients):

        $`v_t=\beta_2 v_{t-1}+\left(1-\beta_2\right)\left(\nabla_\theta J(\theta)\right)^2`$


    Step 2: Correct bias (since initial values are zero):
    - $`\hat{m}_t=\frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t=\frac{v_t}{1-\beta_2^t}`$


    Step 3: Parameter update:
    - $`\theta=\theta-\frac{\alpha}{\sqrt{\hat{v}_t}+\epsilon} \hat{m}_t`$


        Typical default values:
        - $`\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}`$


- [To be reviewed] **AdamW** decouples weight decay from Adam's adaptive gradient update. Adding an L2 penalty to the loss and applying decoupled weight decay are generally different for adaptive optimizers. AdamW is widely used, but its generalization advantage and appropriate decay rate remain empirical. [AdamW paper](https://arxiv.org/abs/1711.05101)

- [To be reviewed] **Muon** applies an approximate orthogonalization, commonly using Newton-Schulz iterations, to momentum updates for suitable matrix-shaped parameters. Embeddings, output heads, vectors, or unsupported tensors may use another optimizer, depending on the recipe. It changes matrix-update geometry; it is not a coordinatewise adaptive-rate method like Adam, and orthogonal updates do not prove diverse learned representations. [Author implementation](https://github.com/KellerJordan/Muon)

[To be reviewed] **Optimizer comparison.** State counts below refer to a common optimizer's persistent auxiliary tensors, excluding parameters, gradients, master weights, activations, and temporary workspaces. Options and implementations can change the count.

| Method | Coordinatewise adaptive scale? | Common auxiliary state | Update / interpretation |
| --- | --- | --- | --- |
| SGD without momentum | No | None | Current gradient direction |
| Momentum SGD | No | One velocity | Accumulated gradient history |
| RMSprop, uncentered/no momentum | Yes | One squared-gradient average | Coordinatewise rescaling |
| Adam | Yes | First and second moments | Momentum plus coordinatewise rescaling |
| AdamW | Yes | First and second moments | Adam with decoupled decay |
| Muon | No Adam-style coordinatewise scale | Momentum plus implementation workspaces | Approximate matrix-update orthogonalization |

Stability and generalization require a task-specific comparison; they are not fixed “Medium/High” attributes. [Optimizer options](https://docs.pytorch.org/docs/stable/optim.html), [Muon implementation](https://github.com/KellerJordan/Muon)


### Black-Box Optimization
- Black-box optimization (BBO) refers to optimizing an objective function without an explicit formula or gradient information for that function. BBO has become increasingly important with the rise of expensive simulations and automated experiments – scenarios where each function evaluation is costly or time-consuming, and derivatives are unavailable.

- [To be reviewed] Random Search

    - **Mechanism:** sample configurations independently from a specified search distribution. Use log-uniform sampling for scale parameters such as learning rate; uniform sampling spends most trials near the upper end of a wide numerical interval.
    - **Example:** if only learning rate and weight decay matter among ten hyperparameters, 50 random trials explore about 50 settings for each relevant coordinate, whereas a Cartesian grid spends trials recombining irrelevant coordinates. Keep the final test set outside the search loop. [Random search paper](https://jmlr.org/papers/v13/bergstra12a.html)

- [To be reviewed] Evolutionary Algorithms

    - Maintain a population, evaluate fitness, and generate new candidates through selection, mutation, and sometimes crossover. Evolution strategies can adapt a continuous search distribution; genetic algorithms also accommodate discrete architectures and rules.
    - **Experimental design:** compare best validation score against equal evaluation or wall-clock budgets, including population initialization. In a noisy simulator, a lucky candidate may survive because of evaluation noise; reevaluate finalists with independent seeds. Evolution is useful when gradients are unavailable, but population search can be expensive. [CMA-ES author's resources](https://cma-es.github.io/)

- [To be reviewed] Bayesian Optimization

    - Fit a probabilistic surrogate to observed configurations and scores, then maximize an acquisition function balancing predicted quality and uncertainty. For maximization, expected improvement is $`\operatorname{EI}(x)=\mathbb{E}[\max(f(x)-f_{\mathrm{best}},0)]`$ under the surrogate posterior.
    - **Example:** when one model evaluation takes two GPU-hours, spend computation choosing the next configuration rather than sampling blindly. Gaussian-process BO is especially attractive for expensive, relatively low-dimensional continuous searches; high-dimensional conditional spaces and strongly nonstationary objectives require different surrogates or search structure. [BO tutorial](https://arxiv.org/abs/1807.02811)

- [To be reviewed] Derivative-Free Optimization Methods (Direct Search and Surrogate Algorithms)

    - Direct-search methods compare function values at candidate points; model-based methods fit a local approximation inside a trust region. Nelder-Mead uses a simplex, while methods such as BOBYQA use local quadratic models. These methods require objective evaluations, rather than analytic gradients.
    - **Example:** tune three continuous simulator parameters with deterministic outputs using a local method from several initial points. A local optimum is not a global-optimality certificate. Noisy evaluations, poor scaling between coordinates, discontinuities, and large dimension can make the comparison unreliable. [SciPy optimization reference](https://docs.scipy.org/doc/scipy/reference/optimize.html)

- [To be reviewed] Reinforcement Learning-based Approaches

    - A controller can sequentially choose design decisions, such as architecture components, and receive validation performance as a terminal reward. A policy-gradient estimator has the form $`\nabla_\phi\mathbb{E}[R(a)]=\mathbb{E}[(R(a)-b)\nabla_\phi\log\pi_\phi(a)]`$, where the baseline reduces variance.
    - **Distinction:** an ordinary hyperparameter trial is not automatically an RL problem. RL becomes useful when decisions have sequential structure or the controller transfers across tasks. Account for controller training and failed trials when comparing against random search. [Neural architecture search with RL](https://arxiv.org/abs/1611.01578)



### Model evaluation and selection
- Evaluation
    - TP, FP, TN, FN

    - Metrics:
        - Accuracy: Overall correctness
        - Precision ($`\frac{T P}{T P+F P}`$): Out of predicted positives, how many were actually positive?
        - Recall/sensitivity/TPR ($`\frac{T P}{T P+F N}`$): 	Out of actual positives, how many were predicted correctly?
        - Specificity/TNR ($`\frac{T N}{T N+F P}`$): Out of actual negatives, how many were correctly predicted negative?
        - F1-score ($`2 \times \frac{\text { Precision } \times \text { Recall }}{\text { Precision }+ \text { Recall }}`$): Harmonic mean balancing precision and recall. For multi-class problems, use macro or weighted averages.
        - **How to choose**: Accuracy is misleading with imbalanced data. Precision, if avoiding false positives is crucial (e.g., spam detection). Recall (Sensitivity), if missing a positive case is costly (e.g., cancer detection, fraud detection). F1-score, if both precision and recall matter equally. Specificity, if correctly identifying negative cases is essential (medical tests).
        - Precision vs Recall: Precision clearly matters when the cost of a FP is high. (e.g., classifying email as spam—high precision avoids wrongly marking important emails). Recall (TPR) clearly matters when the cost of missing positives (FN) is very high (e.g., disease diagnosis—high recall ensures positive cases aren't missed).

    - ROC (Receiver Operating Characteristic) curve (TPR vs FPR, threshold selection)
        - [To be reviewed] ROC plots TPR against FPR as the decision threshold changes. Threshold choice balances costs, prevalence, and constraints; maximizing TPR and minimizing FPR simultaneously is generally impossible. Inspect the operating region relevant to deployment.
    - AUC (model comparison)
        - [To be reviewed] ROC-AUC summarizes ranking across thresholds: it equals the probability that a randomly drawn positive scores above a randomly drawn negative, with half credit for ties under the standard convention. It does not establish probability calibration or performance at one deployment threshold. [ROC-AUC reference](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
        - [To be reviewed] Range **[0, 1]**. A score independent of labels has expected AUC 0.5; a perfectly reversed ranking has AUC 0. Reversing a continuous score transforms AUC to $`1-\mathrm{AUC}`$ under consistent tie handling.
    - Confusion matrix: Confusion matrix clearly visualizes all predictions vs actual classes:

    <div align="center">
        <img src="../figs/confusion_matrix.png" width="50%">
    </div>

- Bias/Variance [[ref](https://traintestsplit.com/bias-vs-variance-in-machine-learning/)]
    - [To be reviewed] For squared-error regression, bias at an input is the difference between the estimator's average prediction over repeated training samples and the true regression function. Squared bias is one component of expected prediction error; it is not simply the number of modeling assumptions.

    - [To be reviewed] Variance measures the spread of fitted predictions across repeated training datasets at a fixed input. It depends on data, model class, regularization, and fitting procedure. High variance can contribute to overfitting, but parameter count alone does not determine it.

    - [To be reviewed] Under squared loss with suitable finite moments, expected test error decomposes into noise, squared bias, and variance. The textbook complexity trade-off is useful but not a universal monotone law; modern interpolating models can display double descent. Evaluate held-out behavior rather than inferring it from parameter count. [Double-descent study](https://arxiv.org/abs/1812.11118)

    <div align="center">
        <img src="../figs/bias-vs-variance.png" width="60%">
    </div>

    - Underfitting/overfitting
        - Underfitting occurs when the model is too simple to capture underlying data trends. Solution:Increase model complexity (e.g., add features, use more complex models, reduce regularization).
        - Overfitting occurs when the model captures noise or random fluctuations (memorize every single detail) in training data. Solution: Reduce model complexity (e.g., remove features, apply regularization, use simpler models); Increase training data size; Early stopping.
    - Regularization: Regularization, such as L1 (LASSO) or L2 (Ridge), can be used to control model complexity to prevent overfitting and tackle high variance. They work by adding a penalty term to the magnitude of the coefficients.
        General form of regularized loss function:

        $`\operatorname{Loss}(w)=\operatorname{Loss}_{\text{original}}(w)+\lambda\cdot{R}(w)`$


        - L0 is a technique in ML to encourage sparsity in a model's parameters. It penalizes the number of **non-zero parameters** in a model.

            $`\|\mathbf{w}\|_0=\sum_{i=1}^n \mathbf{1}\left[w_i \neq 0\right]`$

            Use case: Feature Selection: L0 regularization is particularly useful in scenarios where the number of features is large, and only a small subset is expected to be relevant. It helps to automatically select a subset of features that contribute significantly to the model's performance.

        - L1 penalizes **absolute magnitude** of weight:

            $`R(w)=\sum\left|w_i\right|`$.

            Encourages sparsity through geometry.

        - L2 penalizes **square magnitude** of weights:

            $`R(w)=\sum\left(w_i^2\right)`$

            Penalizes large weights heavily, encourges smaller, diffuse weigths. Handles correlated features better.

        - L-infinity penalizes the **largest absolute weight** (maximum norm):

            $`R(w)=\max \left|w_i\right|`$

            It's used when you need to control the model's sensitivity to prevent any one feature from dominating the predictions.

        - When prefer L1 over L2? For explicit feature selection or interpretability
        - When prefer L2 over L1? L2 (Ridge) shrinks correlated features toward each other, effectively managing correlated features.
        - [To be reviewed] As $`\lambda\to\infty`$, penalized coefficients typically approach zero under the stated objective, while an unpenalized intercept need not. L1 can make coefficients exactly zero at finite penalty values; L2 generally shrinks them continuously. There is no parameterization-independent sense in which L1 always reaches zero “faster.”
        - Is L0 convex? Why does it matter? No, it's non-convex. This matters due to optimization difficulty and computational complexity.




    - [To be reviewed] Feature selection can reduce variance or computation but can increase bias by discarding useful information. Fit the selection procedure inside each training fold; selecting features on all data before cross-validation leaks evaluation information. Use task-appropriate held-out selection criteria.
        - Backward Elimination: Start with all features, then iteratively remove the least significant feature.
        - Forward Selection: Start with no features, then teratively add the most significant feature until no improvement.
        - Recursive Feature Elimination (RFE): Iteratively fits a model and removes least important features based on coefficients or importance metrics.
        - Embedded Methods (L1 Regularization / Lasso):
            - Feature selection happens simultaneously with model fitting

- Data
    - Missing data
        - [To be reviewed] MICE iterates conditional imputation models and, in a proper multiple-imputation workflow, generates multiple plausible completed datasets to represent uncertainty. A deterministic chained-regression fill produces one completion, not multiple-imputation uncertainty by itself. Check missingness assumptions and pool downstream uncertainty appropriately. [MICE reference](https://www.jstatsoft.org/article/view/v045i03)
    - [To be reviewed] Imbalance means class frequencies differ; its consequences depend on the objective, prevalence, sampling, and error costs. Assess per-class metrics, precision-recall behavior, and calibration. Fit resampling only on training folds and account for altered class priors when interpreting probabilities. [Resampling leakage](https://imbalanced-learn.org/stable/common_pitfalls.html)
        - Oversampling: Duplicate minority examples
            - SMOTE (Synthetic Minority Oversampling Technique): 1. Chooses a random minority instance; 2. Finds k nearest minority neighbors; 3. Creates synthetic points along the line segments joining neighbors.
            - ADASYN (Adaptive Synthetic Sampling): Generates more synthetic data for harder-to-classify minority instances (closer to majority class boundary).
        - Undersampling: Remove majority examples strategically.
            - ENN (Edited Nearest Neighbor): Removes majority samples misclassified by a k-NN classifier (noisy or ambiguous instances).

    - Distribution shifts: A distribution shift happens when the data seen during training differs from the data encountered during deployment or testing. It breaks the core assumption that train and test data come from the same distribution.
        - Covariate Shift (X changed): This occurs when the input distribution $`P(x)`$ changes, but the conditional distribution $`P(y \mid x)`$ stays the same. Example: A spam detection model trained on emails from last year might see different phrasing this year, even if what qualifies as spam hasn’t changed.

        - Label Shift (Y changed): Here, the label distribution $`P(y)`$ changes, but the distribution of features given the label $`P(x \mid y)`$ remains the same. Example: In medical data, the proportion of patients with a certain condition may increase in a new population.
        - Concept Drift (f changed): The conditional distribution $`P(y \mid x)`$ itself changes. Example: A recommendation model may degrade as user preferences evolve.

- Sampling
    - [To be reviewed] Uniform sampling without replacement selects a subset with equal inclusion probabilities. `random.sample(data, k)` requires a suitable population representation; its cost depends on population type, size, and implementation. Do not assign a universal $`O(k)`$ bound including access or materialization of the input.
    - [To be reviewed] Negative sampling replaces an expensive all-candidate computation with sampled comparisons. In Word2Vec it uses a binary discrimination objective, not an unbiased exact-softmax replacement. With $`k`$ sampled negatives and embedding width $`d`$, update arithmetic is roughly $`O(kd)`$ rather than $`O(Vd)`$ for a full vocabulary comparison. The sampling distribution changes the learned objective. [Word2Vec negative sampling](https://arxiv.org/abs/1310.4546)
    - Reservoir sampling: Technique to uniformly sample $`k`$ items from a stream or large unknown-size dataset in one pass.
        - Use Cases: Sampling from streaming data or very large files/databases.
        - Algorithm: Maintain a reservoir of size $`k`$. For first $`k`$ items: directly fill reservoir. For each subsequent item $`i>k`$ : With probability $`k / i`$, randomly replace an item in the reservoir.
        - Time: $`O(N)`$, single pass.
        - Memory: $`O(k)`$, fixed space.
    - Stratified sampling: Technique where the dataset is divided into distinct subgroups (strata) based on certain attributes (e.g., age, gender, class), and samples are drawn from each subgroup proportionally or with specific representation.
        - Code: df.groupby(stratify_col, group_keys=False).apply(lambda x: x.sample(frac=frac))
    - Questions
        - When would you prefer reservoir sampling over uniform random sampling? If you're sampling tweets from a live Twitter stream (which is continuously updating and has unknown total size), reservoir sampling is the ideal method to guarantee uniformity while using fixed memory.
        - How does stratified sampling reduce variance? If you're conducting a survey on voting preferences and you stratify by age groups (young, middle-aged, elderly), each subgroup's responses tend to be more similar internally than across the entire population. This homogeneity decreases the variance of estimates significantly compared to simple random sampling.




- Model Selection
    - K-fold Cross-validation is a resampling procedure used to evaluate machine learning models and tune hyperparameters.
    - $`K=5`$ or $`K=10`$ are most common.
    - Trade-off (Bias vs. Variance):
        - Large $`K`$ :
            - [To be reviewed] More folds train each model on a larger fraction of the data, often reducing training-size bias in the performance estimate.
            - [To be reviewed] Computation grows with the number of fitted models. Validation errors across folds are dependent; the variance of a cross-validation estimate is not universally increasing in $`K`$.
        - Small $`K`$ :
            - [To be reviewed] Fewer folds generally reduce the number of fits and computational cost.
            - [To be reviewed] Each fit uses less training data, which can increase training-size bias. The estimator's variability depends on the dataset, learner, split design, and metric; it cannot be ordered by fold count alone.
    - How to handle imbalanced datasets in K-fold CV? Use Stratified K-fold, ensuring each fold maintains the proportion of classes from the original dataset.
    - Does K-fold cross-validation prevent overfitting? No, it helps evaluate and select models accurately but doesn't inherently prevent overfitting. Use regularization, early stopping, or simpler models to control overfitting.
    - [To be reviewed] Ordinary shuffled K-fold can leak future information in temporal data. Use forward/blocked time splits, group-aware splits, and gaps when required by the deployment task. Cross-validation itself is not inherently incompatible with time series. [Cross-validation design](https://scikit-learn.org/stable/modules/cross_validation.html)

- Hyper-Parameter Tuning
    - Grid search: Exhaustively evaluates all possible combinations of hyper-parameters provided in a predefined grid.
    - Random search: Randomly samples hyper-parameter values within predefined ranges.
    - Bayesian optimization: Uses past evaluation results to model the hyper-parameter space.
        - Popular libraries: `Hyperopt`, `Optuna`, `BayesSearchCV`.
    - Other autoML packages: `auto-sklearn`, `h2o`.






## ML Algorithms
<a id="algorithm-categories"></a>
### Algorithm Categories [[Super Quick Overview](https://www.youtube.com/watch?v=E0Hmnixke2g)]
- Types by (labeled) data
    - **Supervised Learning** is a type of machine learning where the algorithm is trained on a labeled dataset. Labeled data consists of input-output pairs, meaning the algorithm learns to map input data to corresponding output labels. The goal is for the algorithm to generalize from the training data and make accurate predictions on new, unseen data.
        - Classification: Predicts categorical outputs (e.g., spam detection, image recognition). Algorithms: Logistic regression, Decision tree, RF,KNN, SVM.
            - Loss function [[YouTube](https://www.youtube.com/watch?v=v_ueBW_5dLg)]: These losses measure how correctly and confidently a model classifies examples into categories.
                - logistic loss/ binary cross entropy: Used for binary classification problems. For a predicted probability $`p`$ and ground truth label $`y \in\{0,1\}`$ : $`\mathrm{BCE}=-[y \log (p)+(1-y) \log (1-p)]`$.
                    - [To be reviewed] A sigmoid output parameterizes a binary probability in [0, 1]; binary cross-entropy is the loss, not the operation that produces probabilities. A fused loss on logits is usually numerically preferable.
                    - Penalizes confident but wrong predictions heavily.
                    - Works well when used with sigmoid activation in the output layer.

                - categorial cross entropy: Extends cross-entropy loss to multi-class classification problems. For a ground truth one-hot encoded vector $`\mathbf{y}`$ and a prediction vector $`\hat{\mathbf{y}}`$ : $`\mathrm{CCE}=-\sum_{c=1}^C y_c \log \left(\hat{y}_c\right)`$.
                    - Used together with a softmax activation in the output layer.
                    - Encourages the model to assign high probability to the correct class.

                - [To be reviewed] KL divergence

                    - For discrete distributions, $`D_{\mathrm{KL}}(p\|q)=\sum_i p_i\log(p_i/q_i)`$. It is nonnegative but asymmetric; if $`p_i>0`$ and $`q_i=0`$, the divergence is infinite. Cross-entropy satisfies $`H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)`$, so minimizing it over $`q`$ with fixed targets minimizes forward KL.
                    - **Example:** knowledge distillation uses the teacher's full soft distribution, preserving distinctions that a one-hot label discards. State the direction of KL explicitly; reversing it changes which distribution supplies the expectation and can change optimization behavior. [Distillation paper](https://arxiv.org/abs/1503.02531)


                - hinge loss: Commonly used for Support Vector Machines (SVMs) and for certain deep learning tasks that involve margin-based classification. For a prediction score $`s`$ and a true label $`y \in\{-1,1\}`$ : $`\text { Hinge Loss }=\max (0,1-y \cdot s)`$.
                    - Focuses on the margin between classes.
                    - Can also be adapted for multi-class classification with variations like multi-class hinge loss.

        - Regression: Predicts continuous numerial outputs. (e.g., house price prediction, stock market forecasting). Algorithms: Linear regression, RF.
            - Loss function:These losses measure how close predicted values are to true values numerically.
                - [To be reviewed] MSE

                    - $`\operatorname{MSE}=n^{-1}\sum_i (y_i-\hat y_i)^2`$. Its population minimizer is the conditional mean when the second moment exists. Squaring residuals makes large errors disproportionately influential.
                    - **Example:** residuals 1 and 10 contribute 1 and 100. MSE is appropriate for expected squared error, but an extreme corrupted label can dominate an otherwise clean batch. Gaussian observation noise with fixed variance gives the same optimum as maximum likelihood. [Regression losses](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

                - [To be reviewed] RMSE

                    - $`\operatorname{RMSE}=\sqrt{\operatorname{MSE}}`$ has the target's units. On one fixed evaluation set, it ranks models identically to MSE because the square root is monotone.
                    - **Pitfall:** averaging fold RMSEs is not the same as taking the square root of pooled squared errors. For unequal fold sizes, aggregate squared-error totals and observation counts before taking the square root when the desired estimand is sample-level RMSE. The globally coupled square root also changes gradient scaling if optimized directly.

                - [To be reviewed] MAE

                    - $`\operatorname{MAE}=n^{-1}\sum_i |y_i-\hat y_i|`$. Its population minimizer is a conditional median; its residual gradient has constant magnitude away from zero. A Laplace observation model with fixed scale yields an equivalent maximum-likelihood objective.
                    - **Example:** with targets 0, 0, and 100 for the same features, squared loss prefers the mean 33.33, while absolute loss prefers 0. Choose the loss from the cost of mistakes and the intended conditional statistic, rather than assuming robustness is always desirable.

                - [To be reviewed] Huber Loss

                    - For residual $`r=\hat y-y`$, Huber loss is $`\ell_\delta(r)=r^2/2`$ when $`|r|\leq\delta`$, and $`\delta(|r|-\delta/2)`$ otherwise. It has a quadratic center and linear tails with a continuous first derivative.
                    - **Example:** choose the transition scale relative to standardized target residuals, then tune it on held-out data. A threshold of 1 has completely different meaning for errors measured in dollars and millions of dollars. Huber limits the influence of large residuals but cannot repair systematically biased labels. [Huber loss reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)

                - [To be reviewed] Log-Cosh Loss

                    - $`\ell(r)=\log\cosh(r)`$ behaves like $`r^2/2`$ near zero and $`|r|-\log 2`$ for large magnitude. Its derivative $`\tanh(r)`$ is bounded, providing smooth robust optimization.
                    - **Implementation:** use $`|r|+\log(1+e^{-2|r|})-\log 2`$ to avoid overflow from computing `cosh` directly. An explicit residual scale is needed when units vary. Like other symmetric robust losses, it does not generally estimate the conditional mean under asymmetric noise. [TensorFlow LogCosh reference](https://www.tensorflow.org/api_docs/python/tf/keras/losses/LogCosh)


    - **Unsupervised Learning** is a type of machine learning that learns from data without human supervision. Unlike supervised learning, unsupervised machine learning models are given unlabeled data and allowed to discover patterns and insights without any explicit guidance or instruction. (e.g., Customer segmentation, dimensionality reduction)
        - **Clustering** is a technique for exploring raw, unlabeled data and breaking it down into groups (or clusters) based on similarities or differences. It is used in a variety of applications, including customer segmentation, fraud detection, and image analysis. Clustering algorithms split data into natural groups by finding similar structures or patterns in uncategorized data.
            - [To be reviewed] K-means

                - Partition observations by nearest centroid and minimize squared Euclidean reconstruction error; choose feature scaling and multiple initializations deliberately. See the detailed K-means section below. Example: grouping customers by annual spend without scaling can effectively ignore a purchase-frequency feature measured on a much smaller numerical scale.

            - [To be reviewed] DBSCAN

                - Identify density-connected regions and label unsupported points as noise. Unlike K-means, DBSCAN does not require a prespecified number of clusters, but its distance metric and density thresholds strongly influence the answer. A single global threshold can struggle when genuine clusters have very different densities. See the detailed DBSCAN section below.

            - [To be reviewed] Hierarchical clustering.

                - Construct a nested grouping represented by a dendrogram, then choose a cut using the task and a stated linkage criterion. The same data can yield different hierarchies under single, complete, average, or Ward linkage; the linkage is part of the model, rather than a plotting option.

        - Association rule mining is a rule-based approach to reveal interesting relationships between data points in large datasets. Unsupervised learning algorithms search for frequent if-then associations—also called rules—to discover correlations and co-occurrences within the data and the different connections between data objects.

        It is most commonly used to analyze retail baskets or transactional datasets to represent how often certain items are purchased together. These algorithms uncover customer purchasing patterns and previously hidden relationships between products that help inform recommendation engines or other cross-selling opportunities. You might be most familiar with these rules from the “Frequently bought together” and “People who bought this item also bought” sections on your favorite online retail shop.

        [To be reviewed] Association rules identify observed co-occurrence patterns. In clinical datasets, a rule's confidence is an empirical conditional frequency under the sampled population, not automatically a calibrated diagnostic probability or causal relationship. Diagnostic claims require supervised validation and appropriate sampling assumptions.

        [To be reviewed] Apriori, Eclat, and FP-growth are common frequent-pattern mining approaches. Their relative suitability depends on support thresholds, sparsity, and implementation; avoid an unsupported universal popularity ranking.

        - Dimensionality  Reduction is an unsupervised learning technique that reduces the number of features, or dimensions, in a dataset.
            - [To be reviewed] PCA

                - PCA finds orthogonal directions of maximal variance after centering. It is useful for compression and conditioning, but a low-variance direction can still contain the discriminative signal. See the detailed PCA entry under Dimension reduction techniques.

            - [To be reviewed] t-SNE

                - t-SNE is primarily a local-neighborhood visualization method. Use multiple seeds and perplexities; do not infer class counts or quantitative global distances solely from separated islands. See the detailed t-SNE entry under Dimension reduction techniques.

    - **Semi-supervised Learning** falls between unsupervised learning (without any labeled training data) and supervised learning (with completely labeled training data). It combines small amounts of labeled data with large amounts of unlabeled data. Useful when labeling data is expensive or difficult. The model leverages labeled data to guide learning, while also making use of unlabeled data to generalize better and improve performance.
    - [To be reviewed] **Reinforcement Learning** optimizes decisions under rewards in an environment or a specified decision process. Online RL collects new interactions; offline RL learns from an existing interaction dataset, so new environment interaction is not part of every RL training run.
    - [To be reviewed] **Self-supervised Learning** constructs training targets from the observed data, such as masked tokens, next tokens, or augmented-view correspondence. This differs from pseudo-labeling, where a model predicts labels for otherwise unlabeled examples; both can appear in one pipeline.

- Parametric vs non-parametric algorithms (if we assume a fixed functional form or not)

    (Functional form = Mathematical shape or structure that the model assumes to represent how the input relates to the output.)
    - [To be reviewed] **Parametric models** use a parameterization whose dimension is fixed for a specified model as sample size grows; examples include fixed-architecture neural networks and logistic regression. This does not guarantee simplicity, interpretability, few parameters, or protection against overfitting.

    - [To be reviewed] **Nonparametric model families** permit effective complexity to grow with available data, for example unconstrained trees or neighbor-based estimators. They still impose assumptions through metrics, kernels, smoothness, partition rules, and regularization. Fixed-size restrictions can bound a particular implementation's capacity; “nonparametric” does not mean assumption-free.

- Linear vs Nonlinear algorithms
    - [To be reviewed] Linear regression is linear in its coefficients; logistic regression makes the log-odds linear in the supplied features. Fixed nonlinear feature maps can make either model nonlinear in raw inputs. A linear SVM has a linear decision function in its chosen feature representation.
    - Nonlinear algorithms (Decision Trees, Random Forest, Neural Networks, Kernel SVM) capture more complex patterns and relationships, offering flexibility but potentially at the cost of interpretability and efficiency.


### Supervised Learning
- K-Nearest Neighbors (KNN) [[Code](https://www.youtube.com/watch?v=ngLyX54e1LU&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=1)]
    - Nature: non-parametric, instance-based, lazy :smile:. It directly predicts the output for a new instance by identifying the k closest data points in the training set, using a suitable distance metric (typically Euclidean distance for continuous features).
    - Curse of Dimensionality: High-dimensional spaces reduce the discriminative power of distance metrics, often leading to performance degradation unless accompanied by dimensionality reduction techniques.
    - Time Complexity: Prediction requires computing distances to many or all training samples, which can be computationally expensive for large datasets.

- Linear Algorithms
    - **Linear Regression** [[Code](https://www.youtube.com/watch?v=4swNt7PiamQ&list=PLqnslRFeH2Upcrywf-u2etjdxxkL8nl7E&index=2)]  is a supervised statistical model to predict dependent variable quantity based on independent variables. Linear regression is a parametric model and the objective of linear regression is that it has to learn coefficients using the training data and predict the target value given only independent values.
        - Assumptions:
            - [To be reviewed] Specify a conditional-mean model such as $`\mathbb{E}[y\mid X]=X\beta`$ and the exogeneity assumption $`\mathbb{E}[\epsilon\mid X]=0`$ when claiming unbiased OLS coefficients.
            - [To be reviewed] Dependence across observations affects standard errors and effective sample size. Independent errors are a common textbook condition; clustered or serial dependence requires suitable covariance estimators or an explicit dependence model.
            - [To be reviewed] Homoscedasticity is required for the usual Gauss-Markov efficiency statement and conventional covariance formula, not for computing OLS. Heteroscedasticity-robust inference can be appropriate under its own assumptions.
            - [To be reviewed] Gaussian errors support exact small-sample t/F inference under the classical linear model. Normality is not needed to compute OLS or to establish unbiasedness given exogeneity; asymptotic inference has separate conditions.
            - [To be reviewed] Full column rank identifies a unique unregularized coefficient vector. Strong collinearity can inflate coefficient uncertainty; it does not by itself make predictions impossible. Perfect collinearity requires constraints, a generalized inverse, or regularization. [Linear-model reference](https://www.statsmodels.org/stable/regression.html)
        - Formulation: Linear regression models the relationship between a dependent variable $`y`$ and one or more independent variables $`X`$ as a linear combination: $`y=X \beta+\epsilon,`$, where $`\beta`$ is a vector of coefficients and $`\epsilon`$ is the error term.
        - Regularization: Extensions such as Ridge (L2) and Lasso (L1) regression add penalty terms to handle multicollinearity and promote sparsity.

    - Logistic Regression
        - Formulation: Logistic regression models the probability of a binary outcome by applying the logistic (sigmoid) function to a linear combination of input features: $`P(y=1 \mid X)=\frac{1}{1+e^{-X \beta}} .`$
        - [To be reviewed] Parameters can be fitted by minimizing negative log-likelihood, often with regularization. Under complete separation, unregularized logistic regression need not have a finite MLE; solver convergence and penalty choice matter.
        - Interpretability: Coefficients can be interpreted in terms of odds ratios, offering intuitive insights into feature effects.

        - The cost for a single training example using the binary cross-entropy (log-loss) is: $`\ell(w, b)=-[y \log (\hat{y})+(1-y) \log (1-\hat{y})]`$.
        - For $`m`$ training examples, the overall cost is: $`J(w, b)=\frac{1}{m} \sum_{i=1}^m \ell^{(i)}(w, b)=-\frac{1}{m} \sum_{i=1}^m\left[y^{(i)} \log \left(\hat{y}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)\right]`$
        - Derivation of gradients
            - The cost for one sample is: $`\ell(w, b)=-[y \log (\hat{y})+(1-y) \log (1-\hat{y})], \quad \text { with } \hat{y}=\sigma(z) \text { and } z=w^T x+b`$.
            - Compute the derivative of the cost with respect to $`\hat{y}`$ : $`\frac{\partial \ell}{\partial \hat{y}}=-\left(\frac{y}{\hat{y}}-\frac{1-y}{1-\hat{y}}\right)`$
            - The derivative of the sigmoid is: $`\frac{d \hat{y}}{d z}=\hat{y}(1-\hat{y}) .`$
            - Then by the chain rule: $`\frac{\partial \ell}{\partial z}=\frac{\partial \ell}{\partial \hat{y}} \cdot \frac{d \hat{y}}{d z}`$. When you work through the algebra, the product simplifies to: $`\frac{\partial \ell}{\partial z}=\hat{y}-y`$.
            - Since $`z=w^T x+b`$, the derivative of $`z`$ with respect to $`w_j`$ is: $`\frac{\partial z}{\partial w_j}=x_j`$. Then: $`\frac{\partial \ell}{\partial w_j}=\frac{\partial \ell}{\partial z} \cdot \frac{\partial z}{\partial w_j}=(\hat{y}-y) x_j`$.
            - Similarly, $`\frac{\partial \ell}{\partial b}=\frac{\partial \ell}{\partial z} \cdot \frac{\partial z}{\partial b}=(\hat{y}-y)`$.



    - Support Vector Machines
        - Formulation: SVMs seek a hyperplane that maximally separates classes with the greatest margin. For linearly separable data, the optimization problem is: $`\min _{w, b} \frac{1}{2}\|w\|^2 \quad \text { s.t. } y_i\left(w^T x_i+b\right) \geq 1, \quad \forall i`$.
        - [To be reviewed] Soft-margin SVM introduces nonnegative slack for margin violations, which include correctly classified points inside the margin as well as misclassifications. The regularization parameter balances margin and total violation penalty.
        - Kernel Trick: For nonlinear classification, kernel functions (e.g., Gaussian, polynomial) implicitly map input data to a higher-dimensional space where linear separation is feasible.
        - Dual Formulation: The Lagrange dual formulation and application of Karush-Kuhn-Tucker (KKT) conditions are central to SVM theory, providing insights into support vectors and decision boundaries.
        - The optimization is convex, ensuring global optimum, but can be computationally challenging for large datasets.


    - Naive Bayes
        [[Coding](https://www.youtube.com/watch?v=TLInuAorxqE)]
        - Foundational Principle: Based on Bayes' theorem: $`P(C \mid x)=\frac{P(x \mid C) P(C)}{P(x)}`$, Naive Bayes assumes conditional independence among features given the class label.
            - $`P(C)`$ is the prior probability of class $`C`$.
            - $`P(x \mid C)`$ is the likelihood of observing features $`x`$ under class $`C`$.
            - $`P(C \mid x)`$ is the posterior probability: what we want.
            - Because $`P(x)`$ is the same across classes when comparing them, we only need $`\arg \max _C P(x \mid C) P(C)`$.

        - Choosing a Model
            - Multinomial NB: features are non‑negative counts (e.g. word counts in text).
            - Bernoulli NB: binary features (presence/absence).
            - Gaussian NB: continuous features, modeled with class‑conditional Gaussians.

        - Why naive? The term "naive" in Naive Bayes refers to the algorithm's assumption that all features (predictor variables) used for classification are mutually independent given the class label. This assumption is considered "naive" because, in real-world scenarios, features often exhibit some degree of dependency, and assuming independence can be an oversimplification of the true underlying relationships. Despite the “naive” assumption being seldom true in practice, the method often performs well due to its simplicity and effectiveness in high-dimensional settings.

    - Linear discriminant analysis (LDA)
        - Concept & Assumptions: LDA models class conditional densities as multivariate Gaussians with equal covariance matrices for all classes. Under this assumption, the discriminant function is linear in the features: $`\delta_k(x)=x^T \Sigma^{-1} \mu_k-\frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k+\log \pi_k`$. Here, $`\mu_k`$ is the mean vector, $`\Sigma`$ the common covariance matrix, and $`\pi_k`$ the prior probability of class $`k`$.
        - [To be reviewed] With the true Gaussian class parameters and priors known, the LDA rule is Bayes-optimal for 0–1 classification loss under its assumptions. A finite-sample fitted plug-in classifier is an estimate of that rule, not a guarantee of Bayes-optimal performance.
        - Dimensionality Reduction: LDA also serves as a dimensionality reduction technique by projecting data onto a subspace that maximizes between-class variance while minimizing within-class variance (related to Fisher’s Linear Discriminant).
        - Applications: Particularly effective when the Gaussian assumption holds and when class covariances are similar.

- Decision Trees

    - Structure: Decision trees recursively partition the input feature space by selecting splits that maximize a certain criterion (e.g., information gain, Gini impurity for classification, or mean squared error reduction for regression).
    - Algorithm details
        - Recursive Partitioning: At each node, the algorithm selects the feature and threshold that best “purifies” the subsets.
        - Stopping Criteria & Pruning: Trees can easily overfit; thus, criteria such as maximum depth, minimum samples per leaf, or post-pruning (e.g., cost-complexity pruning) are essential.
        - Interpretability: Decision trees produce models that are easily interpretable; however, their hierarchical nature can also be sensitive to small changes in data (high variance).
    - Decision trees are highly flexible non-parametric models. Their analysis often involves bias–variance trade-offs: while they are low bias (fit complex relationships well), they can suffer from high variance if unpruned.

    - [To be reviewed] Leaves

        - A leaf stores the fitted response for a region of feature space: typically class proportions for classification and a mean for squared-error regression. These are piecewise-constant estimates, so a regression tree cannot extrapolate a linear trend beyond its observed leaf values.
        - **Example:** a leaf containing eight positive and two negative training examples predicts a positive-class probability of 0.8 before any smoothing or calibration. A leaf containing one positive example also predicts 1.0, but carries much greater estimation uncertainty. Minimum leaf size controls this variance. [Decision trees](https://scikit-learn.org/stable/modules/tree.html)

    - [To be reviewed] Training algorithm+stop criteria

        - CART greedily chooses the feature and threshold maximizing weighted impurity reduction. It does not jointly optimize the entire tree. Classification commonly uses Gini impurity or entropy; squared-error regression minimizes within-node variance.
        - Stop using maximum depth, minimum split/leaf counts, or minimum impurity decrease. Fit these choices inside cross-validation. **Example:** a split isolating one mislabeled point may improve training purity while harming held-out performance; a minimum leaf-size constraint prevents that particular memorization pattern.

    - [To be reviewed] Inference

        - Route an example from the root to one leaf by applying its feature-threshold comparisons. Prediction cost is proportional to path depth, not the number of training observations. A balanced depth-10 tree uses about ten comparisons; a badly unbalanced tree can be much deeper.
        - Record the decision path for debugging. Missing-value routing and categorical support are implementation-specific; training and serving must apply the same feature encoding and missing-data semantics.

    - [To be reviewed] Pruning

        - Minimal cost-complexity pruning trades training impurity against leaf count: $`R_\alpha(T)=R(T)+\alpha |\mathcal{L}(T)|`$. Increasing $`\alpha`$ favors smaller subtrees. Select it on validation folds rather than the final test set.
        - **Example:** compare a deep tree and its pruned subtrees using mean validation score and variability across folds. A tiny training improvement from a rare branch may disappear under resampling. [Cost-complexity pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning)



- Ensemble methods [[Wiki](https://en.wikipedia.org/wiki/Ensemble_learning)] [[Document](https://scikit-learn.org/stable/modules/ensemble.html)]
    - Definition: Ensemble methods aim to combine the predictions of multiple base learners to achieve improved generalization performance and robustness compared to single models.

    - Bagging: Variance Reduction through Bootstrap Aggregation
        - Concept: A set of base learners (often decision trees) is trained on different bootstrap samples of the data. Their predictions are then aggregated (by averaging or voting).
        - Bagging primarily focuses on reducing variance. It typically employs complex base learners (e.g., deep decision trees) which have low bias but high variance. By training these models on different bootstrap samples of the data and then averaging their predictions, Bagging smooths out the predictions and reduces the overall variance of the ensemble, often without significantly affecting the bias.
        - [To be reviewed] **Random Forest**

            - Combine bootstrap-trained trees with random feature subsets at each split to reduce correlation between trees. For $`M`$ equal-variance predictors with pairwise correlation $`\rho`$, their average has variance $`\sigma^2[\rho+(1-\rho)/M]`$: adding trees cannot remove the correlated component.
            - **Example:** if one highly predictive feature is always available, many trees can choose similar early splits. Feature subsampling encourages alternative partitions. Tune leaf size and feature subsampling; more trees mainly reduce Monte Carlo variation. Out-of-bag evaluation does not replace a time-based split when deployment predicts the future. [Random forests](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)

    - Boosting: Sequential Learning for Bias Reduction
        - Boosting represents a family of ensemble algorithms that construct a strong predictive model by iteratively training a sequence of base learners (typically weak learners), where each new learner focuses on correcting the errors made by the ensemble built so far. Unlike Bagging's parallel approach, Boosting is inherently sequential.
        - [To be reviewed] Boosting can reduce approximation bias by adding learners sequentially, but its bias/variance and noise sensitivity depend on loss, base learners, shrinkage, subsampling, and stopping. There is no universal ordering that boosting overfits more than bagging.
        - [To be reviewed] **Adaboost**

            - In binary AdaBoost with labels in $`\{-1,+1\}`$, a weak learner with weighted error $`\epsilon_t`$ receives coefficient $`\alpha_t=\tfrac12\log[(1-\epsilon_t)/\epsilon_t]`$. Update example weights proportionally to $`w_i\exp[-\alpha_t y_i h_t(x_i)]`$; mistakes gain weight.
            - **Example:** error 0.2 gives coefficient about 0.693 under this convention. Persistently mislabeled examples may absorb increasing weight, so inspect label noise. State the binary formulation explicitly: multiclass variants use different coefficients and conditions. [AdaBoost reference](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)

        - [To be reviewed]  [[**Gradient Boosting**](https://www.youtube.com/watch?v=en2bmeB4QUo)]

            - At stage $`t`$, fit a base learner to negative loss gradients and update $`F_t(x)=F_{t-1}(x)+\eta\gamma_t h_t(x)`$. Residual fitting is exactly the squared-loss case; for logistic loss the targets are functional gradients rather than ordinary residuals.
            - **Example:** with squared loss, a prediction of 7 for a target of 10 leaves a residual of 3 for the next learner. Learning rate, number of iterations, tree complexity, and subsampling jointly control fit. Use validation-based early stopping. [Friedman's gradient boosting paper](https://doi.org/10.1214/aos/1013203451)

        - [To be reviewed] **XGBoost**

            - XGBoost optimizes a regularized boosting objective using first- and second-order loss information. With leaf gradient sum $`G_j`$, Hessian sum $`H_j`$, and L2 penalty $`\lambda`$, the unconstrained optimum is $`w_j^*=-G_j/(H_j+\lambda)`$. Split gain balances the child and parent scores against a leaf-complexity penalty.
            - Sparse-aware split search and default directions handle missing entries; a missing value is not automatically the numerical value zero. **Example:** for sparse purchase features, distinguish “no observed purchase” from an unavailable measurement in the data contract. [XGBoost paper](https://arxiv.org/abs/1603.02754)


    - Stacking
        - Concept: Stacking (or stacked generalization) involves training a meta-learner to combine the outputs of several base models, allowing for more flexible aggregation methods that can capture dependencies between predictions. In stacking, a model learns the optimal way to combine predictions.
        - Architecture
            - Base level (Level 0):Multiple heterogeneous models (different algorithms). Each trained on the same training data. Examples: Random Forests, SVMs, Neural Networks, Gradient Boosting.
            - Meta level: A model that takes base models' predictions as inputs. Learns to combine these predictions optimally. Can be any algorithm (logistic regression is common for simplicity and interpretability)

                - [To be reviewed] **Leakage control:** train the meta-model on out-of-fold predictions. Each training row's level-0 prediction must come from a model that did not train on that row. Then refit the base estimators on all development data for deployment. In time series, out-of-fold construction must also respect chronology. Otherwise the meta-model learns from unrealistically optimistic base predictions. [Stacking implementation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)

    - Voting
        - Voting is one of the simplest ensemble techniques where multiple models make predictions independently, and the final prediction is determined by combining these individual predictions. The key idea is that different models may excel in different regions of the input space, so combining them can lead to better overall performance.
        - [To be reviewed] Hard Voting (Majority Voting)

            - Choose the class receiving the most votes, optionally with predefined weights. Hard voting discards confidence: probabilities 0.51 and 0.99 both become one vote for the same class.
            - **Example:** three binary classifiers voting positive, positive, negative produce a positive decision. An ensemble of nearly identical models offers little diversity, and shared training leakage is not repaired by voting.

        - [To be reviewed] Soft Voting (Weighted Average)

            - Average predicted class probabilities, then choose a class or apply an application-specific threshold. Comparable probability scales matter; assess calibration on held-out data before averaging.
            - **Example:** probabilities 0.51, 0.51, and 0.01 average to approximately 0.343, so equal-weight soft voting chooses negative even though hard voting chooses positive. [Voting classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)

        - [To be reviewed] Weighted Voting

            - Assign nonnegative model weights and normalize by their sum for probability averaging. Tune weights on validation data or learn a regularized stacking model; never choose them from test-set performance.
            - **Example:** a domain-specialist model can receive higher weight on a validated target population, but its advantage may disappear under distribution shift. Report individual-model and ensemble results to show whether the additional inference cost buys a reproducible improvement.



- While we can also divide the supervised learning classification algorithms by Training Methods into **Generative vs. Discriminative Models**
    - Discriminative models[[wiki](https://en.wikipedia.org/wiki/Discriminative_model)], also referred to as conditional models, are a class of models frequently used for classification.
        - Discriminative models learn the conditional distribution $`P(y \mid x)`$ (or decision boundary), focusing solely on predicting labels from observed inputs.
        - [To be reviewed] Discriminative models support binary, multiclass, structured, and continuous-output prediction. The defining distinction is conditional prediction or decision modeling, not a restriction to binary classification.
        - logistic regression (LR), Support Vector Machines (SVMs), Decision Trees & Random Forests, KNN, conditional random fields (CRFs).
    - [To be reviewed] In classification, a generative model may learn $`p(x,y)`$ or class-conditional densities with class priors. More generally, generative models learn a distribution or sampling process over observations, possibly with latent variables or conditioning; an unconditional GAN or VAE need not model labeled $`p(x,y)`$.
        - Naive Bayes, Gaussian Mixture Models (GMMs), Hidden Markov Models (HMMs), Latent Dirichlet Allocation (LDA), Boltzmann Machines / RBMs, Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), Normalizing Flows, Diffusion Models

----


### Unsupervised Learning

 [[Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)] is a framework in machine learning where, in contrast to supervised learning, algorithms learn patterns exclusively from unlabeled data.

- Clustering
    - K-means
        - [To be reviewed] K-means minimizes $`J=\sum_{j=1}^k\sum_{x_i\in S_j}\|x_i-\mu_j\|^2`$, where $`\mu_j=|S_j|^{-1}\sum_{x_i\in S_j}x_i`$. It favors compact groups under a shared Euclidean metric, but does not impose equal cluster sizes. Unequal density, scale, anisotropy, and outliers can make the induced partition scientifically misleading. [K-means reference](https://scikit-learn.org/stable/modules/clustering.html#k-means)
        - Algorithm details
            - [To be reviewed] Initialization of centroids

                - K-means++ selects the first center randomly and subsequent centers with probability proportional to squared distance from the nearest existing center. It reduces poor starts but does not guarantee the globally optimal clustering in each run.
                - **Example:** compare multiple initializations at a fixed $`k`$, using final inertia and cluster stability. Standardize features when Euclidean distance should not be dominated by a feature's measurement units. [K-means++ paper](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)

            - Assignment Step (E-step): Assign each data point $`x_i`$ to the cluster $`S_j`$ whose centroid $`\mu_j`$ is nearest (typically using squared Euclidean distance).
            - [To be reviewed] Update step

                - Replace each center with the mean of its assigned points. Assignment and update steps do not increase the squared-distance objective, but convergence can reach a local optimum.
                - **Example:** points 1, 2, and 6 assigned to one one-dimensional cluster move its center to 3. Empty clusters need an explicit reinitialization policy. A lower training inertia at larger $`k`$ is expected and alone does not establish a better scientific explanation.

            - Iteration: Repeat steps 2 and 3 until convergence criteria are met.
            - WCSS - elbow method for k optimization.
        - Evaluation Metrics
            - Silhouette Score: Measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation). For a point $`i`$, let $`a(i)`$ be the average distance to other points in its cluster, and $`b(i)`$ be the minimum average distance to points in another cluster. The score is $`s(i)=(b(i)-a(i)) / \max (a(i), b(i))`$. Scores range from -1 (misclassified) to +1 (well-clustered), with 0 indicating points near cluster boundaries. The overall score is the average $`s(i)`$ over all points. Higher scores are better.
    - [To be reviewed] Hierarchical clustering

        - Agglomerative clustering starts with individual points and repeatedly merges clusters according to a linkage criterion. Single linkage uses the nearest pair and can create chains; complete linkage uses the farthest pair; average linkage averages cross-cluster distances. Ward linkage minimizes the increase in within-cluster squared error and requires Euclidean geometry.
        - **Example:** a narrow bridge of points can join two groups under single linkage while complete linkage keeps them apart. A dendrogram exposes the hierarchy, but its heights depend on linkage and feature scaling. Check sensitivity before treating a cut as a natural taxonomy. [Clustering documentation](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)

    - [[DBSCAN](https://www.youtube.com/watch?v=4AW_5nYQkuc&ab_channel=KilianWeinberger)] is a density-based clustering algorithm that groups together points that are closely packed, marking points in low-density regions as outliers (noise).
        - Parameters
            - eps ( $`\epsilon`$ ): The maximum distance (radius) between two points for one to be considered as in the neighborhood of the other.
            - min_samples (MinPts): The minimum number of points required within the e-neighborhood (including the point itself) for a point to be considered a core point.
        - [To be reviewed] A core point has at least `min_samples` observations, including itself, within $`N_\epsilon(p)=\{q\in D:\operatorname{dist}(p,q)\leq\epsilon\}`$. A border point can join a cluster without itself being a core point.
        - Algorithm
            - 1. Start with an arbitrary unvisited point $`p`$.
            - 2. Mark $`p`$ as visited.
            - [To be reviewed] 3. Retrieve the $`\epsilon`$-neighborhood $`N_\epsilon(p)`$.
            - [To be reviewed] 4. If $`|N_\epsilon(p)|<\mathrm{min\_samples}`$, mark $`p`$ as noise provisionally; later expansion can relabel it as a border point.
            - [To be reviewed] 5. If $`|N_\epsilon(p)|\geq\mathrm{min\_samples}`$, create a cluster, add $`p`$, and initialize its expansion set with the neighborhood.
            - 6. Expand the cluster: For each point q in the seed set:
                - [To be reviewed] If $`q`$ is unvisited, mark it visited and retrieve $`N_\epsilon(q)`$. If $`q`$ is core, add its neighbors to the expansion set; avoid repeated processing. Border-point assignment can depend on traversal order when it touches multiple clusters.
                - If q is not yet a member of any cluster, add q to cluster C.
            - 7. Repeat step 6 until the seed set is empty.
            - 8. Go back to step 1 and pick another unvisited point, until all points are visited.
        - [[Interview Questions](https://www.youtube.com/watch?v=cCw1yqqvOCI&ab_channel=UnfoldDataScience)]
    - [To be reviewed] Gaussian Mixture Models ([[GMMs](https://www.youtube.com/watch?v=hJLaHWaLsyg&ab_channel=AtifAdib)])

        - A GMM models $`p(x)=\sum_{k=1}^K\pi_k\mathcal{N}(x;\mu_k,\Sigma_k)`$. EM alternates posterior responsibilities with weighted updates to component parameters. Unlike K-means, assignments are soft and full covariance matrices permit elliptical components.
        - **Example:** a point between two overlapping customer populations can have responsibilities 0.6 and 0.4. Multiple restarts and covariance regularization matter: unconstrained likelihood can diverge when a component collapses onto one observation. Select covariance structure and component count using held-out density and appropriate model-selection criteria. [Gaussian mixtures](https://scikit-learn.org/stable/modules/mixture.html)



- [To be reviewed] Latent Mixture Models

    - Introduce an unobserved component variable $`z`$: $`p(x)=\sum_k p(z=k)p(x\mid z=k)`$. Gaussian mixtures are one choice of emission family; mixtures of Bernoulli or multinomial distributions suit other observations. EM exploits tractable conditional responsibilities, whereas more complicated latent models may need variational inference or sampling.
    - **Example:** a mixture of two coin biases can describe heterogeneous users, but one binary observation per user cannot identify both biases and their mixture weight. Mixture labels are interchangeable, so interpret identifiable distributions and predictions before attaching semantic names to components.

- Hidden Markov Models(HMMs)
    - [To be reviewed] Markov processes

        - In a first-order HMM, the latent state satisfies $`p(z_t\mid z_{1:t-1})=p(z_t\mid z_{t-1})`$. Observations are conditionally independent given their corresponding states. This is a modeling assumption, not a property guaranteed by sequential data.
        - **Example:** a latent “active” or “inactive” customer state can persist across days while observed event counts fluctuate. A standard HMM implies geometric state-duration distributions; use explicit-duration models when dwell times have different structure. [HMM reference](https://hmmlearn.readthedocs.io/en/stable/tutorial.html)

    - [To be reviewed] Transition probability and emission probability

        - The transition matrix $`A_{ij}=p(z_t=j\mid z_{t-1}=i)`$ has rows summing to one; an emission distribution specifies $`p(x_t\mid z_t=j)`$. Include an initial-state distribution. Forward-backward inference computes marginal state probabilities, and Baum-Welch applies EM to estimate parameters.
        - **Example:** “active” may transition to itself with probability 0.95 while emitting a Poisson event count. Do not confuse a noisy low count with a definite state transition; posterior inference combines emissions with temporal persistence.

    - [To be reviewed] Viterbi algorithm

        - Viterbi finds the highest-probability complete state sequence using dynamic programming: $`\delta_t(j)=\log p(x_t\mid j)+\max_i[\delta_{t-1}(i)+\log A_{ij}]`$, with backpointers. Dense transitions give time complexity $`O(TK^2)`$.
        - **Distinction:** the most probable complete path need not equal the sequence of marginally most probable states from forward-backward. Use log-space arithmetic or scaling to avoid numerical underflow on long sequences.

- Dimension reduction techniques
    - Linear/Subspace Methods
        - [To be reviewed] PCA

            - Center the training matrix and retain the leading right singular vectors. PCA solves a rank-constrained squared reconstruction problem; projected coordinates preserve directions of maximal sample variance, which need not be the most predictive directions.
            - **Example:** a large nuisance variation in sensor brightness can dominate the first component while a small diagnostic signal is discarded. Fit centering, scaling, and components on training folds only. Whitening changes the metric by rescaling component variances, so treat it as a separate modeling decision. [PCA documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)

        - [To be reviewed] Independent Component Analysis (ICA)

            - ICA assumes observations are linear mixtures of statistically independent sources and seeks an unmixing transform. Independence is stronger than PCA's decorrelation. Under the standard identifiable model, sources are non-Gaussian with at most one Gaussian source; their order and scale remain ambiguous.
            - **Example:** two microphones record mixtures of two speakers. PCA can whiten the observations, but ICA uses higher-order distributional structure to separate candidate sources. If sources are strongly dependent, the interpretation can fail. [FastICA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)

    - Nonlinear/ Manifold Methods
        - [To be reviewed] Isomap

            - Build a neighborhood graph, estimate geodesic distances by shortest paths, then apply multidimensional scaling. This aims to preserve manifold distances rather than straight-line distances in the ambient space.
            - **Example:** on a rolled “Swiss roll,” nearby points across different layers may be close in Euclidean distance but far along the surface. Too many neighbors can create shortcuts across folds; too few can disconnect the graph. [Isomap documentation](https://scikit-learn.org/stable/modules/manifold.html#isomap)

        - [To be reviewed] T-SNE

            - t-SNE matches local similarity distributions by minimizing $`D_{\mathrm{KL}}(P\|Q)`$, with a heavy-tailed low-dimensional distribution. Perplexity controls a notion of neighborhood size; initialization and optimization affect the result.
            - **Example:** a plot can show separated islands even when the original distribution has continuous structure. Cluster area, inter-island distance, and empty space are not automatically faithful quantitative summaries. Vary seeds and perplexity, and verify conclusions in the original representation. [t-SNE paper](https://www.jmlr.org/papers/v9/vandermaaten08a.html)

        - [To be reviewed] UMAP

            - UMAP constructs a weighted neighborhood graph and optimizes a low-dimensional representation to preserve its fuzzy connectivity. `n_neighbors` controls neighborhood scale; `min_dist` influences how tightly embedded points can pack.
            - **Example:** use a fitted transform to visualize new examples, but monitor changes in the original feature distribution: a plausible-looking projection is not evidence that a new example is in-distribution. Check neighborhood preservation and parameter sensitivity; neither UMAP nor t-SNE establishes the number of real-world classes from a plot alone. [UMAP paper](https://arxiv.org/abs/1802.03426)

    - [To be reviewed] Autoencoders

        - Learn an encoder and decoder minimizing reconstruction loss with a bottleneck or another constraint. A sufficiently expressive unconstrained network can memorize the identity map; useful representations depend on the inductive bias and training distribution.
        - **Example:** a denoising autoencoder reconstructs a clean image from a corrupted input. Evaluate frozen embeddings on a downstream task and compare against PCA at a comparable representation size. Reconstruction quality and semantic usefulness are different evaluation targets; probabilistic generation requires additional assumptions, as in VAEs.



----
## XAI
<a id="feature-importance"></a>
### [To be reviewed] Feature importance

- Separate **global reliance** from **local attribution**. Permutation importance measures the drop in held-out score when a feature is shuffled; impurity importance measures accumulated training split improvements and favors some high-cardinality features.
- **Example:** two nearly duplicate predictors can each have low permutation importance because the other substitutes for it. Shuffle a meaningful feature group, or use conditional perturbations if the scientific question requires dependence to be preserved. Report the metric, evaluation split, perturbation distribution, and uncertainty across repetitions. These methods explain fitted-model behavior; they do not identify intervention effects. [Permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html)

<a id="shap"></a>
### [To be reviewed] SHAP

- SHAP represents a prediction as $`f(x)=\phi_0+\sum_i\phi_i`$, with Shapley allocations of marginal contributions under a specified coalition-value function. The background distribution and the treatment of missing features determine what the attribution means.
- **Example:** a baseline log-odds of -1 plus contributions 1.5 and -0.2 yields log-odds 0.3, not a probability of 0.3. State the output scale. Correlated features can receive very different allocations under conditional and interventional formulations; local additivity does not resolve causal identification. Check stability to background sampling and feature grouping. [SHAP paper](https://arxiv.org/abs/1705.07874)

<a id="attention-based"></a>
### [To be reviewed] Attention-based

- Attention weights describe how a particular module mixes representations. They are not sufficient evidence of an input feature's causal effect: values, residual paths, other heads, and later nonlinearities also affect the output.
- **Example:** a high attention weight on a token with a near-zero value vector may have little effect on the final prediction. Compare attention visualizations with controlled ablations, output sensitivity, and adversarially altered attention patterns. Faithfulness depends on the architecture and explanation question. [Attention is not Explanation](https://arxiv.org/abs/1902.10186)

- [To be reviewed] RETAIN

    - RETAIN applies reverse-time attention at two levels: scalar visit weights and vector-valued variable weights. Its context is $`c=\sum_t\alpha_t(\beta_t\odot v_t)`$. With linear visit embeddings and a linear output logit, one can decompose the logit into additive contributions of observed variables, conditional on the computed attention weights.
    - **Example:** a recent diagnosis code may receive a larger contribution than an older visit. This supports auditing the model's prediction; it does not imply that changing the diagnosis code would change the patient's underlying outcome. [RETAIN paper](https://arxiv.org/abs/1608.05745)
