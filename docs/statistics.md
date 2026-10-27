# Statistics

[Handbook contents](../README.md#table-of-contents) | [Next: Machine Learning](machine-learning.md)

## Contents

- [Probability and Conditioning](#stats-probability)
- [Expectation, Variance, and Dependence](#stats-moments)
- [Sampling, Estimation, and Uncertainty](#stats-uncertainty)
- [Hypothesis Test](#hypothesis-test)
    - [Effect Size, Power, and Multiplicity](#stats-testing)
- [MLE](#mle)
- [Bayesian Inference, MAP, and Prediction](#stats-bayes)
- [Data Design, Experiments, and Causality](#stats-experiments)
- [Information Theory and Probability Calibration](#stats-information)

This chapter supplies the statistical foundations used by [ML](machine-learning.md), [DL](deep-learning.md), and [evaluation](benchmarking-and-evaluation.md). Read the assumptions alongside the formulas; identifying the sampling unit and target population is as important as choosing an estimator.

<a id="stats-probability"></a>
## [To be reviewed] Probability and Conditioning

A random variable maps outcomes to values. Discrete probabilities sum to one; continuous densities integrate to one, but a density value is not a point probability. Joint, marginal, and conditional distributions answer different questions. For discrete events with positive denominator, $`P(A\mid B)=P(B\mid A)P(A)/P(B)`$. The chain rule factors a joint distribution into successive conditionals without assuming independence; this underlies [autoregressive language modeling](large-language-models.md#pre-training).

**Base-rate example:** a target occurs in 1% of cases. A classifier flags 90% of targets and 5% of nontargets. Among flagged cases, the target probability is $`0.9(0.01)/[0.9(0.01)+0.05(0.99)]\approx0.154`$. High sensitivity does not imply high precision; deployment prevalence matters.

| Distribution | Useful modeling role | Assumption to check |
| --- | --- | --- |
| Bernoulli / binomial | One binary outcome / a count of n such outcomes | Binomial assumes independent trials with a common success probability |
| Categorical | One of K mutually exclusive classes | Multilabel outcomes need a different joint specification |
| Gaussian | Continuous noise or a parameter prior | Convenient algebra does not establish normality of data |
| Poisson | Counts over a specified exposure | Its conditional mean equals its variance; overdispersion needs attention |

<a id="stats-moments"></a>
## [To be reviewed] Expectation, Variance, and Dependence

Expectation is a probability-weighted average; variance measures squared deviation from that average. Covariance measures linear co-movement. Independence implies zero covariance when moments exist; the converse fails: if X is uniform on [-1, 1] and Y = X squared, covariance is zero although Y is determined by X. Correlation is also not causation.

Linearity of expectation does not require independence; the variance of a sum additionally depends on covariance.

Two reusable identities are $`\mathbb E[Y]=\mathbb E[\mathbb E[Y\mid X]]`$ and $`\operatorname{Var}(Y)=\mathbb E[\operatorname{Var}(Y\mid X)]+\operatorname{Var}(\mathbb E[Y\mid X])`$, under the corresponding integrability/finite-variance conditions. They separate within-group variability from between-group variability. Nonlinear transformations do not generally commute with expectation: $`\mathbb E[X^2]=(\mathbb E[X])^2+\operatorname{Var}(X)`$. These ideas support [bias/variance analysis](machine-learning.md#model-evaluation-and-selection) and [RL baselines](large-language-models.md#rl-credit-assignment). [MIT probability lectures](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/60eede37f87fd886d4ba574e1cf26617_MIT6_041F10_lec_slides.pdf)

<a id="stats-uncertainty"></a>
## [To be reviewed] Sampling, Estimation, and Uncertainty

A population parameter is the target; an estimator is a sample-dependent rule; its realized value is an estimate. Under iid sampling with finite, nonzero variance, the law of large numbers supports sample-mean consistency and the central limit theorem gives an asymptotically normal standardized mean. The observations themselves need not be Gaussian. Dependence and heavy tails can invalidate the usual approximation.

**SD versus SE:** standard deviation describes observation spread; standard error describes estimator variability across repeated samples. For n iid Bernoulli observations, a plug-in estimate is $`\widehat{\operatorname{SE}}(\hat p)=\sqrt{\hat p(1-\hat p)/n}`$. With 800 successes out of 1,000, SE is about 0.0126; a rough 95% normal interval is [0.775, 0.825]. Use a suitable binomial interval near boundaries or with small counts. Frequentist 95% coverage describes the interval procedure across samples, not a posterior probability for one fixed parameter. [Confidence-interval interpretation](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)

**Bootstrap:** resample the appropriate independent units and recompute the statistic. Use paired task outcomes for model differences; preserve user/repository clusters or temporal dependence when required. Resampling tokens does not create independent tasks. Fixed-checkpoint task uncertainty and variability from retraining/seeds are distinct. See [paired evaluation and clustering](benchmarking-and-evaluation.md#eval-statistics).

## Hypothesis Test
[[Wiki](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)]
A statistical hypothesis test is a method of statistical inference used to decide whether the data provide **sufficient evidence to reject** a particular hypothesis. A statistical hypothesis test typically involves a calculation of a test statistic. Then a decision is made, either by comparing the test statistic to a critical value or equivalently by evaluating a p-value computed from the test statistic.

- The p-value is the probability, under the specified null model and sampling assumptions, of a test statistic at least as extreme as observed under the prespecified tail/extremeness rule.
    - [To be reviewed] A p-value is a tail probability for a specified test statistic under the null model; it is not the probability that the null is true, the probability of the observed dataset, or an effect-size measure. Report the effect estimate, uncertainty, assumptions, and multiplicity/selection protocol. [ASA statement](https://doi.org/10.1080/00031305.2016.1154108)

<a id="stats-testing"></a>
### [To be reviewed] Effect Size, Power, and Multiplicity

The Type I error rate is the probability of rejection under the null; the Type II error rate is the probability of failing to reject under a specified alternative. Power is one minus the Type II error rate and depends on effect size, noise, sample size, and the test. A nonsignificant result does not establish equivalence; equivalence needs a prespecified margin and appropriate analysis. Statistical significance does not establish practical importance.

Testing many metrics/checkpoints or repeatedly inspecting results changes false-positive control. Prespecify primary outcomes, a minimum useful effect, sample size/stopping rules, and an appropriate multiple-testing or sequential procedure. Family-wise error and false discovery rate control different quantities. Keep exploratory results distinct from confirmation. [Statistical learning: inference and multiple testing](https://www.statlearning.com/)

## MLE
- Definition: Maximum Likelihood Estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution by maximizing a likelihood function. It's a fundamental technique in both statistics and machine learning.
- For observed data x and parameter theta, maximize $`L(\theta;x)=p_\theta(x)`$ (or its log), where p denotes the joint probability mass or density appropriate to the model. Existence and uniqueness of a finite maximizer require separate conditions; see the [logistic-regression separation example](machine-learning.md#supervised-learning).
- [To be reviewed] A likelihood treats the observed data as fixed and varies the model parameter. For continuous observations it is typically a joint density, not the probability of an exact dataset, and it need not integrate to one over parameters. MLE applies to general parameterized models, beyond means and standard deviations.

<a id="stats-bayes"></a>
## [To be reviewed] Bayesian Inference, MAP, and Prediction

Bayesian inference updates a prior using data: $`p(\theta\mid D)\propto p(D\mid\theta)p(\theta)`$. MAP maximizes this posterior density; MLE maximizes likelihood alone. Gaussian coefficient priors can yield L2 penalties and Laplace priors L1 penalties under matching likelihood/scaling conventions. MAP is a parameterization-dependent point estimate, not posterior averaging or the same operation as AdamW decay.

**Example:** a Bernoulli probability with Beta(2, 2) prior and 8 successes in 10 independent trials has posterior Beta(10, 4). Its next-trial predictive success probability is 10/14, compared with MLE 8/10. This shrinkage follows the specified prior; it is not a universal correction for biased data. A Bayesian credible interval describes posterior probability conditional on the model/prior; it is different from frequentist coverage. [MIT statistics readings](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_statistics.pdf)

<a id="stats-experiments"></a>
## [To be reviewed] Data Design, Experiments, and Causality

Specify the target population, sampling/assignment unit, and available information before splitting data. Repeated records from one user or near-duplicate documents are not independent evidence. Fit preprocessing and model selection inside the development protocol; keep the final test outside it. Missingness and selection can distort conclusions even with a large dataset.

An A/B experiment estimates a defined outcome contrast under randomized assignment, not merely a difference between two logged groups. Randomize/analyze at an appropriate unit, check assignment balance, account for interference and delayed outcomes, and distinguish assigned treatment from actual exposure. An observed association $`P(Y\mid A=a)`$ generally differs from the outcome under an intervention on A; observational identification needs assumptions such as consistency, exchangeability, and positivity.

**Recommendation example:** promoted items receive more exposure and clicks. A model predicting those clicks need not identify the incremental effect of promotion. Compare properly randomized policies or justify an observational identification strategy; see [exposure bias and online experiments](recommendation-systems.md#rec-evaluation). [Hernán and Robins: Causal Inference](https://miguelhernan.org/whatifbook)

<a id="stats-information"></a>
## [To be reviewed] Information Theory and Probability Calibration

Entropy is expected surprisal; KL is nonnegative but asymmetric, rather than a distance metric. For categorical target distribution p and prediction q, cross-entropy satisfies $`H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)`$. With fixed p, minimizing expected log loss fits the conditional distribution within the model class; finite data, misspecification, and distribution shift remain. See [ML losses](machine-learning.md#algorithm-categories) and [LLM likelihood/perplexity](large-language-models.md#evaluation-and-monitoring) for implementations. Low predictive entropy can mean confident error rather than factual correctness.

**Calibration:** among comparable binary predictions near 0.8, roughly 80% should be positive in the evaluated population. Discrimination/ranking and calibration are different: ROC-AUC does not determine probability accuracy. Use held-out log loss/Brier score and reliability plots; these scores assess overall probabilistic prediction, not calibration alone. Binned calibration errors depend on binning and sample size.

Temperature scaling fits a positive scalar on held-out logits. It preserves multiclass argmax labels while changing probabilities, so accuracy can stay unchanged as calibration changes. Evaluate on separate data and relevant slices; one fitted temperature need not calibrate every subgroup or a shifted population. [Guo et al., calibration](https://proceedings.mlr.press/v70/guo17a.html)
