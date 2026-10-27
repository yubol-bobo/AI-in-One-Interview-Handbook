# Statistics

[Handbook contents](../README.md#table-of-contents) | [Next: Machine Learning](machine-learning.md)

## Contents

- [Hypothesis Test](#hypothesis-test)
- [MLE](#mle)


## Hypothesis Test
[[Wiki](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)]
A statistical hypothesis test is a method of statistical inference used to decide whether the data provide **sufficient evidence to reject** a particular hypothesis. A statistical hypothesis test typically involves a calculation of a test statistic. Then a decision is made, either by comparing the test statistic to a critical value or equivalently by evaluating a p-value computed from the test statistic.

- The p-value is the probability, under the specified null model and sampling assumptions, of a test statistic at least as extreme as observed under the prespecified tail/extremeness rule.
    - [To be reviewed] A p-value is a tail probability for a specified test statistic under the null model; it is not the probability that the null is true, the probability of the observed dataset, or an effect-size measure. Report the effect estimate, uncertainty, assumptions, and multiplicity/selection protocol. [ASA statement](https://doi.org/10.1080/00031305.2016.1154108)

## MLE
- Definition: Maximum Likelihood Estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution by maximizing a likelihood function. It's a fundamental technique in both statistics and machine learning.
- For observed data x and parameter theta, maximize $`L(\theta;x)=p_\theta(x)`$ (or its log), where p denotes the joint probability mass or density appropriate to the model. Existence and uniqueness of a finite maximizer require separate conditions; see the [logistic-regression separation example](machine-learning.md#supervised-learning).
- [To be reviewed] A likelihood treats the observed data as fixed and varies the model parameter. For continuous observations it is typically a joint density, not the probability of an exact dataset, and it need not integrate to one over parameters. MLE applies to general parameterized models, beyond means and standard deviations.
