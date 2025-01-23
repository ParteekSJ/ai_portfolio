---
title: "Probability Distributions & their conjugate"
date: "2025-01-19"
summary: "Probability Distributions & their Conjugate."
description: "This blog discusses about probability distribution and their conjugate."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---


## Bernoulli Distribution
A Bernoulli random variable takes only two possible outcomes, typically denoted as 0 or 1. The probability of observing $X=1$ is given by a parameter $\lambda\in[0,1]$ known as the "probability of success".  Thus 
$$ \begin{align*} 
p(X=0|\lambda) = 1-\lambda \\\\ 
p(X=1|\lambda) = \lambda\end{align*}$$

The probability mass function (PMF) can be written as $$p(x) = \lambda^x\cdot (1-\lambda)^x$$.
This is the simplest discrete distribution, often used to model binary outcomes like coin flips (head/tail), success/failure in a trial, or on/off states

## Categorical Distribution
Categorical Distribution generalizes the Bernoulli distribution from two outcomes to $K$ possible outcomes. It's the distribution of a random variable $X$ that can take exactly one of $K$ discrete states, where each state $k$ is associated with a probability $\lambda\_k$.
- Each $\lambda\_k \in [0,1]$ and $\sum\_{k=1}^K \lambda\_k = 1$
- $X$ is often represented as a one-hot vector $\mathbf{x}$, where exactly one element is 1 and all others are 0.

For example, if $K=5$, a possible outcome is $\mathbf{x}=[0,0,1,0,0]^T$ means that the third state has occurred. The parameters $\Lambda=[\lambda\_1, \lambda\_2, \lambda\_3, \lambda\_4, \lambda\_5]^T$ represent the probability that the random variable takes each corresponding value. The PMF can be written as follows: 
$$ p(\boldsymbol{X}=\boldsymbol{e}\_k|\lambda) = \lambda\_k $$ or $$ \begin{align*} p(\boldsymbol{x}) = \prod\_{k=1}^K \lambda\_k^{x\_k} = \lambda\_k \\ p(\boldsymbol{x}) = \text{Cat}\_x[\lambda] \end{align*} $$
In the following equation, $p(\boldsymbol{x}) = \prod\_{k=1}^K \lambda\_k^{x\_k} = \lambda\_k$, $x\_k=1$ if and only if the $k \in K$ is selected. The remaining terms cancel out.
## Univariate Normal Distribution
This is also known as Gaussian distribution. This distribution is used to describe a single continuous variable, $X$, i.e., $x\in\mathbb{R}$. It is parameterized by two values
- $\mu \in \mathbb{R}$ (*mean*)
- $\sigma^2 > 0$ (*variance*)
The PDF can be described as follows: $$ p(X=a|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp - \frac{(a-\mu)^2}{2\sigma^2}, a\in\mathbb{R} $$OR
$$ p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp - \frac{(x-\mu)^2}{2\sigma^2} $$
## Multivariate Normal Distribution
This distribution can be used to describe $D$-dimensional continuous variable $\boldsymbol{X}$, i.e., $\boldsymbol{x}\in\mathbb{R}^D$. Such a distribution is parameterized by
- $\boldsymbol{\mu} \in \mathbb{R}^D$ ($D$-dimensional mean), and
- $D\times D$ positive definite covariance matrix $\Sigma\in\mathbb{R}\_{+}^{D\times D}$
The PDF can be described as follows: $$ p(\boldsymbol{X}=\boldsymbol{a}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})=\frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\{-0.5(\boldsymbol{a}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{a}-\boldsymbol{\mu})\},\boldsymbol{a}\in\mathbb{R}^D $$


## Conjugate Distributions (Distribution of the Parameters)
Conjugate distributions model the parameters of the probability distribution.  It is analogous to prior distribution of a parameter of a distribution. Product of a probability distribution and its conjugate has the same form as the conjugate times a constant.

> **Parameters of the conjugate distribution** are known as **hyperparameters** because they control the parameter distributions. These parameters aren't learned from data but empirically set.

For example:
1. For a Bernoulli Distribution with domain: $x\in\{0,1\}$, the parameters are best modeled by beta distribution
2. For a Bernoulli Distribution with domain: $x\in\{1,2,\dots,K\}$, the parameters are best modeled by Dirichlet distribution.
3. For a univariate normal distribution with domain $x\in\mathbb{R}$, the parameters are modeled by normal inverse gamma distribution.
4. For a multivariate normal distribution with domain $\boldsymbol{x}\in\mathbb{R}^k$, the parameters are modeled by normal inverse Wishart distribution.

## Importance of Conjugate Distribution
Conjugate distribution is crucial to learning/modeling the parameters $\theta$ of a probability distribution given observed data, i.e., $p(\theta|x)$. Recalling, Bayes' rule we obtain: $$ p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)} = \frac{p(x|\theta) p(\theta)}{\int p(x,\theta) d\theta} = \frac{p(x|\theta) p(\theta)}{\int p(x|\theta) p(\theta) d\theta}  $$


We choose a prior distribution $p(\theta)$ that is conjugate to the likelihood. If the prior is not a conjugate to the likelihood, then the posterior distribution $p(\theta|x)$ doesn't have a closed-form solution and hence becomes **intractable**. Thus, this implies that the posterior must have the same form as the conjugate prior distribution, i.e., closed-form. The normalizer (**evidence**) also consists of the likelihood and the prior distribution. The posterior must be a distribution which implies that evidence must equal constant $\kappa$ from conjugate relation.

Marginalizing over parameters, i.e., **Predictive Distribution** $$p(x^\*|\boldsymbol{x})  = \int p(x^\*|\theta) p(\theta|x) d\theta$$ where 
- $\boldsymbol{x}$ denotes the observed data points, and 
- $x^\*$ is the new, unobserved data point you want to predict

$p(x^\*|\boldsymbol{x})$ represents the predictive distribution in a Bayesian framework. This distribution allows you to predict a future observation $x^\*$ after having observed some data $\boldsymbol{x}$, by integrating over the uncertainty in the model parameters $\theta$. To predict a new data point, we use $p(x^\*|\boldsymbol{x})  = \int p(x^\*|\theta) p(\theta|x) d\theta$ where
- $p(x^\*|\theta)$ - Likelihood of the new observation $x^\*$ given the particular value of the parameters $\theta$. If we knew $\theta$ exactly, this would be straightforward to compute.
 - $p(\theta|\boldsymbol{x})$ - Posterior distribution of $\theta$ given observed data $\boldsymbol{x}$. It represents what you believe about the parameter values, incorporating your prior knowledge and the observed data
 - $\int p(x^\*|\theta) p(\theta|x) d\theta$ - Since we aren't certain about $\theta$, we weigh the likelihood of $x^\*$ for each possible parameter setting $\theta$ by how probable that parameter setting is, according to the posterior. In other words, you average (integrate) over all plausible parameter values, weighted by their posterior probability, to obtain a final predictive distribution for $x^\*$.

In classical (frequentist) statistics, one might plug in a single estimate $\hat{\theta}$ (like the maximum likelihood estimate or a point estimate from a fitted model) into the likelihood $p(x^\*|\hat{\theta})$ to predict. However, this ignores uncertainty in $\theta$. If there is substantial uncertainty about the parameters, a single $\hat{\theta}$ could be misleading.

The Bayesian predictive distribution $p(x^\*|\boldsymbol{x})$, however, integrates over all possible parameter values according to their posterior distribution, leading to a more robust and honest representation of uncertainty. This approach naturally incorporates parameter uncertainty into predictions. It means that if you're unsure about $\theta$, your predictive distribution $p(x^\*|\boldsymbol{x})$ will be more spread out, reflecting that uncertainty.

Instead of using a single fixed parameter estimate (*such as MLE*), we hold a distribution over $\theta$. Each time we incorporate new data, this distribution (the posterior) changes. We then predict the future outcome by integrating over this revised posterior. This naturally updates as we get more data. Initially, the predictive distribution might be uncertain and close to 50/50 (if the prior is centered at 0.5). As data shows more heads, both the posterior mean $\theta$ and predictive probability for the next head become larger. We are using the entire posterior distribution of parameters to make predictions, not just a single point estimate.

![alt text](/assets/posts/distribution_and_conjugate/L1-PredictiveDistribution.png#dark#small "Predictive Distribution Demonstration.")


Marginalizing over parameters is used because
1. accounts for uncertainty in the parameters $\theta$.
2. provides a complete and coherent way to make predictions about new data $x^\*$ in a Bayesian context.
3. avoids overconfidence that can arise by conditioning on a single, possibly imprecise parameter estimate.

- $p(\theta|x)$ (*posterior distribution*) is chosen as conjugate to the other term.
- Integral becomes easy -- the product becomes a constant times a distribution.

The derivation of the predictive distribution integration is as follows: 

$$ \begin{align*} 
p(x^\*)&=\frac{p(x^\*,x)}{p(x)}  \\\\ 
\end{align*}$$


<!-- ![[L1-ConjugateDist-Proof.png]] -->

## Conjugate Distribution: Beta Distribution
> Beta distribution is the conjugate distribution of **Bernoulli distribution**. 

It’s often used as a prior distribution for the parameter of a Bernoulli or Binomial distribution in a Bayesian framework. The distribution is defined over the Bernoulli distribution parameter $\lambda\in[0,1]$, i.e., $$\begin{align*} p(\lambda) = \frac{\Gamma[\alpha+\beta]}{\Gamma[\alpha]\Gamma[\beta]}\lambda^{\alpha-1} (1-\lambda)^{\beta-1} \\ p(\lambda)=\text{Beta}\_\lambda[\alpha,\beta] \end{align*} $$For the beta distribution, parameters $\alpha,\beta$ are the hyperparameters that we can fiddle with. 
- $\alpha$ can be viewed as representing the number of "successes" you’ve seen (or your prior guess). 
- $\beta$ can be viewed as representing the number of "failures".

The gamma function is defined as follows; $$ \begin{align**} \Gamma(z)=\int\_{0}^{\infty}t^{z-1}e^{-t}dt,\quad z\in\mathbb{C} \\ \Gamma(n)=(n-1)!,\quad n\in\mathbb{R}\_{>0}  \end{align**} $$If we have no prior knowledge of the prior distribution, we can use $\alpha,\beta=1,1$. 
![alt text](/assets/posts/distribution_and_conjugate/L1-Beta-BinomialDist.png#dark#small "Beta-Binomial Distribution.")



## Conjugate Distribution: Dirichlet Distribution
Dirichlet distribution is the conjugate distribution of **Categorical distribution**. It is defined over the $K$ parameters of the categorical distribution, $\lambda\_k\in[0,1]$, where $\sum\_k \lambda\_k = 1$. The conjugate distribution can be defined as follows: $$ \begin{aligned}&p(\lambda\_{1},...,\lambda\_{K})=\frac{\Gamma[\sum\_{k=1}^{K}\alpha\_{k}]}{\prod\_{k=1}^{K}\Gamma[\alpha\_{k}]}\prod\_{k=1}^{K}\lambda\_{k}^{\alpha\_{k}-1},\\& p(\lambda\_{1},...,\lambda\_{K})=\mathrm{Dir}\_{\lambda\_{1...K}}[\alpha\_{1},...\alpha\_{K}] \end{aligned} $$
The hyperparameters that we have for the Dirichlet distribution are $[\alpha\_1,\dots,\alpha\_k]$. If we have no prior information, all alphas can be set to 1.

![alt text](/assets/posts/distribution_and_conjugate/L1-Dirichlet-CategoricalDist.png#dark#small "Dirichlet Distribution.")



Vertices of the simplex can be used to denote the individual categories.
## Conjugate Distribution: Normal Inverse Gamma Distribution
Normal Inverse Gamma distribution is the conjugate distribution of **Univariate Normal distribution**. This distribution is described over parameters $\mu,\sigma^2 >0$ of the univariate distribution. $$ \begin{aligned}&p(\mu,\sigma^2\mid\mu\_0,\lambda,\alpha,\beta)=\frac{\sqrt{\lambda}}{\sigma\sqrt{2\pi}}\frac{\beta^\alpha}{\Gamma(\alpha)}(\sigma^2)^{-(\alpha+1)}\exp\left(-\frac{\beta}{\sigma^2}\right)\exp\left(-\frac{\lambda(\mu-\mu\_0)^2}{2\sigma^2}\right)\\&p(\mu,\sigma^{2})=\text{ NormInvGam}\_{\mu,\sigma^{2}}[\mu\_0,\lambda,\alpha,\beta]\end{aligned} $$Such a conjugate distribution consists of 4 hyperparameters $\alpha,\beta,\gamma,\delta$. 

- $\mu\_0 \in \mathbb{R}$ is the prior mean of $μ$.
- $\lambda>0$ influences the confidence in $μ$​.
- $α>0$ and $β>0$ shape the distribution over $σ^2$.
![[L1-Normal-Inverse-Gamma-Dist\_UnivaraiateNormalDist.png]]
![alt text](/assets/posts/distribution_and_conjugate/L1-Normal-Inverse-Gamma-Dist_UnivaraiateNormalDist.png#dark#small "Normal Inverse Gamma Distribution.")



## Conjugate Distribution: Normal Inverse Wishart
Normal Inverse Wishart is the conjugate distribution of **Multivariate Normal distribution**. It is defined on the parameters $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ of the multivariate distribution. $$\begin{aligned}&p(\boldsymbol{\mu},\boldsymbol{\Sigma})=\frac{\gamma^{D/2}|\boldsymbol{\Psi}|^{\alpha/2}\exp[-0.5\left(\mathrm{Tr}[\boldsymbol{\Psi}\boldsymbol{\Sigma}^{-1}]+\gamma(\boldsymbol{\mu}-\boldsymbol{\delta})^{T}\boldsymbol{\Sigma}^{-1}(\boldsymbol{\mu}-\boldsymbol{\delta})\right)]}{2^{\alpha D/2}(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{(\alpha+D+2)/2}\Gamma\_{D}[\alpha/2]}\\&p(\boldsymbol{\mu},\boldsymbol{\Sigma})=\mathrm{NorlWis}\_{\boldsymbol{\mu},\boldsymbol{\Sigma}}[\alpha,\boldsymbol{\Psi},\gamma,\boldsymbol{\delta}]\end{aligned}  $$
Such a distribution consists of 4 hyperparameters:
 - a positive scalar $\alpha$
 - a positive definite matrix $\Psi \in \mathbb{R}\_+^{D\times D}$
 - a positive scalar $\gamma$
 - vector $\delta \in \mathbb{R}^D$

![alt text](/assets/posts/distribution_and_conjugate/L1-Normal-Inverse-Wishart-Dist_MultivariateNormalDist.png#dark#small "Normal Inverse Wishart Distribution.")

By changing the hyperparameters for the Normal Inverse Wishart distribution, we get different distributions for the mean and variance vectors.


