---
title: "KL Divergence"
date: "2024-11-12"
summary: "KL Divergence Explained (with Derivation)."
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

## Introduction
KL divergence is a measure of how one probability distribution is different from a second, reference distribution. Such a dissimilarity quantification is essential in variational inference.

Consider the random variable $X$ with its possible states $\{x_1,x_2,\dots,x_n\}$. We have two probability distributions over the random variable $X$: $p_\theta$ and $q_\phi$. Given the states that the random variable $X$ can take, we compute the log probabilities (log-likelihoods) instead of probabilities to avoid computational issues with very small values.

For each state $x_i$, one way to understand the dissimilarity between the two distributions is to consider the difference in their log probabilities $$ \log p_\theta(x_i) - \log q_\phi(x_i), \forall \space i \in1,\dots, n $$If the difference results in zero for all $x_i$, then the distributions $p_\theta$ and $q_\phi$ are identical. This difference can also be expressed as $$  \log p_\theta(x_i) - \log q_\phi(x_i)  = \log \left[ \frac{p_\theta (x_i)}{q_\phi(x_i)}\right] $$The ratio ${p_\theta (x_i)} / {q_\phi(x_i)}$ is referred to as **likelihood ratio**. 
The entire term is referred to as **log-likelihood ratio**.

## Expected Value (Concept 1)
The expected value (mean) of a random variable $X$ under the distribution $p\_\theta$ is defined as: $$ \mathbb{E}\_{p\_\theta}[X] = \sum\_{i=1}^\theta x\_i p\_\theta(x\_i)$$where $x\_i$ is the **state** of the random variable $X$ and $p\_\theta(x\_i)$ is the probability of that state.
Similarly, the expected value of a function of a random variable, i.e., $h(X)$ can be computed as follows: $$ \mathbb{E}\_{p\_\theta}[h(X)] = \sum\_{i=1}^\theta h(x\_i) p\_\theta( x\_i)$$For continuous random variables, the sums are replaced with integrals over the probability density function: $$  \mathbb{E}\_{p\_\theta}[h(X)] = \int\_{-\infty}^{\infty}  h(x) p\_\theta( x) dx $$

___

Returning to the KL divergence derivation, the log-likelihood ratio is a function of the random variable $X$.  Thus the expected log-likelihood ratio under $p\_\theta$ is calculated as follows: $$ \begin{aligned}
D\_{KL}(p\_\theta\parallel q\_\phi) & =\mathbb{E}\_{p\_\theta}\left[\log\left(\frac{p\_\theta(X)}{q\_\phi(X)}\right)\right] \\
 & =\sum\_{i=1}^np\_\theta(x\_i)\log\left(\frac{p\_\theta(x\_i)}{q\_\phi(x\_i)}\right)
\end{aligned} $$
For continuous random variables $$ D\_{KL}(p\_\theta \parallel  q\_\phi)=\int\_{-\infty}^\infty p\_\theta(x)\log\left(\frac{p\_\theta(x)}{q\_\phi(x)}\right)dx $$Computing these expressions exactly can be challenging, especially when $n$ is large or the 
integrals are intractable.

For both discrete and random variable formulations, there would be a lot of computational problems. This stems from the fact that the summation is until $\infty$, and/or the integration is from $-\infty$ to $\infty$.

## Law of Large Numbers (Concept 2)
The Law of Large Numbers states that the sample average of a function of a random variable converges to its expected value as the number of samples $N$ approaches infinity, i.e., $N\to\infty$ 
$$   \frac{1}{N}\sum_{i=1}^N h(x_i) \approx \mathbb{E}_p [h(X)]$$ where $x_i$ are independent samples drawn from the distribution $p$.
As $N$ gets larger, the average tends to get closer to the true expected value of the random variable $X$. $N$ has to be a large number for the approximation to be held true.


Using the Law of Large Numbers, we can approximate the KL divergence by sampling from $p\_\theta$: $$ D\_{KL}(p\_\theta \parallel  q\_\phi) = \mathbb{E}\_{p\_\theta} \left[ \log \left( \frac{p\_\theta(X)}{q\_\phi(X)} \right) \right] \approx  \frac{1}{N}\sum\_{i=1}^N \log \left( \frac{p\_\theta(x\_i)}{q\_\phi(x\_i)} \right)$$Similarly, we can approximate $D\_{KL}(q\_\phi \parallel  p\_\theta)$ by sampling from $q\_\phi$:
$$ D\_{KL}(q\_\phi \parallel  p\_\theta) = \mathbb{E}\_{q\_\phi} \left[ \log \left( \frac{q\_\phi(X)}{p\_\theta(X)} \right) \right] \approx  \frac{1}{N}\sum\_{i=1}^N \log \left( \frac{q\_\phi(x\_i)}{p\_\theta(x\_i)} \right)$$

> **Note:** In practice, we often choose to sample from the distribution that is easier to sample from or compute.


$$ \begin{gather*} D_{KL}(p_\theta \parallel  q_\phi) \neq D_{KL}(q_\phi \parallel  p_\theta) \end{gather*}$$KL-divergence is not a symmetric metric.  This asymmetry is why it's called a divergence rather than a distance metric.

## Forward vs. Reverse KL Divergence

Depending on which distribution is the reference, and which is the approximation. 
- **Forward KL divergence** $D_{KL}(p_\theta \parallel  q_\phi)$: Minimizing this tends to produce a **mean-seeking** behavior, where the approximating distribution $q_\phi$ spreads out to cover the support of $p_\theta$. 
- **Reverse KL divergence** $D_{KL}(q_\phi \parallel  p_\theta)$: Minimizing this tends to produce a **mode-seeking** behavior, where $q_\phi$​ focuses on the modes (peaks) of $p_\theta$​.


**Example:** Suppose $p$ is a bimodal distribution, and we want to approximate it with a unimodal distribution $q$:
- Minimizing **forward KL divergence** $D_{KL}(p \parallel  q)$ will result in $q$ trying to cover both modes, often centering between them.
- Minimizing **reverse KL divergence** $D_{KL}(q\parallel  p)$ will result in $q$ focusing on one of the modes, effectively ignoring the other.

The choice between forward and reverse KL divergence depends on the application and desired properties of the approximation.

**Cross-Entropy Loss** implicitly involves the **forward KL divergence**. In classification tasks, minimizing cross-entropy between the true labels and predicted probabilities is equivalent to minimizing the forward KL divergence between the empirical distribution and the model's predicted distribution.

> Reverse KL divergence is mostly used when it comes to density estimation tasks.

![alt text](/assets/posts/kl_divergence/forward_reverse_kldiv.svg#dark#small "Using Forward & Reverse KL Divergence to Approximate a Bimodal Distribution")



<!-- ## Properties of KL Divergence
1. Non-Negativity (Gibbs Inequality): $\mathrm{KL}(P\parallel Q) \geq 0$. Equality holds iff $P=Q$ almost everywhere.
2. Non Symmetry: $\mathrm{KL}(P\parallel Q) \neq \mathrm{KL}(Q\parallel P)$
3. Additivity for Independent Distributions: If $P(x,y)=P(x)P(y)$ and $Q(x,y)=Q(x)Q(y)$ then $$ \mathrm{KL}(P(x,y) \parallel  Q(x,y))=\mathrm{KL}(P(x) \parallel  Q(x))+\mathrm{KL}(P(y) \parallel  Q(y)) $$
4. Relation to Entropy: KL Divergence can be expressed in terms of entropy: $$ \mathrm{KL}(P\parallel Q)=H(P,Q)-H(P) $$ where 
	- $H(P)$ is the entropy of the distribution $P$.
	- $H(P,Q)$ is the cross-entropy between the distributions $P$ and $Q$.


--- -->