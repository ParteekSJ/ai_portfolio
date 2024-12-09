---
title: "Cross Entropy Loss"
date: "2024-10-05"
summary: "BCE and CE Loss Explained."
description: "This blog discusses about different types of normalization used in deep learning."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---


<!-- - negative log likelihood
- optimization surface
- derivation
- sample intuitive example. -->

## Introduction

The **Cross-Entropy Loss** is a loss function commonly used in classification tasks. The formula for the cross-entropy between two distributions $P^∗$ (true distribution) and $P$ (model distribution) is: $$H(P^\*|P) = -\sum_i P^\*(i) \log P(i)$$
Imagine we've been given an image of an animal. We've been tasked to classify this image into one of $N$ animal categories. Modeling the neural network's output as a probability distribution introduces the idea of uncertainty in our predictions. This approach allows us to quantify the uncertainty we have in our predictions.

The input to the model $h_\theta$ can be described as $x_i$. The output of the model is a distribution $P(y | x_i ; \theta)$ over the possible classes $y$. If we have the true class distribution, i.e., $P^\*(y|x\_i)$ (*ground truth labels*), we can optimize the parameters $\theta$ of model such that model distribution matches the true class distribution as closely as possible, i.e.,  $$P(y | x\_i ; \theta) \approx P^\*(y|x\_i)$$The Kullback-Leibler (KL) divergence between $P$ and $P^\*$ can be defined as follows: $$ D\_{\text{KL}}(P^\* \parallel P) = \sum\_i P^\*(i) \log \left(\frac{P^\*(i)}{P(i)}\right) $$
Intuitively, minimizing $D_{KL}(p^\*(y|x_i) \parallel p(y|x_i;\theta))$ seems the right choice for making $P \approx P^\*$. Expanding the KL Divergence yields,
$$ \begin{align*}D_{KL}  &= \sum_y P^\*(y|x\_i) \log \left( \frac{P^\*(y|x\_i)}{P(y|x\_i;\theta)} \right) \\\\ &= \sum\_y P^\*(y|x\_i) \left[ \log P^\*(y|x\_i) - \log P(y|x\_i;\theta) \right] \\\\&= \sum\_y P^\*(y|x\_i) \log P^\*(y|x\_i) - \sum_y P^\*(y|x\_i) \log P(y|x\_i;\theta) \\\\ &= H(P^\*) - H(P^\*, P) \end{align*} $$



Here,
 - $H(P^\*) = \sum_y P^\*(y|x\_i) \log P^\*(y|x\_i)$ is the **entropy** of the true/label distribution, and
 - $H(P^\*, P)=\sum_y P^\*(y|x\_i) \log P(y|x\_i;\theta)$ is the **cross-entropy** between $P^\*$ and $P$.

Since we aim to find $\theta$ that minimizes $D_{KL}(P^\*(y|x\_i) \parallel P(y|x\_i;\theta))$, and $H(P^\*)$ doesn't depend on $\theta$, we can discard $H(P^\*)$ when optimizing with respect to $\theta$ parameters. hence, the optimization becomes
$$ \arg\min_\theta D_{\text{KL}} (P^\* \parallel P) \equiv \arg\min_\theta H(P^\*, P)$$
To ensure that the model output $P(y|x_i;\theta)$ is a valid probability distribution, we need to enforce the following:
1. Non-negative Outputs: $P(y|x_i;\theta) \geq 0 \space \forall \space y$
2. Normalization: $\sum_y P(y|x_i;\theta) = 1$
To satisfy these constraints, we use the **softmax function** to convert the model's raw outputs (logits) into probabilities  $$ P(y|x_i;\theta) = \frac{\exp(s_y)}{\sum_{k=1}^N \exp(s_k)}$$where $s_y$​ is the logit (unnormalized score) for class $y$.

The **Cross-Entropy Loss** will thus try to minimize the KL divergence between the label distribution $P^∗$ and the predicted distribution $P$.

## Cross Entropy Loss vs. MSE Loss

![alt text](/assets/posts/cross_entropy/mse_vs_ce.svg#dark#small "Mean Squared Error vs Cross Entropy Function Loss & Gradients.")

In the following diagram, we can see the graphs of both loss functions. As the neural network's predictive ability decreases, i.e., the model outputs incorrect predictions the loss from cross entropy is greater than MSE. In addition, the gradient of the cross entropy loss is much steeper than the MSE loss. Further the predicted value is from the ground truth, higher the loss value.

- Cross-Entropy Loss penalizes the model more when it is confidently wrong.  If the model assigns a high probability to an incorrect class, the loss increases significantly. MSE treats all errors uniformly and does not adequately penalize confident but incorrect predictions.  Cross-Entropy Loss provides well-behaved gradients (*right figure*)that are conducive to efficient learning, especially when combined with the softmax activation function.

- Gradient Magnitude: The gradients of cross-entropy loss remain significant even when the predicted probabilities are close to 0 or 1, preventing the vanishing gradient problem.
MSE can lead to vanishing gradients in classification tasks, particularly when activation functions like sigmoid or softmax are used. This can slow down learning or cause the model to converge poorly.

The Cross Entropy Loss and the MSE Loss functions are described as follows: $$ \begin{gather*} \mathcal{L}\_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^N (y\_i - \hat{y}\_i^2) \\\\ \mathcal{L}\_{\text{CE}} = -\sum_{i=1}^N y_i \log \hat{y}_i \end{gather*}$$


