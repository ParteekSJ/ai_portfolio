---
title: "ResNet Explained"
date: "2023-08-29"
summary: "ResNet Model Explained."
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

<!-- **marginally better** - something being only slightly or minimally better than something else, but not significantly or substantially better -->


**MAIN CONTRIBUTION**: Reformulate the layers to learn a residual function with respect to the layer inputs, instead of learning unreferenced functions. In deeper networks, training and testing errors tend to increase due to optimization difficulties—a problem that ResNet addresses

> Whenever there is a drop in the loss function / error rate, that corresponds to learning rate decay. This is with reference to Figure 1.

## Introduction
- The quality of features in a CNN can be enriched by increasing the number of stacked layers. Theoretically, the deeper the network, the greater the potential benefit.
- Most networks before ResNet that performed well on the ImageNet dataset had depths ranging between 16 and 30 layers.
- Deeper networks also have a larger receptive field, meaning they can capture more extensive spatial hierarchies in images.
- However, simply increasing the depth of the network doesn't necessarily lead to better performance.

**ISSUES WITH INCREASED DEPTH**
- **Vanishing/Exploding Gradients**: Deeper networks suffer from gradients that either vanish (become too small) or explode (become too large), hampering convergence from the beginning. This issue can be mitigated by:
	- Normalized Initialization (initialize parameters $\sim\mathcal{N}(0,0.02)$) 
	- Intermediate Normalization Layers (BN)
- However, even when these networks are able to converge, another problem emerges.

**DEGRADATION PROBLEM**
- As network depth increases, accuracy saturates and then degrades rapidly. Surprisingly, this degradation isn't caused by overfitting.
- The authors experimentally concluded that adding more layers to a network leads to higher training error.
> **Degradation Error**: An optimization issue indicating that not all systems are easy to optimize.

{{< figure src="/assets/papers/resnet/resnet-depth-implications.png">}}

The degradation problem suggests that not all architectures are easy to optimize.


> Proposed Solution: **Deep Residual Learning Framework**. 

{{< figure src="/assets/papers/resnet/residual_block.png">}}


- Instead of hoping each few **stacked layers** directly fit a desired underlying mapping $\mathcal{H}(\mathbf{x})$, we explicitly let this layers fit a **residual mapping** $\mathcal{F}(\mathbf{x}):= \mathcal{H}(\mathbf{x}) - \mathbf{x}$.
- The original mapping is recast as $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$. 
- The authors hypothesize that it is easier to optimize the residual mapping $\mathcal{F}(\mathbf{x}) + \mathbf{x}$ than to optimize the original, unreferenced mapping $\mathcal{H}(\mathbf{x})$.
- Ultimately, both unreferenced function $\mathcal{H}(\mathbf{x})$ and recasted mapping $\mathcal{H}(\mathbf{x}) = \mathcal{F}(\mathbf{x}) + \mathbf{x}$ are different ways of expressing the same underlying unknown function. The hypothesis is that it is easier to optimize the latter than the former. Easier to obtain the final result.
- In the extreme case where the identity mapping is optimal, it's easier for the network to push the residual function towards zero, $\mathcal{F}(\mathbf{x}) \to 0$, than to fit an identity mapping using a stack of non-linear layers.

**HOW TO IMPLEMENT THIS IN CODE?**

- Practically, the formulation $\mathcal{F}(\mathbf{x}) + \mathbf{x}$ can be realized using feed-forward neural networks with shortcut connections.
- These shortcut connections perform identity mapping, adding the input $\mathbf{x}$ to the output of the stacked layers $\mathcal{F}(\mathbf{x})$. This is depicted in the figure above.
- No extra parameters are added, nor is there additional computational complexity.


### Results
- Extremely deep residual networks are easy to optimize, whereas their plain (non-residual) counterparts exhibit higher errors
- Deep residual networks benefit from accuracy gains due to increased depth without suffering from optimization difficulties

### Deep Residual Learning
> Residual Mapping: Difference b/w the desired output $\mathcal{H}(\mathbf{x})$ and the input $\mathbf{x}$, i.e., $\mathcal{H}(\mathbf{x}) - \mathbf{x}$ rather than learning the entire mapping $\mathcal{H}(\mathbf{x})$ from scratch.

### Intuition
Consider a scenario where you want to predict the value of a complex function $\mathcal{H}(\mathbf{x})$ given an input $\mathbf{x}$. Instead of trying to directly learn the mapping $\mathcal{H}(\mathbf{x})$ using a stack of layers, which can be challenging for a neural network, you can re-parameterize the problem as learning the residual function $$\begin{gather*} \mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x} \\ \mathcal{H}(\mathbf{x})= \mathcal{F}(\mathbf{x}) + \mathbf{x} \end{gather*}$$The intuition is that if the input $\mathbf{x}$ already contains some information about the desired output $\mathcal{H}(\mathbf{x})$, then the residual function $\mathcal{F}(\mathbf{x})$ should be easier to learn than the original mapping $\mathcal{H}(\mathbf{x})$. This is because the residual function only needs to learn the "residual" or the "correction" that needs to be added to the input $\mathbf{x}$ to obtain the desired output $\mathcal{H}(\mathbf{x})$.

### Residual Learning
- $\mathcal{H}(\mathbf{x})$ denotes the underlying mapping to be estimated/fit by a few stacked layers (***not necessarily the entire network***). Here, $\mathbf{x}$ denotes the input to these layers.
- If multiple non-linear layers can approximate complicated functions such as $\mathcal{H}(\mathbf{x})$, then they can asymptotically approximate the residual functions $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$.
- Instead of hoping each few **stacked layers** directly fit a desired underlying mapping $\mathcal{H}(\mathbf{x})$, we explicitly let this layers fit a residual function $\mathcal{F}(\mathbf{x}) := \mathcal{H}(\mathbf{x}) - \mathbf{x}$. 
- Original function to estimate using the stack of non-linear layers just becomes $\mathcal{H}(\mathbf{x})= \mathcal{F}(\mathbf{x}) + \mathbf{x}$. 

> The model now aims to find $\mathcal{F}(\mathbf{x})$, the residual that needs to be added to the input $\mathbf{x}$ to obtain the underlying mapping $\mathcal{H}(\mathbf{x})$.

- The degradation problem suggests that solvers (*optimizers*) might struggle to approximate the identity mapping using multiple non-linear layers. However, with the residual learning formulation, if the identity mapping is optimal, the network can simply push the residual function towards zero, $\mathcal{F}(\mathbf{x}) \to 0$, resulting in $\mathcal{H}(\mathbf{x}) \approx \mathbf{x}$.

	<!-- NEED TO IMPL¸EMENT THIS IN CODE AND CHECK. that the learned residual functions in general have small responses, suggesting that identity mappings provide reasonable preconditioning. -->
### Identity Mapping By Shortcuts
- The residual learning paradigm is applied to every few stacked layers.
- A **building block** is defined by $$ \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x} $$
- Here, $\mathbf{x}$ and $\mathbf{y}$ denote the input and output vectors of the layers considered.
- $\mathcal{F}(\mathbf{x},W_i)$ is the residual mapping to be learned.

Operation of $\mathcal{F}$ and $\mathbf{x}$ is performed by a shortcut connection and element-wise addition. These "shortcut connections" introduce neither extra parameters nor computational complexity.

### Handling Dimensionality Differences
 Dimensions of $\mathcal{F}$ and $\mathbf{x}$ must be equal to make the element-wise addition possible. 
 If they're not, we can perform a linear projection $W_s$ by the shortcut connections to match the dimensions, i.e., $$ \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s\mathbf{x} $$
 > $W_s$ is only used for matching dimensions.
 > $\mathcal{F}(\mathbf{x}, \{W_i\})$ can represent multiple convolutional layers.
 
 In the case of convolutional layers, dimensionality difference is adjusted as follows:
```python
nn.Conv2D(prev_channel_dim, new_channel_dim, kernel_size=1, stride=2)
```
- `kernel_size=1` is used to increase/decrease the channel dimensions.
- `stride=2` is used to reduce spatial dimensions. For example: $56\times56 \to 28\times28$.

## Network Architectures
### Plain Network
- Plain networks are inspired by VGG's architecture.
- Convolutional Layers mostly have 3x3 filters and follow 2 design rules:
	- for the same output feature map size, the layers have the same number of filters.
	- if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer. 
	- For instance, suppose the feature map size is $56\times56$ and number of filters is $64$. Once the feature map size is halved, the number of filters is doubled, i.e., $128$ $28\times28$ filters
### Residual Network
- Shortcut connections turn the network into its counterpart residual version.
- Identity shortcuts can be directly used when the input and output are of the same dimensions.
- When the dimensions [*channel dimensionality*] increase,
	1. Shortcut still performs identity mapping, with extra zeros padded for increasing dimensions. No extra parameters are introduced. $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \text{PAD}(\mathbf{x})$.
	2. Projection shortcut can be used to match dimensions (done via $1\times 1$ convolutions). This is done via the projection layer $W_s$, i.e., $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s\mathbf{x}$. 

Both these options are performed with a stride $s=2$.
1x1 CONV to increase channel dimensions, and/or $s=2$ to reduce spatial dimensions.
### Implementation
- We adopt batch normalization right after each convolution and before activation, i.e., 
  `CONV -> BN -> RELU`

Remaining implementation details are outlined in the paper.
## Experiments

{{< figure src="/assets/papers/resnet/resnet_arch_outlined.png">}}



 <!-- - [ ] We also verify that the backward propagated gradients exhibit healthy norms with BN. So neither forward nor backward signals vanish. NEED TO UNDERSTAND AND IMPLEMENT THIS. -->
## Identity vs. Projection Shortcuts
>**parameter-free identity shortcuts** = zero padding. 

Prior to this section, experiments with using padding shortcuts is done. 
Authors now investigate projection shortcuts.

Three Options
1. Zero padding shortcuts are used for increasing dimensions. All shortcuts are parameter free, i.e., $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \text{PAD}(\mathbf{x})$
2. Projection Shortcuts are used for increasing dimensions, and other shortcuts are identity, i.e., $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s(\mathbf{x})$. These could be $1\times1$ Convolutional Layers.
3. All shortcuts are projections.
#### Observations
- Option B is slightly better than A. 
   REASON (*hypothesized*): Zero Padded dimensions in A have no residual learning.
- Option C is marginally better than B. 
   REASON: Extra parameters introduced by the projection shortcuts.
Authors will not be using OPTION C since the performance difference is small.
## Deeper Bottleneck Architectures

{{< figure src="/assets/papers/resnet/bottleneck_block.png">}}

Authors alter the Basic Block architecture (Fig. 5 Left) in order to make the networks even deeper. Due to concerns on the training time and computational complexity, a bottleneck block is created ([*modified Basic Block for deeper architectures*]). 

### Why Bottleneck?
Deeper non-bottleneck ResNets also gain accuracy from increased depth, but aren't as economical as bottleneck ResNets. Usage of bottleneck design is mainly due to practical considerations.
> Bottleneck Blocks were initially used to make blocks less computationally expensive to increase the number of layers, i.e., ResNet50, ResNet101, ResNet152.


Parameter Free Identity Shortcuts [Zero padding Shortcuts] are important for the bottleneck architectures, i.e., no linear projection layer such as $W_s$ is used since the residual $\mathcal{F}$ goes through a $1\times1$ CONV that makes that the dimensionality of $\mathcal{F}$ and $\mathbf{x}$ are the same.

For each residual function $\mathcal{F}$, we use a stack of 3 layers instead of 2. The three layers are $1\times 1$, $3\times 3$ and $1\times 1$ convolutions, where the $1\times 1$ convolutions are responsible for reducing and then increasing (restoring) dimensions. For instance, 
1. For a $256$ dimensional input, the $1\times1$ CONV layer downsamples it to $64$.
2. For a $64$ dimensional input, the $3\times3$ CONV layer maintains the dimensionality $64$.
3. For a $64$ dimensional input, the $1\times1$ CONV layer upsamples it back to $256$.

### ResNet50
- Each 2 layer basic block in the ResNet layer is replaced with a 3-layer bottleneck block. 
- Option B ($1\times1$ CONV) is used for increasing channel dimensions.
The 50/101/152-layer ResNets are more accurate than the 34-layer ones by considerable margins.

