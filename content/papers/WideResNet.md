---
title: "WideResNet Explained"
date: "2024-09-05"
summary: "WideResNet Model Explained."
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---


- In ResNet, each fraction of improved accuracy is obtained by nearly doubling the layers, i.e., increasing depth. Training of such deep NN leads to a problem of diminishing <mark>feature reuse</mark>.
- The implications of increasing depth beyond a certain point are as follows:
	- Added layers are not contributing significantly to the network's performance, i.e., *diminishing feature reuse*.
	- Training process becomes slow and computationally expensive.
- To mitigate the above problem, the authors introduce a novel architecture where they **widen** the convolutional layers in the residual blocks. In simpler terms, they
	- decrease depth (reduce total number of layers) and 
	- increase width (increase number of channels in each convolutional layer // increase the channel dimension)


Important Distinction
- **Depth** = Number of Layers in the Network.
- **Width** = Number of Channels/Filters in each convolutional layer.

# Introduction
- Training deep networks has the vanishing/exploding gradients and degradation problem. Some mitigation strategies for these are
	- better optimizers.
	- well-designed initialization strategies.
	- skip connections.
	- knowledge transfer. 
	- layer-wise training.
- Residual links ($\mathcal{F}(\mathbf{x}) + \mathbf{x}$) speed up convergence of deep networks.
## Width vs. Depth in Neural Networks
Authors of ResNet tried to make the network as thin as possible (small channel/feature map sizes) in favor of increasing their depth and having less parameters. Also, introduced a bottleneck block which can be referred in [(2015) ResNet - Deep Residual Learning for Image Recognition](/blogs/001-resnet/) paper explanation.

Authors of ResNet paper wanted to make their network deeper. For that, they introduced the Bottleneck blocks since it was getting computationally expensive. 

> ResNet with identity mapping that allows to train very deep networks is at the same time a weakness of the residual networks.

## Problem: Diminishing Feature Reuse
As gradients flow through the network there is nothing to force it to go through the residual block weights and hence it can avoid learning anything during training. Therefore, it is possible that 
- there are either only a few blocks that learn useful representations, or 
- many blocks share little information with small contribution to the final goal.

To address the problem of diminishing feature reuse, one proposed idea was to randomly disable residual blocks during training where each residual block has an identity scalar weight associated with it on which dropout is applied.


# Wide Residual Networks
Residual block with identity mapping can be represented as follows: 
$$\mathbf{x}_{l+1} =  \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$$ 

where $\mathbf{x}\_{l}$ and $\mathbf{x}\_{l+1}$ are the input and output of the residual block, $\mathcal{F}$ is the residual function and $\mathcal{W}$ is the weights of the residual block.

ResNet consists of sequentially stacked residual blocks. 
Residual networks consists of two types of blocks:
1. **Basic**: Two consecutive $3\times3$  convolutions with BN and ReLU preceding the convolution, i.e., these operations occur before the convolutional operation. $$ \begin{gather} \text{conv } 3\times 3 - \text{conv } 3\times 3 \\ \text{CONV} \to \text{BN} \to \text{ReLU} \end{gather} $$
2. **Bottleneck**: One $3\times 3$ convolution surrounded by dimensionality reducing and expanding $1\times1$ convolutional layers. $$ \begin{gather} \text{conv } 1\times 1 - \text{conv } 3\times 3 - \text{conv } 1\times 1 \\ \text{CONV} \to \text{BN} \to \text{ReLU} \end{gather}$$
> Compared to original architecture from the ResNet paper, the order of Batch Normalization, ReLU activation and CONV in residual block was changed from `CONV-BN-ReLU` TO `BN-ReLU-CONV`. The latter is shown to train faster and achieve better results.

Bottleneck blocks were initially used to make the networks deeper (*increasing the number of layers*) by making the residual blocks less computationally expensive. However, the authors focus on widening the blocks and hence `BottleNeck` block which makes the residual blocks thinner is not considered at at. We only consider `Basic` block. 

## Three Ways to Increase Representational Power of Residual Blocks
1. **to add more convolutional layers per block**, i.e., increase depth / adding more layers to the residual block and by extension to the network.
2. **to widen convolutional layers by adding from feature planes**, i.e., increase the number of filters in each convolutional layer. default=**64**. 
3. to increase filter sizes in convolutional layers, i.e., alter the kernel sizes in the convolutional layers. Since $3\times3$ filters are shown to be effective in several works, this option is **NOT CONSIDERED.**

Two Factors are introduced
1. $l$ - Deepening Factor (*number of convolutions per block*)
2. $k$ - Widening Factor (*number of feature planes in the block / channel or filter size*)

> Baseline `Basic` block corresponds to $l=2,k=1$, i.e., the block consists of $l=2$ convolutional layers and default channel dimensions/feature planes, i.e., $k=1$. There is no increase in the amount of filters/channels in the `CONV` layers. It stays the same throughout.

## General Structure of a Residual Block:
{{< figure src="/assets/blogs/004-wideresnet-explained/Structure_of_WRN.png">}}
- $k$ denotes the widening factor. Original ResNet architecture is equivalent to $k=1$.
- $l$ denotes the deepening factor. In this case, $l=2$.
- $N$ denotes the number of convolutional blocks $B(M)$ per group, i.e., `CONV2` consists of $N \times B(3,3)$ blocks.
- Downsampling of the spatial dimensions is performed by the first layers in groups `CONV3` and `CONV4`.

The general structure is illustrated in the above picture. 
1. It consists of an initial convolutional layer [`CONV1`] that is followed by $3$ groups [`CONV2`, `CONV3`, `CONV4`]. Each group [`CONV2-4`] consists of $N$ $B(3,3)$ blocks.
2. This is followed by average pooling and final classification layer.
Size of `CONV1` is fixed for all layers, while introduced widening factor $k$ scales the width of residual blocks in three groups [`CONV2-4`].  

## Types of Convolutions Per Residual Block
- $B(M)$ denotes residual block structure, where $M$ is a list with **kernel sizes** of the convolutional layers in the block. Additionally, number of feature planes stays constant for all the blocks (*since `BottleNeck` blocks aren't considered*)
- $B(3,1)$ denotes a residual block with a $3\times 3$ and $1\times 1$ convolutional layers.
- Different combinations such as $B(3,1)$ or $B(1,3)$ or $B(3,1,1)$ can increase or decrease the representational power of the residual blocks. 
- Here are the different residual block structures that were considered
	- $B(3,3)$ - original `BASIC` block.
	- $B(3,1,3)$ - with one extra  $1\times1$ layer.
	- $B(1,3,1)$ - with the same dimensionality for all convolutions, `straightened BottleNeck`, i.e., no dimensionality change.
	- $B(1,3)$ - alternating $1\times 1$ and $3\times 3$ convolutions everywhere.
	- $B(3,1)$ - similar idea to previous block.
	- $B(3,1,1)$ - Network-in-Network style block (*from some paper*).
These were the different types of convolutions that were considered.

## Number of Convolutional Layers Per Residual Block
- Authors also experiment with block deepening factor $l$ to see how it affects performance. The comparision is done amongst different networks such that number of parameters roughly remain the same.
- They experiment networks with different $l$ and $d$ (where $d$ denotes the total number of blocks), while ensuring network complexity is roughly constant, i.e., for an increase in $l$ (*number of CONV layers per residual block*), there should be a decrease in $d$ (*total number of blocks*).

## Width of Residual Blocks
Authors also experiment with the widening factor $k$ of a block.
- While number of parameters increase linearly with $l$ (*deepening factor*) and $d$ (*number of ResNet blocks*), number of parameters and computational complexity are quadratic in $k$. Even though the parameters increase quadratically with $k$, this is fine for the GPU since we're distributing the same tensor from the previous activation across the different feature maps. 
> Widening Factor is easier to parallelize on the GPU. More parameters, more memory, and better results.
- Original ResNet is a WRN with $k=1$ and is referred to as a `THIN` network.
- Networks with $k>1$ is referred to as `WIDE` network.
> WRN-$n$-$k$ denotes a residual network that has a total number of convolutional layers $n$ and widening factor $k$.

For example, WRN-$n$-$k$ is a network with $40$ layers and $k=2$ wider than the original would be denoted as WRN-$40$-$2$. We can also append the block type, i.e., WRN-$40$-$2$-$B(3,3)$.

## Dropout in Residual Networks
- With increase in $k$ [*widening factor*], the authors also looked into different methods of regularization. Default ResNets already have BN that provides a regularization effect by reducing internal covariate shift. However it requires heavy data augmentation (e.g., random cropping, flipping, rotating, etc.) to artificially increase the diversity of the training data and improve generalization. This however isn't always possible.
- To address the issue of overfitting, the dropout layer is placed between the two consecutive 3×3 convolutions and after the ReLU activation function. The purpose of this dropout layer is to perturb (or introduce noise) to the Batch Normalization layer in the next residual block. By randomly dropping out (or setting to zero) some of the activations, the dropout layer prevents the BN layers from overfitting to the specific patterns in the training data.
- Negative Results from inserting dropout in the identity component of the residual block.
  Positive Results using dropout between convolutional layers.

> It is a good idea to have dropout between the CONV layers, i.e., in the residual block; the non-linear part of the network rather than having it in the shortcut (*identity connection*). This forces the network to go through the shortcut connection (*if possible*).

# 3. Experimental Results
- $k=2$ is used through all experiments.
- For all the data preprocessing and other stuff, READ THE PAPER.
## Types of Convolutions in a Block
- Authors use WRN-$40$-$2$ for blocks $B(1,3,1), B(3,1), B(1,3)$ and $B(3,1,1)$. All these blocks have a single $3\times3$ convolutional layer.
- Additionally, they also use WRN-$28$-$2$-$B(3,3)$ and WRN-$22$-$2$-$B(3,1,3)$.
- All these networks are roughly similar in terms of the parameters.
### Results
- Block $B(3,3)$ turned out to be the best by a little margin.
- Block $B(3,1)$ and Block $B(3,1,3)$ are very close to $B(3,3)$ in accuracy having less parameters and less layers.
- $B(3,1,3)$ is faster than others by a small margin.
> With the above results, the authors restrict their attention to WRNs with $3\times3$ convolutions.

## Number of Convolutions Per Block
- Different deepening factor values $l\in[1,2,3,4]$ are tried and experimented with.
- $l$ denotes the number of CONV layers per block.
- WRN-$40$-$2$ with different deepening factor values is tested and the results are as follows:
	- $B(3,3)$, i.e., $l=2$ turned out to be the best, whereas $B(3,3,3)$ and $B(3,3,3,3)$, i.e., $l=3, l=4$, had the worst performance.
	- This is probably due to increased difficulty in optimization as a result of decreased number of residual connections in the last two cases.
> $B(3,3)$ is optimal in terms of number of convolutions per block and hence is considered for all the remaining experiments.

## Width of Residual Blocks
- As $k$ (widening factor) increases, we have to decrease total number of layers in order to maintain roughly the same computational complexity.
- To find the optimal ratio, the authors experimented with $k\in[1,12]$ and depth $\in [16,40]$.

### Results
- All networks with $40, 22$ and $16$ layers see consistent gains in accuracy with increase when width $k$ is increased by $1$ to $12$ times.
- When maintaining the same $k=8$ or $k=10$ and varying depth from $16$ to $28$ there is a consistent improvement, however when we further increase depth to $40$, the accuracy decreases. 
> In simpler terms, WRN-$40$-$8$ loses in accuracy to WRN-$28$-$2$ 

### THIN vs. WIDE Residual Networks
- WRN-$40$-$4$ compares favorably to thin ResNet-$1001$ (*basic block*) as it achieves better accuracy on the CIFAR-10 and CIFAR-100 dataset.
- WRN-$40$-$4$ & ResNet-$1001$ have comparable number of parameters, $8.9\times10^6$ and $10.2\times10^6$, suggesting that depth DOES NOT ADD REGULARIZATION EFFECTS.
- Furthermore, the authors also show that WRN-$40$-$4$ is 8-times faster to train, so evidently depth-to-width ratio in the original thin residual networks is far from optimal.


## Summary
1. Widening consistently improves performance across residual networks of different depth.
2. Increasing both depth $l$ and width $d$ helps until the number of parameters become too high and stronger regularization is needed.
3. There doesn’t seem to be a regularization effect from very high depth in residual networks. Wide Networks ($k>1$) can learn representations that are at least as good as, or potentially better than, thinner networks (fewer channels/filters per layer), even when the total number of parameters is the same.

