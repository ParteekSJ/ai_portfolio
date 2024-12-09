---
title: "Batch Normalization"
date: "2024-10-15"
summary: "Explanation of what Batch Normalization is and how it is used."
description: "This blog discusses about different types of normalization used in deep learning."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

<!-- https://pytorch.org/tutorials/intermediate/custom_function_conv_bn_tutorial.html -->

## Introduction
As models get deeper, it gets prone to facing the issue of exploding/vanishing gradients, that is, the gradients become too small or large hampering the network convergence. If the above problem persists, the distribution of each layer's input keeps varying significantly as the parameters of the previous layers keep getting updated. This slows down the training, thus requiring careful parameter/weight initialization and makes it notoriously hard to train models with saturating non-linearities. A common solution for this problem is to simply standardize the statistics (*means and variances*) of the hidden layers, i.e., all weights of a specific layer follow the same distribution. With such a setting, the optimizer is less likely to get stuck in a saturated regime. Normalizing the layer inputs can help stabilize the network by desensitizing it to different weight initialization schemes and learning rate changes, and ultimately accelerate the network training process by allowing it to converge faster.


## Batch Normalization

This is the most common type of normalization. It ensures that the distributions of activations within a layer has zero mean and unit variance, when averaged across the samples in a mini-batch $\mathcal{B}$. This helps reduce **internal covariate shift**, i.e., abrupt changes in the distribution of the model activations.

The set of equations used in batch normalization are as follows: $$
\begin{gather} 
\boldsymbol{\tilde{z}}_n = \gamma \odot \hat{\boldsymbol{z}}_n + \beta \\\\ 
\hat{\boldsymbol{z}}\_n = \frac{\boldsymbol{z}\_n-\boldsymbol{\mu}\_\mathcal{B}}{\sqrt{\boldsymbol{\sigma}^2\_\mathcal{B} + \epsilon}} = \frac{\boldsymbol{z}\_n-\mathbb{E}[\mathcal{B}]}{\sqrt{\mathbb{V}[\mathcal{B}] + \epsilon}} \\\\ 
\boldsymbol{\mu}\_\mathcal{B} = \frac{1}{|\mathcal{B}|}\sum\_{\boldsymbol{z} \in \mathcal{B}} \boldsymbol{z} \\\\ 
\boldsymbol{\sigma}^2\_\mathcal{B} = \frac{1}{|\mathcal{B}|}\sum\_{\boldsymbol{z} \in \mathcal{B}} (\boldsymbol{z} - \boldsymbol{\mu}\_\mathcal{B})^2
\end{gather}$$ 


where
- $\mathcal{B}$ is the mini-batch consisting of example $n$,
- $\boldsymbol{\mu}_\mathcal{B}$ is the mean of the activations for this batch,
- $\boldsymbol{\sigma}^2_\mathcal{B}$ is the variance of the activations for this batch, 
- $\hat{z}_n$ is the standardized activation vector, i.e., post-activation, $\hat{z}_n = \sigma(z_n)$,
- $\boldsymbol{\tilde{z}}_n$ is the shifted + scaled version; output of the batch norm layer,
- $\beta$ (*additive factor / **shift***) and $\gamma$ (*multiplicative factor / **scale***) are learnable parameters for this layer allowing it to restore the representation power of the network.
- $\epsilon=1e^{-5}$ is a small constant to avoid division by 0.

The transformation defined above is differentiable, and hence we can pass gradients to the parameters $\beta$ (shifting parameter) and $\gamma$ (scaling parameter) and also to the input layer.

Mean and Variance for the input layer can be computed once, since the data is static. However, the empirical means and variances of the internal layers keep changing, as the parameters adapt. (This is sometimes called “**internal covariate shift**”.) This is why we need to recompute $\boldsymbol{\mu}$ and $\sigma^2$ on each mini-batch.

Batch Normalization has a beneficial effect on the gradient flow through the network, by reducing the dependence of the gradients on the scale of the parameters or their initial values. It also induces a regularization effect.

During inference, we may have a single input, and so we can't compute the batch statistics. The standard solution to this is as follows:
1. After training, compute $\boldsymbol{\mu}_l$, and $\boldsymbol{\sigma}^2_l$ for layer $l$ across all the examples in the training set ("using the full batch").
2. Freeze the parameters and add them to the list of other parameters for the layer, namely $\beta_l$ and $\gamma_l$, i.e., we save the mean and standard deviation along the shifting and scaling parameters.
3. At test time, we use these frozen values $\boldsymbol{\mu}_l$, and $\boldsymbol{\sigma}^2_l$, rather than computing statistics from the test batch. Thus, when using a model with BN, we need to specify `model.train()` or `model.eval()` 


## BN From Scratch

```python
import torch
import matplotlib.pyplot as plt


# Step 1: Simulate input data
x = (torch.randn((20, 2)) + 5) * 2  # 20 2D points distributed according to N(5, 2)

# Learnable Parameters (initializing with 1 and 0)
gamma = torch.Tensor([1.0, 1.0])  # scaling factor
beta = torch.Tensor([0.0, 0.0])  # shifting factor
epsilon = 1e-5  # Small constant to avoid division by zero

# Step 2: Compute Mean and Variance for each feature
mean = x.mean(dim=0)
variance = x.var(dim=0, unbiased=False)

# Step 3: Normalize each feature
x_normalized = (x - mean) / torch.sqrt(variance + epsilon)

# Step 4: Apply learnable scale (gamma) and shift (beta)
y = gamma * x_normalized + beta

# Extract original and normalized features
x_feature1, x_feature2 = x[:, 0].numpy(), x[:, 1].numpy()
y_feature1, y_feature2 = y[:, 0].numpy(), y[:, 1].numpy()

# Plot the original and normalized data
plt.figure(figsize=(8, 6))
plt.scatter(x_feature1, x_feature2, color="blue", label="Original Data", s=100)
plt.scatter(y_feature1, y_feature2, color="red", label="Normalized Data", s=100)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
plt.title("Batch Normalization Visualization", fontsize=16)
plt.xlabel("Feature 1", fontsize=14)
plt.ylabel("Feature 2", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Using PyTorch's built-in BatchNormalization
bn_layer = torch.nn.BatchNorm1d(num_features=2, eps=epsilon, affine=True)

# Only allowed if `affine` hyperparameter is set to True.
bn_layer.weight.data = gamma  # Set gamma
bn_layer.bias.data = beta  # Set beta

# Computing BatchNorm1d's output 
output = bn_layer(x)

# Check if the implementation is correct
torch.allclose(output, y, rtol=1e-4) # True

```

Visually, the results are as follows:
![alt text](/assets/posts/batch_normalization/bn-norm-plot.svg#dark#small "Batch Norm being applied to 20 normally distributed points.")

## `nn.BatchNorm2d` in PyTorch

The PyTorch documentation can be found here: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html

The behaviour of `nn.BatchNorm2D` changes with training/evaluation mode. In the training mode, the `nn.BatchNorm2D` layer behaves as follows:
  - **Running Statistics**: The BatchNorm layer updates its running mean and variance using the statistics of the current mini-batch. These running statistics are used for normalization during inference (when the model is in evaluation mode).
  - **Normalization**: During training, the normalization is based on the mean and variance of the current mini-batch.
  - **Learnable Parameters**: The affine transformation parameters (gamma and beta) are updated during backpropagation, provided they are trainable.

In the test mode, the `nn.BatchNorm2D` layer behaves as follows:
  - **Runing Statistics**: The BatchNorm layer *stops updating* the running mean and variance. Instead, it uses the already-computed running mean and variance (accumulated during training) for normalization.
  - **Normalization**: The normalization is based on the stored running mean and variance, ensuring consistency during inference.
  - **Learnable Parameters**: The affine transformation parameters (gamma and beta) remain unchanged, but they are still used to scale and shift the normalized output.

### What is `track_running_stats`?
`track_running_stats` is an argument that we pass to `nn.BatchNorm2D` layer. 

Also by default, during training this layer keeps **running estimates** of its computed mean and variance, which are then used for normalization during evaluation. The running estimates are kept with a default momentum of 0.1. If `track_running_stats` is set to **False**, this layer then does not keep running estimates, and batch statistics are instead used during evaluation time as well.  Mathematically, the update rule for running statistics here is: $$ \hat{x}_{\text{new}} = (1 - \text{momentum}) \times \hat{x} +  \text{momentum} \times {x}_t $$ where $\hat{x}$ is the estimated statistic and $x_t$ is the newly observed value.

> With a momentum of 0.1, each new batch contributes 10% to the running statistics, while the existing running statistics retain 90%. This smoothing ensures that the running estimates are stable and not overly influenced by any single batch.

### Example to Illustrate Running Statistics

The BatchNorm layer computes the mean and variance across the current mini-batch and uses these statistics to normalize the data. During inference (evaluation), we may not have a mini-batch (or it might be a single sample), thus making batch statistics unreliable. To address this, BatchNorm maintains running estimates of the mean and variance, which are accumulated during training. These running statistics are then used for normalization during evaluation to ensure consistent and stable behavior. Some key terms to remember are as follows: 
1. **Running Mean** (`running_mean`): An exponential moving average of the batch means computed during training.
2. **Running Variance** (`running_var`): An exponential moving average of the batch variances computed during training.
3. **Momentum**: Determines the weight given to the new batch statistics when updating the running statistics. A higher momentum gives more weight to recent batches.

Consider the following code: 
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(0)

# Define a simple model with BatchNorm
class SimpleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return x

# Initialize model, loss function, and optimizer
model = SimpleModel(input_dim=3, output_dim=2)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Print initial running stats
print("Initial running_mean:", model.bn.running_mean)
print("Initial running_var:", model.bn.running_var)
print("-" * 50)

# Simulate training
model.train()  # Set to training mode

for epoch in range(3):
    # Generate dummy input and target
    input = torch.randn(4, 3)  # Batch size of 4
    target = torch.randn(4, 2)

    # Forward pass
    output = model(input)
    loss = criterion(output, target)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print running stats after each epoch
    print(f"Epoch {epoch+1}:")
    print("  running_mean:", model.bn.running_mean)
    print("  running_var:", model.bn.running_var)
    print("-" * 50)

# Switch to evaluation mode
model.eval()

# Print running stats before inference
print("After Training:")
print("  running_mean:", model.bn.running_mean)
print("  running_var:", model.bn.running_var)
print("-" * 50)

# Perform inference
with torch.no_grad():
    test_input = torch.randn(2, 3)
    test_output = model(test_input)
    print("Inference output:", test_output)
    print("-" * 50)

# Print running stats after inference to show they haven't changed
print("After Inference (Evaluation Mode):")
print("  running_mean:", model.bn.running_mean)
print("  running_var:", model.bn.running_var)
print("-" * 50)
```

Running this code yields the following (sample) output: 

```yaml
Initial running_mean: tensor([0., 0.])
Initial running_var: tensor([1., 1.])
--------------------------------------------------
Epoch 1:
  running_mean: tensor([-0.0751,  0.0433])
  running_var: tensor([0.9641, 0.9523])
--------------------------------------------------
Epoch 2:
  running_mean: tensor([-0.1604,  0.0845])
  running_var: tensor([0.9233, 0.9224])
--------------------------------------------------
Epoch 3:
  running_mean: tensor([-0.2281,  0.1173])
  running_var: tensor([0.8893, 0.8954])
--------------------------------------------------
After Training:
  running_mean: tensor([-0.2281,  0.1173])
  running_var: tensor([0.8893, 0.8954])
--------------------------------------------------
Inference output: tensor([[ 0.0331, -0.0397],
        [ 0.0034,  0.0036]], grad_fn=<NativeBatchNormBackward0>)
--------------------------------------------------
After Inference (Evaluation Mode):
  running_mean: tensor([-0.2281,  0.1173])
  running_var: tensor([0.8893, 0.8954])
--------------------------------------------------
```

- **Initial Running Stats**: `running_mean` starts at $[0,0]$ and running_var at $[1,1]$.
- **After Each Epoch**:
  - The `running_mean` and `running_var` gradually adjust based on the batch statistics.
  - The updates are influenced by the **momentum** parameter (default is 0.1 in PyTorch), meaning the running stats are a weighted average with more emphasis on older statistics.
- **After Training**: The `running_mean` and `running_var` reflect the accumulated statistics from all training epochs.
- **Inference**: The output is normalized using the `running_mean` and `running_var`. Crucially, after inference, the `running_mean` and `running_var` remain unchanged, demonstrating that evaluation mode does not update these statistics.


## Using BatchNorm in LeNet
This is an illustration of BatchNorm being used in LeNet's model architecture.

```python
class LeNetModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dense1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.bn3 = nn.BatchNorm1d(num_features=120)
        self.dense2 = nn.Linear(in_features=120, out_features=84)
        self.bn4 = nn.BatchNorm1d(num_features=84)
        self.dense3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.sigmoid(self.bn1(self.conv1(x)))  # [B, 1, 28, 28] -> [B, 6, 28, 28]
        x = self.avg_pool1(x)  # [B, 6, 28, 28] -> [B, 6, 14, 14]
        x = F.sigmoid(self.bn2(self.conv2(x)))  # [B, 6, 14, 14] -> [B, 16, 10, 10]
        x = self.avg_pool2(x)  # [B, 16, 10, 10] -> [B, 16, 5, 5]
        x = nn.Flatten(start_dim=1, end_dim=-1)(x)  # [B, 16, 5, 5] -> [B, 400]
        x = F.sigmoid(self.bn3(self.dense1(x)))  # [B, 400] -> [B, 120]
        x = F.sigmoid(self.bn4(self.dense2(x)))  # [B, 120] -> [B, 84]
        x = self.dense3(x)  # [B, 120] -> [B, 10]
        return x
```