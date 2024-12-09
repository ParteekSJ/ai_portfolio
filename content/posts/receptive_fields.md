---
title: "Receptive Fields"
date: "2024-08-12"
summary: "Receptive Fields Explained"
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

## What is Receptive Field?

The “effective receptive field” of a neuron is the area of the original image that influences the activations (output). In other words, it refers to the specific region in the image/feature map that a particular neuron is influenced by. Receptive fields give us a better insight in how CNNs interpret and process spatial hierarchies in data.

 To understand this more intuitively, consider the following example

![alt text](/assets/posts/receptive_fields/rf_example.png#dark#small "Weight Tying.")


Each neuron in the CNN depicted in the picture processes a small patch of the input image, defined by the kernel (or filter) size. As presented, the receptive field of the neuron in the **first convolutional layer** is equivalent to the **kernel size of that layer**.

- Suppose A,B and C are convolutional layers of a CNN (*padding $p=1$ to maintain the dimensions, filter size $k=3\times3$ and stride $s=1$*). 
- The "receptive field" of a "neuron" in a layer would be the **cross-section of the previous layer from which neurons provide inputs**. With this logic, the RF of $B(2,2)$ is as follows:  $$ \text{RF}[B(2,2)] = A(1:3,1:3) \in \mathbb{R}^{3\times 3} $$ The receptive field of $B(2,2)$ is the $3\times 3$ cross-section in $A$.
- Receptive Field of $B(2,4)$ is the $A(3:5,3:5) \in \mathbb{R}^{3\times 3}$. 
- Lastly, the receptive field of $C(3,3) \in \mathbb{R}^{3\times 3}$ is simply $B(2:4, 2:4)$ which itself receives inputs from $A(1:5,1:5) \in \mathbb{R}^{5\times 5}$.

As more convolutional layers are stacked, the receptive field of the neurons in deeper layers grows. This is because each neuron in a deeper layer receives input from multiple neurons in the previous layer, which in turn are influenced by even more extensive areas of the input image. It provides us an insight in the pictorial context captured by the network. 

A larger receptive field allows the network to consider more of the surrounding context of a feature, which is crucial for tasks like object detection and segmentation. However, in practice, it is advised to use a stack of small convolutional filters, rather than using a single large convolutional filter. For instance, rather than using a single filter of size $k=5\times5$, stacking 2 convolutional layers (without pooling) with $3\times3$ filters results in a net $5\times5$ filter. Stacking $3$ such layers would give you an effective receptive size of $7\times7$, and so on.

> Better to stack smaller dimensional convolutional layers than to use a single large filter since the computational complexity/cost of stacking small layers is less that using a large filter. Ultimately, the receptive sizes of both paths are the same. The basic idea is to extract local features and then combine them to make more complex patterns. That translates to local transformations and therefore the idea of receptive fields.

For example, suppose that the input volume has size `32x32x3`, (e.g. an RGB CIFAR-10 image). If the receptive field (or the filter size) is `5x5`, then each neuron in the Conv Layer will have weights to a `5x5x3` region in the input volume, for a total of `5*5*3 = 75` weights (and +1 bias parameter). Notice that the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.

<!-- ## Convolution with a Receptive Field of Size 1×1 in CNNs

A convolution with a receptive field of $1\times 1$ in CNNs refers to a convolutional layer where the filter has the spatial dimensions of $1\times 1$.  -->


## Calculating the Receptive Field

The receptive field (RF) of a "neuron" in a layer indicates the region of the input that affects that "neuron". The formula depends on several factors:
1. Kernel Size ($k$) - Size of the Convolutional Filter.
2. Stride ($s$) - Step Size of the Convolution.
3. Padding ($p$) - Amount of padding added to the input (*prior convolution*)
4. Receptive Field of the Previous Layer ($\text{RF}_\text{{prev}}$): Size of the RF of previous layer.

The formula to compute the receptive field at layer $L$ is as follows: $$ \text{RF}\_L = \text{RF}\_\text{{prev}} + (k-1) \times \prod\_{i=1}^{L-1} s\_i $$
- $\text{RF}_\text{{prev}}$ is the receptive field of the previous layer (or $1$ for the input layer).
- $k$ is the kernel size of the current layer.
- $s_i$ is the stride of the $i$'th layer.



## Code Example
Consider the following `SimpleModel`

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
        )  # Output shape: [B, 1, 64, 64] -> [B, 8, 64, 64]
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1
        )  # Output shape: [B, 8, 64, 64] -> [B, 16, 64, 64]
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Output shape after pooling: [B, 16, 64, 64] -> [B, 16, 32, 32]

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x
```
This simple CNN consists of 2 convolutional layers and 1 pooling layer. 

We write a function which is used to calculate the receptive field size of the convolutional and pooling layers. It is as follows: 
```python
# Function to compute the receptive field size manually (approximation)
def compute_receptive_field(layers):
    rf = 1  # Initial receptive field size
    total_stride = 1  # Initial total stride
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            kernel_size = layer.kernel_size[0]
            stride = layer.stride[0]
            # Update receptive field before updating total stride
            rf = rf + (kernel_size - 1) * total_stride
            total_stride *= stride # product of the strides of all previous layers
        elif isinstance(layer, nn.MaxPool2d):
            kernel_size = layer.kernel_size
            stride = layer.stride
            # Update receptive field before updating total stride
            rf = rf + (kernel_size - 1) * total_stride
            total_stride *= stride # product of the strides of all previous layers
    return rf
```

> Receptive field expansion depends on the previous total stride, not the current layer's stride.


```python
# Define layers
layers = [
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
]
# Compute receptive field
rf = compute_receptive_field(layers)
print(f"Receptive field size: {rf}")
```
Consider a CNN composed of two convolutional layers.
- First Convolutional Layer: Kernel Size $7\times 7$, Stride $2$
- Second Convolutional Layer: Kernel Size $3\times 3$, Stride $2$
  
We start of by initializing the receptive field(`rf`), and `total_stride`. They're both set to a value of $1$.
- **Layer 1**
  - $\text{rf} = \text{rf} + (\text{kernel size} - 1)\times \text{total stride}$
  - $\text{rf} = 1 + (7 - 1) \times 1 = 7$
  - Updating $\text{total stride} = \text{total stride} \times \text{stride} = 1\times 2 = 2$

- **Layer 2**
  - $\text{rf} = \text{rf} + (\text{kernel size} - 1)\times \text{total stride}$
  - $\text{rf} = 7 + (3 - 1) \times 2 = 9 + 2 \times 1 = 11$
  - Updating $\text{total stride} = \text{total stride} \times \text{stride} = 2\times 2 = 4$
  
The final receptive field is $11$, i.e., each "neuron" in the output feature map of the two convolutional layers sees a patch of $5\times 5$ collectively in the previous two layers.

```python
# Define layers
layers = [
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2),
    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
]

# Compute receptive field
rf = compute_receptive_field(layers)
print(f"Receptive field size: {rf}")
```


## ResNet's Receptive Fields
The ResNet architecture is as follows:
![alt text](/assets/posts/receptive_fields/resnet-model.jpg#dark#small "ResNet Model Architecture")

The network begins with a single $7\times 7$ convolutional layer with a stride of 2, followed by batch normalization and a ReLU activation function. As mentioned previously, we can also use a stack of smaller kernel size convolutional layersas it is parameter-efficient, however, using a large filter in the initial stages of the network is still very prevalent. Multiple $3\times 3$ filters can achieve a similar receptive field to a single $7\times 7$ convolution. To be precise, two stacked $3\times 3$ convolutional layers have a receptive field of $5\times 5$, and three $3\times 3$ convolutional layers have a receptive field of $7\times 7$. Models like MobileNet or EfficientNet use smaller kernels ($3\times 3$ or $5\times 5$) in the initial set of layers. 

Some advantages of using large filters in the initial stages of the network architecture are as follows:
1. **Capturing Global Patterns Early**: A $7\times 7$ kernel can capture diverse set of spatial patterns in the input image, such as broader edges or textures, which smaller kernels might miss in the initial stages. Capturing more information in the beginning can provider richer activations/features for subsequent layers to build upon.
2. **Spatial Dimension Reduction**: Often, the first layer uses a stride greater than 1 (e.g., stride 2), which reduces the spatial dimensions of the feature maps. This downsampling decreases computational load for later layers. For instance, if we have an input image of dimensionality $224 \times 224$, and we apply a $7\times 7$ filter with a stride of 2, the output feature dimensionality is $112\times 112$.

Larger kernels in the first layer can increase computational cost but may extract more informative features early on. On the contrary, smaller kernels reduce computation but may require deeper networks to capture the same level of detail.

ADD COMPUTATIONS AND IMAGES OF RECEPTIVE FIELDS IN PRACTICE.

- https://distill.pub/2019/computing-receptive-fields/#solving-receptive-field-region
- https://youtu.be/ip2HYPC_T9Q?si=sz-YvXhb2ewIT12Q