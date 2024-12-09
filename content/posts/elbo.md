---
title: "ELBO: Evidence Lower Bound Optimization"
date: "2024-10-22"
summary: "ELBO Derivation."
description: "This blog discusses about different types of normalization used in deep learning."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

## Problem 
Suppose we're working with the posterior distribution over latent variable, i.e., $p(\mathbf{z}|\mathbf{x}=\mathcal{D})$. In practice, we do not have access to a closed form solution for $p(\mathbf{z}|\mathbf{x}=\mathcal{D})$, especially in high-dimensional observed spaces. Therefore, we approximate the posterior distribution $p(\mathbf{z}|\mathbf{x}=\mathcal{D})$ via a surrogate distribution $q(\mathbf{z}|\mathbf{x}=\mathcal{D})$. 

We consider Directed Graphical Models with observed variables $\mathbf{x} \in \mathbb{R}^N$ and latent variable $\mathbf{z} \in \mathbb{R}^D$ where $D \ll N$.

If we have access to the DGM, we also have access to the joint distribution $p(\mathbf{x}, \mathbf{z})$ where $\mathbf{x}$ denotes images, and $\mathbf{z}$ denotes latent variables such as camera angles, or some property within the image not directly visible. Joint distribution can be defined as follows: $$ p(\mathbf{x},\mathbf{z}) = p(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) $$We observe $\mathbf{x}$ as the dataset $\mathcal{D}=\{x^{(i)}\}_{i=1}^N$ (*images*). Given this, we want to perform variational inference in order to learn more about the latent variables The posterior distribution of the latent variable $\mathbf{z}$ given the observed data point $\mathbf{x}$ is as follows, $$ p(\mathbf{z}|\mathbf{x}=\mathcal{D}) = \frac{p(\mathbf{x}=\mathcal{D}|\mathbf{z}) p(\mathbf{z})}{p(\mathbf{x}=\mathcal{D})} $$The problem in this formulation is the **denominator** term, i.e., the marginal likelihood $p(\mathbf{x}=\mathcal{D})$ which involves integrating over all possible latent variables $\mathbf{z}$.

The marginal likelihood over a single data point is $\mathbf{x}$ is $p(\mathbf{x}) = \int p(\mathbf{x}, \mathbf{z}) , d\mathbf{z}$. For the entire dataset $\mathcal{D}$, the marginal likelihood is $$ p(\mathbf{x}=\mathcal{D}) = \int_{\mathbf{z}} p(\mathbf{x}, \mathbf{z}) d\mathbf{z} $$However, this integration is intractable/incomputable. 
## Solution: Approximate `p(z|x)` using a Surrogate Posterior
Instead, we approximate $p(\mathbf{z}|\mathbf{x})$ with a variational distribution $q(\mathbf{z}|\mathbf{x})$. The term "_**variational**_" comes from variational calculus, as we optimize over a function to perform inference.

To optimize, we need a loss function that returns us the **goodness of the fit**. We use **KL divergence** for this task as it helps express the dissimilarity between two distributions.
$$ q^*(\mathbf{z}) = {\arg\min}\_{q(\mathbf{z})\in\mathcal{Q}} [KL(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}) ]$$
The KL divergence can be defined as follows: $$ KL(q(\mathbf{z} | \mathbf{x}) \| p(\mathbf{z}|\mathbf{x})) = \mathbb{E}\_{q(\mathbf{z}|\mathbf{x})} \left[  \log \frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})} \right] $$Higher this value, further apart the two distributions are. It simply is the expected log likelihood ratio of the two distributions. Here, we're using **REVERSE KL DIVERGENCE**.

However, we cannot compute $p(\mathbf{z}|\mathbf{x})$ due to the intractable marginal likelihood $p(\mathbf{x})$. Instead, we derive a lower bound on $\log p(x)$, known as the **Evidence Lower Bound (ELBO)**.

 In the above formulation, we do not have the posterior $p(\mathbf{z}|\mathbf{x})$. All we have is the joint distribution $p(\mathbf{z},\mathbf{x})$, i.e., $$ p(\mathbf{z}|\mathbf{x}) = \frac{p(\mathbf{x},\mathbf{z})}{p(\mathbf{x})} =  \frac{p(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{p(\mathbf{x})} $$
## Derivation of ELBO
Starting with the KL Divergence; $$ \begin{gather*}
\mathrm{KL}\left(q(\mathbf{z}|\mathbf{x})\parallel p(\mathbf{z}|\mathbf{x})\right)  =\int q(\mathbf{z}|\mathbf{x})\log\left(\frac{q(\mathbf{z}|\mathbf{x})}{p(\mathbf{z}|\mathbf{x})}\right)d\mathbf{z} \\\\
 =\int q(\mathbf{z}|\mathbf{x})\left(\log q(\mathbf{z}|\mathbf{x})-\log p(\mathbf{z}|\mathbf{x})\right)d\mathbf{z}
\end{gather*} $$
Since $p(\mathbf{z}|\mathbf{x}) = p(\mathbf{x},\mathbf{z}) / p(\mathbf{x})$, $$ \begin{gather*}
\mathrm{KL}\left(q(\mathbf{z}|\mathbf{x})\parallel p(\mathbf{z}|\mathbf{x})\right) =\int q(\mathbf{z}|\mathbf{x})\left(\log q(\mathbf{z}|\mathbf{x})-\log\frac{p(\mathbf{x},\mathbf{z})}{p(\mathbf{x})}\right)d\mathbf{z} \\\\
 =\int q(\mathbf{z}|\mathbf{x})\left(\log q(\mathbf{z}|\mathbf{x})-\log p(\mathbf{x},\mathbf{z})+\log p(\mathbf{x})\right)d\mathbf{z}
\end{gather*} $$Since $\log p(\mathbf{x})$ doesn't depend on $\mathbf{z}$, we have  $$ \begin{gather*}
\mathrm{KL}\left(q(\mathbf{z}|\mathbf{x})\parallel p(\mathbf{z}|\mathbf{x})\right)=\int q(\mathbf{z}|\mathbf{x})\left(\log q(\mathbf{z}|\mathbf{x})-\log p(\mathbf{x},\mathbf{z})\right)d\mathbf{z}+\log p(\mathbf{x})\int q(\mathbf{z}|\mathbf{x})d\boldsymbol{z} \\\\
=\mathbb{E}\_{q(\mathbf{z}|\mathbf{x})}\left[\log q(\mathbf{z}|\mathbf{x})-\log p(\mathbf{x},\mathbf{z})\right]+\log p(\mathbf{x})
\end{gather*} $$Since $\mathbb{E}_{\mathbf{z} \in q(\mathbf{z}|\mathbf{x})} =  \int q(\mathbf{z}|\mathbf{x})d\boldsymbol{z} = 1$, the integration simplifies. 
- we know how to compute $q(\mathbf{z}|\mathbf{x})$ since we know the functional form of the surrogate function.
- we also know $p(\mathbf{x}, \mathbf{z})$, i.e., encoder output. It is calculated as: $p(\mathbf{z}|\mathbf{x}) p(\mathbf{x})$.  

Let us define the ELBO as follows: $$ \mathcal{L}(q)=\mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[\log p(\mathbf{x},\mathbf{z})-\log q(\mathbf{z}|\mathbf{x})\right]$$Rewriting the KL divergence, 
$$ \mathrm{KL}\left(q(\mathbf{z}|\mathbf{x})\parallel p(\mathbf{z}|\mathbf{x})\right)=-\mathcal{L}(q)+\log p(\mathbf{x}) $$
Here, $p(\mathbf{x})$ is the marginal distribution. 
Applying $\log$ to the marginal distribution $p(\mathbf{x})$ yields the **evidence**. It is called evidence since it is the log-probability of the data.  If $\log$ is applied to a quantity that is between 0 and 1, we get values $\leq 0$, i.e., $[-\infty, 0]$. We have no ways of computing $p(\mathbf{x})$. 

Since the KL Divergence $\mathrm{KL}\left(q(\mathbf{z}|\mathbf{x})\parallel p(\mathbf{z}|\mathbf{x})\right)$ is non-negative, and the evidence $\log p(\mathbf{x})$ is $\leq 0$ we have $$ \mathcal{L}(q) \leq \log p(\mathbf{x}) $$
This inequality shows that $\mathcal{L}(q)$ is a lower bound on the log evidence $\log p(\mathbf{x})$, hence the name **Evidence Lower Bound (ELBO)**. The ELBO is always smaller than the evidence. 

Our optimization task becomes $$ q^*(\mathbf{z}|\mathbf{x}) =\arg\max_{q(\mathbf{z}|\mathbf{x}) \in\mathcal{Q}}\mathcal{L}(q) $$We can also arrange the equation to express $\log p(\mathbf{x})$: $$ \log p(\mathbf{x})=\mathcal{L}(q)+\mathrm{KL}\left(q(\mathbf{z}|\mathbf{x})\parallel p(\mathbf{z}|\mathbf{x})\right)$$
> Since the KL divergence is non-negative, maximizing $\mathcal{L}(q)$ minimizes the KL divergence between the $q(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{z}|\mathbf{x})$,

$\mathcal{L}(q) =\log p(\mathbf{x}) \iff KL(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))=0$, i.e., the surrogate distribution & the true posterior are the same. This is usually not achieved in variational inference. The best fit is computed by finding a $q$ that maximizes the ELBO, i.e., $\arg\max_{q(\mathbf{z}|\mathbf{x}) \in\mathcal{Q}}\mathcal{L}(q)$

## Code Example
We'll be using a Variational Auto Encoder to provide a code-based explanation of ELBO.

The encoder network approximates the surrogate distribution $q(\mathbf{z}|\mathbf{x}) \approx p(\mathbf{z}|\mathbf{x})$.
```python
# The encoder network approximates the posterior q(z|x)
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        # Input layer to hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Hidden layer to mean of q(z|x)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Hidden layer to log variance of q(z|x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        """
        Forward pass through the encoder to obtain parameters of q(z|x),
        which is a Gaussian distribution with mean mu and variance sigma^2.
        """
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)         # Mean of q(z|x)
        log_var = self.fc_logvar(h)  # Log variance of q(z|x)
        return mu, log_var
```

The re-parameterization trick relates to sampling $\mathbf{z}$ from the surrogate distribution $q(\mathbf{z}|\mathbf{x})$, i.e., $\mathbf{z}\sim q(\mathbf{z}|\mathbf{x})$ for the expectation $\mathbb{E}_{q(\mathbf{z})}$. Via the code we get the mean $\mu$ and the log variance $\log \sigma$. These values are used to sample points from the "*learned*" latent space.
```python
def reparameterize(self, mu, log_var):
    """
    Reparameterization trick to sample z ~ q(z|x).
    Instead of sampling z ~ N(mu, sigma^2), we sample eps ~ N(0,1) and 
    compute z = mu + sigma * eps.
    This allows backpropagation through stochastic nodes.
    """
    std = torch.exp(0.5 * log_var)  # Standard deviation
    eps = torch.randn_like(std)     # Sample from standard normal
    z = mu + eps * std              # Reparameterize
    return z
```

The decoder relates to sampling $p(\mathbf{x}|\mathbf{z})$ in the joint distribution $p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x}|\mathbf{z})p(\mathbf{z})$
```python
# The decoder network models the likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # Latent space to hidden layer
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        # Hidden layer to output layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        Forward pass through the decoder to obtain p(x|z).
        """
        h = F.relu(self.fc1(z))
        # Output probabilities for Bernoulli likelihood
        x_recon = torch.sigmoid(self.fc2(h))  
        return x_recon
```

ELBO Computation
```python
def compute_elbo(self, x, x_recon, mu, log_var):
    """
    Compute the Evidence Lower BOund (ELBO).

    ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))

    The ELBO consists of:
    1. Reconstruction loss: negative expected log-likelihood E_q(z|x)[log p(x|z)]
    2. KL divergence between q(z|x) and the prior p(z)
    """
    # Reconstruction loss (Negative Log-Likelihood)
    # Using Binary Cross Entropy Loss for Bernoulli likelihood p(x|z)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL Divergence between q(z|x) and p(z)
    # For multivariate Gaussians, KL divergence has an analytical solution:
    # KL(q(z|x) || p(z)) = 0.5 * sum( exp(log_var) + mu^2 - 1 - log_var )
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # ELBO is the negative of the sum of reconstruction loss and KL divergence
    # ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
    elbo = - (recon_loss + kl_divergence)
    return elbo, recon_loss, kl_divergence
```
- The ELBO here is $\mathcal{L}(q)=\mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}, \mathbf{z}) - \log q(\mathbf{z})]$.
- The reconstruction term `recon_loss` corresponds to $\mathbb{E}_{q(\mathbf{z})} [\log p(\mathbf{x}|\mathbf{z})]$.
- The KL divergence term corresponds to $\mathbb{E}_{q(\mathbf{z})}[\log q(\mathbf{z}) - \log p(\mathbf{z})]$

The training loop is as follows
```python
# Hyperparameters
input_dim = 28 * 28  # For MNIST images
hidden_dim = 400
latent_dim = 20
batch_size = 128
learning_rate = 1e-3
num_epochs = 50

# Transformations for the MNIST dataset
transform = transforms.Compose(
    [
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Lambda(lambda x: x.view(-1)),  # Flatten the images
    ]
)

# Load the MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate the VAE model and move it to the device
vae = VAE(input_dim, hidden_dim, latent_dim).to(device)

# Define the optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

# Create a directory to save the model checkpoints
checkpoint_dir = "./vae_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "vae_mnist.pth")

# Training loop
for epoch in range(num_epochs):
    vae.train()
    total_loss = 0
    for batch_idx, (x_batch, _) in enumerate(dataloader):
        # Move data to the appropriate device (CPU/GPU)
        x_batch = x_batch.to(device)

        # Forward pass
        x_recon, mu, log_var = vae(x_batch)
        # Compute ELBO
        elbo, recon_loss, kl_divergence = vae.compute_elbo(x_batch, x_recon, mu, log_var)

        # Backward pass and optimization
        optimizer.zero_grad()
        # We minimize -ELBO, which is equivalent to maximizing ELBO
        loss = -elbo
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, "
                f"KL Div: {kl_divergence.item():.4f}"
            )

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Save the model checkpoint
torch.save(vae.state_dict(), checkpoint_path)
print(f"Model saved to {checkpoint_path}")
```
In the above code, we aim to maximize ELBO $\mathcal{L}(q)$ by updating the parameters $q(\mathbf{z}|\mathbf{x})$ and $p(\mathbf{x}|\mathbf{z})$, i.e., encoder and decoder parameters, respectively. The optimization objective is $$ q^*(\mathbf{z}) = {\arg\max}_{q(\mathbf{z})\in\mathcal{Q}} \mathcal{L}(q) $$

The model's reconstruction after training is as follows: 
![alt text](/assets/posts/elbo/mnist-vae-elbo.svg#dark#small "VAE MNIST Generation")

## Connecting Back to ELBO Derivation
- Objective: We aim to approximate $p(\mathbf{z}|\mathbf{x}=\mathcal{D})$ with $q(\mathbf{z}|\mathbf{x})$.
- KL Divergence: We aim to minimize $KL(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}|\mathbf{x}))$
- ELBO: Since $p(\mathbf{x})$ is intractable (*expanding $p(\mathbf{z}|\mathbf{x})=p(\mathbf{x},\mathbf{z})p(\mathbf{x})$*), we maximize the ELBO $\mathcal{L}(q)$ which is a lower bound of the evidence $\log p(\mathbf{x})$
$$ \begin{gather*}
\mathcal{L}(q) & =\mathbb{E}\_{q(\mathbf{z})}\left[\log\left(\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\right)\right] \\\\
 & =\mathbb{E}\_{q(\mathbf{z})}[\log p(\mathbf{x}|\mathbf{z})]+\mathbb{E}\_{q(\mathbf{z})}[\log p(\mathbf{z})]-\mathbb{E}\_{q(\mathbf{z})}[\log q(\mathbf{z})] \\\\
 & =\mathbb{E}\_{q(\mathbf{z})}[\log p(\mathbf{x}|\mathbf{z})]-KL(q(\mathbf{z})||p(\mathbf{z}))
\end{gather*} $$
In code
- $\mathbb{E}_{q(\mathbf{z})}[\log p(\mathbf{x}|\mathbf{z})]$ corresponds to the reconstruction loss (`recon_loss`)
- $KL(q(\mathbf{z})||p(\mathbf{z}))$ corresponds to the KL divergence term (`kl_divergence`)
- The ELBO is computed as a negative sum of these two terms, i.e., `elbo = -(recon_loss + kl_divergence)`

### Notes
1. Reparameterization Trick: Allows gradients to flow through stochastic sampling by expression $\mathbf{z} = \mu + \sigma \cdot \epsilon$, where $\epsilon \in \mathcal{N}(0,I)$
2. KL Divergence for Gaussians: Closed form solution exists. The analytical expression simplifies computation and avoids numerical integration.
3. Maximizing ELBO: By maximizing ELBO, we indirectly minimize the KL divergence between $q(\mathbf{z}|\mathbf{x})$ (*surrogate distribution / encoder output*) and $p(\mathbf{z}|\mathbf{x})$ (*true posterior*)

