---
title: "Weight Tying"
date: "2024-11-26"
summary: "Weight Tying Explained."
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

https://paperswithcode.com/method/weight-tying
This is a technique that improves the performance of language models by tying (**sharing**) the weights of the input embedding and output projection layers. This method has shown to have a significant reduction in perplexity. Additionally, this technique can reduce the size of the language models to less than half of their original size without harming their performance.


https://arxiv.org/abs/1608.05859v3
- focused more on the topmost weight matrix of language models.
- recommend tying the input embedding and this output embedding

## What is Weight Tying?
This is a technique that is used improve the performance of large language models by tying (sharing) the weights of the embedding and softmax layers. To obtain a better understanding, consider the following figure.

![alt text](/assets/posts/weight_tying/weight-tying-illustration.png#dark#small "Weight Tying.")

Since most language models are decoder-only transformer models, the weight tying is performed between the input embedding layer and the output projection layer. There are several benefits of performing weight tying.

## Benefits of Weight Tying

1. **Parameter Efficiency**: De-facto language models have separate weights for the input embedding and the output projection layers, each with dimensionality varying with the vocabulary and embedding size. Weight tying reduces the total number of parameters by sharing these matrices, i.e.,  `Traditional Total Parameters = Embedding Matrix Parameters + Output Matrix Parameters`
With weight tying:  `Total Parameters with Weight Tying = Embedding Matrix Parameters`. This is especially useful for large vocabularies.

2. **Semantic Alignment**: Tying the weights between the input embedding layer and the output projection layer enforces a symmetry between how words are represented when are the fed as tokens to the input embedding layer versus when they are predicted as outputs. This consistency (*in the input and output space*) can help the model generalize better on unseen data. The semantic relationships learned during the embedding of input tokens are directly utilized in output prediction (*right before performing softmax and sampling*). If two words have similar embedding in the input space, the tied weights ensure that the model will also consider them similar in output space potentially improving synonym recognition.
3. **Regularization Effect**: Sharing weights acts as a form of regularization as the it reduces the amount of parameters that the model can tune. In other words, it reduces the model's capacity to overfit by limiting the number of parameters that can be adjusted independently
4. **Improved Metrics & Training**: Studies such as described in the paper [here](https://arxiv.org/abs/1608.05859v3), have shown that weight tying can lead to better performance metrics such as **perplexity** compared to models without weight tying. Despite the reduction in parameters, models with weight tying often match or exceed the performance of larger models without weight tying, suggesting that the additional parameters in traditional models may not contribute effectively to learning. It also improves training and expedites convergence  as fewer parameters make the optimization landscape smootherand easier to navigate for optimizers such as Adam, SGD, etc.


<!-- ## Theoretical Justification
In language models, embeddings serve a dual role, i.e., 
- Encoding: Mapping tokens to continuous vector representations.
- Decoding: Mapping contextualized hidden states back to token probabilities.

In linear algebra, the most natural way to reverse a transformation is by using the transpose of the transformation matrix (assuming orthogonality). Tying weights approximates this by using the same matrix for both encoding and decoding. -->

## Code Example

Consider the following transformer model

```python
class TransformerWithWeightTying(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super(TransformerWithWeightTying, self).__init__()

        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Initializing an embedding layer.
        # Used to embed each token to a `embedding_dim` vector.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # Initializing a PositionalEncoding Layer
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))

        # Initializing a Transformer Model
        self.transformer = nn.Transformer(
            d_model=embedding_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * embedding_dim,
            batch_first=True,
        )

        # Defining the Output Projection Layer (embedding_dim -> vocab_size)
        self.output_layer = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

        # Tie the weights between embedding & output layer
        self.output_layer.weight = self.embedding.weight

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Add embeddings and positional encodings to source and target
        src_embeddings = self.embedding(src) + self.positional_encoding[:, : src.size(1), :]
        tgt_embeddings = self.embedding(tgt) + self.positional_encoding[:, : tgt.size(1), :]

        # Pass through the transformer
        transformer_output = self.transformer(
            src_embeddings,
            tgt_embeddings,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        # Compute logits using the tied output layer
        logits = self.output_layer(transformer_output)
        return logits

    def _generate_square_subsequent_mask(self, sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def generate(self, src, start_token_id, end_token_id, max_length=50):
        src_mask = self._generate_square_subsequent_mask(src.size(1)).to(src.device)

        # Encode the source sequence
        src_embeddings = self.embedding(src) + self.positional_encoding[:, : src.size(1), :]

        # Represents the encoded source information
        memory = self.transformer.encoder(src_embeddings, mask=src_mask)

        # Initialize target sequence with start token
        tgt_tokens = (
            torch.ones((src.size(0), 1), dtype=torch.long, device=src.device) * start_token_id
        )

        # Starting the token generation process..
        for _ in range(max_length):
            # Generating target sequence embeddings.
            tgt_embeddings = (
                self.embedding(tgt_tokens) + self.positional_encoding[:, : tgt_tokens.size(1), :]
            )

            # Generating a `tgt_mask` to ensure that the model cannot look ahead.
            tgt_mask = self._generate_square_subsequent_mask(tgt_tokens.size(1)).to(src.device)

            # Decode the target sequence
            output = self.transformer.decoder(
                tgt=tgt_embeddings,
                memory=memory,
                tgt_mask=tgt_mask,
            )

            # Compute logits, i.e., Output Projection
            logits = self.output_layer(output[:, -1, :])  # Get the last token's logits

            # Apply the softmax function to the logits to obtain probabilities
            probs = F.softmax(logits, dim=-1)

            # Greedy decoding: pick the token with the highest probability
            next_token = torch.argmax(probs, dim=-1).unsqueeze(1)

            # Append the predicted token to the target sequence
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)

            # Check for end token
            if torch.all(next_token == end_token_id):
                break

        return tgt_tokens
```

This model with the following configuration

``` python
vocab_size = 10000
embedding_dim = 512
num_heads = 8
num_layers = 6
max_seq_len = 50
```

has `54,390,544` parameters. If we remove the line from the code that performs weight tying, the model gains `26000` additional parameters, i.e., `54,416,144` parameters.

Suppose we're using the above model to translate English sentences to French.

1. Source Sentence: "I love coding"
2. Tokenized Source: `[101, 102, 103]` (where each number represents a word)
3. Generation Process:
   1. Start with `<SOS>`, i.e., Start of Sentence Token.
   2. tgt_tokens = `[<SOS>]`. This is the initial model output. Once we begin the inference process, we append the model's output to this list.
4. First Iteration: The model predicts the probability distribution for the next word after `<SOS>`. Suppose it predicts high probability for "**J'**" (French for "I"). We append "**J'**" to `tgt_tokens`. Second Iteration: `tgt_tokens = [<SOS>, "J'"]` 
5. The model now predicts the next word after "**J'**". Suppose it predicts "**aime**" ("love"). We append "**aime**" to `tgt_tokens`.
6. Third Iteration: `tgt_tokens = [<SOS>, "J'", "aime"]`. The model predicts the next word. Suppose it predicts **"coder"** ("coding"). We append **"coder"** to `tgt_tokens`.
7. Fourth Iteration: `tgt_tokens = [<SOS>, "J'", "aime", "coder"]`. The model predicts the next word. Suppose it predicts `<EOS>` (end of sequence). We append `<EOS>` and terminate the generation.
8. Final Generated Sequence: `[<SOS>, "J'", "aime", "coder", <EOS>]`

> Translation:"J'aime coder"