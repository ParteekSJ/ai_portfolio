---
title: "Transformer: Attention Is All You Need"
date: "2025-01-15"
summary: "Transformer Model Explained."
description: "Transformer Model Explained."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
draft: False
---

<!-- ![transformer architecture](/assets/papers/transformer/transformer_architecture.png#floatleft, "") -->

{{< figure src="/assets/papers/transformer/transformer_architecture.png" caption="Transformer Architecture." alt="Transformer Architecture" width="500px" height="auto" align="center" class="float-right">}}

In this article, I'll be explaining how transformer models works for the task of text translation between `ENG🇬🇧-ITA🇮🇹`. One sentence pair will be used to demonstrate how the model processes it. This should give the reader an in-depth understand of how input gets transformed as it goes through the model

## Data Preparation

To train a transformer model for the task of translation between the source and target language, the source-target sentence pairs needs to be preprocessed. Suppose, we're performing the task of translating **English** sentences to **Italian**. Consider the following sentence pair
- "en" - the dog jumped over the wall
- "it" - il cane ha saltato il muro

This data needs to be preprocessed before it gets fed to the transformer model. Before the preprocessing, we need to define some constants
- `src_seq_len` - maximum length of the source language sentence that we can use as input. Sentences longer than this value get truncated.
- `tgt_seq_len` - maximum length of the target language sentence.

> For this example, we set `src_seq_len` and `tgt_seq_len` to $20$. This ensures batch uniformity.
- `<SOS>, <EOS>, <PAD>` - These are special tokens that indicate start of sentence, end of sentence, and padding.
  - `<SOS>`: this token is prepended to the sequence to indicate the start of the sentence
  - `<EOS>`: this token is towards the end to indicate the end of the sentence
  - `<PAD>`: this token is used to ensure that the source and target tokenized sequence reaches the `src_seq_len` and `tgt_seq_len`, respectively.

Transformers process batches of sequences in parallel, requiring all sequences in a batch to have the same length. This is where padding (<PAD>) comes in. Special tokens help define the start, end, and boundaries of this process. 

Suppose, the source and target tokenizer assign these special tokens the following IDs: `<SOS> - 0`, `<EOS> - 1` and `<PAD> - 99`

Each sentence is tokenized, i.e., each word is assigned a unique integer based on the source/target **tokenizer's vocabulary**. For the above 2 sentences, we get the following tokens
   - "en": the dog jumped over the wall --> [5, 791, 1207, 121, 5, 726] $\in \mathbb{R}^6$
   - "it" - il cane ha saltato il muro --> [14, 813, 104, 5527, 14, 1326] $\in \mathbb{R}^6$

Once we have the source and target sequence tokenized, we add the special tokens to indicate when the sentence begins and ends.
   - "en": `<SOS>` the dog jumped over the wall `<EOS>`: [**0**, 5, 791, 1207, 121, 5, 726, **1**] $\in \mathbb{R}^8$
   - "it" - `<SOS>` il cane ha saltato il muro: [**0**, 14, 813, 104, 5527, 14, 1326] $\in \mathbb{R}^7$


Given that the maximum source and target sequence length is 20, we need to ensure that the tokenized sequences are $\in \mathbb{R}^{20}$. This ensures that the training/inference can be performed on batched inputs as opposed to single inputs. The reason for this is primarily because sentences in batches aren't homogeneous (*different en-it pairs have differenet lengths*), and consequently, to perform processing on batched inputs, their tokenized sequence dimensions must be the same. Thus, 12 `<PAD>` tokens are added to the source sequence, and 13 `<PAD>` tokens are added to the target sequence. The updated tokenized representation looks as follows:
  - "en": `<SOS>` the dog jumped over the wall <`EOS>` `<PAD>` ... `<PAD>`: [0, 5, 791, 1207, 121, 5, 726, 1, **99, ..., 99**] $\in \mathbb{R}^{20}$
  - "it" - `<SOS>` il cane ha saltato il muro `<PAD>` ... `<PAD>`: [0, 14, 813, 104, 5527, 14, 1326, **99, ..., 99**] $\in \mathbb{R}^{20}$

In addition to the tokenized source and target sequence, we also create a **tokenized ground truth label**. The ground truth label doesn't contain the `<SOS>` token, but contains the `<EOS>` and `<PAD>` tokens. The ground truth label tensor for the above pair is created as follows:
  - Adding the `<EOS>` token: 
    - "it" - il cane ha saltato il muro `<EOS>` : [14, 813, 104, 5527, 14, 1326, **1**] $\in \mathbb{R}^7$
  - Adding the `<PAD>` tokens: 
    - "it" - il cane ha saltato il muro `<EOS>` `<PAD>` ... `<PAD>`  : [14, 813, 104, 5527, 14, 1326, 1, **99, ..., 99**] $\in \mathbb{R}^{20}$
  
The tokenized source sequence tensor, tokenized target sequence tensor and ground truth tensor are used for training the transformer model. In addition to these sequences, we also create masks. This will be discussed in the "Multi-Head Attention" section.






## Input Embedding

Now, we'll focus on the **Input Embedding Layer** of the model. It is a simple lookup table that stores embeddings of a fixed dictionary and size. In other words, each token (numerical representation of the word) gets assigned to a `d_model` dimensional vector. The transformer, by default, uses `d_model=512`.

Given the source sequence: `<SOS>` the dog jumped over the wall <`EOS>` `<PAD>` ... `<PAD>`, its tokenized representation gets mapped from $\mathbb{R}^{20} \to \mathbb{R}^{20 \times 512}$ which simply means each word is now described by a $512$-dimensional vector. The embedding vectors are randomly initialized, and are learned throughout the training process. This layer is implemented via a `torch.nn.Embedding` layer.

## Positional Encoding

Each word in the source sequence is now a vector of dimension `d_model=`$512$. These embedding vectors do not carry any positional information. Unlike traditional RNNs or LSTMs, transformers don't process sequences sequentially but parallelly. In simpler words, we want the model to treat terms that are "close" (in a semantic sense) as closer and terms that are "distant" as distant. This is achieved by combining the input embedding information with the positional embeddings. The positioanl encoding are created using the following formulas:
$$
\begin{aligned}
&PE(pos, 2i) = \sin \left( \frac{pos}{ 10000^{(2i/d_{\text{model}})}} \right) \\\\
&PE(pos, 2i+1) = \cos \left( \frac{pos}{ 10000^{(2i/d_{\text{model}})}} \right)
\end{aligned}
$$

- `pos` denotes the position. Since, the maximum source and target sequence length, `src_seq_len` and `tgt_seq_len`, are 40, our `pos` parameter will range from $[1,...,40]$
- `i` denotes the embedding dimension. Since each token is embedded into $\mathbb{R}^{512}$. Thus, $i$ ranges from  $[0,...512]$. $2i$ denotes all the even embeddings, and $2i+1$ denotes the odd embeddings.

The Positional Encoding function can be coded as follows:
```python
def getPositionEncoding(seq_len, d, n=10000):
    # Defining an empty matrix of dimensionality (max_seq_len, d_model)
    P = np.zeros((seq_len, d))
    # Iterating through (0, 20)
    for k in range(seq_len):
        # Iterating through the embedding dimensions (0, 512)
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator) # odd embedding indices
            P[k, 2 * i + 1] = np.cos(k / denominator) # even embedding indices
    return P
```

Each position/index is mapped to a vector of dimensionality `d_model` i.e., $\mathbb{R}^{512}$. 

![alt text](/assets/papers/transformer/pe.png#dark#small, "")
In the following plot, the $x$-axis denotes the embedding dimension `i`, and the $y$-axis denotes the position `pos`. For each position (row in the plot), there exists a unique pattern that indicates position. The positional encoding matrix has the dimensionality of $\mathbb{R}^{\text{seqlen}\times\text{dmodel}}$, which in our case is $\mathbb{R}^{20 \times 512}$. The positional encoding matrix is concatenated to the input source embedding resulting in a $\in \mathbb{R}^{20 \times 512}$ tensor.

## Encoder Forward Pass

The source input embedding concatenated with the positional encoding matrix gets fed into the Encoder Block. The encoder block can be divided into 2 main parts:
1. Multi-Head Attention + Add & Norm 
2. Feed Forward + Add & Norm 

We'll first talk about Multi-Head Attention. Three different weight matrices of dimensionality (`d_model` * `d_model`) are defined
- $W_Q \in \mathbb{R}^{512 \times 512}$
- $W_K \in \mathbb{R}^{512 \times 512}$
- $W_V \in \mathbb{R}^{512 \times 512}$

These weight matrices implemented as `torch.nn.Linear` layers are applied on the concatenated input embedding tensor $\mathbf{x}\in\mathbb{R}^{20\times 512}$ to generate the query $Q \in \mathbb{R}^{20\times 512}$, keys $K \in \mathbb{R}^{20\times 512}$ and values $V \in \mathbb{R}^{20\times 512}$. The attention operation is performed as follows: $$ \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}}  \right)V $$

Instead of performing a single attention function with $Q \in \mathbb{R}^{20\times 512}$, $K \in \mathbb{R}^{20\times 512}$, and $V \in \mathbb{R}^{20\times 512}$, the authors found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$ , $d_k$ and $d_v$ dimensions, respectively. We define a hyperparameter `num_heads = 8`. Practically, we take the linearly projected $Q \in \mathbb{R}^{20\times 512}$, $Q \in \mathbb{R}^{20\times 512}$, and $V \in \mathbb{R}^{20\times 512}$, and rewrite them  as follows: $$ \begin{align*} Q\in \mathbb{R}^{20\times 512} \equiv \mathbb{R}^{20\times 8 \times 64} \equiv \mathbb{R}^{8\times 20 \times 64} \\\\ K \in \mathbb{R}^{20\times 512} \equiv \mathbb{R}^{20\times 8 \times 64} \equiv \mathbb{R}^{8\times 20 \times 64} \\\\ V\in \mathbb{R}^{20\times 512} \equiv \mathbb{R}^{20\times 8 \times 64} \equiv \mathbb{R}^{8\times 20 \times 64}  \end{align*}$$

This is equivalent to "**linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_k$ , $d_k$ and $d_v$ dimensions, respectively**". The dimensionality $\mathbb{R}^{8\times 20 \times 64}$ can be interpreted as follows: 
- `20` denotes the (maximum) sequence length.
- `8` denotes the number of attention heads, i.e., $h$.
- `64` denotes the embedding dimension per head, i.e., $d\_\text{model} / h =  d_k = d_v$

> Each head has access to the entire sequence, but only a subset of the embedding.



The MultiHead Attention is defined as follows: $$ \begin{align*} \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O \\\\ \text{where} \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \end{align*}$$

In code, we perform
1. Perform $QK^T$ which simply is $\mathbb{R}^{8\times 20 \times 64} \times \mathbb{R}^{8\times 64 \times 20}$ yielding $\mathbb{R}^{8\times 20 \times 20}$ matrix. For each head, we have a `(seq_len, seq_len)` matrix signifying the unnormalized attention scores.
2. Divide by $\sqrt{d_k}$ where $d\_k = d\_{\text{model}} / h = 512/8=64$. This is to avoid the problem of internal covariate shift, i.e., when the gradients become too small resulting in no learning being performed.

Before, we apply the softmax operation to compute the normalized attention scores, a mask is created to ensure that no attention is paid to `<PAD>` tokens. Practically, this is done by setting all `<PAD>` tokens in the resultant $QK^T/\sqrt{d_k} \in \mathbb{R}^{8\times 20\times 20}$ to a value of $-\infty$.

To recap, our source sequence looks as follows: 
- "en": `<SOS>` the dog jumped over the wall <`EOS>` `<PAD>` ... `<PAD>`: [0, 5, 791, 1207, 121, 5, 726, 1, **99, ..., 99**] $\in \mathbb{R}^{20}$.

A mask should be generated such as all tokens with the `ID=99` aren't paid any attention. Hence, our source language sequence mask looks as follows:
{{< figure src="/assets/papers/transformer/attn_masks_src.png" caption="Transformer Architecture." alt="Image description" >}}


The left image shows the output of $QK^T/\sqrt{d_k}$. The right image shows the mask being applied to $QK^T/\sqrt{d_k}$. Once we have the mask applied, we compute the softmax on the resultant matrix $\mathbb{R}^{8\times 20 \times 20}$ over the last dimension giving us a distribution of attention scores for each token. All scores for each row add up to 1.

Once softmax is applied, the $\text{softmax}(QK^T/\sqrt{d_k})$ matrix, it is multiplied with the value matrix resulting in $\mathbb{R}^{8\times 20 \times 20} \to \mathbb{R}^{8\times 20 \times 64}$ output. Now that the attention operation is computed for all the heads, we perform concatenate them, i.e., $\text{Concat}(\text{head}_1, \dots, \text{head}_h)$.

Practically, we do this by reshaping the resultant tensor as follows: $$ \mathbb{R}^{8\times 20\times 64} \to \mathbb{R}^{20\times 8\times 64} \to \mathbb{R}^{20\times 512} $$
Lastly, the resultant matrix is multiplied by a matrix $W_O \in \mathbb{R}^{512\times 512}$ resulting in an output of $\mathbb{R}^{20\times 512}$. This is the output of the Multi-Head Self Attention.

We then have the Add & Norm block where we concatenate the output of the MHSA with the original input to the MHSA block, and then perform Layer Normalization.

Now, we'll move onto the Feed Forward + Add & Norm block. A Position-wise Feed-Forward Networks layer consisting of two linear transformations with a ReLU activation in between is defined. This FFN is applied to each position separately and identically. $$ \text{FFN}(x) = \max(0, xW_1 + b1)W_2 + b_2 $$

These layers can also be viewed as as two convolutions with kernel size 1. In code, these layers are described as `torch.nn.Linear` layers with the input and output dimensionality being `d_model=512`, and the inner-layer has dimensionality being `d_ff=2048`. The resultant output of this FFN block is $\mathbb{R}^{20\times 512}$. This is then fed to the Add & Norm block where we first perform a residual connection, i.e., the output of the FFN block is concatenated with the original input ot the FFN block. All this is then passed throufh a `LayerNorm` layer. The resultant output of this stack is $\mathbb{R}^{20\times 512}$.


Suppose, if `num_encoder_blocks=6`, i.e., we have 6 encoder blocks, this operation is repeated and the dimensionality of the final output that we get from the encoder is $\mathbb{R}^{20\times 512}$.


----

The "Output Embedding" and "Positional Encoding" is similar to what we discussed above.

## Decoder Forward Pass

The target output embedding concatenated with the positional encoding matrix gets fed into the Decoder Block. The encoder block can be divided into 3 main parts:
1. Masked Multi-Head Attention + Add & Norm 
2. Multi-Head Attention + Add & Norm 
3. Feed Forward + Add & Norm


To recap, our target sequence is as follows: "it" - il cane ha saltato il muro `<EOS>` `<PAD>` ... `<PAD>`  : [14, 813, 104, 5527, 14, 1326, 1, **99, ..., 99**] $\in \mathbb{R}^{20}$. This goes through the output embedding layer converting it from $\mathbb{R}^{20} \to \mathbb{R}^{20 \times 512}$. Finally, a positional encoding matrix is generated and concatenated resulting in a $\mathbb{R}^{20 \times 512}$ matrix.

This get fed into the decoder block. The first module of the decoder block is the **Masked Multi-head Attention**. Similar to the MHSA module in the encoder, three different weight matrices of dimensionality (`d_model` * `d_model`) are defined
- $W_Q \in \mathbb{R}^{512 \times 512}$
- $W_K \in \mathbb{R}^{512 \times 512}$
- $W_V \in \mathbb{R}^{512 \times 512}$

These weight matrices implemented as `torch.nn.Linear` layers are applied on the concatenated output embedding tensor $\mathbf{x}\in\mathbb{R}^{20\times 512}$ to generate the query $Q \in \mathbb{R}^{20\times 512}$, keys $K \in \mathbb{R}^{20\times 512}$ and values $V \in \mathbb{R}^{20\times 512}$. Instead of using the entire `d_model` dimension to compute the dot-product scaled attention, we perform multi-head attention where queries, keys and values are linearly projected $h$ times with different, learned linear projections to $d_k$ , $d_k$ and $d_v$.  Practically, we take the linearly projected $Q \in \mathbb{R}^{20\times 512}$, $Q \in \mathbb{R}^{20\times 512}$, and $V \in \mathbb{R}^{20\times 512}$, and rewrite them  as follows: $$ \begin{align*} Q\in \mathbb{R}^{20\times 512} \equiv \mathbb{R}^{20\times 8 \times 64} \equiv \mathbb{R}^{8\times 20 \times 64} \\\\ K \in \mathbb{R}^{20\times 512} \equiv \mathbb{R}^{20\times 8 \times 64} \equiv \mathbb{R}^{8\times 20 \times 64} \\\\ V\in \mathbb{R}^{20\times 512} \equiv \mathbb{R}^{20\times 8 \times 64} \equiv \mathbb{R}^{8\times 20 \times 64}  \end{align*}$$

Masked Multi-Head Attention is computed as follows:
1. Perform $QK^T$ which simply is $\mathbb{R}^{8\times 20 \times 64} \times \mathbb{R}^{8\times 64 \times 20}$ yielding $\mathbb{R}^{8\times 20 \times 20}$ matrix. For each head, we have a `(seq_len, seq_len)` matrix signifying the unnormalized attention scores.
2. Divide by $\sqrt{d_k}$ where $d\_k = d\_{\text{model}} / h = 512/8=64$. This is to avoid the problem of internal covariate shift, i.e., when the gradients become too small resulting in no learning being performed.

Before, we apply the softmax operation to compute the normalized attention scores, a specialized mask is created to ensure 
- each token can pay attention to the tokens that've already appeared before it.
- each `<PAD>`is ignored

To recap, our target sequence looks as follows: 
- "it" - `<SOS>` il cane ha saltato il muro `<PAD>` ... `<PAD>`: [0, 14, 813, 104, 5527, 14, 1326, **99, ..., 99**] $\in \mathbb{R}^{20}$

Hence, our target language sequence mask applied looks as follows:
{{< figure src="/assets/papers/transformer/decoder_masks_tgt.png" caption="Masked Multi-Head Attention." alt="Image description" >}}

As you can see the generated mask ensures that all tokens with the `ID=99`, i.e., `<PAD>` aren't paid any attention. In addition, a token can only compute attention with the tokens that've appeared before it. For example, token `813` can compute the scaled dot product with tokens `0` and `14`. 

Once we have the mask applied, we compute the softmax on the resultant matrix $\mathbb{R}^{8\times 20 \times 20}$ over the last dimension giving us a distribution of attention scores for each token. All scores for each row add up to 1. 

Once softmax is applied, the $\text{softmax}(QK^T/\sqrt{d_k})$ matrix, it is multiplied with the value matrix, $V\in\mathbb{R}^{8\times64}$ resulting in $\mathbb{R}^{8\times 20 \times 20} \to \mathbb{R}^{8\times 20 \times 64}$ output. Now that the attention operation is computed for all the heads, we perform concatenate them, i.e., $\text{Concat}(\text{head}_1, \dots, \text{head}_h)$. Practically, we do this by reshaping the resultant tensor as follows: $$ \mathbb{R}^{8\times 20\times 64} \to \mathbb{R}^{20\times 8\times 64} \to \mathbb{R}^{20\times 512} $$
Lastly, the resultant matrix is multiplied by a matrix $W_O \in \mathbb{R}^{512\times 512}$ resulting in an output of $\mathbb{R}^{20\times 512}$. This is the output of the Multi-Head Self Attention.


Now, we perform Cross Attention where query from the decoder, and keys and value from the last encoder block's output.
{{< figure src="/assets/papers/transformer/crossattn.png" caption="Cross Attention." alt="Image description">}}

It is similar to the above where the query comes from the output of the Masked-MHSA module, and keys and value from the last encoder block's output. The output of the cross attention block is $\mathbb{R}^{20\times 512}$. This gets fed to the Position Wise FFN that consists of 2 linear layers described as follows: $$ \text{FFN}(x) = \max(0, xW_1 + b1)W_2 + b_2 $$

The output of this module is $ \mathbb{R}^{20\times 512} $

## Projection Layer

Finally, the output from the final decoder layer is transformed into a vector of vocabulary size, i.e., `d_vocab`. Then, a softmax is applied over this vector of size $d_{vocab}$, resulting in a probability distribution over all the words in the vocabulary. The Projection Layer is simply a linear layer that maps the final decoder output $$ \mathbb{R}^{20\times 512} \to \mathbb{R}^{20\times d\_\text{vocab}}$$ . The projection layer is implemented as a `torch.nn.Linear` layer.

Using a decoding strategy such as Greedy Decoding or Beam Search, we can pick the next word of the decoder sequence.

## Training the Transformer Model

Let us recap on the structure of the source language sequence, target language sequence, and ground truth label sequence. 
- Source Language Sentences have the following structure: `<SOS> content <EOS> <PAD> ... <PAD>`.
- Target Language Sentences have the following structure: `<SOS> content <PAD> ... <PAD>`. As you can see, it doesn't contain the `<EOS>` token.

The decoder starts generating the target sentence from the `<SOS>` token. During training, the decoder is fed the target sequence shifted right (i.e., starting with `<SOS>`), and it predicts the next token at each step. For example, if the target is "**il cane ha saltato il muro**", the input to the decoder begins with `<SOS>` to predict "**il**".

- Ground Truth Sequence have the following structure: `content <EOS> <PAD> ... <PAD>`. 

The ground truth sequence doesn't contain the `<SOS>` token, but contains the `<EOS>` token. This is what the decoder should predict at the final step of generation. During training, the model compares its predictions to this ground truth, and <EOS> signals when to stop.

> During training, the decoder uses **teacher forcing**: it’s fed the ground truth sequence (shifted right) as input to predict the next token.

Transformer model's output is $\mathbb{R}^{20\times d\_\text{vocab}}$. The cross entropy loss is used to backpropagate gradients which causes the transformer model to learn inherent patterns between the source and language text.

- The transformer's final model output is $\mathbb{R}^{20\times d\_\text{vocab}}$ where $20$ represents the sequence length, and $d_\text{vocab}$ represents the vocabulary size. In other words, we have a distribution $\in\mathbb{R}^{d\_\text{vocab}}$ over all of the 20 tokens.
- The ground truth label has the dimensionality $\mathbb{R}^{20}$.

The loss is computed using the `torch.nn.CrossEntropy` function. It quantifies how well the model predicts the next token.

---
I hope this article helps you get a better understanding of how the transformer model is used for seq2seq translation task. The code can be found on: https://github.com/ParteekSJ/transformer-scratch