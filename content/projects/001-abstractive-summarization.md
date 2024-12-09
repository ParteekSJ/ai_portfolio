
---
title: "Abstractive Summarization of Scientific Articles using Transformers (with Question-Answering)"
date: "2024-07-30"
summary: "Abstractive Summarization of Scientific Articles using Transformers (with QA)."
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

{{< youtube L-ORIv7PLro >}}

> All the trained models can be found in [HuggingFace](https://huggingface.co/parteeksj).
## Project Description
- Utilized [PEGASUS](https://huggingface.co/docs/transformers/en/model_doc/pegasus), [BART](https://huggingface.co/facebook/bart-large), and [FLAN-T5](https://huggingface.co/google/flan-t5-large) and enhanced their performance by fine-tuning a dataset of scientific papers, leveraging LoRA.
- Developed a specialized parsing system to extract data from .tex files originating from arXiv research articles.
- Implemented a user-friendly interface using Streamlit to allow users to input and receive concise summaries of scientific papers.
- Augmented the system with a question-answering module, enabling users to ask specific questions about the paper’s content using [MBZUAI/LaMini-Flan-T5-783M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-783M) model.
- Leveraged vector databases, specifically ChromaDB, to efficiently store and retrieve high-dimensional embeddings of scientific papers, enabling rapid similarity searches and content-based recommendations.
- Uploaded the fine-tuned models and datasets to HuggingFace.
