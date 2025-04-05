---
title: "Aero Engine Blade Anomaly Detection"
date: "2024-10-29"
summary: "Aero Engine Blade Defect Detection using Masked Multi-scale Reconstruction (MMR) model which utilizes a Vision Transformer and Feature Pyramid Network."
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

In this project, we've applied the Masked Multi-scale Reconstruction (MMR) method which can be found [here](https://arxiv.org/pdf/2304.02216v2) on the Aero-engine Blade Anomaly Detection (AeBAD) dataset.  This model utilizes a Vision Transformer (ViT) based encoder-decoder architecture and Feature Pyramid Network to enhance the model’s ability to perform multi-scale reconstruction of image patches. The architecture looks as follows:

![alt text](/assets/projects/aeroengine-blade-project/MMR-Architecture.png#dark#small "Masked Multi-scale Reconstruction (MMR) Model Architecture.")

The code can be found [here](https://github.com/ParteekSJ/Masked-Multiscale-Reconstruction). Below are the outputs of the trained model on the AeBAD dataset.

![alt text](/assets/projects/aeroengine-blade-project/op1.gif#dark#small "Model Output (Scenario 1).")
![alt text](/assets/projects/aeroengine-blade-project/op2.gif#dark#small "Model Output (Scenario 2).")
