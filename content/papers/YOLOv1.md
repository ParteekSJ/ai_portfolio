---
title: "YOLO-V1: You Only Look Once: Unified, Real-Time Object Detection"
date: "2024-12-03"
summary: "YOLO V1 Model Explained."
description: "An LSM Tree overview and Java implementation."
toc: true
readTime: true
autonumber: true
math: true
tags: ["database", "java"]
showTags: false
hideBackToTop: false
---

## One Stage vs. Two Stage Detection
YOLOV1 is a unified architecture , i.e., one stage detector.
- Two Stage Detection System
	- Stage 1: Predict candidate regions which possibly contain objects
	- Stage 2: Classify these regions into appropriate categories and as well as regressing bounding boxes for the proposed regions to tightly fit the underlying object.
	- Example: Faster RCNN (RPN for Stage 1, and Detection Head for Stage 2)
- One Stage
	- Skip the Proposal Generation Step.
	- Directly make fixed number of predictions for multiple categories using a single network given the input image.
	- Reduces complexity of detection pipeline introduced in two stage models.

![alt text](/assets/papers/yolov1/YOLOV1-twostage_vs_onestage.png#dark#small "One Stage vs. Two Stage Models.")

## YOLO Object Detection Algorithm
YOLOV1 frames object detection as a single stage regression problem. Input Image is passed to the YOLO CNN, and this network predicts multiple bounding boxes for the detected objects and class probabilities associated with these predicted bounding boxes in one evaluation.

1. Divide image into $S\times S$ grid cells covering the entire image. In the paper,  the authors use a value of $S=7$, i.e., dividing the image into a grid of $7\times 7$.     ![alt text](/assets/papers/yolov1/YOLOV1-dividing_image_into_sxs_gridcells.png#dark#small, "")
2. Every target object is assigned to one grid cell that contains the **center of the object**. Given the ground truth bounding box, we find the cell which contains the center of the bbox and assign that to that specific object. In the image below, there are two objects - Person and Car. The centers of the ground truth bounding boxes for both the objects lies in grid cell 6 and 5, respectively.![alt text](/assets/papers/yolov1/YOLOV1-gridcell_categorization.png#dark#small, "")
3. Each grid cell predicts $B$ bounding boxes. For ease of visualization, we assume $B=1$. In the paper, $B=2$, i.e., 2 bounding boxes are predicted for each grid cell. ![alt text](/assets/papers/yolov1/YOLOV1-bbox_preds_per_cell.png#dark#small, "")
4. YOLO is trained to have box predictions of each cell as close as possible to the target assigned to that cell. In this image, for the cell which had a target assigned (*person: cell 6, car: cell 5*), we retain those prediction boxes and discard all others. Through training, the YOLO model will learn to have the predictions as close as possible to the ground truth boxes.
![alt text](/assets/papers/yolov1/YOLOV1-retain_predictionbboxes_gridcell.png#dark#small "")

## YOLO Box Predictions
Now we'll dive into the exact values that the model predicts for **each bounding box**. YOLO model predicts **5 parameters for each of the bounding boxes**
- $c_x$: The x-coordinate of the center of the bounding box, relative to the bounds of the grid cell.
- $c_y$: The y-coordinate of the center of the bounding box, relative to the bounds of the grid cell.
- $w$: The width of the bounding box, relative to the entire image width.
- $h$: The height of the bounding box, relative to the entire image height.
- **Confidence**: A scalar value representing the confidence that an object exists within the bounding box and that the bounding box accurately locates it.

### Detailed Explanations
1. $c_x$ and $c_y$
	-  **Relative to Grid Cell**: YOLOv1 divides the input image into $S\times S$ grid (typically $S=7$). Each grid cell is responsible for detecting objects whose centers fall within it. They are the **offset values**, i.e., $x$-translation and $y$-translation from the top-left corner of the assigned grid cell. Offset values are used to denote center of the bounding box **relative** to the top left corner of the grid cell. These offset values will also be normalized between 0 and 1.
	-  **Normalization**: The center coordinates $(x,y)$ are normalized between 0 and 1 within their grid cell. This means $(x,y) \in [0,1]$, where $(0,0)$ is the top-left corner of the grid cell and $(1,1)$ is the bottom-right corner of the grid cell.
2. $w$ and $h$
	-  **Relative to Entire Image**: The width and height are normalized by the total width and height of the image. This means $w,h \in [0,1]$
	-  $w=1$ indicates that the width of the bounding box extends across the entire width of the image. $h=1$ means that the height of the bounding box extends across the entire height of the image. Both $w=1, h=1$ indicates that the bounding box covers the entire image.
	-  **Square Root Transformation**: In YOLOV1, the square root of the width and height is predicted to stabilize the learning process for varying object sizes.
3. Confidence
	-  **Objectness Score**: The confidence score reflects the **probability that a bounding box contains an object** and **how accurate the bounding box is**. It can be viewed as an estimate to how good the model's prediction is .It attempts to capture 2 aspects. 
		-  how confident the model is that the box indeed contains an object
		-  how accurate or good fit the predicted box is, for the object it contains.
	-  Range: The confidence score is between 0 and 1.
 ![alt text](/assets/papers/yolov1/YOLOV1-box_centroid_prediction.png#dark#small "") 
In the image above, the red dot indicates the center of the bounding box for the "car" object. The $c_x,c_y$ value is relative to the top left corner of grid cell 5.  

## YOLO - Grid Cell Level Predictions
At grid cell level, each grid cell has $5\times B$ predictions, i.e., each grid cell predicts $B$ bounding boxes. Each of those bounding boxes consists of $5$ parameters: $w, h, c_x, x_y, \text{conf}$.  ![alt text](/assets/papers/yolov1/YOLOV1-gridecell-predictions.png#dark#small "")
In addition to the above, the model also predicts **class conditional probabilities** for each grid cell. Each grid cell will have $(5\times B)+C$ values. For the PascalVOC dataset with 20 classes, each grid cell will predict $(2\times 2) + 20 = 30$ values. 
For each of the $S\times S$ grid cell, we'll be predicting 30 values, i.e., 10 values for the $B=2$ bounding boxes and $20$ class conditional probabilities. ![alt text](/assets/papers/yolov1/YOLOV1-gridcell_level_predictions.png#dark#small "")
For each grid cell, YOLO predicts one set of class probabilities as we can see in the above image. YOLO will predict multiple boxes $(B>1)$ per grid cell,  but only one predictor box is responsible for that target, the one bounding box with highest IOU with the target box. ![alt text](/assets/papers/yolov1/YOLOV1-30predictions.png#dark#small "")
> In this image, for grid cell 5, the model predicts 20 class conditional probabilities (*distribution over all the classes [VOC DATASET HAS 20 GROUND TRUTH CLASSES] given the detection object*) and 10 bounding box predictions.

### Additional Details
- For each grid cell, YOLO predicts only a **single set** of class probabilities no matter what the value of $B$ is. 
- YOLO will predict multiple bounding boxes $(B>1)$ per grid cell, but only one predictor box is responsible for that target, the one with highest IOU with the target box (*ground truth box*). ![alt text](/assets/papers/yolov1/YOLOV1-multiplebboxpredictions.png#dark#small "")
In the above image the bounding boxes with the highest IOUs are stored whereas the rest of them are discarded.

> From the paper: "YOLO predicts multiple bounding boxes per grid cell. At training time we only want one bounding box predictor to be responsible for each object. We assign one predictor to be "responsible" for predicting an object based on which prediction has the highest IOU with the ground truth. This leads to specialization between bounding box predictors. Each predictor gets better at predicting certain sizes, aspect ratios, or classes of object, improving overall recall."

Localization Loss is calculated for the "responsible" predictor.

- "***predictors***": represents the outputs associated with a specific bounding box prediction. In code, predictors are represented by slices of the output tensor. It can be accessed by indexing into the output tensor after reshaping. Each predictor outputs `(x, y, w, h, confidence)`.
## YOLO Architecture 
Given an image, the model outputs a value of the dimensionality $S\times S \times ((5\times B) + C)$. This is transformed into a $S\times S$ grid with $((5\times B) + C)$ channels as seen in the image below. ![alt text](/assets/papers/yolov1/YOLOV1-modeloutput.png#dark#small "")
If $S=3, B=2$ and $C=20$ (*number of classes*), this CNN will return prediction values which can be transformed into a $3\times 3$ grid output with $(5\times B) + C$ channels. Each output cell (*each cell in the $S\times S$ grid*) is going to have $(5\times B) + C$ values, i.e., bounding box predictions plus conditional class probability distribution.  ![alt text](/assets/papers/yolov1/YOLOV1-modelpredictionoverview.png#dark#small "")
For the model architecture, the authors utilize a custom version of the GoogleNet architecture. Instead of the Inception Module, they replace it with $1\times 1$ and $3\times 3$ convolutional layers. ![alt text](/assets/papers/yolov1/YOLOV1-modelarchitecture-customgooglenetvariant.png#dark#small "")
 ![alt text](/assets/papers/yolov1/YOLOV1-modelarchitecture-customgooglenetvariant2.png#dark#small "")
The authors first pre-train this network on the ImageNet classification task by stacking FC layers and training it on images of dimensionality $224\times 224$. ![alt text](/assets/papers/yolov1/YOLOV1-architecture-imagenet-pretraining.png#dark#small "")
Post-training, they get rid of the FC layers, and add additional convolutional layers prior to model detection training (*specifically 4 convolutional layers*). After these convolutional layers, we have 2 FC layers to predict the $S\times S \times (5\times B + C)$ dimensional tensor.
![alt text](/assets/papers/yolov1/YOLOV1-architecture.png#dark#small "")
The final detection network looks as follows: ![alt text](/assets/papers/yolov1/YOLOV1-architecture_postpretraining.png#dark#small "")
The convolutional layers in the red box denote the additional layers that were added to the pretrained network (*network trained on ImageNet classification task*). The input to this architecture has the dimensionality of $448\times 448$ instead of $224\times 224$ for fine grained visual information. The model outputs a tensor of dimensionality $7\times 7 \times 30$. For the VOC dataset, each cell (*in the grid cell*) will have 30 prediction values.

## YOLO Loss
![alt text](/assets/papers/yolov1/YOLOV1-lossfunctions.png#dark#small "")
The YOLOv1 loss function combines multiple components to penalize errors in
1. **Localization Loss**: Predicting the bounding box coordinates (*x, y offsets from the top-left corner of the grid cell*) accurately (*whether the model has generated correct bounding box coordinates*)
2. **Confidence Score**: Estimating the likelihood that a predicted box contains an object (*whether the predicted bounding box contains an object or not*)
3. **Classification**: Correctly classifying the object within the bounding box (*whether the object inside the predicted bounding box is correctly classified*)
It can be represented as $$ 
\begin{aligned}\text{Loss} & =\lambda\_\mathrm{coord}\sum\_{i=0}^{S^2} \sum\_{j=0}^B 1\_\mathrm{obj}^{ij} \left[(x\_i-\hat{x}\_i)^2+(y\_i-\hat{y}\_i)^2\right] \\\\
 & +\lambda\_\mathrm{coord~}\sum\_{i=0}^{S^2} \sum\_{j=0}^B1\_\mathrm{obj}^{ij} \left[(\sqrt{w\_i}-\sqrt{\hat{w\_i}})^2+(\sqrt{h\_i}-\sqrt{\hat{h\_i}})^2\right] \\\\
 & +\sum\_{i=0}^{S^2}\sum\_{j=0}^B1\_{\mathrm{obj}}^{ij}(C\_i-\hat{C}\_i)^2 \\\\
 & +\lambda\_\mathrm{noobj}\sum\_{i=0}^{S^2}\sum\_{j=0}^B1\_\mathrm{noobj}^{ij}(C\_i-\hat{C}\_i)^2 \\\\
 & +\sum\_{i=0}^{S^2}1\_{\mathrm{obj}}^i\sum\_{c\in\mathrm{classes}}(p\_i(c)-\hat{p}\_i(c))^2
\end{aligned}$$
- $S$: Grid Size (*number of cells along one dimension, e.g., $S=7$*)
- $B$: Number of Bounding boxes predicted per grid cell. (e.g., $B=2$)
- $\lambda_\text{coord}$: Weighting term for localization loss (*typically set to 5*)
- $\lambda_\text{noobj}$: Weighting term for confidence loss when no object is present (*typically set to 0.5*)
- $1^{i,j}_{\text{obj}}$: Indicator function equal to 1 if object appears in cell $i$ and predictor $j$ is responsible for the prediction.
- $1^{i,j}_{\text{noobj}}$: Indicator function equal to 1 if no object is present in cell $i$ for predictor $j$.
- $(x_i, y_i)$: Ground truth center coordinates of the bounding box, relative to the grid cell
- $(\hat{x_i}, \hat{y_i})$: Predicted center coordinates.
- $(w_i, h_i)$: Ground truth width and height, normalized by image dimensions
- $(\hat{w}_i, \hat{h}_i)$: Predicted width and height
- $C_i$: Ground truth confidence score (*usually 1 if object is present*)
- $\hat{C}_i$: Predicted Confidence score
- $p_i(c)$: Ground truth probability of class $c$ in cell $i$
- $\hat{p}_i(c)$: Predicted probability of class $c$ in cell $i$

### Localization Loss 
$$ \begin{aligned} \text{Localization Loss}&=\lambda\_\mathrm{coord}\sum\_{i=0}^{S^2} \sum\_{j=0}^B 1\_\mathrm{obj}^{ij} \left[(x\_i-\hat{x}\_i)^2+(y\_i-\hat{y}\_i)^2\right] \\\\ & +\lambda\_\mathrm{coord~}\sum\_{i=0}^{S^2}  \sum\_{j=0}^B1\_\mathrm{obj}^{ij} \left[(\sqrt{w\_i}-\sqrt{\hat{w\_i}})^2+(\sqrt{h\_i}-\sqrt{\hat{h\_i}})^2\right]
\end{aligned}
$$
This loss penalizes the model when the predicted bounding box coordinates deviate from the ground truth bounding box coordinates. The components of this loss are as follows: 
1. Coordinate Error
	- $(x_i - \hat{x}_i)^2$: Error in $x$-coordinate
	- $(y_i - \hat{y}_i)^2$: Error in $y$-coordinate
2. Size Errors (with Square Root)
	- $(\sqrt{w_i} - \sqrt{\hat{w}}_i)^2$: Error in the width.
	- $(\sqrt{h_i} - \sqrt{\hat{h}}_i)^2$: Error in the height.

We only want to calculate this loss for boxes which are responsible for some target ground truth and ignore the rest. This sum would be over the predicted boxes of cells assigned with some target that has maximum IOU with the target box. The indicator function filters only those boxes. $1_\mathrm{obj}^{ij}$ is 1 if a cell $i$ is assigned a target and box $j$ is responsible for that target. 


The reason we use **square roots** of width and height balances the loss between large and small boxes, preventing dominance of large errors due to large objects.
> $1^{i,j}_{\text{obj}}$ ensures that the loss is calculated only for the predictor responsible for the object in that grid cell.

![alt text](/assets/papers/yolov1/YOLOV1-widthheight_squareroot.png#dark#small "")
### Why Square Root for Width, Height?
Let us consider a hypothetical example where all our predicted height and width parameters are off from their target by an offset of $0.1$. Not using square root for width and height difference computation is detrimental when computing bound box offset for small objects. MSE with square root penalizes small bounding boxes offsets higher than large bounding box offsets
![alt text](/assets/papers/yolov1/YOLOV1-widthheight_squareroot_why.png#dark#small "")
> Small deviations on large boxes matter much less than small deviations on small boxes.

### Confidence Loss
The confidence loss is as follows; $$ \begin{aligned} \text{Confidence Loss} &= \sum\_{i=0}^{S^2}\sum\_{j=0}^B1\_{\mathrm{obj}}^{ij}(C\_i-\hat{C}\_i)^2 \\\\
  &+\lambda\_\mathrm{noobj}\sum\_{i=0}^{S^2}\sum\_{j=0}^B1\_\mathrm{noobj}^{ij}(C\_i-\hat{C}\_i)^2  \end{aligned}$$This loss penalizes the model for incorrect confidence scores. The components of this loss are as follows: 
1. Object Present $(1^{i,j}\_{\text{obj}})$:
	- $(C\_i - \hat{C}\_i)^2$: Error in the confidence score when an object is present.
2. No Object Present $(1^{i,j}\_{\text{noobj}})$:
	- $(C\_i - \hat{C}\_i)^2$: Error in the confidence score when no object is present.
	- Weighted by $\lambda\_{\text{noobj}}$ to reduce the impact of many background boxes

Confidence Scores $\hat{C}\_i$ quantifies two things
1. confidence of the model that the box indeed contains an object
2. how accurate or good fit the predicted box is, for the object is contains. 
Ideally, $\hat{C}\_i$ should be high when an object is present and low when its not. 
![alt text](/assets/papers/yolov1/YOLOV1-confidenceloss.png#dark#small "")
For all no object boxes, the confidence scores are calculated as follows: $$ \text{Pr(Object)} * \text{IOU}\_{\text{truth}}^{\text{pred}} $$


We want to ensure that the confidence of the responsible predictor boxes are closer to their target values. In addition, we also train the model to predict the confidence scores of boxes which are not assigned to any target object as 0. The second term takes care of that.

### Classification Loss
The classification loss is as follows: $$ \text{Classification Loss = } \sum_{i=0}^{S^2}1\_{\mathrm{obj}}^i\sum\_{c\in\mathrm{classes}}(p\_i(c)-\hat{p}\_i(c))^2 $$This loss penalizes the model when it incorrectly predicts the class of the object. The components of this loss function are as follows: 
- $(p\_i(c) - \hat{p}\_i(c))^2$ denotes the error in predicted class probabilities.
- Calculated for each class $c$ and only in grid cells where an object is present, i.e., $1^{i,j}_{\text{obj}}$

![alt text](/assets/papers/yolov1/YOLOV1-combined_loss.png#dark#small "")
- For most of the images we would only have few cells that would be assigned with some target and a lot of background cells. This would cause the gradient from confidence score of boxes with no objects overpower those gradients produced by the term in the loss function that contains the objects. To mitigate this, and to add more preference to the localization error instead of classification error we use 2 terms: $\lambda_{\text{coord}}$ and $\lambda_{\text{noobj}}$. ![alt text](/assets/papers/yolov1/YOLOV1-loss_with_labels.png#dark#small "")
