# 2025 CVDL Final project

## Introduction
The goal of this project is to build an image classification model that can automatically identify the disease status of cassava leaves from photographs. Specifically, the task is to classify each image into one of five categories: four known cassava leaf diseases or a healthy class. The dataset comprises 21367 labeled images of cassava leaves, exhibiting variations in lighting, angles, and backgrounds to reflect real-world scenarios. A notable challenge is the class imbalance, with certain diseases like Cassava Mosaic Disease (CMD) being more prevalent than other four categories.  
To talk about the importance of cassava leaf disease classification, cassava leaf disease classification first enhances the global food security problem and improves the economic well-being of smallholder farmers; at the same time, it also drives innovation in image classification technologies by addressing real-world challenges such as data imbalance, limited computational resources, and deployment on low-power devices in rural agricultural settings.  
In this task, we observe that over the past four years, many have attempted to tackle this classification problem, yet no one has been able to surpass an accuracy of 0.92. Therefore, we study the new model architectures and learn from the top-ranking solutions on the leaderboard to explore whether it is possible to overcome this challenging problem.
One key difficulty we encountered is the class imbalance in the dataset, some disease categories are significantly underrepresented, which can lead to biased predictions and reduced model generalization. Our approach aims to mitigate this issue through balanced training strategies and ensemble methods to improve robustness across all classes. Overall, our advanced approach is that we use new model architectures different from the previous works and use ensemble methods to integrate those models to achieve the best performance in this task.  

## Model Overview
![image](https://github.com/user-attachments/assets/8748b959-f510-4777-974f-f5e960f3ec11)

## Repo Guide

* Each folder contains the training implemenation of individual included model.
* For inference, upload the notebook to kaggle and download the require dataset.
* dataset
* https://www.kaggle.com/datasets/tsaitsungwei/convnextv2-small
* https://www.kaggle.com/datasets/tsaitsungwei/testrestnet
* https://www.kaggle.com/datasets/dennislin0906/5fold-effxl-2
* https://www.kaggle.com/datasets/dennislin0906/5fold-eff-l-v2
* https://www.kaggle.com/datasets/tsaitsungwei/swintransformer
* https://www.kaggle.com/datasets/tsaitsungwei/mobilenet
## Performance
![image](https://github.com/user-attachments/assets/9c4c1eba-17a9-403a-bac4-f7ac209f5bda)
