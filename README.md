# GAN-Evaluation-Synthetic-vs-Real-Data
This repository contains the code used to evaluate the performance of Generative Adversarial Networks (GANs) in generating synthetic data and assessing its utility in comparison with real data using traditional machine learning models.
## Why use GAN?
Generative Adversarial Networks (GANs) are a class of neural networks designed to generate synthetic data that closely resembles real data. GANs consist of two main components: the generator and the discriminator. The generator creates synthetic data, while the discriminator evaluates it against real data. Through this adversarial process, the generator learns to produce increasingly realistic data, making GANs an excellent tool for data augmentation, particularly when real data is scarce or difficult to obtain.
## Synthetic Data Explanation
Synthetic data refers to data that is artificially generated rather than obtained by direct measurement. In this project, we use GANs to generate synthetic data that mimics the characteristics of real data. This experiment aims to determine how well traditional machine learning models, like Support Vector Machines (SVMs), perform when trained on this synthetic data compared to real data. Additionally, we evaluate the effects of mixing synthetic data with real data to see if it can enhance model performance.
## Further Details
This repository documents the code aspects of a research paper currently in process of publication. We encourage trying and testing out the model with different datasets for different results. 

To check out results, go to the [Results.md](Results.md) page.
For running code, check out the [How-to-run.txt](How-to-run.txt) page.
