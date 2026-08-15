<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,35:0F766E,70:0284C7,100:38BDF8&height=240&section=header&text=Satellite%20Vision%20Transformer&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=EuroSAT%20Classification%20%7C%20Transfer%20Learning%20%7C%20Hugging%20Face&descSize=16&descAlignY=60" />

<img src="https://readme-typing-svg.herokuapp.com?font=Inter&weight=800&size=24&duration=2800&pause=900&color=38BDF8&center=true&vCenter=true&width=950&lines=State-of-the-Art+Computer+Vision;Google+ViT+%2B+Hugging+Face+Transformers;Remote+Sensing+%2B+Land+Use+Classification;98%25+Validation+Accuracy+in+1+Epoch" />

<br/>

<a href="#">
  <img src="https://img.shields.io/badge/GitHub-Repository-111827?style=for-the-badge&logo=github&logoColor=white" />
</a>

<img src="https://img.shields.io/badge/Status-Production%20Ready-10B981?style=for-the-badge" />
<img src="https://img.shields.io/badge/Accuracy-98.05%25-7C3AED?style=for-the-badge" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/Kaggle_GPU-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />

</div>


## ✨ Project Overview

**Satellite Vision Transformer** is a deep learning computer vision pipeline designed to classify land use from space. 

Instead of relying on traditional Convolutional Neural Networks (CNNs), this project leverages a **Vision Transformer (ViT)**. By utilizing Transfer Learning on a pre-trained Google ViT (`vit-base-patch16-224-in21k`), the model rapidly adapts to geospatial data, identifying 10 distinct topological classes (e.g., Forests, Highways, Industrial Areas) with near-perfect accuracy in a fraction of standard training time.


## 🎯 Primary Objectives

* Implement state-of-the-art Transformer architectures for a non-text (Computer Vision) modality.
* Utilize the Hugging Face ecosystem (`transformers`, `datasets`, `evaluate`) for streamlined data streaming and model training.
* Handle complex image preprocessing, including tensor conversion, resizing, and color normalization via `AutoImageProcessor`.
* Execute memory-efficient GPU training using Gradient Accumulation to prevent Out-of-Memory (OOM) failures.
* Demonstrate high-accuracy transfer learning for enterprise-level remote sensing and climate tech applications.


## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph Cloud Data Ingestion
        A[Hugging Face Hub] --> B[EuroSAT Dataset]
        B --> C[21,600 Satellite Images]
    end

    subgraph Image Preprocessing Pipeline
        C --> D[Resize to 224x224]
        D --> E[RGB Tensor Conversion]
        E --> F[Statistical Normalization]
    end

    subgraph Vision Transformer Architecture
        F --> G[Split into 16x16 Pixel Patches]
        G --> H[Pre-trained ViT Base Layers]
        H --> I[Custom 10-Class Classification Head]
    end

    subgraph Training & Inference
        I --> J[Kaggle GPU Training Loop]
        J --> K[Validation Metrics]
        K --> L[Real-time Image Inference]
    end
