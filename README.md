# 🏆 Amazon ML Challenge 2025: Multimodal Product Price Prediction

This repository contains our team's solution for the **Amazon ML Challenge 2025**.  
The goal was to predict product prices using a multimodal dataset consisting of **text descriptions (`catalog_content`)** and **product images (`image_link`)**.  
Our approach involved a phased strategy — starting with a robust text-only baseline and progressing to a final multimodal model that incorporates both text and image features.

**Final Multimodal OOF SMAPE:** `49.40%`

---

## 🚀 Project Summary

This project tells a story of **rapid development**, **strategic adaptation**, and **real-world problem-solving** under a tight deadline.  
We successfully built a complete end-to-end pipeline — from data ingestion and cleaning to advanced feature engineering, model training, and cloud-based deep learning.

### 🔹 Key Achievements

- **Advanced Feature Engineering:** Developed a comprehensive pipeline to extract features like *Item Pack Quantity (IPQ)*, brand names, product categories, and weight/volume from noisy text data.  
- **Hybrid Cloud/Local Workflow:** Strategically used **Google Colab’s T4 GPU** for extracting image embeddings and a **local multi-core CPU** for Gradient Boosting model training.  
- **Robust Modeling:** Implemented a stable **5-fold cross-validation** strategy to train and validate models, ensuring reliable performance.  
- **Significant Performance Gains:** Improved SMAPE from an initial baseline of `~50.01%` to a final multimodal score of `~49.40%`, validating the importance of image and feature fusion.

---

## 🛠️ Technical Approach

Our solution is built on a modular pipeline that processes **text** and **image** data separately before fusing them for a final prediction.

### 1. 🧩 Feature Engineering  
**File:** `src/data/feature_engineering.py`

Feature engineering was the cornerstone of our performance improvement.  
We built a single, powerful pipeline applying the following transformations:

- **Basic Text Features:** `content_length`, `word_count`, `digit_count`, etc.  
- **Advanced Text Features:**
  - *Brand Detection:* Identified known brands from a curated list.  
  - *Category Heuristics:* Classified products into food, beauty, etc., based on keywords.  
  - *Premium/Value Indicators:* Scored text based on words like `organic`, `premium`, `combo`, and `value pack`.  
  - *Enhanced IPQ Extraction:* Used prioritized regex patterns for robust Item Pack Quantity detection.  
  - *Weight/Volume Extraction:* Normalized units to kilograms and liters.  
  - *Interaction Features:* Created features such as `ipq * weight_kg` and `text_len / ipq`.

---

### 2. 🖼️ Image Feature Extraction  
**File:** `scripts/extract_image_embeddings.py`

To process the 150K+ images, we used a **pre-trained EfficientNet-B0** model.

- **Model:** EfficientNet-B0 (ImageNet pre-trained)  
- **Process:** The classification head was removed, and a **1280-dimensional embedding** was extracted for each image.  
- **Execution:** This step was performed on a **Google Colab T4 GPU**, reducing processing time from multiple days to a few hours.

---

### 3. 🤖 Modeling  
**File:** `scripts/train_multimodal.py`

Our final model combined all engineered features into one fusion dataset.

- **Algorithm:** LightGBM (Light Gradient Boosting Machine)  
- **Features Used:**
  - **TF-IDF Features:** 30K sparse features (`ngram_range=(1, 2)`)  
  - **Image Embeddings:** 1280 dense features from EfficientNet-B0  
  - **Tabular Features:** ~32 numerical features from the engineered pipeline  
- **Fusion Strategy:** *Early Fusion* — concatenated all three feature types before 5-fold CV training.

---

## 📊 Results

Our phased approach allowed us to quantify each stage’s contribution:

| Model Version | Key Features | OOF SMAPE |
|----------------|---------------|------------|
| Initial Baseline | Basic Text + TF-IDF | ~50.01% |
| Enhanced Baseline | Advanced Text + TF-IDF | 49.59% |
| Final Multimodal | Text + TF-IDF + Image Embeddings | **49.40%** |

This clearly demonstrates the incremental value added by advanced feature engineering and multimodal fusion.

---

