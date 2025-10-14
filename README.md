Amazon ML Challenge 2025: Multimodal Product Price Prediction
This repository contains our team's solution for the Amazon ML Challenge 2025. The goal was to predict product prices using a multimodal dataset consisting of text descriptions (catalog_content) and product images (image_link). Our approach involved a phased strategy, starting with a robust text-only baseline and progressing to a final multimodal model that incorporates both text and image features.

Final Multimodal OOF SMAPE: 49.40%

🚀 Project Summary
This project tells a story of rapid development, strategic adaptation, and real-world problem-solving under a tight deadline. We successfully built a complete end-to-end pipeline, from data ingestion and cleaning to advanced feature engineering, model training, and cloud-based deep learning.

Key achievements include:

Advanced Feature Engineering: Developed a comprehensive pipeline to extract features like Item Pack Quantity (IPQ), brand names, product categories, and weight/volume from noisy text data.

Hybrid Cloud/Local Workflow: Strategically used Google Colab's T4 GPU for the computationally intensive task of extracting image embeddings and leveraged a powerful local multi-core CPU for the Gradient Boosting model training.

Robust Modeling: Implemented a stable 5-fold cross-validation strategy to train and validate our models, ensuring reliable performance estimates.

Significant Performance Gains: Improved our model's SMAPE score from an initial baseline of ~50.01% to a final multimodal score of ~49.40%, proving the value of our feature engineering and the inclusion of image data.

🛠️ Technical Approach
Our solution is built on a modular pipeline that processes text and image data separately before fusing them for a final prediction.

1. Feature Engineering (src/data/feature_engineering.py)
This was the cornerstone of our performance improvement. We built a single, powerful pipeline that applies the following transformations:

Basic Text Features: content_length, word_count, digit_count, etc.

Advanced Text Features:

Brand Detection: Identified known brands from a predefined list.

Category Heuristics: Classified products into categories like food, beauty, etc., based on keywords.

Premium/Value Indicators: Scored text based on words like organic, premium, combo, and value pack.

Enhanced IPQ Extraction: Used a prioritized list of regular expressions to robustly extract the Item Pack Quantity.

Weight/Volume Extraction: Normalized weights to kilograms and volumes to liters.

Interaction Features: Created new features by combining existing ones (e.g., ipq * weight_kg, text_len / ipq).

2. Image Feature Extraction (scripts/extract_image_embeddings.py)
To process the 150,000 images, we used a pre-trained deep learning model.

Model: EfficientNet-B0 (pre-trained on ImageNet).

Process: The model was used as a feature extractor. The final classification layer was removed, and the model was used to generate a 1280-dimensional embedding (vector) for each product image.

Execution: This entire process was run on a Google Colab T4 GPU for efficiency, which turned a multi-day task into a multi-hour one.

3. Modeling (scripts/train_multimodal.py)
Our final model combines all engineered features.

Algorithm: LightGBM (Light Gradient Boosting Machine), which is highly effective for tabular and mixed-density data.

Features Used:

TF-IDF Features: A sparse matrix of 30,000 text features (ngram_range=(1, 2)).

Image Embeddings: A dense matrix of 1280 features from EfficientNet-B0.

Tabular Features: A dense matrix of ~32 advanced numerical features from our engineering pipeline.

Strategy: We used an Early Fusion approach, concatenating all three feature types into a single wide dataset before training the LightGBM model with 5-fold cross-validation.

📊 Results
Our phased approach allowed us to measure the impact of each major component:

Model Version

Key Features

OOF SMAPE

Initial Baseline

Basic Text Features + TF-IDF

~50.01%

Enhanced Baseline

Advanced Text Features + TF-IDF

49.59%

Final Multimodal

Enhanced Text + TF-IDF + Image Embeddings

49.40%

This clearly demonstrates that our feature engineering provided a significant boost, and the addition of image embeddings provided a further, measurable improvement.

⚙️ How to Run This Project
Setup:

Clone the repository: git clone <repo_url>

Create and activate a virtual environment.

Install dependencies: pip install -r requirements.txt

Download Data:

Place train.csv and test.csv in the data/raw/ directory.

Run the image downloader script to fetch all product images:

python scripts/download_images.py --data data/raw/train.csv --output data/raw/train_images
python scripts/download_images.py --data data/raw/test.csv --output data/raw/test_images

Extract Image Embeddings:

This step is computationally intensive and is best run on a machine with a GPU (e.g., Google Colab).

python scripts/extract_image_embeddings.py

Train the Final Model & Generate Submission:

Run the multimodal training script. This will perform all feature engineering and train the 5-fold model.

python scripts/train_multimodal.py

The final submission file will be saved to outputs/submissions/test_out_multimodal.csv.
