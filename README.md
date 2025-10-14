Amazon ML Challenge 2025: Multimodal Product Price Prediction

This repository contains our team’s solution for the Amazon ML Challenge 2025, where the goal was to predict product prices using both text descriptions (catalog_content) and product images (image_link).

Our approach followed a phased strategy, beginning with a robust text-only baseline and extending to a multimodal model that fuses text and image features.

Final Multimodal OOF SMAPE: 49.40%

Project Summary

This project reflects a 3-day journey of rapid experimentation, debugging, and strategic pivots under pressure.
We built a complete, reproducible end-to-end pipeline — from raw data to multimodal prediction — blending NLP, Computer Vision, and classical ML.

Key Highlights:

• Feature Engineering: Extracted structured features like IPQ, brand names, and weights from noisy text.

• Hybrid Cloud/Local Setup: Used Google Colab (T4 GPU) for image embeddings and local CPU for LightGBM training.

• Cross-Validation: Implemented a stable 5-fold CV pipeline ensuring robust metrics.

• Performance: Improved SMAPE from ~50.01% → 49.59% → 49.40%, validating both text and multimodal improvements.

• Technical Approach

Our solution follows a modular design, processing text and images separately and merging their representations for final prediction.

1. Feature Engineering (src/data/feature_engineering.py)

Extracted meaningful signals from messy product text.

Basic: content_length, word_count, digit_count

Advanced:

Brand detection via keyword matching

Category heuristics (food, beauty, electronics, etc.)

IPQ & weight extraction using regex-based parsing

“Premium” keyword scoring

Interaction features (e.g., ipq * weight_kg, len/ipq)

2. Image Embeddings (scripts/extract_image_embeddings.py)

Processed 150K images using a pre-trained EfficientNet-B0 model.

Used as a feature extractor (1280-D embeddings).

Computed efficiently on Google Colab (T4 GPU).

Saved embeddings for fusion with tabular/text data.

3. Modeling (scripts/train_multimodal.py)

Combined TF-IDF, engineered features, and image embeddings.

Model: LightGBM (efficient for tabular + sparse data)

Fusion: Early concatenation of all features

Validation: 5-Fold CV with early stopping

Output: OOF predictions, test submission, and performance logs

📊 Results
Model Version	Key Features	OOF SMAPE
Baseline	TF-IDF + Basic Text Features	~50.01%
Enhanced Text	Advanced Text + TF-IDF	49.59%
Final Multimodal	Text + TF-IDF + Image Embeddings	49.40%

Feature engineering provided the largest performance boost, while image features added measurable improvement.

⚙️ How to Run
# 1. Clone repository
git clone <repo_url>
cd amazon-ml-challenge-2025

# 2. Setup environment
pip install -r requirements.txt

# 3. Prepare data
# Place train.csv and test.csv inside data/raw/

# 4. (Optional) Download product images
python scripts/download_images.py --data data/raw/train.csv --output data/raw/train_images
python scripts/download_images.py --data data/raw/test.csv --output data/raw/test_images

# 5. Extract image embeddings (GPU recommended)
python scripts/extract_image_embeddings.py

# 6. Train multimodal model and generate predictions
python scripts/train_multimodal.py


Output files (OOF, metrics, and test predictions) will be saved in:

outputs/models/
outputs/submissions/

👥 Team

Team Members:

Hari Om Singh
Ayush Raj
Aarohan Garg

🧩 Acknowledgment

Special thanks to Amazon ML Challenge 2025 organizers for creating a platform that blended real-world problem solving with AI innovation.
