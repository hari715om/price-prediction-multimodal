import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# Add project root to path to find our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.feature_engineering import engineer_all_features

print("--- STARTING EMERGENCY PREDICTION SCRIPT ---")

# --- 1. Load Data ---
print("\n[1/5] Loading raw train and test data...")
train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')

# --- 2. Run Feature Engineering ---
# We need to run this on both train and test to get the text and numeric features
print("\n[2/5] Running feature engineering pipeline...")
train_features_df = engineer_all_features(train_df)
test_features_df = engineer_all_features(test_df)

# --- 3. Re-create the TF-IDF Vectorizer ---
# This is the critical step. We fit it on the training text.
print("\n[3/5] Re-creating TF-IDF features...")
tfidf_config = {
    'max_features': 50000,
    'ngram_range': (1, 2),
    'min_df': 3,
    'max_df': 0.9,
    'sublinear_tf': True
}
vectorizer = TfidfVectorizer(**tfidf_config)
# Fit on train data
print("   - Fitting vectorizer on training data...")
X_train_text = vectorizer.fit_transform(train_features_df['cleaned_text'].fillna(''))
# Transform test data
print("   - Transforming test data...")
X_test_text = vectorizer.transform(test_features_df['cleaned_text'].fillna(''))

# --- 4. Prepare Final Test Dataset ---
print("\n[4/5] Preparing final test dataset for prediction...")
# Load the scaler that was saved correctly
with open('outputs/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Use the same list of numeric columns
numeric_cols = [
    'content_len_chars', 'content_len_words', 'num_digits', 'num_special_chars', 'has_known_brand',
    'brand_count', 'category_score', 'premium_score', 'value_score', 'is_premium', 'is_value_pack',
    'ipq', 'ipq_confidence', 'weight_kg', 'volume_l', 'has_weight', 'has_volume', 'max_number',
    'min_number', 'avg_number', 'number_count', 'has_decimal', 'digit_ratio', 'special_char_ratio',
    'avg_word_length', 'ipq_x_weight', 'ipq_x_volume', 'text_len_per_ipq', 'words_per_ipq',
    'premium_x_brand', 'value_x_ipq', 'weight_volume_ratio'
]
# We need to handle a small typo fix if 'ip_x_volume' exists from an old version
if 'ip_x_volume' in test_features_df.columns and 'ipq_x_volume' not in numeric_cols:
    numeric_cols[numeric_cols.index('ipq_x_weight')+1] = 'ip_x_volume'
else:
    numeric_cols = [col for col in numeric_cols if col in test_features_df.columns]


X_test_numeric = scaler.transform(test_features_df[numeric_cols].fillna(0))

# Combine all test features
X_test = hstack([X_test_text, csr_matrix(X_test_numeric)])

# --- 5. Load Models and Predict ---
print("\n[5/5] Loading models and generating submission file...")
test_preds_log = np.zeros(len(test_df))
models = []
for fold in range(1, 6):
    model_path = f'outputs/models/baseline_lgbm_fold{fold}.txt'
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        models.append(model)
        print(f"   - Loading model from fold {fold}...")
        test_preds_log += model.predict(X_test)

if models:
    test_preds_log /= len(models)
    predictions = np.expm1(test_preds_log)
    
    submission = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': predictions})
    submission_path = 'outputs/submissions/EMERGENCY_submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print("\n" + "="*80)
    print("✅ EMERGENCY SUBMISSION FILE CREATED SUCCESSFULLY!")
    print(f"   File saved at: {submission_path}")
    print("   SUBMIT THIS FILE NOW.")
    print("="*80)
else:
    print("\nERROR: No trained models (.txt files) found in 'outputs/models/'.")