import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import pickle
from tqdm import tqdm

from src.data.feature_engineering import engineer_all_features
from src.utils.metrics import smape

print("="*80)
print("MULTIMODAL MODEL: Text + Image + Tabular Features")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")
train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================================
# 2. EXTRACT TEXT FEATURES
# ============================================================================
print("\n[2/7] Extracting text features (IPQ, weights, volumes)...")
train_df = engineer_all_features(train_df)
test_df = engineer_all_features(test_df)

print(f"✓ IPQ extracted - Mean: {train_df['ipq'].mean():.2f}, Max: {train_df['ipq'].max()}")

# ============================================================================
# 3. LOAD IMAGE EMBEDDINGS
# ============================================================================
print("\n[3/7] Loading image embeddings...")

train_emb_index = pd.read_csv('data/processed/embedding_index_train.csv')
test_emb_index = pd.read_csv('data/processed/embedding_index_test.csv')

def load_embedding(sample_id, dataset='train'):
    """Load embedding for a sample, return zeros if missing."""
    emb_path = Path(f'data/processed/embeddings/{dataset}/{sample_id}.npy')
    if emb_path.exists():
        return np.load(emb_path)
    else:
        return np.zeros(1280)  # EfficientNet-B0 output dim

print("Loading train embeddings...")
train_embeddings = np.array([
    load_embedding(sid, 'train') 
    for sid in tqdm(train_df['sample_id'], desc="Train")
])

print("Loading test embeddings...")
test_embeddings = np.array([
    load_embedding(sid, 'test') 
    for sid in tqdm(test_df['sample_id'], desc="Test")
])

print(f"✓ Train embeddings shape: {train_embeddings.shape}")
print(f"✓ Test embeddings shape: {test_embeddings.shape}")

# ============================================================================
# 4. CREATE TF-IDF FEATURES
# ============================================================================
print("\n[4/7] Creating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=30000,  # Reduced from 50k to save memory
    ngram_range=(1, 2),
    min_df=3,
    dtype=np.float32
)

X_train_tfidf = vectorizer.fit_transform(train_df['catalog_content'])
X_test_tfidf = vectorizer.transform(test_df['catalog_content'])

print(f"✓ TF-IDF shape: {X_train_tfidf.shape}")

# ============================================================================
# 5. CREATE TABULAR FEATURES
# ============================================================================
print("\n[5/7] Creating tabular features...")

tabular_features = [
    # Basic Features
    'content_len_chars', 'content_len_words', 'num_digits', 'num_special_chars',

    # Brand Features
    'has_known_brand', 'brand_count',

    # Category Features
    'category_score',

    # Premium/Value Features
    'premium_score', 'value_score', 'is_premium', 'is_value_pack',

    # Enhanced IPQ Features
    'ipq', 'ipq_confidence',

    # Weight/Volume Features
    'weight_kg', 'volume_l', 'has_weight', 'has_volume',

    # Numeric Indicator Features
    'max_number', 'min_number', 'avg_number', 'number_count', 'has_decimal',

    # Length Ratio Features
    'digit_ratio', 'special_char_ratio', 'avg_word_length',

    # Interaction Features
    'ipq_x_weight', 'ipq_x_volume', 'text_len_per_ipq', 'words_per_ipq',
    'premium_x_brand', 'value_x_ipq', 'weight_volume_ratio'
]

X_train_tabular = train_df[tabular_features].values.astype(np.float32)
X_test_tabular = test_df[tabular_features].values.astype(np.float32)

# Scale tabular features
scaler = StandardScaler()
X_train_tabular = scaler.fit_transform(X_train_tabular)
X_test_tabular = scaler.transform(X_test_tabular)

print(f"✓ Tabular features shape: {X_train_tabular.shape}")

# ============================================================================
# 6. CONCATENATE ALL FEATURES
# ============================================================================
print("\n[6/7] Concatenating multimodal features...")

# TF-IDF (sparse) + Image embeddings (dense) + Tabular (dense)
X_train_dense = np.hstack([train_embeddings, X_train_tabular])
X_test_dense = np.hstack([test_embeddings, X_test_tabular])

X_train = hstack([X_train_tfidf, csr_matrix(X_train_dense)])
X_test = hstack([X_test_tfidf, csr_matrix(X_test_dense)])

print(f"✓ Final feature shape: {X_train.shape}")
print(f"  - TF-IDF: {X_train_tfidf.shape[1]} dims")
print(f"  - Image embeddings: 1280 dims")
print(f"  - Tabular: {len(tabular_features)} dims")

# Target
y_train = np.log1p(train_df['price'].values)

# ============================================================================
# 7. TRAIN WITH 5-FOLD CV
# ============================================================================
print("\n[7/7] Training multimodal model with 5-fold CV...")

# Create stratified folds based on price bins
train_df['price_bin'] = pd.qcut(train_df['price'], q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': -1
}

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
fold_scores = []
models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, train_df['price_bin']), 1):
    print(f"\n{'='*80}")
    print(f"FOLD {fold}/5")
    print(f"{'='*80}")
    
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=5000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(100),
            lgb.early_stopping(200)
        ]
    )
    
    # OOF predictions
    oof_preds[val_idx] = model.predict(X_val)
    
    # Test predictions
    test_preds += model.predict(X_test) / 5
    
    # Calculate SMAPE
    fold_smape = smape(
        np.expm1(y_val),
        np.expm1(oof_preds[val_idx])
    )
    fold_scores.append(fold_smape)
    print(f"\nFold {fold} SMAPE: {fold_smape:.4f}%")
    
    models.append(model)

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)

oof_smape = smape(train_df['price'].values, np.expm1(oof_preds))
print(f"Overall OOF SMAPE: {oof_smape:.4f}%")
print(f"Fold scores: {[f'{s:.4f}%' for s in fold_scores]}")
print(f"Mean ± Std: {np.mean(fold_scores):.4f}% ± {np.std(fold_scores):.4f}%")
print("="*80)

# ============================================================================
# SAVE ARTIFACTS
# ============================================================================
output_dir = Path('outputs/models/multimodal')
output_dir.mkdir(parents=True, exist_ok=True)

# Save OOF predictions
oof_df = pd.DataFrame({
    'sample_id': train_df['sample_id'],
    'price': train_df['price'],
    'price_pred': np.expm1(oof_preds)
})
oof_df.to_csv(output_dir / 'multimodal_oof.csv', index=False)
print(f"\n✓ Saved OOF predictions: {output_dir / 'multimodal_oof.csv'}")

# Save models
for i, model in enumerate(models, 1):
    model.save_model(str(output_dir / f'multimodal_fold{i}.txt'))
print(f"✓ Saved {len(models)} fold models")

# Save preprocessing artifacts
with open(output_dir / 'vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open(output_dir / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Saved vectorizer and scaler")

# Save test predictions
test_df['price'] = np.expm1(test_preds)
test_df[['sample_id', 'price']].to_csv('outputs/submissions/test_out_multimodal.csv', index=False)
print(f"✓ Saved test predictions: outputs/submissions/test_out_multimodal.csv")

print("\n" + "="*80)
print("✅ MULTIMODAL TRAINING COMPLETE!")
print("="*80)