import os
import sys

# Dynamically add the project root to PYTHONPATH
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pickle
import json
import os
from pathlib import Path
import sys
sys.path.append('..')
from src.data.preprocess_text import TextPreprocessor
from src.utils.metrics import smape, smape_percent
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
class BaselineModel:
    """TF-IDF + LightGBM baseline model for price prediction"""
    
    def __init__(self, config: dict = None):
        self.config = config or self.default_config()
        self.preprocessor = TextPreprocessor()
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.models = []  # List of models from each fold
        self.feature_importance = None
    
    @staticmethod
    def default_config():
        return {
            'tfidf': {
                'max_features': 50000,
                'ngram_range': (1, 2),
                'min_df': 3,
                'max_df': 0.9,
                'sublinear_tf': True
            },
            'lgbm': {
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
                'n_estimators': 5000,
                'early_stopping_rounds': 200,
                'verbose': -1,
                'random_state': 42,
                'n_jobs': -1
            },
            'cv': {
                'n_splits': 5,
                'random_state': 42
            }
        }
    
    def prepare_features(self, df: pd.DataFrame, is_train: bool = True) -> tuple:
        """
        Prepare text and numeric features
        
        Returns:
            (text_features_sparse, numeric_features_array, target_array or None)
        """
        print("\nExtracting text features...")
        df_features = self.preprocessor.process_dataframe(df)
        
        # TF-IDF features
        print("Creating TF-IDF features...")
        if is_train:
            self.tfidf_vectorizer = TfidfVectorizer(**self.config['tfidf'])
            text_features = self.tfidf_vectorizer.fit_transform(df_features['cleaned_text'].fillna(''))
        else:
            text_features = self.tfidf_vectorizer.transform(df_features['cleaned_text'].fillna(''))
        
        print(f"TF-IDF shape: {text_features.shape}")
        
        # Numeric features
        numeric_cols = [
            'ipq',
            'content_len_chars',
            'content_len_words',
            'title_len',
            'title_word_count',
            'title_caps_ratio',
            'num_digits',
            'num_special_chars',
            'has_ipq',
            'has_brand_candidate'
        ]
        
        numeric_features = df_features[numeric_cols].fillna(0).astype(float)
        
        # Scale numeric features
        if is_train:
            numeric_features_scaled = self.scaler.fit_transform(numeric_features)
        else:
            numeric_features_scaled = self.scaler.transform(numeric_features)
        
        # Target (log-transformed)
        if is_train and 'price' in df.columns:
            target = np.log1p(df['price'].values)
        else:
            target = None
        
        return text_features, numeric_features_scaled, target, df_features
    
    def train(self, train_df: pd.DataFrame, save_dir: str = '../outputs/models'):
        """Train baseline model with cross-validation"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("=" * 80)
        print("TRAINING BASELINE MODEL: TF-IDF + LightGBM")
        print("=" * 80)
        
        # Prepare features
        text_features, numeric_features, target, df_features = self.prepare_features(train_df, is_train=True)
        
        # Create price bins for stratified CV
        n_bins = min(10, len(train_df) // 1000)
        price_bins = pd.qcut(train_df['price'], q=n_bins, labels=False, duplicates='drop')
        
        # Initialize CV
        skf = StratifiedKFold(**self.config['cv'], shuffle=True)
        
        # Store OOF predictions
        oof_preds = np.zeros(len(train_df))
        oof_scores = []
        
        # Feature importance aggregation
        feature_importance_list = []
        
        print(f"\nTraining with {self.config['cv']['n_splits']}-fold CV...")
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, price_bins), 1):
            print(f"\n{'='*80}")
            print(f"FOLD {fold}/{self.config['cv']['n_splits']}")
            print(f"{'='*80}")
            
            # Split data
            X_train_text = text_features[train_idx]
            X_val_text = text_features[val_idx]
            X_train_num = numeric_features[train_idx]
            X_val_num = numeric_features[val_idx]
            y_train = target[train_idx]
            y_val = target[val_idx]
            
            # Combine features (convert sparse to dense for numeric concat)
            from scipy.sparse import hstack, csr_matrix
            X_train = hstack([X_train_text, csr_matrix(X_train_num)])
            X_val = hstack([X_val_text, csr_matrix(X_val_num)])
            
            # Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.config['lgbm'],
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=self.config['lgbm']['early_stopping_rounds']),
                    lgb.log_evaluation(period=100)
                ]
            )
            
            # Save model
            self.models.append(model)
            
            # Predict on validation set
            val_preds_log = model.predict(X_val, num_iteration=model.best_iteration)
            oof_preds[val_idx] = val_preds_log
            
            # Calculate SMAPE on original scale
            val_preds_original = np.expm1(val_preds_log)
            val_true_original = np.expm1(y_val)
            fold_smape = smape(val_true_original, val_preds_original)
            fold_smape_pct = smape_percent(val_true_original, val_preds_original)
            
            oof_scores.append(fold_smape)
            
            print(f"\nFold {fold} SMAPE: {fold_smape_pct:.4f}%")
            
            # Feature importance
            importance = model.feature_importance(importance_type='gain')
            feature_importance_list.append(importance)
        
        # Overall OOF score
        oof_preds_original = np.expm1(oof_preds)
        train_true_original = train_df['price'].values
        overall_smape = smape(train_true_original, oof_preds_original)
        overall_smape_pct = smape_percent(train_true_original, oof_preds_original)
        
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 80)
        print(f"Overall OOF SMAPE: {overall_smape_pct:.4f}%")
        print(f"Fold scores: {[f'{s*100:.4f}%' for s in oof_scores]}")
        print(f"Mean ± Std: {np.mean(oof_scores)*100:.4f}% ± {np.std(oof_scores)*100:.4f}%")
        print("=" * 80)
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            'sample_id': train_df['sample_id'],
            'price': train_true_original,
            'price_pred_oof': oof_preds_original
        })
        oof_df.to_csv(os.path.join(save_dir, 'baseline_oof.csv'), index=False)
        print(f"\n✓ Saved OOF predictions: {save_dir}/baseline_oof.csv")
        
        # Save models
        for fold, model in enumerate(self.models, 1):
            model.save_model(os.path.join(save_dir, f'baseline_lgbm_fold{fold}.txt'))
        print(f"✓ Saved {len(self.models)} fold models")
        
        # Save vectorizer and scaler
        with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        print("✓ Saved TF-IDF vectorizer and scaler")
        
        # Save metrics
        metrics = {
            'model': 'TF-IDF + LightGBM',
            'oof_smape': float(overall_smape),
            'oof_smape_percent': float(overall_smape_pct),
            'fold_scores': [float(s) for s in oof_scores],
            'mean_score': float(np.mean(oof_scores)),
            'std_score': float(np.std(oof_scores)),
            'n_folds': len(oof_scores),
            'config': self.config
        }
        
        with open(os.path.join(save_dir, 'baseline_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print("✓ Saved metrics")
        
        return overall_smape, oof_df
    
    def predict(self, test_df: pd.DataFrame, models_dir: str = '../outputs/models') -> np.ndarray:
        """Generate predictions using trained models"""
        
        print("\nGenerating predictions on test set...")
        
        # Prepare features
        text_features, numeric_features, _, _ = self.prepare_features(test_df, is_train=False)
        
        # Combine features
        from scipy.sparse import hstack, csr_matrix
        X_test = hstack([text_features, csr_matrix(numeric_features)])
        
        # Average predictions from all folds
        predictions_log = np.zeros(len(test_df))
        
        for fold, model in enumerate(self.models, 1):
            fold_preds = model.predict(X_test, num_iteration=model.best_iteration)
            predictions_log += fold_preds
            print(f"  Fold {fold} predictions generated")
        
        predictions_log /= len(self.models)
        
        # Convert back to original scale
        predictions = np.expm1(predictions_log)
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def load_models(self, models_dir: str = '../outputs/models'):
        """Load trained models, vectorizer, and scaler"""
        
        print("Loading trained models...")
        
        # Load vectorizer and scaler
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load fold models
        self.models = []
        fold = 1
        while True:
            model_path = os.path.join(models_dir, f'baseline_lgbm_fold{fold}.txt')
            if os.path.exists(model_path):
                model = lgb.Booster(model_file=model_path)
                self.models.append(model)
                fold += 1
            else:
                break
        
        print(f"✓ Loaded {len(self.models)} fold models")
        
        return self


def analyze_errors(oof_df: pd.DataFrame, train_df: pd.DataFrame, output_dir: str = '../outputs/analysis'):
    """Analyze prediction errors"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate errors
    oof_df['error'] = oof_df['price_pred_oof'] - oof_df['price']
    oof_df['abs_error'] = np.abs(oof_df['error'])
    oof_df['pct_error'] = (oof_df['abs_error'] / oof_df['price']) * 100
    
    # Sort by error
    worst_predictions = oof_df.nlargest(100, 'abs_error')
    
    # Merge with original data to get catalog content
    worst_predictions = worst_predictions.merge(
        train_df[['sample_id', 'catalog_content', 'image_link']], 
        on='sample_id', 
        how='left'
    )
    
    # Save worst predictions
    worst_predictions.to_csv(
        os.path.join(output_dir, 'worst_oof_predictions.csv'), 
        index=False
    )
    
    print(f"\n✓ Saved worst predictions: {output_dir}/worst_oof_predictions.csv")
    
    # Error statistics
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    print(f"Mean Absolute Error: ₹{oof_df['abs_error'].mean():.2f}")
    print(f"Median Absolute Error: ₹{oof_df['abs_error'].median():.2f}")
    print(f"Mean Percentage Error: {oof_df['pct_error'].mean():.2f}%")
    print(f"Median Percentage Error: {oof_df['pct_error'].median():.2f}%")
    
    # Error by price quantile
    oof_df['price_quantile'] = pd.qcut(oof_df['price'], q=10, labels=False, duplicates='drop')
    error_by_quantile = oof_df.groupby('price_quantile').agg({
        'price': ['min', 'max', 'mean'],
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).round(2)
    
    print("\nError by Price Quantile:")
    print(error_by_quantile)
    
    return worst_predictions


# ==================== MAIN SCRIPT ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train baseline TF-IDF + LightGBM model')
    parser.add_argument('--train', type=str, default='../data/raw/train.csv', help='Path to train.csv')
    parser.add_argument('--test', type=str, default='../data/raw/test.csv', help='Path to test.csv')
    parser.add_argument('--output', type=str, default='../outputs/models', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'both'], default='both')
    
    args = parser.parse_args()
    
    # Initialize model
    baseline = BaselineModel()
    
    if args.mode in ['train', 'both']:
        # Load training data
        print(f"Loading training data from {args.train}...")
        train_df = pd.read_csv(args.train)
        print(f"Train shape: {train_df.shape}")
        
        # Train model
        oof_smape, oof_df = baseline.train(train_df, save_dir=args.output)
        
        # Analyze errors
        worst_preds = analyze_errors(oof_df, train_df, output_dir='../outputs/analysis')
    
    if args.mode in ['predict', 'both']:
        # Load models if only predicting
        if args.mode == 'predict':
            baseline.load_models(models_dir=args.output)
        
        # Load test data
        print(f"\nLoading test data from {args.test}...")
        test_df = pd.read_csv(args.test)
        print(f"Test shape: {test_df.shape}")
        
        # Generate predictions
        predictions = baseline.predict(test_df, models_dir=args.output)
        
        # Create submission file
        submission = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'price': predictions
        })
        
        # Save submission
        submission_path = os.path.join(args.output, 'baseline_test_out.csv')
        submission.to_csv(submission_path, index=False)
        
        print(f"\n✓ Saved predictions: {submission_path}")
        print(f"✓ Number of predictions: {len(submission)}")
        print(f"✓ Price range: ₹{predictions.min():.2f} - ₹{predictions.max():.2f}")
        print(f"✓ Mean predicted price: ₹{predictions.mean():.2f}")
    
    print("\n" + "=" * 80)
    print("✅ PHASE D COMPLETE!")
    print("=" * 80)