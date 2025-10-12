import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import os

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE)
    
    Formula: SMAPE = (1/n) * Σ |y_pred - y_true| / ((|y_true| + |y_pred|)/2)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        SMAPE value (between 0 and 2)
    """
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    smape_value = np.mean(numerator / denominator)
    
    return smape_value


def smape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate SMAPE as percentage (0-200%)
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        SMAPE percentage
    """
    return smape(y_true, y_pred) * 100


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all regression metrics
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'smape': float(smape(y_true, y_pred)),
        'smape_percent': float(smape_percent(y_true, y_pred)),
        'mae': float(mae(y_true, y_pred)),
        'rmse': float(rmse(y_true, y_pred)),
        'mape': float(mape(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }


def calculate_metrics_by_quantile(y_true: np.ndarray, y_pred: np.ndarray, 
                                  n_quantiles: int = 10) -> pd.DataFrame:
    """
    Calculate metrics by price quantiles
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_quantiles: Number of quantiles
    
    Returns:
        DataFrame with metrics per quantile
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Create quantiles
    df['quantile'] = pd.qcut(df['y_true'], q=n_quantiles, labels=False, duplicates='drop')
    
    # Calculate metrics per quantile
    results = []
    for q in sorted(df['quantile'].unique()):
        q_data = df[df['quantile'] == q]
        
        metrics = {
            'quantile': q,
            'n_samples': len(q_data),
            'price_min': q_data['y_true'].min(),
            'price_max': q_data['y_true'].max(),
            'price_mean': q_data['y_true'].mean(),
            'smape_percent': smape_percent(q_data['y_true'].values, q_data['y_pred'].values),
            'mae': mae(q_data['y_true'].values, q_data['y_pred'].values),
            'rmse': rmse(q_data['y_true'].values, q_data['y_pred'].values)
        }
        results.append(metrics)
    
    return pd.DataFrame(results)


class ExperimentLogger:
    """Log and track experiments"""
    
    def __init__(self, log_dir: str = '../outputs/metrics'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.log_dir / 'experiments_log.csv'
        
        # Initialize experiments log if doesn't exist
        if not self.experiments_file.exists():
            self._init_experiments_log()
    
    def _init_experiments_log(self):
        """Initialize experiments log CSV"""
        df = pd.DataFrame(columns=[
            'experiment_id',
            'timestamp',
            'model_name',
            'description',
            'smape_percent',
            'mae',
            'rmse',
            'r2',
            'config_path',
            'artifacts_path'
        ])
        df.to_csv(self.experiments_file, index=False)
    
    def log_experiment(self, 
                      experiment_name: str,
                      model_name: str,
                      metrics: Dict[str, float],
                      config: Dict = None,
                      description: str = '',
                      artifacts: Dict[str, str] = None) -> str:
        """
        Log a new experiment
        
        Args:
            experiment_name: Name/ID of experiment
            model_name: Model type (e.g., 'TF-IDF+LightGBM')
            metrics: Dictionary of metrics
            config: Model configuration
            description: Description of experiment
            artifacts: Dictionary of artifact paths
        
        Returns:
            experiment_id
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Save detailed metrics to JSON
        metrics_detail = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'description': description,
            'metrics': metrics,
            'config': config,
            'artifacts': artifacts
        }
        
        metrics_path = self.log_dir / f'{experiment_id}.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_detail, f, indent=2)
        
        # Append to experiments log
        log_entry = {
            'experiment_id': experiment_id,
            'timestamp': timestamp,
            'model_name': model_name,
            'description': description,
            'smape_percent': metrics.get('smape_percent', np.nan),
            'mae': metrics.get('mae', np.nan),
            'rmse': metrics.get('rmse', np.nan),
            'r2': metrics.get('r2', np.nan),
            'config_path': str(metrics_path),
            'artifacts_path': str(artifacts.get('model_dir', '')) if artifacts else ''
        }
        
        # Append to CSV
        df = pd.read_csv(self.experiments_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(self.experiments_file, index=False)
        
        print(f"\n✓ Logged experiment: {experiment_id}")
        print(f"✓ Detailed metrics saved: {metrics_path}")
        
        return experiment_id
    
    def get_best_experiments(self, metric: str = 'smape_percent', n: int = 10) -> pd.DataFrame:
        """Get top N experiments by metric"""
        df = pd.read_csv(self.experiments_file)
        
        if metric.lower() in ['smape', 'smape_percent', 'mae', 'rmse']:
            # Lower is better
            return df.nsmallest(n, metric)
        else:
            # Higher is better (e.g., r2)
            return df.nlargest(n, metric)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments"""
        df = pd.read_csv(self.experiments_file)
        return df[df['experiment_id'].isin(experiment_ids)]
    
    def get_experiment_details(self, experiment_id: str) -> Dict:
        """Get detailed information about an experiment"""
        # Find the JSON file
        json_files = list(self.log_dir.glob(f'{experiment_id}*.json'))
        
        if not json_files:
            raise ValueError(f"No experiment found with ID: {experiment_id}")
        
        with open(json_files[0], 'r') as f:
            return json.load(f)


class CVLogger:
    """Log cross-validation results"""
    
    def __init__(self):
        self.fold_results = []
    
    def log_fold(self, fold: int, metrics: Dict[str, float], 
                 model_path: str = None, predictions: np.ndarray = None):
        """Log results from a single fold"""
        fold_data = {
            'fold': fold,
            'metrics': metrics,
            'model_path': model_path,
            'predictions': predictions
        }
        self.fold_results.append(fold_data)
    
    def get_summary(self) -> Dict:
        """Get summary statistics across all folds"""
        if not self.fold_results:
            return {}
        
        # Extract metrics
        metrics_by_fold = {}
        for result in self.fold_results:
            for metric_name, metric_value in result['metrics'].items():
                if metric_name not in metrics_by_fold:
                    metrics_by_fold[metric_name] = []
                metrics_by_fold[metric_name].append(metric_value)
        
        # Calculate summary statistics
        summary = {}
        for metric_name, values in metrics_by_fold.items():
            summary[f'{metric_name}_mean'] = np.mean(values)
            summary[f'{metric_name}_std'] = np.std(values)
            summary[f'{metric_name}_min'] = np.min(values)
            summary[f'{metric_name}_max'] = np.max(values)
        
        summary['n_folds'] = len(self.fold_results)
        
        return summary
    
    def save_results(self, output_path: str):
        """Save CV results to file"""
        results_data = {
            'fold_results': [
                {
                    'fold': r['fold'],
                    'metrics': r['metrics'],
                    'model_path': r['model_path']
                }
                for r in self.fold_results
            ],
            'summary': self.get_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ Saved CV results: {output_path}")


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics\n" + "=" * 80)
    
    # Example data
    y_true = np.array([100, 200, 300, 400, 500])
    y_pred = np.array([110, 190, 320, 380, 510])
    
    print("\nTest Data:")
    print(f"True:      {y_true}")
    print(f"Predicted: {y_pred}")
    
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test experiment logger
    print("\n" + "=" * 80)
    print("Testing Experiment Logger")
    
    logger = ExperimentLogger(log_dir='../outputs/metrics')
    
    exp_id = logger.log_experiment(
        experiment_name='baseline_test',
        model_name='TF-IDF+LightGBM',
        metrics=metrics,
        config={'lr': 0.05, 'n_estimators': 1000},
        description='Test baseline model'
    )
    
    print(f"\nLogged experiment: {exp_id}")
    
    print("\n✅ Metrics test complete!")