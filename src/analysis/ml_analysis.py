"""
Machine Learning Analysis for Treatment Response Prediction

Uses Random Forest and XGBoost to predict treatment response based on
cell population frequencies.
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class ResponsePredictor:
    """Machine learning models to predict treatment response."""
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        Initialize predictor.
        
        Args:
            db_connection: Active SQLite connection
        """
        self.conn = db_connection
        self.scaler = StandardScaler()
        self.rf_model = None
        self.xgb_model = None
        self.feature_names = ['b_cell', 'cd8_t_cell', 'cd4_t_cell', 'nk_cell', 'monocyte']
    
    def prepare_data(
        self,
        indication: str = 'melanoma',
        treatment: str = 'miraclib',
        sample_type: str = 'PBMC'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for machine learning.
        
        Args:
            indication: Disease indication
            treatment: Treatment name
            sample_type: Sample type
            
        Returns:
            (X, y) tuple: Features and labels
        """
        query = """
        WITH total_counts AS (
            SELECT sample_id, SUM(count) as total_count
            FROM cell_counts
            GROUP BY sample_id
        ),
        pivoted_data AS (
            SELECT 
                s.sample_id,
                s.response,
                cp.population_name,
                (cc.count * 100.0 / tc.total_count) as percentage
            FROM cell_counts cc
            JOIN samples s ON cc.sample_id = s.sample_id
            JOIN cell_populations cp ON cc.population_id = cp.population_id
            JOIN total_counts tc ON cc.sample_id = tc.sample_id
            WHERE 
                s.indication = ?
                AND s.treatment = ?
                AND s.sample_type = ?
                AND s.response IN ('yes', 'no')
        )
        SELECT 
            sample_id,
            response,
            MAX(CASE WHEN population_name = 'b_cell' THEN percentage END) as b_cell,
            MAX(CASE WHEN population_name = 'cd8_t_cell' THEN percentage END) as cd8_t_cell,
            MAX(CASE WHEN population_name = 'cd4_t_cell' THEN percentage END) as cd4_t_cell,
            MAX(CASE WHEN population_name = 'nk_cell' THEN percentage END) as nk_cell,
            MAX(CASE WHEN population_name = 'monocyte' THEN percentage END) as monocyte
        FROM pivoted_data
        GROUP BY sample_id, response
        """
        
        df = pd.read_sql_query(
            query,
            self.conn,
            params=(indication, treatment, sample_type)
        )
        
        # Prepare features and labels
        X = df[self.feature_names]
        y = (df['response'] == 'yes').astype(int)  # 1 for responders, 0 for non-responders
        
        return X, y
    
    def train_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.3,
        random_state: int = 42
    ) -> Dict:
        """
        Train Random Forest and XGBoost models.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Dictionary with results for both models
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # ═══════════════════════════════════════════════════════════
        # RANDOM FOREST
        # ═══════════════════════════════════════════════════════════
        print("\nTraining Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight='balanced'
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Predictions
        rf_train_pred = self.rf_model.predict(X_train_scaled)
        rf_test_pred = self.rf_model.predict(X_test_scaled)
        rf_test_proba = self.rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.rf_model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='roc_auc'
        )
        
        results['random_forest'] = {
            'model': self.rf_model,
            'train_accuracy': accuracy_score(y_train, rf_train_pred),
            'test_accuracy': accuracy_score(y_test, rf_test_pred),
            'test_precision': precision_score(y_test, rf_test_pred),
            'test_recall': recall_score(y_test, rf_test_pred),
            'test_f1': f1_score(y_test, rf_test_pred),
            'test_roc_auc': roc_auc_score(y_test, rf_test_proba),
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'predictions': rf_test_pred,
            'probabilities': rf_test_proba,
            'feature_importance': dict(zip(self.feature_names, self.rf_model.feature_importances_)),
            'confusion_matrix': confusion_matrix(y_test, rf_test_pred),
            'y_test': y_test
        }
        
        # ═══════════════════════════════════════════════════════════
        # XGBOOST
        # ═══════════════════════════════════════════════════════════
        print("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        )
        
        self.xgb_model.fit(X_train_scaled, y_train)
        
        # Predictions
        xgb_train_pred = self.xgb_model.predict(X_train_scaled)
        xgb_test_pred = self.xgb_model.predict(X_test_scaled)
        xgb_test_proba = self.xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.xgb_model, X_train_scaled, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring='roc_auc'
        )
        
        results['xgboost'] = {
            'model': self.xgb_model,
            'train_accuracy': accuracy_score(y_train, xgb_train_pred),
            'test_accuracy': accuracy_score(y_test, xgb_test_pred),
            'test_precision': precision_score(y_test, xgb_test_pred),
            'test_recall': recall_score(y_test, xgb_test_pred),
            'test_f1': f1_score(y_test, xgb_test_pred),
            'test_roc_auc': roc_auc_score(y_test, xgb_test_proba),
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'predictions': xgb_test_pred,
            'probabilities': xgb_test_proba,
            'feature_importance': dict(zip(self.feature_names, self.xgb_model.feature_importances_)),
            'confusion_matrix': confusion_matrix(y_test, xgb_test_pred),
            'y_test': y_test
        }
        
        # Store split data for visualization
        results['split_data'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        return results
    
    def print_results(self, results: Dict) -> None:
        """Print formatted results."""
        print("\n" + "="*80)
        print("MACHINE LEARNING RESULTS: RESPONSE PREDICTION")
        print("="*80)
        
        print(f"\nDataset Size:")
        print(f"  Training samples: {len(results['split_data']['y_train'])}")
        print(f"  Test samples: {len(results['split_data']['y_test'])}")
        print(f"  Responders: {results['split_data']['y_test'].sum()}")
        print(f"  Non-responders: {len(results['split_data']['y_test']) - results['split_data']['y_test'].sum()}")
        
        for model_name in ['random_forest', 'xgboost']:
            model_results = results[model_name]
            
            print(f"\n{'─'*80}")
            print(f"{model_name.upper().replace('_', ' ')}")
            print(f"{'─'*80}")
            
            print(f"\nPerformance Metrics:")
            print(f"  Train Accuracy: {model_results['train_accuracy']:.4f}")
            print(f"  Test Accuracy:  {model_results['test_accuracy']:.4f}")
            print(f"  Precision:      {model_results['test_precision']:.4f}")
            print(f"  Recall:         {model_results['test_recall']:.4f}")
            print(f"  F1-Score:       {model_results['test_f1']:.4f}")
            print(f"  ROC-AUC:        {model_results['test_roc_auc']:.4f}")
            print(f"\n  Cross-Val ROC-AUC: {model_results['cv_roc_auc_mean']:.4f} ± {model_results['cv_roc_auc_std']:.4f}")
            
            print(f"\nFeature Importance:")
            sorted_features = sorted(
                model_results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in sorted_features:
                print(f"  {feature:15s}: {importance:.4f}")
            
            print(f"\nConfusion Matrix:")
            cm = model_results['confusion_matrix']
            print(f"  [[TN={cm[0,0]:3d}  FP={cm[0,1]:3d}]")
            print(f"   [FN={cm[1,0]:3d}  TP={cm[1,1]:3d}]]")
        
        print("\n" + "="*80 + "\n")
    
    def create_visualizations(self, results: Dict, save_dir: str = "outputs") -> None:
        """Create visualization plots."""
        from pathlib import Path
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Create a 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Machine Learning Results: Treatment Response Prediction', fontsize=16, fontweight='bold')
        
        # ═══════════════════════════════════════════════════════════
        # Plot 1: ROC Curves
        # ═══════════════════════════════════════════════════════════
        ax = axes[0, 0]
        
        for model_name, color, label in [
            ('random_forest', 'blue', 'Random Forest'),
            ('xgboost', 'red', 'XGBoost')
        ]:
            y_test = results[model_name]['y_test']
            y_proba = results[model_name]['probabilities']
            
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = results[model_name]['test_roc_auc']
            
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # ═══════════════════════════════════════════════════════════
        # Plot 2: Feature Importance Comparison
        # ═══════════════════════════════════════════════════════════
        ax = axes[0, 1]
        
        rf_importance = pd.Series(results['random_forest']['feature_importance']).sort_values()
        xgb_importance = pd.Series(results['xgboost']['feature_importance']).sort_values()
        
        x = np.arange(len(self.feature_names))
        width = 0.35
        
        # Sort by RF importance
        sorted_features = rf_importance.index
        
        ax.barh(x - width/2, [rf_importance[f] for f in sorted_features], width, label='Random Forest', color='blue', alpha=0.7)
        ax.barh(x + width/2, [xgb_importance[f] for f in sorted_features], width, label='XGBoost', color='red', alpha=0.7)
        
        ax.set_ylabel('Cell Population', fontsize=11)
        ax.set_xlabel('Feature Importance', fontsize=11)
        ax.set_title('Feature Importance Comparison', fontsize=12, fontweight='bold')
        ax.set_yticks(x)
        ax.set_yticklabels([f.replace('_', ' ').title() for f in sorted_features])
        ax.legend()
        ax.grid(True, axis='x', alpha=0.3)
        
        # ═══════════════════════════════════════════════════════════
        # Plot 3: Confusion Matrices
        # ═══════════════════════════════════════════════════════════
        for idx, (model_name, title) in enumerate([
            ('random_forest', 'Random Forest'),
            ('xgboost', 'XGBoost')
        ]):
            ax = axes[1, idx]
            
            cm = results[model_name]['confusion_matrix']
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Responder', 'Responder'],
                yticklabels=['Non-Responder', 'Responder'],
                cbar_kws={'label': 'Count'}
            )
            
            ax.set_xlabel('Predicted', fontsize=11)
            ax.set_ylabel('Actual', fontsize=11)
            ax.set_title(f'{title} - Confusion Matrix', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{save_dir}/ml_analysis_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    # Test ML analysis
    import sqlite3
    
    print("Testing ML predictor...")
    
    conn = sqlite3.connect("data/processed/loblaw_trial.db")
    predictor = ResponsePredictor(conn)
    
    # Prepare data
    print("\n1. Preparing data...")
    X, y = predictor.prepare_data()
    print(f"   ✓ Features shape: {X.shape}")
    print(f"   ✓ Labels: {y.sum()} responders, {len(y) - y.sum()} non-responders")
    
    # Train models
    print("\n2. Training models...")
    results = predictor.train_models(X, y)
    
    # Print results
    predictor.print_results(results)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    predictor.create_visualizations(results)
    
    conn.close()
    print("\n✓ ML analysis test complete!")