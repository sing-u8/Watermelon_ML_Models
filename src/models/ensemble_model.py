#!/usr/bin/env python3
"""
Ensemble models for watermelon sweetness prediction.

This module implements various ensemble strategies:
- Simple Voting Regressor (equal weights)
- Weighted Average (performance-based weights)
- Stacking (meta-learner approach)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import joblib
import logging
from pathlib import Path
import yaml

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import KFold

# Import our traditional ML models
from src.models.traditional_ml import WatermelonGBT, WatermelonSVM, WatermelonRandomForest

logger = logging.getLogger(__name__)


class WatermelonEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble model for watermelon sweetness prediction.
    
    Combines multiple traditional ML models using various strategies:
    - voting: Simple average of predictions
    - weighted: Weighted average based on validation performance
    - stacking: Meta-learner trained on base model predictions
    """
    
    def __init__(self, 
                 ensemble_strategy: str = 'weighted',
                 meta_learner: str = 'ridge',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize WatermelonEnsemble.
        
        Args:
            ensemble_strategy: 'voting', 'weighted', or 'stacking'
            meta_learner: For stacking - 'linear', 'ridge', 'lasso'
            cv_folds: Number of CV folds for weight calculation
            random_state: Random seed for reproducibility
        """
        self.ensemble_strategy = ensemble_strategy
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize base models
        self.base_models = self._initialize_base_models()
        self.model_weights_ = None
        self.meta_model_ = None
        self.is_fitted_ = False
        
    def _initialize_base_models(self) -> Dict[str, BaseEstimator]:
        """Initialize base models with optimal hyperparameters."""
        # Use optimized hyperparameters from tuning results
        rf_config = {
            'model': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt'
            }
        }
        
        gbt_config = {
            'model': {
                'n_estimators': 200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8
            }
        }
        
        svm_config = {
            'model': {
                'kernel': 'rbf',
                'C': 10,
                'gamma': 'scale',
                'epsilon': 0.01
            }
        }
        
        base_models = {
            'random_forest': WatermelonRandomForest(config=rf_config, random_state=self.random_state),
            'gradient_boosting': WatermelonGBT(config=gbt_config, random_state=self.random_state),
            'svm': WatermelonSVM(config=svm_config, random_state=self.random_state)
        }
        return base_models
    
    def _get_meta_learner(self) -> BaseEstimator:
        """Get meta-learner for stacking."""
        if self.meta_learner == 'linear':
            return LogisticRegression(random_state=self.random_state)
        elif self.meta_learner == 'ridge':
            return RidgeClassifier(alpha=1.0, random_state=self.random_state)
        elif self.meta_learner == 'lasso':
            return SGDClassifier(loss='log_loss', penalty='l1', alpha=0.1, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown meta_learner: {self.meta_learner}")
    
    def _calculate_model_weights(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate model weights based on cross-validation performance."""
        logger.info("모델 가중치 계산 중...")
        
        cv_scores = {}
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for name, model in self.base_models.items():
            logger.info(f"  {name} 교차 검증 중...")
            # Use accuracy for scoring (higher is better)
            scores = cross_val_score(model, X, y, cv=kfold, 
                                   scoring='accuracy', n_jobs=-1)
            cv_scores[name] = np.mean(scores)  # Accuracy score
            logger.info(f"    CV Accuracy: {cv_scores[name]:.4f}")
        
        # Calculate weights proportional to accuracy
        # Better performance (higher accuracy) gets higher weight
        total_accuracy = sum(cv_scores.values())
        weights = {name: acc / total_accuracy for name, acc in cv_scores.items()}
        
        logger.info("계산된 모델 가중치:")
        for name, weight in weights.items():
            logger.info(f"  {name}: {weight:.4f}")
        
        return weights
    
    def _generate_stacking_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate features for stacking using cross-validation."""
        logger.info("스태킹을 위한 특징 생성 중...")
        
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        stacking_features = np.zeros((n_samples, n_models))
        
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        for i, (name, model) in enumerate(self.base_models.items()):
            logger.info(f"  {name} 예측 생성 중...")
            
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # Train model on training fold
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                # Predict on validation fold
                val_pred = model_clone.predict(X_val)
                stacking_features[val_idx, i] = val_pred
        
        return stacking_features
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WatermelonEnsemble':
        """
        Fit the ensemble model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            self: Fitted ensemble model
        """
        logger.info(f"🤖 {self.ensemble_strategy} 앙상블 모델 훈련 시작")
        
        # Fit all base models
        logger.info("기본 모델들 훈련 중...")
        for name, model in self.base_models.items():
            logger.info(f"  {name} 훈련 중...")
            model.fit(X, y)
        
        if self.ensemble_strategy == 'voting':
            # Simple voting - equal weights
            self.model_weights_ = {name: 1.0 / len(self.base_models) 
                                 for name in self.base_models.keys()}
            
        elif self.ensemble_strategy == 'weighted':
            # Weighted average based on CV performance
            self.model_weights_ = self._calculate_model_weights(X, y)
            
        elif self.ensemble_strategy == 'stacking':
            # Stacking with meta-learner
            stacking_features = self._generate_stacking_features(X, y)
            self.meta_model_ = self._get_meta_learner()
            self.meta_model_.fit(stacking_features, y)
            logger.info(f"메타 모델 ({self.meta_learner}) 훈련 완료")
            
        else:
            raise ValueError(f"Unknown ensemble_strategy: {self.ensemble_strategy}")
        
        self.is_fitted_ = True
        logger.info("앙상블 모델 훈련 완료")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all base models
        base_predictions = {}
        for name, model in self.base_models.items():
            base_predictions[name] = model.predict(X)
        
        if self.ensemble_strategy in ['voting', 'weighted']:
            # Weighted voting for classification
            # For binary classification, we'll use probability predictions
            predictions = np.zeros(X.shape[0])
            for name, pred in base_predictions.items():
                predictions += self.model_weights_[name] * pred
            # Round to nearest integer for classification
            return np.round(predictions).astype(int)
            
        elif self.ensemble_strategy == 'stacking':
            # Use meta-learner for final prediction
            stacking_features = np.column_stack(list(base_predictions.values()))
            return self.meta_model_.predict(stacking_features)
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from base models that support it."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted first")
        
        importance_dict = {}
        
        # Random Forest importance
        if hasattr(self.base_models['random_forest'], 'feature_importances_'):
            importance_dict['random_forest'] = self.base_models['random_forest'].feature_importances_
        
        # Gradient Boosting importance
        if hasattr(self.base_models['gradient_boosting'], 'feature_importances_'):
            importance_dict['gradient_boosting'] = self.base_models['gradient_boosting'].feature_importances_
        
        return importance_dict
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble configuration."""
        info = {
            'ensemble_strategy': self.ensemble_strategy,
            'meta_learner': self.meta_learner if self.ensemble_strategy == 'stacking' else None,
            'base_models': list(self.base_models.keys()),
            'is_fitted': self.is_fitted_
        }
        
        if self.model_weights_ is not None:
            info['model_weights'] = self.model_weights_
            
        return info


class EnsembleTrainer:
    """Trainer class for ensemble models with comprehensive evaluation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize EnsembleTrainer.
        
        Args:
            config_path: Path to ensemble configuration file
        """
        self.config = self._load_config(config_path)
        self.ensemble_models = {}
        self.evaluation_results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load ensemble configuration."""
        default_config = {
            'ensemble_strategies': ['voting', 'weighted', 'stacking'],
            'meta_learners': ['ridge', 'linear', 'lasso'],
            'cv_folds': 5,
            'random_state': 42
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def train_all_ensembles(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, WatermelonEnsemble]:
        """
        Train all ensemble configurations.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dict of trained ensemble models
        """
        logger.info("🚀 모든 앙상블 구성 훈련 시작")
        
        ensemble_configs = []
        
        # Add voting and weighted ensembles
        for strategy in ['voting', 'weighted']:
            if strategy in self.config['ensemble_strategies']:
                ensemble_configs.append({
                    'name': f'ensemble_{strategy}',
                    'strategy': strategy,
                    'meta_learner': None
                })
        
        # Add stacking ensembles with different meta-learners
        if 'stacking' in self.config['ensemble_strategies']:
            for meta_learner in self.config['meta_learners']:
                ensemble_configs.append({
                    'name': f'ensemble_stacking_{meta_learner}',
                    'strategy': 'stacking',
                    'meta_learner': meta_learner
                })
        
        # Train each ensemble configuration
        for config in ensemble_configs:
            logger.info(f"\n=== {config['name']} 훈련 중 ===")
            
            ensemble = WatermelonEnsemble(
                ensemble_strategy=config['strategy'],
                meta_learner=config['meta_learner'],
                cv_folds=self.config['cv_folds'],
                random_state=self.config['random_state']
            )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_pred = ensemble.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
            
            logger.info(f"검증 성능 - 정확도: {val_accuracy:.4f}, F1-score: {val_f1:.4f}")
            
            # Store results
            self.ensemble_models[config['name']] = ensemble
            self.evaluation_results[config['name']] = {
                'val_accuracy': val_accuracy,
                'val_f1_score': val_f1,
                'config': config
            }
        
        # Find best ensemble
        best_ensemble_name = max(self.evaluation_results.keys(),
                               key=lambda k: self.evaluation_results[k]['val_accuracy'])
        
        logger.info(f"\n🏆 최고 앙상블: {best_ensemble_name}")
        logger.info(f"   정확도: {self.evaluation_results[best_ensemble_name]['val_accuracy']:.4f}")
        logger.info(f"   F1-score: {self.evaluation_results[best_ensemble_name]['val_f1_score']:.4f}")
        
        return self.ensemble_models
    
    def evaluate_on_test(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all ensembles on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict of test evaluation results
        """
        logger.info("📊 테스트 세트 평가 중...")
        
        test_results = {}
        
        for name, ensemble in self.ensemble_models.items():
            test_pred = ensemble.predict(X_test)
            
            test_accuracy = accuracy_score(y_test, test_pred)
            test_f1 = f1_score(y_test, test_pred, average='weighted')
            test_precision = precision_score(y_test, test_pred, average='weighted')
            test_recall = recall_score(y_test, test_pred, average='weighted')
            
            test_results[name] = {
                'test_accuracy': test_accuracy,
                'test_f1_score': test_f1,
                'test_precision': test_precision,
                'test_recall': test_recall
            }
            
            logger.info(f"{name}: 정확도={test_accuracy:.4f}, F1-score={test_f1:.4f}")
        
        return test_results
    
    def save_best_ensemble(self, save_dir: str) -> str:
        """
        Save the best performing ensemble model.
        
        Args:
            save_dir: Directory to save the model
            
        Returns:
            Path to saved model
        """
        if not self.evaluation_results:
            raise ValueError("No ensemble models trained yet")
        
        # Find best ensemble based on validation accuracy
        best_name = max(self.evaluation_results.keys(),
                       key=lambda k: self.evaluation_results[k]['val_accuracy'])
        
        best_ensemble = self.ensemble_models[best_name]
        
        save_path = Path(save_dir) / f"{best_name}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(best_ensemble, save_path)
        logger.info(f"최고 앙상블 모델 저장: {save_path}")
        
        # Save ensemble info
        info_path = save_path.with_suffix('.yaml')
        ensemble_info = best_ensemble.get_ensemble_info()
        ensemble_info['validation_results'] = self.evaluation_results[best_name]
        
        with open(info_path, 'w', encoding='utf-8') as f:
            yaml.dump(ensemble_info, f, default_flow_style=False, allow_unicode=True)
        
        return str(save_path)


def load_ensemble_model(model_path: str) -> WatermelonEnsemble:
    """
    Load a saved ensemble model.
    
    Args:
        model_path: Path to saved ensemble model
        
    Returns:
        Loaded ensemble model
    """
    ensemble = joblib.load(model_path)
    logger.info(f"앙상블 모델 로드 완료: {model_path}")
    return ensemble


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # This would typically be run from a separate script
    print("🤖 WatermelonEnsemble 모듈이 성공적으로 로드되었습니다!")
    print("사용 방법:")
    print("1. EnsembleTrainer 초기화")
    print("2. train_all_ensembles() 호출")
    print("3. evaluate_on_test() 호출")
    print("4. save_best_ensemble() 호출") 