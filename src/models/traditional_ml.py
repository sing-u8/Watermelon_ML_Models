"""
🍉 수박 음 높낮이 분류 - 전통적인 머신러닝 모델 클래스

scikit-learn 기반 GBT, SVM, Random Forest 분류 모델 구현
- 공통 인터페이스 제공
- 특징 스케일링 통합
- 교차 검증 및 성능 평가
- 특징 중요도 분석
- 모델 저장/로드 기능
"""

import os
import yaml
import pickle
import joblib
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseWatermelonModel(BaseEstimator, ClassifierMixin, ABC):
    """
    수박 음 높낮이 분류 모델을 위한 추상 베이스 클래스
    
    모든 전통적인 ML 분류 모델이 공통으로 구현해야 하는 인터페이스 정의
    scikit-learn의 BaseEstimator와 ClassifierMixin을 상속받아 
    GridSearchCV/RandomizedSearchCV와 호환됩니다.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, model_name: str = "base_model"):
        """
        베이스 모델 초기화
        
        Args:
            config: 모델 설정 딕셔너리
            model_name: 모델 이름
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.feature_names = None
        self.training_history = {}
        self.classes_ = None
        
        # 스케일러 초기화
        self._init_scaler()
        
    def _init_scaler(self):
        """특징 스케일링을 위한 스케일러 초기화"""
        scaler_type = self.config.get('preprocessing', {}).get('feature_scaling', {}).get('method', 'standard')
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}. Using StandardScaler.")
            self.scaler = StandardScaler()
    
    @abstractmethod
    def _create_model(self) -> Any:
        """모델 인스턴스 생성 (하위 클래스에서 구현)"""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """
        모델 훈련
        
        Args:
            X: 훈련 특징
            y: 훈련 타겟 (0: 낮음, 1: 높음)
            X_val: 검증 특징 (선택사항)
            y_val: 검증 타겟 (선택사항)
        """
        logger.info(f"Training {self.model_name} model...")
        
        # 입력 검증
        X, y = self._validate_input(X, y)
        
        # 클래스 정보 저장
        self.classes_ = np.unique(y)
        
        # 특징 스케일링
        X_scaled = self._fit_transform_features(X)
        
        # 모델 생성 및 훈련
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        
        # 훈련 기록
        self.training_history['train_score'] = self.model.score(X_scaled, y)
        
        if X_val is not None and y_val is not None:
            if self.scaler is not None:
                X_val_scaled = self.scaler.transform(X_val)
                self.training_history['val_score'] = self.model.score(X_val_scaled, y_val)
        
        self.is_fitted = True
        logger.info(f"{self.model_name} training completed.")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        분류 예측 수행
        
        Args:
            X: 예측할 특징
            
        Returns:
            예측된 클래스 (0: 낮음, 1: 높음)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        if self.model is None:
            raise ValueError("Model is not initialized.")
        
        if self.scaler is None:
            raise ValueError("Scaler is not initialized.")
        
        X = self._validate_input_single(X)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        분류 확률 예측 수행
        
        Args:
            X: 예측할 특징
            
        Returns:
            각 클래스에 대한 확률
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        if self.model is None:
            raise ValueError("Model is not initialized.")
        
        if self.scaler is None:
            raise ValueError("Scaler is not initialized.")
        
        X = self._validate_input_single(X)
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict_proba(X_scaled)
        
        return probabilities
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                      scoring: str = 'accuracy') -> Dict[str, Any]:
        """
        교차 검증 수행
        
        Args:
            X: 특징
            y: 타겟 (0: 낮음, 1: 높음)
            cv: 교차 검증 폴드 수
            scoring: 평가 지표
            
        Returns:
            교차 검증 결과
        """
        logger.info(f"Performing {cv}-fold cross validation for {self.model_name}...")
        
        X, y = self._validate_input(X, y)
        X_scaled = self._fit_transform_features(X)
        
        # 모델 생성
        model = self._create_model()
        
        # 교차 검증 실행
        cv_results = cross_validate(
            model, X_scaled, y,
            cv=cv,
            scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
            return_train_score=True,
            n_jobs=-1
        )
        
        # 결과 정리
        results = {
            'test_accuracy': cv_results['test_accuracy'],
            'test_f1': cv_results['test_f1'],
            'test_precision': cv_results['test_precision'],
            'test_recall': cv_results['test_recall'],
            'test_roc_auc': cv_results['test_roc_auc'],
            'train_accuracy': cv_results['train_accuracy'],
            'train_f1': cv_results['train_f1'],
            'train_precision': cv_results['train_precision'],
            'train_recall': cv_results['train_recall'],
            'train_roc_auc': cv_results['train_roc_auc'],
        }
        
        # 통계 요약
        for metric in ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_roc_auc',
                      'train_accuracy', 'train_f1', 'train_precision', 'train_recall', 'train_roc_auc']:
            values = results[metric]
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        
        logger.info(f"Cross validation completed. Test Accuracy: {results['test_accuracy_mean']:.3f} ± {results['test_accuracy_std']:.3f}")
        
        return results
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        모델 성능 평가
        
        Args:
            X: 테스트 특징
            y: 테스트 타겟 (0: 낮음, 1: 높음)
            
        Returns:
            평가 메트릭 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")
        
        X, y = self._validate_input(X, y)
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # 분류 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'f1_score': f1_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'roc_auc': roc_auc_score(y, probabilities[:, 1]),
        }
        
        # 혼동 행렬
        cm = confusion_matrix(y, predictions)
        metrics['confusion_matrix'] = cm
        
        # 분류 리포트
        report = classification_report(y, predictions, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        특징 중요도 반환
        
        Returns:
            특징 중요도 배열
        """
        if not self.is_fitted or self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning(f"{self.model_name} does not support feature importance.")
            return None
    
    def save_model(self, filepath: str):
        """
        모델 저장
        
        Args:
            filepath: 저장할 파일 경로
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving.")
        
        # 저장할 객체 준비
        save_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'classes_': self.classes_,
            'config': self.config,
            'model_name': self.model_name,
            'training_history': self.training_history
        }
        
        # 모델 저장
        joblib.dump(save_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        모델 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            로드된 모델 인스턴스
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # 모델 로드
        save_data = joblib.load(filepath)
        
        # 모델 인스턴스 생성
        model_instance = cls(config=save_data.get('config'))
        model_instance.model = save_data['model']
        model_instance.scaler = save_data['scaler']
        model_instance.feature_names = save_data.get('feature_names')
        model_instance.classes_ = save_data.get('classes_')
        model_instance.training_history = save_data.get('training_history', {})
        model_instance.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        return model_instance
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """입력 데이터 검증"""
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")
        
        # 클래스 검증
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected 2 classes, got {len(unique_classes)}")
        
        if not np.array_equal(unique_classes, [0, 1]):
            raise ValueError("Classes must be 0 and 1")
        
        return X, y
    
    def _validate_input_single(self, X: np.ndarray) -> np.ndarray:
        """단일 예측을 위한 입력 검증"""
        if X is None:
            raise ValueError("X cannot be None")
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        return X
    
    def _fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        """특징 스케일링 수행"""
        if self.scaler is not None:
            return self.scaler.fit_transform(X)
        return X
    
    def __str__(self) -> str:
        return f"{self.model_name} (fitted: {self.is_fitted})"
    
    def __repr__(self) -> str:
        return self.__str__()


class WatermelonGBT(BaseWatermelonModel):
    """Gradient Boosting Classifier for watermelon pitch classification"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, random_state: int = 42):
        super().__init__(config, "WatermelonGBT")
        self.random_state = random_state
    
    def _create_model(self) -> GradientBoostingClassifier:
        """Gradient Boosting Classifier 생성"""
        gbt_config = self.config.get('gradient_boosting', {})
        
        return GradientBoostingClassifier(
            n_estimators=gbt_config.get('n_estimators', 100),
            learning_rate=gbt_config.get('learning_rate', 0.1),
            max_depth=gbt_config.get('max_depth', 3),
            min_samples_split=gbt_config.get('min_samples_split', 2),
            min_samples_leaf=gbt_config.get('min_samples_leaf', 1),
            subsample=gbt_config.get('subsample', 1.0),
            random_state=self.random_state
        )


class WatermelonSVM(BaseWatermelonModel):
    """Support Vector Classifier for watermelon pitch classification"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, random_state: int = 42):
        super().__init__(config, "WatermelonSVM")
        self.random_state = random_state
    
    def _create_model(self) -> SVC:
        """Support Vector Classifier 생성"""
        svm_config = self.config.get('svm', {})
        
        return SVC(
            kernel=svm_config.get('kernel', 'rbf'),
            C=svm_config.get('C', 1.0),
            gamma=svm_config.get('gamma', 'scale'),
            probability=True,  # 확률 예측을 위해 필요
            random_state=self.random_state
        )


class WatermelonRandomForest(BaseWatermelonModel):
    """Random Forest Classifier for watermelon pitch classification"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, random_state: int = 42):
        super().__init__(config, "WatermelonRandomForest")
        self.random_state = random_state
    
    def _create_model(self) -> RandomForestClassifier:
        """Random Forest Classifier 생성"""
        rf_config = self.config.get('random_forest', {})
        
        return RandomForestClassifier(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', None),
            min_samples_split=rf_config.get('min_samples_split', 2),
            min_samples_leaf=rf_config.get('min_samples_leaf', 1),
            max_features=rf_config.get('max_features', 'sqrt'),
            random_state=self.random_state
        )


class ModelFactory:
    """모델 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseWatermelonModel:
        """
        모델 타입에 따른 모델 인스턴스 생성
        
        Args:
            model_type: 모델 타입 ('gbt', 'svm', 'rf')
            config: 모델 설정
            
        Returns:
            모델 인스턴스
        """
        model_type = model_type.lower()
        
        if model_type == 'gbt':
            return WatermelonGBT(config)
        elif model_type == 'svm':
            return WatermelonSVM(config)
        elif model_type == 'rf':
            return WatermelonRandomForest(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_all_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, BaseWatermelonModel]:
        """
        모든 모델 인스턴스 생성
        
        Args:
            config: 모델 설정
            
        Returns:
            모델 딕셔너리
        """
        models = {
            'gbt': WatermelonGBT(config),
            'svm': WatermelonSVM(config),
            'rf': WatermelonRandomForest(config)
        }
        return models


def load_config(config_path: str = "configs/models.yaml") -> Dict[str, Any]:
    """
    설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        설정 딕셔너리
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.warning(f"Configuration file not found: {config_path}. Using default settings.")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return {}


def test_models():
    """모델 테스트 함수"""
    # 샘플 데이터 생성
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # 설정 로드
    config = load_config()
    
    # 모든 모델 테스트
    models = ModelFactory.create_all_models(config)
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # 교차 검증
        cv_results = model.cross_validate(X, y, cv=3)
        print(f"CV Accuracy: {cv_results['test_accuracy_mean']:.3f} ± {cv_results['test_accuracy_std']:.3f}")
        
        # 훈련 및 평가
        model.fit(X, y)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        print(f"Training Accuracy: {accuracy_score(y, predictions):.3f}")
        print(f"F1 Score: {f1_score(y, predictions):.3f}")
        print(f"ROC AUC: {roc_auc_score(y, probabilities[:, 1]):.3f}")


if __name__ == "__main__":
    test_models() 