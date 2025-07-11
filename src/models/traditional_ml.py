"""
🍉 수박 당도 예측 - 전통적인 머신러닝 모델 클래스

scikit-learn 기반 GBT, SVM, Random Forest 모델 구현
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

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    max_error, explained_variance_score
)
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseWatermelonModel(BaseEstimator, RegressorMixin, ABC):
    """
    수박 당도 예측 모델을 위한 추상 베이스 클래스
    
    모든 전통적인 ML 모델이 공통으로 구현해야 하는 인터페이스 정의
    scikit-learn의 BaseEstimator와 RegressorMixin을 상속받아 
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
            y: 훈련 타겟
            X_val: 검증 특징 (선택사항)
            y_val: 검증 타겟 (선택사항)
        """
        logger.info(f"Training {self.model_name} model...")
        
        # 입력 검증
        X, y = self._validate_input(X, y)
        
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
        예측 수행
        
        Args:
            X: 예측할 특징
            
        Returns:
            예측 결과
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
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                      scoring: str = 'neg_mean_absolute_error') -> Dict[str, Any]:
        """
        교차 검증 수행
        
        Args:
            X: 특징
            y: 타겟
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
            scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
            return_train_score=True,
            n_jobs=-1
        )
        
        # 결과 정리
        results = {
            'test_mae': -cv_results['test_neg_mean_absolute_error'],
            'test_mse': -cv_results['test_neg_mean_squared_error'],
            'test_r2': cv_results['test_r2'],
            'train_mae': -cv_results['train_neg_mean_absolute_error'],
            'train_mse': -cv_results['train_neg_mean_squared_error'],
            'train_r2': cv_results['train_r2'],
        }
        
        # 통계 요약
        for metric in ['test_mae', 'test_mse', 'test_r2', 'train_mae', 'train_mse', 'train_r2']:
            values = results[metric]
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
        
        logger.info(f"Cross validation completed. Test MAE: {results['test_mae_mean']:.3f} ± {results['test_mae_std']:.3f}")
        
        return results
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        모델 성능 평가
        
        Args:
            X: 특징
            y: 실제 타겟
            
        Returns:
            평가 메트릭 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation.")
        
        predictions = self.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'r2': r2_score(y, predictions),
            'mape': mean_absolute_percentage_error(y, predictions),
            'median_ae': median_absolute_error(y, predictions),
            'max_error': max_error(y, predictions),
            'explained_variance': explained_variance_score(y, predictions)
        }
        
        # 사용자 정의 메트릭
        metrics['accuracy_0_5'] = self._accuracy_within_threshold(y, predictions, 0.5)
        metrics['accuracy_1_0'] = self._accuracy_within_threshold(y, predictions, 1.0)
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        특징 중요도 반환 (지원하는 모델만)
        
        Returns:
            특징 중요도 배열 (없으면 None)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance.")
        
        if self.model is not None and hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            logger.warning(f"{self.model_name} does not support feature importance.")
            return None
    
    def save_model(self, filepath: str):
        """
        모델 저장
        
        Args:
            filepath: 저장할 파일 경로 (.pkl)
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model_data, filepath)
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
        model_data = joblib.load(filepath)
        
        # 인스턴스 생성
        instance = cls(config=model_data['config'], model_name=model_data['model_name'])
        
        # 데이터 복원
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.training_history = model_data['training_history']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """입력 데이터 검증 (훈련용)"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        return X, y
    
    def _validate_input_single(self, X: np.ndarray) -> np.ndarray:
        """입력 데이터 검증 (예측용)"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        return X
    
    def _fit_transform_features(self, X: np.ndarray) -> np.ndarray:
        """특징 스케일링 (fit & transform)"""
        if self.scaler is None:
            raise ValueError("Scaler is not initialized.")
        return self.scaler.fit_transform(X)
    
    def _accuracy_within_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
        """임계값 내 정확도 계산"""
        return float(np.mean(np.abs(y_true - y_pred) <= threshold))
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"{self.model_name}(fitted={self.is_fitted})"
    
    def __repr__(self) -> str:
        """객체 표현"""
        return self.__str__()


class WatermelonGBT(BaseWatermelonModel):
    """
    수박 당도 예측을 위한 Gradient Boosting Trees 모델
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, random_state: int = 42):
        super().__init__(config, "WatermelonGBT")
        self.random_state = random_state
    
    def _create_model(self) -> GradientBoostingRegressor:
        """GBT 모델 생성"""
        gbt_config = self.config.get('gradient_boosting', {})
        base_params = gbt_config.get('base_params', {})
        base_params['random_state'] = self.random_state
        
        return GradientBoostingRegressor(**base_params)


class WatermelonSVM(BaseWatermelonModel):
    """
    수박 당도 예측을 위한 Support Vector Machine 모델
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, random_state: int = 42):
        super().__init__(config, "WatermelonSVM")
        self.random_state = random_state
    
    def _create_model(self) -> SVR:
        """SVM 모델 생성"""
        svm_config = self.config.get('svm', {})
        base_params = svm_config.get('base_params', {})
        
        return SVR(**base_params)


class WatermelonRandomForest(BaseWatermelonModel):
    """
    수박 당도 예측을 위한 Random Forest 모델
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, random_state: int = 42):
        super().__init__(config, "WatermelonRandomForest")
        self.random_state = random_state
    
    def _create_model(self) -> RandomForestRegressor:
        """Random Forest 모델 생성"""
        rf_config = self.config.get('random_forest', {})
        base_params = rf_config.get('base_params', {})
        base_params['random_state'] = self.random_state
        
        return RandomForestRegressor(**base_params)


class ModelFactory:
    """
    모델 생성을 위한 팩토리 클래스
    """
    
    @staticmethod
    def create_model(model_type: str, config: Optional[Dict[str, Any]] = None) -> BaseWatermelonModel:
        """
        모델 타입에 따라 모델 인스턴스 생성
        
        Args:
            model_type: 모델 타입 ('gbt', 'svm', 'random_forest')
            config: 모델 설정
            
        Returns:
            모델 인스턴스
        """
        if model_type.lower() in ['gbt', 'gradient_boosting']:
            return WatermelonGBT(config)
        elif model_type.lower() in ['svm', 'support_vector_machine']:
            return WatermelonSVM(config)
        elif model_type.lower() in ['rf', 'random_forest']:
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
            모델 인스턴스 딕셔너리
        """
        return {
            'gbt': WatermelonGBT(config),
            'svm': WatermelonSVM(config),
            'random_forest': WatermelonRandomForest(config)
        }


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
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}. Using default settings.")
        return {}
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def test_models():
    """모델 클래스 테스트 함수"""
    logger.info("Testing traditional ML models...")
    
    # 가상 데이터 생성
    np.random.seed(42)
    X = np.random.randn(100, 51)  # 51개 특징
    y = np.random.uniform(8.0, 13.0, 100)  # 8-13 Brix 범위
    
    # 훈련/테스트 분할
    split_idx = 80
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 설정 로드
    config = load_config()
    
    # 모든 모델 테스트
    models = ModelFactory.create_all_models(config)
    
    for model_name, model in models.items():
        logger.info(f"\nTesting {model_name.upper()} model:")
        
        # 훈련
        model.fit(X_train, y_train, X_test, y_test)
        
        # 예측
        predictions = model.predict(X_test)
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # 평가
        metrics = model.evaluate(X_test, y_test)
        logger.info(f"Test MAE: {metrics['mae']:.3f}")
        logger.info(f"Test R²: {metrics['r2']:.3f}")
        
        # 교차 검증
        cv_results = model.cross_validate(X_train, y_train, cv=3)
        logger.info(f"CV MAE: {cv_results['test_mae_mean']:.3f} ± {cv_results['test_mae_std']:.3f}")
        
        # 특징 중요도
        importance = model.get_feature_importance()
        if importance is not None:
            logger.info(f"Feature importance shape: {importance.shape}")
            logger.info(f"Top 3 important features: {np.argsort(importance)[-3:]}")
        
        # 모델 저장/로드 테스트
        save_path = f"models/test_{model_name}.pkl"
        model.save_model(save_path)
        
        loaded_model = model.__class__.load_model(save_path)
        test_pred = loaded_model.predict(X_test[:5])
        logger.info(f"Loaded model prediction (first 5): {test_pred}")
        
        # 임시 파일 삭제
        if os.path.exists(save_path):
            os.remove(save_path)
    
    logger.info("\nAll models tested successfully! ✅")


if __name__ == "__main__":
    test_models() 