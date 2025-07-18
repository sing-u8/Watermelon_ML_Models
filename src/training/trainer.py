"""
🍉 수박 당도 예측 - 훈련 파이프라인

전통적인 ML 모델들의 통합 훈련 관리
- 다중 모델 동시 훈련
- 데이터 로딩 및 검증
- 교차 검증 관리
- 훈련 로깅 및 결과 추적
- 모델 저장 및 비교
"""

import os
import time
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, f1_score, precision_score, recall_score

from ..models.traditional_ml import (
    BaseWatermelonModel, ModelFactory, load_config
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingResults:
    """
    훈련 결과를 저장하고 관리하는 클래스
    """
    
    def __init__(self):
        self.results = {}
        self.best_model = None
        self.best_metric = 0.0  # F1-score는 높을수록 좋음
        self.training_start_time: Optional[datetime] = None
        self.training_end_time: Optional[datetime] = None
    
    def add_model_result(self, model_name: str, result: Dict[str, Any]):
        """모델 결과 추가"""
        self.results[model_name] = result
        
        # 최고 성능 모델 업데이트 (F1-score 기준, 분류 문제)
        if 'val_f1_score' in result and result['val_f1_score'] > self.best_metric:
            self.best_metric = result['val_f1_score']
            self.best_model = model_name
    
    def get_summary(self) -> pd.DataFrame:
        """결과 요약을 DataFrame으로 반환"""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'model': model_name,
                'train_mae': result.get('train_mae', np.nan),
                'val_mae': result.get('val_mae', np.nan),
                'test_mae': result.get('test_mae', np.nan),
                'train_r2': result.get('train_r2', np.nan),
                'val_r2': result.get('val_r2', np.nan),
                'test_r2': result.get('test_r2', np.nan),
                'training_time': result.get('training_time', np.nan),
                'cv_mae_mean': result.get('cv_mae_mean', np.nan),
                'cv_mae_std': result.get('cv_mae_std', np.nan)
            })
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, save_dir: str):
        """결과 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 요약 테이블 저장
        summary_df = self.get_summary()
        summary_path = os.path.join(save_dir, 'training_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # 전체 결과 저장
        results_path = os.path.join(save_dir, 'detailed_results.yaml')
        with open(results_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.results, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Training results saved to {save_dir}")


class MLTrainer:
    """
    전통적인 ML 모델들의 통합 훈련 관리자
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, log_file: Optional[str] = None):
        """
        훈련자 초기화
        
        Args:
            config: 설정 딕셔너리
            log_file: 로그 파일 경로
        """
        self.config = config or load_config()
        self.models = {}
        self.results = TrainingResults()
        
        # 로깅 설정
        if log_file:
            self._setup_file_logging(log_file)
        
        # 모델 생성
        self._create_models()
        
        logger.info("MLTrainer initialized successfully")
    
    def _setup_file_logging(self, log_file: str):
        """파일 로깅 설정"""
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    def _create_models(self):
        """설정에 따라 모델들 생성"""
        self.models = ModelFactory.create_all_models(self.config)
        logger.info(f"Created models: {list(self.models.keys())}")
    
    def load_data(self, train_path: str, val_path: str, test_path: str, 
                  target_column: str = 'sweetness') -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        데이터 로드 및 검증
        
        Args:
            train_path: 훈련 데이터 경로
            val_path: 검증 데이터 경로  
            test_path: 테스트 데이터 경로
            target_column: 타겟 컬럼명
            
        Returns:
            데이터 딕셔너리 및 특징 이름 리스트
        """
        logger.info("Loading training data...")
        
        # 데이터 로드
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        logger.info(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        
        # 특징과 타겟 분리
        feature_columns = [col for col in train_df.columns if col != target_column]
        
        data = {
            'X_train': np.array(train_df[feature_columns].values),
            'y_train': np.array(train_df[target_column].values),
            'X_val': np.array(val_df[feature_columns].values),
            'y_val': np.array(val_df[target_column].values),
            'X_test': np.array(test_df[feature_columns].values),
            'y_test': np.array(test_df[target_column].values)
        }
        
        # 데이터 품질 검증
        self._validate_data(data)
        
        logger.info(f"Data loaded successfully. Features: {len(feature_columns)}")
        return data, feature_columns
    
    def _validate_data(self, data: Dict[str, np.ndarray]):
        """데이터 품질 검증"""
        for name, array in data.items():
            if np.any(np.isnan(array)):
                raise ValueError(f"NaN values found in {name}")
            if np.any(np.isinf(array)):
                raise ValueError(f"Infinite values found in {name}")
        
        logger.info("Data validation passed ✅")
    
    def train_single_model(self, model_name: str, data: Dict[str, np.ndarray], 
                          feature_names: List[str], perform_cv: bool = True) -> Dict[str, Any]:
        """
        단일 모델 훈련
        
        Args:
            model_name: 모델 이름
            data: 데이터 딕셔너리
            feature_names: 특징 이름 리스트
            perform_cv: 교차 검증 수행 여부
            
        Returns:
            훈련 결과 딕셔너리
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name} model...")
        
        start_time = time.time()
        
        # 모델 훈련
        model.fit(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        
        training_time = time.time() - start_time
        
        # 성능 평가
        train_metrics = model.evaluate(data['X_train'], data['y_train'])
        val_metrics = model.evaluate(data['X_val'], data['y_val'])
        test_metrics = model.evaluate(data['X_test'], data['y_test'])
        
        # 결과 정리 (분류 메트릭)
        result = {
            'train_accuracy': train_metrics['accuracy'],
            'train_f1_score': train_metrics['f1_score'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_score': val_metrics['f1_score'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_score': test_metrics['f1_score'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'training_time': training_time,
            'feature_names': feature_names
        }
        
        # 교차 검증
        if perform_cv:
            cv_config = self.config.get('cross_validation', {})
            cv_folds = cv_config.get('cv_folds', 5)
            
            cv_results = model.cross_validate(
                data['X_train'], data['y_train'], 
                cv=cv_folds
            )
            
            result.update({
                'cv_accuracy_mean': cv_results['test_accuracy_mean'],
                'cv_accuracy_std': cv_results['test_accuracy_std'],
                'cv_f1_score_mean': cv_results['test_f1_mean'],
                'cv_f1_score_std': cv_results['test_f1_std']
            })
        
        # 특징 중요도
        importance = model.get_feature_importance()
        if importance is not None:
            # 상위 10개 중요 특징
            top_indices = np.argsort(importance)[-10:]
            result['feature_importance'] = {
                'importance_values': importance.tolist(),
                'top_features': [feature_names[i] for i in top_indices],
                'top_importance': [importance[i] for i in top_indices]
            }
        
        logger.info(f"{model_name} training completed in {training_time:.2f}s. "
                   f"Val F1: {val_metrics['f1_score']:.3f}, Test F1: {test_metrics['f1_score']:.3f}")
        
        return result
    
    def train_all_models(self, data: Dict[str, np.ndarray], feature_names: List[str], 
                        model_subset: Optional[List[str]] = None, 
                        perform_cv: bool = True) -> TrainingResults:
        """
        모든 모델 훈련
        
        Args:
            data: 데이터 딕셔너리
            feature_names: 특징 이름 리스트
            model_subset: 훈련할 모델 subset (None이면 전체)
            perform_cv: 교차 검증 수행 여부
            
        Returns:
            훈련 결과 객체
        """
        self.results.training_start_time = datetime.now()
        
        models_to_train = model_subset or list(self.models.keys())
        total_models = len(models_to_train)
        
        logger.info(f"Starting training for {total_models} models...")
        
        for i, model_name in enumerate(models_to_train, 1):
            logger.info(f"[{i}/{total_models}] Training {model_name}...")
            
            try:
                result = self.train_single_model(
                    model_name, data, feature_names, perform_cv
                )
                self.results.add_model_result(model_name, result)
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        self.results.training_end_time = datetime.now()
        total_time = (self.results.training_end_time - self.results.training_start_time).total_seconds()
        
        logger.info(f"All models training completed in {total_time:.2f}s")
        logger.info(f"Best model: {self.results.best_model} (F1: {self.results.best_metric:.3f})")
        
        return self.results
    
    def save_models(self, save_dir: str, save_best_only: bool = False):
        """
        훈련된 모델들 저장
        
        Args:
            save_dir: 저장 디렉토리
            save_best_only: 최고 성능 모델만 저장 여부
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if save_best_only and self.results.best_model:
            # 최고 성능 모델만 저장
            model_name = self.results.best_model
            model = self.models[model_name]
            
            if model.is_fitted:
                model_path = os.path.join(save_dir, f"best_model_{model_name}.pkl")
                model.save_model(model_path)
                logger.info(f"Best model ({model_name}) saved to {model_path}")
        else:
            # 모든 훈련된 모델 저장
            for model_name, model in self.models.items():
                if model.is_fitted:
                    model_path = os.path.join(save_dir, f"{model_name}_model.pkl")
                    model.save_model(model_path)
            
            logger.info(f"All trained models saved to {save_dir}")
    
    def get_model_comparison(self) -> pd.DataFrame:
        """모델 성능 비교 테이블 생성"""
        summary_df = self.results.get_summary()
        
        if summary_df.empty:
            return summary_df
        
        # 성능 순위 추가
        summary_df['mae_rank'] = summary_df['val_mae'].rank()
        summary_df['r2_rank'] = summary_df['val_r2'].rank(ascending=False)
        
        # 정렬
        summary_df = summary_df.sort_values('val_mae').reset_index(drop=True)
        
        return summary_df
    
    def print_summary(self):
        """훈련 결과 요약 출력"""
        logger.info("\n" + "="*60)
        logger.info("🍉 WATERMELON SWEETNESS PREDICTION - TRAINING SUMMARY")
        logger.info("="*60)
        
        if not self.results.results:
            logger.info("No training results available.")
            return
        
        summary_df = self.get_model_comparison()
        
        # 최고 성능 모델
        best_row = summary_df.iloc[0]
        logger.info(f"🏆 BEST MODEL: {best_row['model']}")
        logger.info(f"   Validation MAE: {best_row['val_mae']:.3f} Brix")
        logger.info(f"   Test MAE: {best_row['test_mae']:.3f} Brix")
        logger.info(f"   R² Score: {best_row['val_r2']:.3f}")
        
        # 목표 달성 여부
        target_mae = self.config.get('performance_targets', {}).get('mae_target', 1.0)
        target_r2 = self.config.get('performance_targets', {}).get('r2_target', 0.8)
        
        mae_achieved = best_row['test_mae'] < target_mae
        r2_achieved = best_row['val_r2'] > target_r2
        
        logger.info(f"\n📊 PERFORMANCE TARGETS:")
        logger.info(f"   MAE < {target_mae} Brix: {'✅ ACHIEVED' if mae_achieved else '❌ NOT ACHIEVED'}")
        logger.info(f"   R² > {target_r2}: {'✅ ACHIEVED' if r2_achieved else '❌ NOT ACHIEVED'}")
        
        # 전체 모델 비교
        logger.info(f"\n📈 ALL MODELS COMPARISON:")
        for _, row in summary_df.iterrows():
            logger.info(f"   {row['model']:15} | MAE: {row['val_mae']:.3f} | R²: {row['val_r2']:.3f} | Time: {row['training_time']:.2f}s")
        
        logger.info("="*60)


def create_trainer_from_config(config_path: str = "configs/models.yaml", 
                              log_file: str = "experiments/training.log") -> MLTrainer:
    """
    설정 파일에서 훈련자 생성
    
    Args:
        config_path: 설정 파일 경로
        log_file: 로그 파일 경로
        
    Returns:
        MLTrainer 인스턴스
    """
    config = load_config(config_path)
    return MLTrainer(config, log_file)


def quick_train(train_path: str = "data/splits/full_dataset/train.csv",
                val_path: str = "data/splits/full_dataset/val.csv",
                test_path: str = "data/splits/full_dataset/test.csv",
                target_column: str = "sweetness",
                save_dir: str = "models/saved",
                results_dir: str = "experiments") -> TrainingResults:
    """
    빠른 훈련 실행 함수
    
    Args:
        train_path: 훈련 데이터 경로
        val_path: 검증 데이터 경로
        test_path: 테스트 데이터 경로
        target_column: 타겟 컬럼명
        save_dir: 모델 저장 디렉토리
        results_dir: 결과 저장 디렉토리
        
    Returns:
        훈련 결과
    """
    # 훈련자 생성
    trainer = create_trainer_from_config()
    
    # 데이터 로드
    data, feature_names = trainer.load_data(
        train_path, val_path, test_path, target_column
    )
    
    # 모든 모델 훈련
    results = trainer.train_all_models(data, feature_names)
    
    # 결과 출력
    trainer.print_summary()
    
    # 모델 및 결과 저장
    trainer.save_models(save_dir, save_best_only=True)
    results.save_results(results_dir)
    
    return results


def test_trainer():
    """훈련자 클래스 테스트"""
    logger.info("Testing MLTrainer...")
    
    # 가상 데이터 생성
    np.random.seed(42)
    n_samples = 100
    n_features = 51
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.uniform(8.0, 13.0, n_samples)
    
    # 데이터 분할
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    data = {
        'X_train': X[:train_size],
        'y_train': y[:train_size],
        'X_val': X[train_size:train_size+val_size],
        'y_val': y[train_size:train_size+val_size],
        'X_test': X[train_size+val_size:],
        'y_test': y[train_size+val_size:]
    }
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # 훈련자 생성
    config = load_config()
    trainer = MLTrainer(config)
    
    # 모든 모델 훈련
    results = trainer.train_all_models(data, feature_names, perform_cv=False)
    
    # 결과 출력
    trainer.print_summary()
    
    # 결과 저장 테스트
    results.save_results("experiments/test")
    
    # 모델 저장 테스트
    trainer.save_models("models/test", save_best_only=True)
    
    logger.info("MLTrainer test completed successfully! ✅")


if __name__ == "__main__":
    test_trainer() 