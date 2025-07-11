"""
하이퍼파라미터 튜닝 모듈

이 모듈은 전통적인 ML 모델들의 하이퍼파라미터 최적화를 위한 도구들을 제공합니다.
GridSearchCV와 RandomizedSearchCV를 모두 지원하며, 다중 모델 병렬 튜닝이 가능합니다.

Classes:
    HyperparameterTuner: 하이퍼파라미터 튜닝을 위한 메인 클래스
    TuningResult: 튜닝 결과를 저장하는 데이터 클래스
"""

import os
import yaml
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score
)
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.base import BaseEstimator
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 프로젝트 모듈 import
from src.models.traditional_ml import WatermelonGBT, WatermelonSVM, WatermelonRandomForest
from src.evaluation.evaluator import ModelEvaluator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """하이퍼파라미터 튜닝 결과를 저장하는 데이터 클래스"""
    model_name: str
    best_params: Dict[str, Any]
    best_score: float
    best_estimator: BaseEstimator
    cv_results: Dict[str, Any]
    search_time: float
    total_fits: int
    method: str  # 'grid' or 'random'
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (best_estimator 제외)"""
        result = asdict(self)
        # best_estimator는 직렬화할 수 없으므로 제외
        del result['best_estimator']
        del result['cv_results']  # cv_results도 너무 클 수 있으므로 제외
        return result


class HyperparameterTuner:
    """
    하이퍼파라미터 튜닝을 위한 메인 클래스
    
    이 클래스는 다양한 ML 모델들에 대해 GridSearchCV와 RandomizedSearchCV를
    사용한 하이퍼파라미터 최적화를 수행합니다.
    """
    
    def __init__(self, 
                 config_path: str = "configs/hyperparameter_search.yaml",
                 scoring: str = "neg_mean_absolute_error",
                 cv: int = 5,
                 n_jobs: int = -1,
                 verbose: int = 1,
                 random_state: int = 42):
        """
        HyperparameterTuner 초기화
        
        Args:
            config_path: 하이퍼파라미터 검색 설정 파일 경로
            scoring: 평가 지표 ('neg_mean_absolute_error', 'r2', etc.)
            cv: 교차 검증 폴드 수
            n_jobs: 병렬 처리 작업 수 (-1은 모든 CPU 사용)
            verbose: 로깅 레벨
            random_state: 랜덤 시드
        """
        self.config_path = config_path
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        
        # 결과 저장
        self.tuning_results: Dict[str, TuningResult] = {}
        self.best_models: Dict[str, BaseEstimator] = {}
        
        # 모델 매핑
        self.model_classes = {
            'gradient_boosting': WatermelonGBT,
            'svm': WatermelonSVM,
            'random_forest': WatermelonRandomForest
        }
        
        # 설정 로드
        self.search_config = self._load_search_config()
        
        logger.info(f"HyperparameterTuner 초기화 완료")
        logger.info(f"Scoring: {self.scoring}, CV: {self.cv}, n_jobs: {self.n_jobs}")
    
    def _load_search_config(self) -> Dict[str, Any]:
        """하이퍼파라미터 검색 설정 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"검색 설정 로드 완료: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            return self._get_default_search_config()
        except Exception as e:
            logger.error(f"설정 파일 로드 오류: {e}")
            return self._get_default_search_config()
    
    def _get_default_search_config(self) -> Dict[str, Any]:
        """기본 하이퍼파라미터 검색 설정"""
        return {
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 9],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly'],
                'degree': [2, 3, 4],  # poly kernel용
                'epsilon': [0.01, 0.1, 0.2]
            },
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
        }
    
    def tune_single_model(self,
                         model_name: str,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         method: str = "grid",
                         n_iter: int = 100) -> TuningResult:
        """
        단일 모델에 대한 하이퍼파라미터 튜닝
        
        Args:
            model_name: 모델 이름 ('gradient_boosting', 'svm', 'random_forest')
            X_train: 훈련 특징
            y_train: 훈련 타겟
            method: 튜닝 방법 ('grid' or 'random')
            n_iter: RandomizedSearchCV의 iteration 수
            
        Returns:
            TuningResult: 튜닝 결과
        """
        logger.info(f"=== {model_name} 하이퍼파라미터 튜닝 시작 ({method}) ===")
        
        if model_name not in self.model_classes:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        if model_name not in self.search_config:
            raise ValueError(f"검색 설정을 찾을 수 없습니다: {model_name}")
        
        # 모델 인스턴스 생성
        model_class = self.model_classes[model_name]
        model = model_class(random_state=self.random_state)
        
        # 하이퍼파라미터 그리드
        param_grid = self.search_config[model_name]
        
        start_time = time.time()
        
        # 검색 방법에 따라 다른 검색 수행
        if method == "grid":
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True,
                error_score='raise'
            )
        elif method == "random":
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                return_train_score=True,
                error_score='raise'
            )
        else:
            raise ValueError(f"지원하지 않는 방법: {method}")
        
        # 튜닝 실행
        try:
            search.fit(X_train, y_train)
            search_time = time.time() - start_time
            
            # 결과 생성
            result = TuningResult(
                model_name=model_name,
                best_params=search.best_params_,
                best_score=search.best_score_,
                best_estimator=search.best_estimator_,
                cv_results=search.cv_results_,
                search_time=search_time,
                total_fits=len(search.cv_results_['mean_test_score']),
                method=method
            )
            
            logger.info(f"{model_name} 튜닝 완료:")
            logger.info(f"  최고 점수: {search.best_score_:.4f}")
            logger.info(f"  최적 파라미터: {search.best_params_}")
            logger.info(f"  소요 시간: {search_time:.2f}초")
            logger.info(f"  총 시도 횟수: {result.total_fits}")
            
            return result
            
        except Exception as e:
            logger.error(f"{model_name} 튜닝 중 오류 발생: {e}")
            raise
    
    def tune_all_models(self,
                       X_train: np.ndarray,
                       y_train: np.ndarray,
                       method: str = "grid",
                       n_iter: int = 100,
                       parallel: bool = False) -> Dict[str, TuningResult]:
        """
        모든 모델에 대한 하이퍼파라미터 튜닝
        
        Args:
            X_train: 훈련 특징
            y_train: 훈련 타겟
            method: 튜닝 방법 ('grid' or 'random')
            n_iter: RandomizedSearchCV의 iteration 수
            parallel: 모델별 병렬 처리 여부
            
        Returns:
            Dict[str, TuningResult]: 모델별 튜닝 결과
        """
        logger.info(f"=== 전체 모델 하이퍼파라미터 튜닝 시작 ({method}) ===")
        logger.info(f"대상 모델: {list(self.model_classes.keys())}")
        
        results = {}
        
        if parallel:
            # 병렬 처리
            with ThreadPoolExecutor(max_workers=len(self.model_classes)) as executor:
                future_to_model = {
                    executor.submit(
                        self.tune_single_model, 
                        model_name, X_train, y_train, method, n_iter
                    ): model_name 
                    for model_name in self.model_classes.keys()
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        result = future.result()
                        results[model_name] = result
                        self.tuning_results[model_name] = result
                        self.best_models[model_name] = result.best_estimator
                    except Exception as e:
                        logger.error(f"{model_name} 병렬 튜닝 실패: {e}")
        else:
            # 순차 처리
            for model_name in self.model_classes.keys():
                try:
                    result = self.tune_single_model(
                        model_name, X_train, y_train, method, n_iter
                    )
                    results[model_name] = result
                    self.tuning_results[model_name] = result
                    self.best_models[model_name] = result.best_estimator
                except Exception as e:
                    logger.error(f"{model_name} 튜닝 실패: {e}")
                    continue
        
        logger.info(f"=== 전체 모델 튜닝 완료 ===")
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, TuningResult]):
        """튜닝 결과 요약 출력"""
        logger.info("\n=== 하이퍼파라미터 튜닝 결과 요약 ===")
        
        # 점수 기준으로 정렬
        sorted_results = sorted(
            results.items(), 
            key=lambda x: x[1].best_score, 
            reverse=True if self.scoring == 'r2' else False
        )
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            logger.info(f"{i}. {model_name}:")
            logger.info(f"   점수: {result.best_score:.4f}")
            logger.info(f"   시간: {result.search_time:.2f}초")
            logger.info(f"   시도: {result.total_fits}회")
    
    def evaluate_tuned_models(self,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            X_val: Optional[np.ndarray] = None,
                            y_val: Optional[np.ndarray] = None) -> Dict[str, Dict[str, float]]:
        """
        튜닝된 모델들의 성능 평가
        
        Args:
            X_test: 테스트 특징
            y_test: 테스트 타겟
            X_val: 검증 특징 (선택사항)
            y_val: 검증 타겟 (선택사항)
            
        Returns:
            Dict[str, Dict[str, float]]: 모델별 성능 지표
        """
        logger.info("=== 튜닝된 모델 성능 평가 ===")
        
        evaluator = ModelEvaluator()
        evaluation_results = {}
        
        for model_name, model in self.best_models.items():
            logger.info(f"평가 중: {model_name}")
            
            # 예측
            y_pred_test = model.predict(X_test)
            
            # 평가 메트릭 계산
            test_metrics = evaluator.calculate_regression_metrics(y_test, y_pred_test)
            
            result = {'test': test_metrics}
            
            # 검증 세트가 있다면 추가 평가
            if X_val is not None and y_val is not None:
                y_pred_val = model.predict(X_val)
                val_metrics = evaluator.calculate_regression_metrics(y_val, y_pred_val)
                result['validation'] = val_metrics
            
            evaluation_results[model_name] = result
            
            # 결과 출력
            logger.info(f"  테스트 MAE: {test_metrics['mae']:.4f}")
            logger.info(f"  테스트 R²: {test_metrics['r2']:.4f}")
        
        return evaluation_results
    
    def save_results(self, 
                    output_dir: str = "experiments/hyperparameter_tuning",
                    save_models: bool = True):
        """
        튜닝 결과 저장
        
        Args:
            output_dir: 출력 디렉토리
            save_models: 모델 저장 여부
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 타임스탬프
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 결과 요약 저장
        summary = {}
        for model_name, result in self.tuning_results.items():
            summary[model_name] = result.to_dict()
        
        summary_path = os.path.join(output_dir, f"tuning_summary_{timestamp}.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"튜닝 요약 저장: {summary_path}")
        
        # 최고 성능 모델 저장
        if save_models and self.best_models:
            models_dir = os.path.join(output_dir, f"best_models_{timestamp}")
            os.makedirs(models_dir, exist_ok=True)
            
            for model_name, model in self.best_models.items():
                model_path = os.path.join(models_dir, f"{model_name}_tuned.pkl")
                joblib.dump(model, model_path)
                logger.info(f"모델 저장: {model_path}")
        
        # 설정 파일도 백업
        config_backup_path = os.path.join(output_dir, f"search_config_{timestamp}.yaml")
        with open(config_backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.search_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"설정 백업: {config_backup_path}")
        logger.info(f"모든 결과가 {output_dir}에 저장되었습니다.")
    
    def get_best_model(self, metric: str = "best_score") -> Tuple[str, BaseEstimator]:
        """
        최고 성능 모델 반환
        
        Args:
            metric: 비교 기준 ('best_score')
            
        Returns:
            Tuple[str, BaseEstimator]: (모델 이름, 모델 객체)
        """
        if not self.tuning_results:
            raise ValueError("튜닝 결과가 없습니다. tune_all_models()를 먼저 실행하세요.")
        
        # 점수 기준으로 최고 모델 선택
        best_model_name = max(
            self.tuning_results.keys(),
            key=lambda name: self.tuning_results[name].best_score
            if self.scoring == 'r2' else -self.tuning_results[name].best_score
        )
        
        best_model = self.best_models[best_model_name]
        
        logger.info(f"최고 성능 모델: {best_model_name}")
        logger.info(f"점수: {self.tuning_results[best_model_name].best_score:.4f}")
        
        return best_model_name, best_model
    
    def compare_with_baseline(self, baseline_results: Dict[str, Dict[str, float]]):
        """
        기본 모델 대비 개선 정도 비교
        
        Args:
            baseline_results: 기본 모델의 성능 결과
        """
        logger.info("\n=== 기본 모델 대비 개선 정도 ===")
        
        for model_name in self.tuning_results.keys():
            if model_name in baseline_results:
                tuned_score = abs(self.tuning_results[model_name].best_score)
                baseline_score = baseline_results[model_name].get('test', {}).get('mae', 0)
                
                if baseline_score > 0:
                    improvement = ((baseline_score - tuned_score) / baseline_score) * 100
                    logger.info(f"{model_name}: {improvement:+.2f}% 개선")
                else:
                    logger.info(f"{model_name}: 기본 성능 데이터 없음")


def main():
    """하이퍼파라미터 튜닝 실행 예제"""
    # 데이터 로드 (예제)
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # 실제 데이터 로드 코드로 교체 필요
    train_path = "data/splits/full_dataset/train.csv"
    val_path = "data/splits/full_dataset/val.csv"
    test_path = "data/splits/full_dataset/test.csv"
    
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
        # 특징과 타겟 분리
        X_train = train_df.drop('sweetness', axis=1).values
        y_train = train_df['sweetness'].values
        X_val = val_df.drop('sweetness', axis=1).values
        y_val = val_df['sweetness'].values
        X_test = test_df.drop('sweetness', axis=1).values
        y_test = test_df['sweetness'].values
        
        # 튜너 초기화
        tuner = HyperparameterTuner()
        
        # 하이퍼파라미터 튜닝 실행
        results = tuner.tune_all_models(X_train, y_train, method="random", n_iter=50)
        
        # 성능 평가
        evaluation_results = tuner.evaluate_tuned_models(X_test, y_test, X_val, y_val)
        
        # 결과 저장
        tuner.save_results()
        
        # 최고 모델 선택
        best_name, best_model = tuner.get_best_model()
        print(f"최고 성능 모델: {best_name}")
        
    else:
        print("데이터 파일을 찾을 수 없습니다.")


if __name__ == "__main__":
    main() 