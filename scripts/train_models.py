#!/usr/bin/env python3
"""
🍉 수박 당도 예측 모델 훈련 스크립트

전통적인 ML 모델(GBT, SVM, Random Forest)을 훈련하고 평가하는 메인 스크립트입니다.

Usage:
    python scripts/train_models.py [--config CONFIG_PATH] [--quick] [--no-viz]

Author: Watermelon ML Team
Date: 2025-01-15
"""

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# 프로젝트 루트 디렉토리를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.trainer import MLTrainer, create_trainer_from_config
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ResultVisualizer
from src.models.traditional_ml import ModelFactory

# 경고 메시지 억제
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'experiments' / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """필요한 디렉토리들을 생성합니다."""
    directories = [
        PROJECT_ROOT / 'experiments',
        PROJECT_ROOT / 'models' / 'saved',
        PROJECT_ROOT / 'experiments' / 'results',
        PROJECT_ROOT / 'experiments' / 'plots'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"디렉토리 설정 완료: {len(directories)}개")


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 파일 로드 완료: {config_path}")
        return config
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        raise


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """훈련/검증/테스트 데이터셋을 로드합니다."""
    try:
        base_path = PROJECT_ROOT / 'data' / 'splits' / 'full_dataset'
        
        train_df = pd.read_csv(base_path / 'train.csv')
        val_df = pd.read_csv(base_path / 'val.csv')
        test_df = pd.read_csv(base_path / 'test.csv')
        
        logger.info(f"데이터셋 로드 완료:")
        logger.info(f"  - 훈련: {len(train_df)}개")
        logger.info(f"  - 검증: {len(val_df)}개") 
        logger.info(f"  - 테스트: {len(test_df)}개")
        logger.info(f"  - 당도 범위: {train_df['sweetness'].min():.1f} ~ {train_df['sweetness'].max():.1f} Brix")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {e}")
        raise


def print_training_summary(config: Dict[str, Any]) -> None:
    """훈련 설정 요약을 출력합니다."""
    print("\n" + "="*80)
    print("🍉 수박 당도 예측 모델 훈련 시작")
    print("="*80)
    
    print(f"📅 훈련 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 성능 목표:")
    print(f"   - MAE < {config['performance']['target_mae']:.1f} Brix")
    print(f"   - R² > {config['performance']['target_r2']:.2f}")
    
    print(f"🤖 훈련 모델:")
    for model_name in config['models'].keys():
        print(f"   - {model_name}")
    
    print(f"⚙️  훈련 설정:")
    print(f"   - 교차 검증: {config['cross_validation']['n_folds']}-fold")
    print(f"   - 스케일러: {config['preprocessing']['scaler_type']}")
    print(f"   - Random State: {config['global']['random_state']}")
    print("="*80 + "\n")


def evaluate_and_visualize(
    trainer: MLTrainer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict[str, Any],
    create_visualizations: bool = True
) -> Dict[str, Any]:
    """모델을 평가하고 시각화합니다."""
    
    logger.info("모델 평가 시작...")
    
    # 특징과 타겟 분리
    feature_cols = [col for col in train_df.columns if col != 'sweetness']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['sweetness'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['sweetness'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['sweetness'].values
    
    # 평가자 초기화
    evaluator = ModelEvaluator()
    all_results = {}
    
    # 각 모델 평가
    for model_name, model in trainer.models.items():
        logger.info(f"  {model_name} 평가 중...")
        
        # 예측
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        # 훈련 세트 평가
        train_results = evaluator.evaluate_model_performance(
            y_train, y_train_pred, model_name, "훈련"
        )
        
        # 검증 세트 평가
        val_results = evaluator.evaluate_model_performance(
            y_val, y_val_pred, model_name, "검증"
        )
        
        # 테스트 세트 평가
        test_results = evaluator.evaluate_model_performance(
            y_test, y_test_pred, model_name, "테스트"
        )
        
        # 결과 저장
        all_results[model_name] = {
            'train': train_results,
            'val': val_results,
            'test': test_results,
            'predictions': {
                'train': y_train_pred,
                'val': y_val_pred,
                'test': y_test_pred
            }
        }
    
    # 시각화 생성
    if create_visualizations:
        logger.info("시각화 생성 중...")
        visualizer = ResultVisualizer()
        
        # 성능 비교 차트
        performance_data = []
        for model_name, results in all_results.items():
            for dataset_type in ['train', 'val', 'test']:
                result = results[dataset_type]
                performance_data.append({
                    'Model': model_name,
                    'Dataset': dataset_type,
                    'MAE': result.mae,
                    'R2': result.r2,
                    'RMSE': result.rmse
                })
        
        performance_df = pd.DataFrame(performance_data)
        
        # 성능 비교 플롯
        fig_performance = visualizer.plot_model_comparison(performance_df)
        fig_performance.write_html(PROJECT_ROOT / 'experiments' / 'plots' / 'model_performance_comparison.html')
        
        # 각 모델별 예측 vs 실제 플롯
        for model_name, results in all_results.items():
            # 테스트 세트 예측 vs 실제
            fig_pred = visualizer.plot_predictions_vs_actual(
                y_test, results['predictions']['test'],
                f"{model_name} - 테스트 세트 예측 vs 실제"
            )
            fig_pred.write_html(
                PROJECT_ROOT / 'experiments' / 'plots' / f'{model_name}_predictions_vs_actual.html'
            )
            
            # 잔차 플롯
            fig_residual = visualizer.plot_residuals(
                y_test, results['predictions']['test'],
                f"{model_name} - 잔차 분석"
            )
            fig_residual.write_html(
                PROJECT_ROOT / 'experiments' / 'plots' / f'{model_name}_residuals.html'
            )
        
        # 특징 중요도 (Random Forest와 GBT만)
        for model_name, model in trainer.models.items():
            if hasattr(model.model, 'feature_importances_'):
                importance_dict = model.get_feature_importance()
                if importance_dict:
                    fig_importance = visualizer.plot_feature_importance(
                        importance_dict, f"{model_name} - 특징 중요도"
                    )
                    fig_importance.write_html(
                        PROJECT_ROOT / 'experiments' / 'plots' / f'{model_name}_feature_importance.html'
                    )
        
        logger.info(f"시각화 완료: experiments/plots/ 디렉토리")
    
    return all_results


def print_results_summary(all_results: Dict[str, Any], config: Dict[str, Any]) -> str:
    """결과 요약을 출력하고 최고 성능 모델을 반환합니다."""
    
    print("\n" + "="*80)
    print("📊 모델 성능 요약")
    print("="*80)
    
    # 테스트 성능 요약 테이블
    print(f"{'모델':<20} {'MAE':<8} {'R²':<8} {'RMSE':<8} {'목표달성':<10}")
    print("-" * 70)
    
    best_model = None
    best_mae = float('inf')
    target_mae = config['performance']['target_mae']
    target_r2 = config['performance']['target_r2']
    
    for model_name, results in all_results.items():
        test_result = results['test']
        mae = test_result.mae
        r2 = test_result.r2
        rmse = test_result.rmse
        
        # 목표 달성 여부
        mae_ok = "✅" if mae < target_mae else "❌"
        r2_ok = "✅" if r2 > target_r2 else "❌"
        goal_status = f"{mae_ok} {r2_ok}"
        
        print(f"{model_name:<20} {mae:<8.3f} {r2:<8.3f} {rmse:<8.3f} {goal_status:<10}")
        
        # 최고 성능 모델 찾기 (MAE 기준)
        if mae < best_mae:
            best_mae = mae
            best_model = model_name
    
    print("-" * 70)
    print(f"🏆 최고 성능 모델: {best_model} (MAE: {best_mae:.3f})")
    
    # 목표 달성 요약
    print(f"\n🎯 성능 목표 달성 현황:")
    models_meeting_mae = sum(1 for results in all_results.values() if results['test'].mae < target_mae)
    models_meeting_r2 = sum(1 for results in all_results.values() if results['test'].r2 > target_r2)
    total_models = len(all_results)
    
    print(f"   - MAE < {target_mae}: {models_meeting_mae}/{total_models} 모델")
    print(f"   - R² > {target_r2}: {models_meeting_r2}/{total_models} 모델")
    
    if best_mae < target_mae:
        print(f"   ✅ 주요 목표 달성! (MAE < {target_mae})")
    else:
        print(f"   ❌ 주요 목표 미달성 (MAE >= {target_mae})")
    
    print("="*80 + "\n")
    
    return best_model


def save_best_model(trainer: MLTrainer, best_model_name: str) -> None:
    """최고 성능 모델을 저장합니다."""
    
    try:
        best_model = trainer.models[best_model_name]
        
        # 모델 저장
        model_path = PROJECT_ROOT / 'models' / 'saved' / 'best_model.pkl'
        best_model.save_model(str(model_path))
        
        # 스케일러 저장 (있는 경우)
        if hasattr(best_model, 'scaler') and best_model.scaler is not None:
            scaler_path = PROJECT_ROOT / 'models' / 'saved' / 'scaler.pkl'
            import joblib
            joblib.dump(best_model.scaler, scaler_path)
            logger.info(f"스케일러 저장: {scaler_path}")
        
        # 모델 설정 저장
        config_path = PROJECT_ROOT / 'models' / 'saved' / 'model_config.yaml'
        model_config = {
            'best_model': best_model_name,
            'model_type': type(best_model).__name__,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(best_model.feature_names_) if hasattr(best_model, 'feature_names_') else 51,
            'performance': {
                'test_mae': float(trainer.latest_results[best_model_name]['test'].mae),
                'test_r2': float(trainer.latest_results[best_model_name]['test'].r2)
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(model_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"최고 성능 모델 저장 완료:")
        logger.info(f"  - 모델: {model_path}")
        logger.info(f"  - 설정: {config_path}")
        
    except Exception as e:
        logger.error(f"모델 저장 실패: {e}")


def save_training_results(
    all_results: Dict[str, Any],
    training_summary: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """훈련 결과를 파일로 저장합니다."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 성능 요약 CSV
        performance_data = []
        for model_name, results in all_results.items():
            for dataset_type in ['train', 'val', 'test']:
                result = results[dataset_type]
                performance_data.append({
                    'model': model_name,
                    'dataset': dataset_type,
                    'mae': result.mae,
                    'mse': result.mse,
                    'rmse': result.rmse,
                    'r2': result.r2,
                    'mape': result.mape,
                    'brix_accuracy_0_5': result.domain_metrics.get('brix_accuracy_0_5', 0),
                    'brix_accuracy_1_0': result.domain_metrics.get('brix_accuracy_1_0', 0),
                    'performance_grade': result.performance_grade
                })
        
        performance_df = pd.DataFrame(performance_data)
        performance_path = PROJECT_ROOT / 'experiments' / 'results' / f'performance_summary_{timestamp}.csv'
        performance_df.to_csv(performance_path, index=False)
        
        # 상세 결과 YAML
        detailed_results = {
            'experiment_info': {
                'timestamp': timestamp,
                'config_used': config,
                'training_summary': training_summary
            },
            'model_results': {}
        }
        
        for model_name, results in all_results.items():
            detailed_results['model_results'][model_name] = {
                'train': results['train'].to_dict(),
                'val': results['val'].to_dict(),
                'test': results['test'].to_dict()
            }
        
        detailed_path = PROJECT_ROOT / 'experiments' / 'results' / f'detailed_results_{timestamp}.yaml'
        with open(detailed_path, 'w', encoding='utf-8') as f:
            yaml.dump(detailed_results, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"훈련 결과 저장 완료:")
        logger.info(f"  - 성능 요약: {performance_path}")
        logger.info(f"  - 상세 결과: {detailed_path}")
        
    except Exception as e:
        logger.error(f"결과 저장 실패: {e}")


def main():
    """메인 실행 함수"""
    
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='수박 당도 예측 모델 훈련')
    parser.add_argument('--config', default='configs/models.yaml', help='설정 파일 경로')
    parser.add_argument('--quick', action='store_true', help='빠른 테스트 모드 (작은 하이퍼파라미터)')
    parser.add_argument('--no-viz', action='store_true', help='시각화 건너뛰기')
    
    args = parser.parse_args()
    
    try:
        # 초기 설정
        setup_directories()
        
        # 설정 로드
        config_path = PROJECT_ROOT / args.config
        config = load_config(config_path)
        
        # 빠른 모드 설정 조정
        if args.quick:
            logger.info("빠른 테스트 모드 활성화")
            if 'gradient_boosting' in config['models']:
                config['models']['gradient_boosting']['n_estimators'] = 50
            if 'random_forest' in config['models']:
                config['models']['random_forest']['n_estimators'] = 50
            config['cross_validation']['n_folds'] = 3
        
        # 데이터 로드
        train_df, val_df, test_df = load_datasets()
        
        # 훈련 요약 출력
        print_training_summary(config)
        
        # 트레이너 생성 및 훈련
        logger.info("모델 훈련 시작...")
        trainer = create_trainer_from_config(config)
        
        # 특징과 타겟 분리
        feature_cols = [col for col in train_df.columns if col != 'sweetness']
        X_train = train_df[feature_cols].values
        y_train = train_df['sweetness'].values
        X_val = val_df[feature_cols].values  
        y_val = val_df['sweetness'].values
        
        # 훈련 실행
        training_results = trainer.train_models(X_train, y_train, X_val, y_val)
        
        logger.info("훈련 완료! 평가 시작...")
        
        # 평가 및 시각화
        all_results = evaluate_and_visualize(
            trainer, train_df, val_df, test_df, config, 
            create_visualizations=not args.no_viz
        )
        
        # 결과 요약 출력
        best_model_name = print_results_summary(all_results, config)
        
        # 최고 성능 모델 저장
        save_best_model(trainer, best_model_name)
        
        # 훈련 결과 저장
        training_summary = training_results.get_summary()
        save_training_results(all_results, training_summary, config)
        
        # 완료 메시지
        print("\n🎉 모델 훈련 및 평가 완료!")
        print(f"📁 결과 파일: experiments/results/")
        print(f"📊 시각화: experiments/plots/")
        print(f"🏆 최고 모델: models/saved/")
        
        return 0
        
    except Exception as e:
        logger.error(f"훈련 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # 메모리 정리
        import gc
        gc.collect()
        logger.info("메모리 정리 완료")


if __name__ == "__main__":
    sys.exit(main()) 