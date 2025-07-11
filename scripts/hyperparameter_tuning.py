#!/usr/bin/env python3
"""
하이퍼파라미터 튜닝 실행 스크립트

이 스크립트는 수박 당도 예측 모델의 하이퍼파라미터를 최적화합니다.
GridSearchCV와 RandomizedSearchCV를 모두 지원하며, 
기본 모델 대비 성능 개선을 추적합니다.

사용법:
    python scripts/hyperparameter_tuning.py --method random --n_iter 50
    python scripts/hyperparameter_tuning.py --method grid --strategy quick

작성자: ML Team
생성일: 2025-01-15
"""

import os
import sys
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime
from pathlib import Path
import gc

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 프로젝트 모듈
from src.training.hyperparameter_tuner import HyperparameterTuner
from src.training.trainer import MLTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import ResultVisualizer
from sklearn.preprocessing import StandardScaler

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "experiments/hyperparameter_tuning",
        "models/tuned",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"디렉토리 생성/확인: {directory}")


def load_data():
    """데이터 로드 및 전처리"""
    logger.info("=== 데이터 로드 시작 ===")
    
    # 데이터 파일 경로
    train_path = "data/splits/full_dataset/train.csv"
    val_path = "data/splits/full_dataset/val.csv"
    test_path = "data/splits/full_dataset/test.csv"
    
    # 파일 존재 여부 확인
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    
    # 데이터 로드
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"훈련 데이터: {train_df.shape}")
    logger.info(f"검증 데이터: {val_df.shape}")
    logger.info(f"테스트 데이터: {test_df.shape}")
    
    # 특징과 타겟 분리
    X_train = train_df.drop('sweetness', axis=1).values
    y_train = train_df['sweetness'].values
    X_val = val_df.drop('sweetness', axis=1).values
    y_val = val_df['sweetness'].values
    X_test = test_df.drop('sweetness', axis=1).values
    y_test = test_df['sweetness'].values
    
    # 특징 스케일링 (SVM을 위해 필수)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("데이터 스케일링 완료")
    logger.info(f"특징 수: {X_train_scaled.shape[1]}")
    logger.info(f"당도 범위: {np.min(y_train):.2f} ~ {np.max(y_train):.2f} Brix")
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': train_df.drop('sweetness', axis=1).columns.tolist()
    }


def load_baseline_results():
    """기본 모델 결과 로드"""
    logger.info("기본 모델 결과 로드 시도...")
    
    baseline_path = "models/saved/training_summary_simple.yaml"
    
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r', encoding='utf-8') as f:
            baseline_data = yaml.safe_load(f)
        
        # 결과 구조 변환
        baseline_results = {}
        for model_name, metrics in baseline_data.get('test_results', {}).items():
            baseline_results[model_name] = {'test': metrics}
        
        logger.info("기본 모델 결과 로드 완료")
        for model, metrics in baseline_results.items():
            mae = metrics['test'].get('mae', 0)
            r2 = metrics['test'].get('r2', 0)
            logger.info(f"  {model}: MAE={mae:.4f}, R²={r2:.4f}")
        
        return baseline_results
    else:
        logger.warning("기본 모델 결과를 찾을 수 없습니다.")
        return {}


def run_hyperparameter_tuning(data, method='random', strategy='medium', n_iter=50):
    """하이퍼파라미터 튜닝 실행"""
    logger.info(f"=== 하이퍼파라미터 튜닝 시작 ({method}) ===")
    
    # 튜너 초기화
    tuner = HyperparameterTuner(
        config_path="configs/hyperparameter_search.yaml",
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # 전체 모델 튜닝 실행
    start_time = datetime.now()
    
    try:
        if method == 'grid':
            # GridSearchCV 사용
            # strategy에 따라 설정 변경 필요 (추후 구현)
            results = tuner.tune_all_models(
                data['X_train'], 
                data['y_train'], 
                method='grid',
                parallel=False  # 메모리 절약을 위해 순차 처리
            )
        elif method == 'random':
            # RandomizedSearchCV 사용
            results = tuner.tune_all_models(
                data['X_train'], 
                data['y_train'], 
                method='random',
                n_iter=n_iter,
                parallel=False  # 메모리 절약을 위해 순차 처리
            )
        else:
            raise ValueError(f"지원하지 않는 방법: {method}")
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        logger.info(f"전체 튜닝 완료 시간: {total_time:.2f}초")
        
        # 메모리 정리
        gc.collect()
        
        return tuner, results
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 튜닝 중 오류 발생: {e}")
        raise


def evaluate_tuned_models(tuner, data):
    """튜닝된 모델 성능 평가"""
    logger.info("=== 튜닝된 모델 성능 평가 ===")
    
    # 성능 평가
    evaluation_results = tuner.evaluate_tuned_models(
        data['X_test'], 
        data['y_test'],
        data['X_val'],
        data['y_val']
    )
    
    # 결과 요약 출력
    logger.info("\n튜닝된 모델 성능 요약:")
    for model_name, metrics in evaluation_results.items():
        test_mae = metrics['test']['mae']
        test_r2 = metrics['test']['r2']
        val_mae = metrics['validation']['mae']
        val_r2 = metrics['validation']['r2']
        
        logger.info(f"  {model_name}:")
        logger.info(f"    테스트  - MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
        logger.info(f"    검증    - MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return evaluation_results


def compare_with_baseline(tuner, evaluation_results, baseline_results):
    """기본 모델과 성능 비교"""
    logger.info("=== 기본 모델 대비 성능 개선 ===")
    
    if not baseline_results:
        logger.warning("기본 모델 결과가 없어 비교를 건너뜁니다.")
        return
    
    improvements = {}
    
    for model_name in evaluation_results.keys():
        if model_name in baseline_results:
            # 튜닝된 모델 성능
            tuned_mae = evaluation_results[model_name]['test']['mae']
            tuned_r2 = evaluation_results[model_name]['test']['r2']
            
            # 기본 모델 성능
            baseline_mae = baseline_results[model_name]['test']['mae']
            baseline_r2 = baseline_results[model_name]['test']['r2']
            
            # 개선 정도 계산
            mae_improvement = ((baseline_mae - tuned_mae) / baseline_mae) * 100
            r2_improvement = ((tuned_r2 - baseline_r2) / baseline_r2) * 100 if baseline_r2 > 0 else 0
            
            improvements[model_name] = {
                'mae_improvement': mae_improvement,
                'r2_improvement': r2_improvement,
                'baseline_mae': baseline_mae,
                'tuned_mae': tuned_mae,
                'baseline_r2': baseline_r2,
                'tuned_r2': tuned_r2
            }
            
            logger.info(f"  {model_name}:")
            logger.info(f"    MAE: {baseline_mae:.4f} → {tuned_mae:.4f} ({mae_improvement:+.2f}%)")
            logger.info(f"    R²:  {baseline_r2:.4f} → {tuned_r2:.4f} ({r2_improvement:+.2f}%)")
    
    return improvements


def save_comprehensive_results(tuner, evaluation_results, improvements, data, method, strategy):
    """포괄적인 결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiments/hyperparameter_tuning/tuning_{method}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"결과 저장 디렉토리: {results_dir}")
    
    # 1. 튜닝 결과 저장
    tuner.save_results(results_dir, save_models=True)
    
    # 2. 평가 결과 저장
    evaluation_path = os.path.join(results_dir, "evaluation_results.yaml")
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        yaml.dump(evaluation_results, f, default_flow_style=False, allow_unicode=True)
    
    # 3. 개선 결과 저장
    if improvements:
        improvement_path = os.path.join(results_dir, "improvements.yaml")
        with open(improvement_path, 'w', encoding='utf-8') as f:
            yaml.dump(improvements, f, default_flow_style=False, allow_unicode=True)
    
    # 4. 스케일러 저장
    scaler_path = os.path.join(results_dir, "scaler.pkl")
    joblib.dump(data['scaler'], scaler_path)
    
    # 5. 실험 메타데이터 저장
    metadata = {
        'timestamp': timestamp,
        'method': method,
        'strategy': strategy,
        'data_shape': {
            'train': data['X_train'].shape,
            'val': data['X_val'].shape,
            'test': data['X_test'].shape
        },
        'feature_count': len(data['feature_names']),
        'target_range': {
            'min': float(data['y_train'].min()),
            'max': float(data['y_train'].max())
        }
    }
    
    metadata_path = os.path.join(results_dir, "experiment_metadata.yaml")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)
    
    logger.info("모든 결과가 저장되었습니다.")
    return results_dir


def generate_summary_report(tuner, evaluation_results, improvements, results_dir):
    """요약 보고서 생성"""
    logger.info("요약 보고서 생성 중...")
    
    # 최고 성능 모델 선택
    best_model_name, best_model = tuner.get_best_model()
    best_metrics = evaluation_results[best_model_name]['test']
    
    # 보고서 작성
    report = f"""
# 하이퍼파라미터 튜닝 결과 보고서

생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 실험 개요

- **목적**: 수박 당도 예측 모델의 하이퍼파라미터 최적화
- **데이터**: 51개 특징, 146개 샘플
- **평가 지표**: MAE (Mean Absolute Error), R² (R-squared)

## 최고 성능 모델

**모델**: {best_model_name}
- **테스트 MAE**: {best_metrics['mae']:.4f} Brix
- **테스트 R²**: {best_metrics['r2']:.4f}
- **테스트 RMSE**: {best_metrics['rmse']:.4f} Brix

## 모든 모델 성능

"""
    
    # 모든 모델 성능 추가
    for model_name, metrics in evaluation_results.items():
        test_metrics = metrics['test']
        report += f"""
### {model_name}
- MAE: {test_metrics['mae']:.4f} Brix
- R²: {test_metrics['r2']:.4f}
- RMSE: {test_metrics['rmse']:.4f} Brix
"""
    
    # 개선 사항 추가
    if improvements:
        report += "\n## 기본 모델 대비 개선\n\n"
        for model_name, imp in improvements.items():
            report += f"""
### {model_name}
- MAE 개선: {imp['mae_improvement']:+.2f}%
- R² 개선: {imp['r2_improvement']:+.2f}%
"""
    
    # 성능 목표 달성 여부
    report += f"""
## 성능 목표 달성 여부

- **MAE < 1.0 Brix**: {'✅ 달성' if best_metrics['mae'] < 1.0 else '❌ 미달성'} ({best_metrics['mae']:.4f})
- **R² > 0.8**: {'✅ 달성' if best_metrics['r2'] > 0.8 else '❌ 미달성'} ({best_metrics['r2']:.4f})

## 결론

최고 성능 모델인 **{best_model_name}**이 MAE {best_metrics['mae']:.4f} Brix, R² {best_metrics['r2']:.4f}의 
{'우수한' if best_metrics['mae'] < 1.0 and best_metrics['r2'] > 0.8 else '양호한'} 성능을 보였습니다.
"""
    
    # 보고서 저장
    report_path = os.path.join(results_dir, "TUNING_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"요약 보고서 저장: {report_path}")
    
    # 콘솔에도 핵심 내용 출력
    print("\n" + "="*60)
    print("🎯 하이퍼파라미터 튜닝 완료!")
    print("="*60)
    print(f"최고 성능 모델: {best_model_name}")
    print(f"테스트 MAE: {best_metrics['mae']:.4f} Brix")
    print(f"테스트 R²: {best_metrics['r2']:.4f}")
    print(f"목표 달성: {'✅' if best_metrics['mae'] < 1.0 and best_metrics['r2'] > 0.8 else '⚠️'}")
    print(f"결과 저장: {results_dir}")
    print("="*60)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='하이퍼파라미터 튜닝 실행')
    parser.add_argument('--method', choices=['grid', 'random'], default='random',
                        help='튜닝 방법 (default: random)')
    parser.add_argument('--strategy', choices=['quick', 'medium', 'thorough'], default='medium',
                        help='검색 전략 (default: medium)')
    parser.add_argument('--n_iter', type=int, default=50,
                        help='RandomizedSearchCV 반복 횟수 (default: 50)')
    parser.add_argument('--no_baseline', action='store_true',
                        help='기본 모델과 비교하지 않음')
    
    args = parser.parse_args()
    
    logger.info("🍉 수박 당도 예측 모델 하이퍼파라미터 튜닝 시작")
    logger.info(f"설정: method={args.method}, strategy={args.strategy}, n_iter={args.n_iter}")
    
    try:
        # 1. 환경 설정
        setup_directories()
        
        # 2. 데이터 로드
        data = load_data()
        
        # 3. 기본 모델 결과 로드
        baseline_results = {} if args.no_baseline else load_baseline_results()
        
        # 4. 하이퍼파라미터 튜닝 실행
        tuner, tuning_results = run_hyperparameter_tuning(
            data, 
            method=args.method, 
            strategy=args.strategy, 
            n_iter=args.n_iter
        )
        
        # 5. 성능 평가
        evaluation_results = evaluate_tuned_models(tuner, data)
        
        # 6. 기본 모델과 비교
        improvements = compare_with_baseline(tuner, evaluation_results, baseline_results)
        
        # 7. 결과 저장
        results_dir = save_comprehensive_results(
            tuner, evaluation_results, improvements, data, args.method, args.strategy
        )
        
        # 8. 요약 보고서 생성
        generate_summary_report(tuner, evaluation_results, improvements, results_dir)
        
        # 메모리 정리
        del tuner, data
        gc.collect()
        
        logger.info("🎉 하이퍼파라미터 튜닝이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 하이퍼파라미터 튜닝 중 오류 발생: {e}")
        sys.exit(1)
    finally:
        # 최종 메모리 정리
        gc.collect()


if __name__ == "__main__":
    main() 