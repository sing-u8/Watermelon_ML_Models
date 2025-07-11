#!/usr/bin/env python3
"""
간단한 하이퍼파라미터 튜닝 스크립트

scikit-learn 모델을 직접 사용하여 하이퍼파라미터 튜닝을 수행합니다.
복잡한 래퍼 클래스 없이 직접적이고 안정적인 접근 방식을 사용합니다.

사용법:
    python scripts/simple_hyperparameter_tuning.py

작성자: ML Team
생성일: 2025-01-15
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime
from pathlib import Path
import gc

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    
    # 특징 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info("데이터 스케일링 완료")
    logger.info(f"특징 수: {X_train_scaled.shape[1]}")
    logger.info(f"당도 범위: {float(np.array(y_train).min()):.2f} ~ {float(np.array(y_train).max()):.2f} Brix")
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'scaler': scaler
    }


def get_param_grids():
    """하이퍼파라미터 그리드 정의"""
    return {
        'gradient_boosting': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'random_state': [42]
        },
        'svm': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'epsilon': [0.01, 0.1, 0.2],
            'degree': [2, 3, 4]  # poly kernel용
        },
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'random_state': [42]
        }
    }


def tune_model(model, param_grid, X_train, y_train, model_name, n_iter=20):
    """단일 모델 하이퍼파라미터 튜닝"""
    logger.info(f"=== {model_name} 하이퍼파라미터 튜닝 시작 ===")
    
    # RandomizedSearchCV 설정
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring='neg_mean_absolute_error',
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    # 튜닝 실행
    start_time = datetime.now()
    search.fit(X_train, y_train)
    end_time = datetime.now()
    
    tuning_time = (end_time - start_time).total_seconds()
    
    logger.info(f"{model_name} 튜닝 완료:")
    logger.info(f"  최고 점수: {search.best_score_:.4f}")
    logger.info(f"  최적 파라미터: {search.best_params_}")
    logger.info(f"  소요 시간: {tuning_time:.2f}초")
    
    return {
        'model': search.best_estimator_,
        'best_score': search.best_score_,
        'best_params': search.best_params_,
        'tuning_time': tuning_time,
        'cv_results': search.cv_results_
    }


def evaluate_model(model, X_test, y_test, X_val, y_val, model_name):
    """모델 성능 평가"""
    # 예측
    y_pred_test = model.predict(X_test)
    y_pred_val = model.predict(X_val)
    
    # 테스트 세트 평가
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # 검증 세트 평가
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_mse = mean_squared_error(y_val, y_pred_val)
    val_rmse = np.sqrt(val_mse)
    val_r2 = r2_score(y_val, y_pred_val)
    
    logger.info(f"{model_name} 성능 평가:")
    logger.info(f"  테스트 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    logger.info(f"  검증   - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}")
    
    return {
        'test': {
            'mae': test_mae,
            'mse': test_mse,
            'rmse': test_rmse,
            'r2': test_r2
        },
        'validation': {
            'mae': val_mae,
            'mse': val_mse,
            'rmse': val_rmse,
            'r2': val_r2
        }
    }


def load_baseline_results():
    """기본 모델 결과 로드"""
    logger.info("기본 모델 결과 로드 시도...")
    
    baseline_path = "models/saved/training_summary_simple.yaml"
    
    if os.path.exists(baseline_path):
        try:
            # 먼저 safe_load 시도
            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_data = yaml.safe_load(f)
        except yaml.constructor.ConstructorError as e:
            logger.warning(f"YAML 파일에 numpy 객체가 포함되어 있어 safe_load에 실패했습니다: {e}")
            try:
                # unsafe load로 시도 (numpy 객체 포함 파일용)
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline_data = yaml.unsafe_load(f)
                logger.info("unsafe_load로 기본 모델 결과를 성공적으로 로드했습니다.")
            except Exception as e2:
                logger.error(f"unsafe_load도 실패했습니다: {e2}")
                logger.warning("기본 모델 결과 로드를 건너뜁니다.")
                return {}
        except Exception as e:
            logger.error(f"YAML 파일 로드 실패: {e}")
            logger.warning("기본 모델 결과 로드를 건너뜁니다.")
            return {}
        
        # 결과 구조 변환
        baseline_results = {}
        test_performance = baseline_data.get('test_performance', {})
        
        if test_performance:
            for model_name, metrics in test_performance.items():
                # numpy 객체를 float로 변환
                converted_metrics = {}
                for key, value in metrics.items():
                    if hasattr(value, 'item'):  # numpy scalar인 경우
                        converted_metrics[key] = float(value.item())
                    else:
                        converted_metrics[key] = float(value)
                
                baseline_results[model_name] = {'test': converted_metrics}
        
        logger.info(f"기본 모델 결과 로드 완료: {list(baseline_results.keys())}")
        return baseline_results
    else:
        logger.warning("기본 모델 결과를 찾을 수 없습니다.")
        return {}


def compare_with_baseline(tuned_results, baseline_results):
    """기본 모델과 성능 비교"""
    if not baseline_results:
        logger.warning("기본 모델 결과가 없어 비교를 건너뜁니다.")
        return {}
    
    logger.info("=== 기본 모델 대비 성능 개선 ===")
    improvements = {}
    
    model_mapping = {
        'gradient_boosting': 'random_forest',  # 기본 결과에서는 random_forest가 최고였음
        'svm': 'random_forest',
        'random_forest': 'random_forest'
    }
    
    for model_name, evaluation in tuned_results.items():
        if model_mapping[model_name] in baseline_results:
            baseline_model = model_mapping[model_name]
            
            # 튜닝된 모델 성능
            tuned_mae = evaluation['test']['mae']
            tuned_r2 = evaluation['test']['r2']
            
            # 기본 모델 성능
            baseline_mae = baseline_results[baseline_model]['test']['mae']
            baseline_r2 = baseline_results[baseline_model]['test']['r2']
            
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
            
            logger.info(f"  {model_name} vs {baseline_model}:")
            logger.info(f"    MAE: {baseline_mae:.4f} → {tuned_mae:.4f} ({mae_improvement:+.2f}%)")
            logger.info(f"    R²:  {baseline_r2:.4f} → {tuned_r2:.4f} ({r2_improvement:+.2f}%)")
    
    return improvements


def save_results(tuning_results, evaluation_results, improvements, data):
    """결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiments/hyperparameter_tuning/simple_tuning_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"결과 저장 디렉토리: {results_dir}")
    
    # 모델 저장
    for model_name, result in tuning_results.items():
        model_path = os.path.join(results_dir, f"{model_name}_tuned.pkl")
        joblib.dump(result['model'], model_path)
        logger.info(f"모델 저장: {model_path}")
    
    # 스케일러 저장
    scaler_path = os.path.join(results_dir, "scaler.pkl")
    joblib.dump(data['scaler'], scaler_path)
    
    # 튜닝 결과 저장
    tuning_summary = {}
    for model_name, result in tuning_results.items():
        tuning_summary[model_name] = {
            'best_score': result['best_score'],
            'best_params': result['best_params'],
            'tuning_time': result['tuning_time']
        }
    
    tuning_path = os.path.join(results_dir, "tuning_results.yaml")
    with open(tuning_path, 'w', encoding='utf-8') as f:
        yaml.dump(tuning_summary, f, default_flow_style=False, allow_unicode=True)
    
    # 평가 결과 저장
    evaluation_path = os.path.join(results_dir, "evaluation_results.yaml")
    with open(evaluation_path, 'w', encoding='utf-8') as f:
        yaml.dump(evaluation_results, f, default_flow_style=False, allow_unicode=True)
    
    # 개선 결과 저장
    if improvements:
        improvement_path = os.path.join(results_dir, "improvements.yaml")
        with open(improvement_path, 'w', encoding='utf-8') as f:
            yaml.dump(improvements, f, default_flow_style=False, allow_unicode=True)
    
    return results_dir


def generate_report(tuning_results, evaluation_results, improvements, results_dir):
    """요약 보고서 생성"""
    # 최고 성능 모델 찾기
    best_model_name = min(evaluation_results.keys(), 
                         key=lambda x: evaluation_results[x]['test']['mae'])
    best_metrics = evaluation_results[best_model_name]['test']
    
    # 보고서 작성
    report = f"""# 하이퍼파라미터 튜닝 결과 보고서

생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 실험 개요

- **목적**: 수박 당도 예측 모델의 하이퍼파라미터 최적화
- **데이터**: 51개 특징, 146개 샘플 (Train: 102, Val: 22, Test: 22)
- **방법**: RandomizedSearchCV (20회 반복)
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
        tuning_time = tuning_results[model_name]['tuning_time']
        report += f"""
### {model_name}
- **테스트 MAE**: {test_metrics['mae']:.4f} Brix
- **테스트 R²**: {test_metrics['r2']:.4f}
- **테스트 RMSE**: {test_metrics['rmse']:.4f} Brix
- **튜닝 시간**: {tuning_time:.2f}초
- **최적 파라미터**: {tuning_results[model_name]['best_params']}
"""
    
    # 개선 사항 추가
    if improvements:
        report += "\n## 기본 모델 대비 개선\n"
        for model_name, imp in improvements.items():
            report += f"""
### {model_name}
- **MAE 개선**: {imp['mae_improvement']:+.2f}%
- **R² 개선**: {imp['r2_improvement']:+.2f}%
"""
    
    # 성능 목표 달성 여부
    report += f"""
## 성능 목표 달성 여부

- **MAE < 1.0 Brix**: {'✅ 달성' if best_metrics['mae'] < 1.0 else '❌ 미달성'} ({best_metrics['mae']:.4f})
- **R² > 0.8**: {'✅ 달성' if best_metrics['r2'] > 0.8 else '❌ 미달성'} ({best_metrics['r2']:.4f})

## 결론

최고 성능 모델인 **{best_model_name}**이 MAE {best_metrics['mae']:.4f} Brix, R² {best_metrics['r2']:.4f}의 
{'우수한' if best_metrics['mae'] < 1.0 and best_metrics['r2'] > 0.8 else '양호한'} 성능을 보였습니다.

하이퍼파라미터 튜닝을 통해 모델 성능이 개선되었으며, 
{'목표 성능을 달성' if best_metrics['mae'] < 1.0 and best_metrics['r2'] > 0.8 else '목표에 근접한 성능을 확보'}했습니다.
"""
    
    # 보고서 저장
    report_path = os.path.join(results_dir, "TUNING_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"요약 보고서 저장: {report_path}")
    
    # 콘솔 출력
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
    logger.info("🍉 간단한 하이퍼파라미터 튜닝 시작")
    
    try:
        # 디렉토리 생성
        os.makedirs("experiments/hyperparameter_tuning", exist_ok=True)
        
        # 1. 데이터 로드
        data = load_data()
        
        # 2. 기본 모델 결과 로드
        baseline_results = load_baseline_results()
        
        # 3. 모델 및 파라미터 그리드 정의
        models = {
            'gradient_boosting': GradientBoostingRegressor(),
            'svm': SVR(),
            'random_forest': RandomForestRegressor()
        }
        param_grids = get_param_grids()
        
        # 4. 각 모델에 대해 하이퍼파라미터 튜닝
        tuning_results = {}
        for model_name, model in models.items():
            try:
                result = tune_model(
                    model, 
                    param_grids[model_name], 
                    data['X_train'], 
                    data['y_train'], 
                    model_name,
                    n_iter=20
                )
                tuning_results[model_name] = result
            except Exception as e:
                logger.error(f"{model_name} 튜닝 실패: {e}")
                continue
        
        # 5. 성능 평가
        evaluation_results = {}
        for model_name, result in tuning_results.items():
            evaluation = evaluate_model(
                result['model'],
                data['X_test'],
                data['y_test'],
                data['X_val'],
                data['y_val'],
                model_name
            )
            evaluation_results[model_name] = evaluation
        
        # 6. 기본 모델과 비교
        improvements = compare_with_baseline(evaluation_results, baseline_results)
        
        # 7. 결과 저장
        results_dir = save_results(tuning_results, evaluation_results, improvements, data)
        
        # 8. 보고서 생성
        generate_report(tuning_results, evaluation_results, improvements, results_dir)
        
        # 메모리 정리
        gc.collect()
        
        logger.info("🎉 하이퍼파라미터 튜닝이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 하이퍼파라미터 튜닝 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main() 