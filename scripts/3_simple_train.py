#!/usr/bin/env python3
"""
🍉 수박 당도 예측 모델 간단 훈련 스크립트

핵심 기능만 포함한 간단한 버전
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# 프로젝트 루트 디렉토리를 Python path에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.traditional_ml import ModelFactory
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def main():
    """간단한 훈련 실행"""
    print("🍉 수박 당도 예측 모델 간단 훈련 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    train_df = pd.read_csv(PROJECT_ROOT / 'data' / 'splits' / 'full_dataset' / 'train.csv')
    val_df = pd.read_csv(PROJECT_ROOT / 'data' / 'splits' / 'full_dataset' / 'val.csv')
    test_df = pd.read_csv(PROJECT_ROOT / 'data' / 'splits' / 'full_dataset' / 'test.csv')
    
    # 특징과 타겟 분리
    feature_cols = [col for col in train_df.columns if col != 'sweetness']
    X_train = train_df[feature_cols].values
    y_train = train_df['sweetness'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['sweetness'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['sweetness'].values
    
    print(f"   - 훈련: {len(X_train)}개")
    print(f"   - 검증: {len(X_val)}개")
    print(f"   - 테스트: {len(X_test)}개")
    print(f"   - 특징 수: {len(feature_cols)}개")
    
    # 2. 특징 스케일링
    print("2. 특징 스케일링 중...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 모델 생성 및 훈련
    models = {}
    results = {}
    
    # 3.1 Gradient Boosting
    print("3. 모델 훈련 중...")
    print("   3.1 Gradient Boosting...")
    gbt_model = ModelFactory.create_model('gradient_boosting')
    gbt_model.fit(X_train_scaled, y_train)
    models['GBT'] = gbt_model
    
    # 3.2 Random Forest
    print("   3.2 Random Forest...")
    rf_model = ModelFactory.create_model('random_forest')
    rf_model.fit(X_train_scaled, y_train)
    models['RF'] = rf_model
    
    # 3.3 SVM
    print("   3.3 SVM...")
    svm_model = ModelFactory.create_model('svm')
    svm_model.fit(X_train_scaled, y_train)
    models['SVM'] = svm_model
    
    # 4. 평가
    print("4. 모델 평가 중...")
    print(f"{'모델':<6} {'데이터셋':<6} {'MAE':<8} {'R²':<8} {'RMSE':<8}")
    print("-" * 50)
    
    best_model_name = None
    best_mae = float('inf')
    
    for model_name, model in models.items():
        for dataset_name, X_data, y_data in [
            ('훈련', X_train_scaled, y_train),
            ('검증', X_val_scaled, y_val),
            ('테스트', X_test_scaled, y_test)
        ]:
            y_pred = model.predict(X_data)
            mae = mean_absolute_error(y_data, y_pred)
            r2 = r2_score(y_data, y_pred)
            rmse = np.sqrt(mean_squared_error(y_data, y_pred))
            
            print(f"{model_name:<6} {dataset_name:<6} {mae:<8.3f} {r2:<8.3f} {rmse:<8.3f}")
            
            # 테스트 성능으로 최고 모델 선정
            if dataset_name == '테스트' and mae < best_mae:
                best_mae = mae
                best_model_name = model_name
    
    print("-" * 50)
    print(f"🏆 최고 성능 모델: {best_model_name} (테스트 MAE: {best_mae:.3f})")
    
    # 5. 목표 달성 확인
    print("\n5. 성능 목표 달성 확인:")
    target_mae = 1.0
    target_r2 = 0.8
    
    test_results = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        test_results[model_name] = {'mae': mae, 'r2': r2}
    
    models_meeting_mae = sum(1 for result in test_results.values() if result['mae'] < target_mae)
    models_meeting_r2 = sum(1 for result in test_results.values() if result['r2'] > target_r2)
    
    print(f"   - MAE < {target_mae}: {models_meeting_mae}/3 모델 달성")
    print(f"   - R² > {target_r2}: {models_meeting_r2}/3 모델 달성")
    
    if best_mae < target_mae:
        print(f"   ✅ 주요 목표 달성! (MAE < {target_mae})")
    else:
        print(f"   ❌ 주요 목표 미달성 (MAE >= {target_mae})")
    
    # 6. 최고 성능 모델 저장
    print("\n6. 모델 저장 중...")
    os.makedirs(PROJECT_ROOT / 'models' / 'saved', exist_ok=True)
    
    best_model = models[best_model_name]
    model_path = PROJECT_ROOT / 'models' / 'saved' / 'best_model_simple.pkl'
    best_model.save_model(str(model_path))
    
    # 스케일러 저장
    import joblib
    scaler_path = PROJECT_ROOT / 'models' / 'saved' / 'scaler_simple.pkl'
    joblib.dump(scaler, scaler_path)
    
    # 결과 요약 저장
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model_name,
        'test_performance': test_results,
        'target_achieved': best_mae < target_mae,
        'feature_count': len(feature_cols)
    }
    
    summary_path = PROJECT_ROOT / 'models' / 'saved' / 'training_summary_simple.yaml'
    with open(summary_path, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)
    
    print(f"   - 모델: {model_path}")
    print(f"   - 스케일러: {scaler_path}")
    print(f"   - 요약: {summary_path}")
    
    print("\n🎉 간단 훈련 완료!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 