import sys
import os
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent.parent))

from src.data.audio_loader import AudioLoader
from src.data.preprocessor import AudioPreprocessor
from src.data.feature_extractor import AudioFeatureExtractor

# 1. 최신 production 모델 디렉토리 자동 탐색
def get_latest_model_dir():
    production_dir = Path('models/production')
    latest_link = production_dir / 'latest'
    if latest_link.exists() and latest_link.is_symlink():
        return latest_link.resolve()
    model_dirs = [d for d in production_dir.iterdir() if d.is_dir() and d.name.startswith('final_model_')]
    if not model_dirs:
        raise FileNotFoundError("production 폴더에 모델이 없습니다.")
    model_dirs = sorted(model_dirs, key=lambda d: d.stat().st_mtime, reverse=True)
    return model_dirs[0]

# 2. 모델, 스케일러, 특징 리스트 로드
def load_model_assets(model_dir):
    model = joblib.load(model_dir / 'watermelon_sweetness_model.pkl')
    scaler = joblib.load(model_dir / 'feature_scaler.pkl')
    with open(model_dir / 'selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    return model, scaler, selected_features

# 3. 원본 샘플 메타데이터 로드 (모드별로 샘플 수 조정)
def load_metadata(mode='all'):
    meta_path = Path('data/watermelon_metadata.csv')
    if not meta_path.exists():
        raise FileNotFoundError(f"메타데이터 파일이 없습니다: {meta_path}")
    
    metadata = pd.read_csv(meta_path)
    
    if mode == 'all':
        # 전체 샘플 (173개)
        return metadata
    elif mode == 'train':
        # 훈련에 사용된 샘플만 (121개)
        # 메타데이터의 처음 121개를 훈련 샘플로 가정
        return metadata.head(121)
    else:
        raise ValueError(f"지원하지 않는 모드입니다: {mode}. 'all' 또는 'train'을 사용하세요.")

# 4. 예측 및 평가
def main():
    # 명령행 인수 파싱
    parser = argparse.ArgumentParser(description='수박 당도 예측 모델 평가')
    parser.add_argument('--mode', choices=['all', 'train'], default='all',
                       help='평가할 샘플 모드: all(전체 173개) 또는 train(훈련용 121개)')
    args = parser.parse_args()
    
    print(f"=== 수박 당도 예측 모델 평가 ({args.mode} 모드) ===")
    print(f"평가 모드: {args.mode}")
    if args.mode == 'all':
        print("평가 대상: 전체 원본 샘플 (173개)")
    else:
        print("평가 대상: 훈련에 사용된 샘플 (121개)")
    
    print("\n최신 production 모델 자동 로드 중...")
    model_dir = get_latest_model_dir()
    print(f"모델 디렉토리: {model_dir}")
    model, scaler, selected_features = load_model_assets(model_dir)

    metadata = load_metadata(args.mode)
    print(f"로드된 샘플 수: {len(metadata)}개")
    
    loader = AudioLoader(sample_rate=16000, mono=True)
    preprocessor = AudioPreprocessor()
    feature_extractor = AudioFeatureExtractor()

    actual = []
    predicted = []
    watermelon_ids = []
    errors = []
    failed = 0

    for idx, row in metadata.iterrows():
        file_path = str(row['file_path'])
        true_sweetness = row['sweetness']
        watermelon_id = row['watermelon_id']
        try:
            audio_data, sr = loader.load_audio(file_path)
            processed_audio, _ = preprocessor.preprocess_audio(audio_data, sr)
            all_features = feature_extractor.extract_all_features(processed_audio, sr)
            feature_names = pd.Index(feature_extractor.get_feature_names())
            feature_df = pd.DataFrame([all_features], columns=feature_names)
            selected_feature_values = feature_df[selected_features].values
            scaled_features = scaler.transform(selected_feature_values)
            pred = model.predict(scaled_features)[0]
            actual.append(true_sweetness)
            predicted.append(pred)
            watermelon_ids.append(watermelon_id)
            errors.append(abs(true_sweetness - pred))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed += 1

    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n=== {args.mode.upper()} 모드 평가 결과 ===")
    print(f"평가 샘플 수: {len(actual)}개")
    print(f"MAE: {mae:.4f} Brix")
    print(f"RMSE: {rmse:.4f} Brix")
    print(f"R²: {r2:.4f}")
    if failed > 0:
        print(f"처리 실패 샘플: {failed}개")

    # 결과 저장
    result_df = pd.DataFrame({
        'watermelon_id': watermelon_ids,
        'actual_sweetness': actual,
        'predicted_sweetness': predicted,
        'abs_error': np.abs(actual - predicted)
    })
    result_csv = model_dir / f'{args.mode}_mode_predictions.csv'
    result_df.to_csv(result_csv, index=False)
    print(f"예측 결과 저장: {result_csv}")

    # Visualization
    plt.figure(figsize=(15, 5))
    
    # (1) Actual vs Predicted Scatter Plot
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=actual, y=predicted, alpha=0.7)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', label='y=x')
    plt.xlabel('Actual Sweetness (Brix)')
    plt.ylabel('Predicted Sweetness (Brix)')
    plt.title(f'Actual vs Predicted Sweetness ({args.mode.upper()} Mode)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # (2) Residual Plot
    plt.subplot(1, 3, 2)
    residuals = predicted - actual
    sns.scatterplot(x=actual, y=residuals, alpha=0.7)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Actual Sweetness (Brix)')
    plt.ylabel('Residual (Predicted - Actual)')
    plt.title(f'Residual Plot ({args.mode.upper()} Mode)')
    plt.grid(True, alpha=0.3)
    
    # (3) Error Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(errors, bins=20, kde=True)
    plt.xlabel('Absolute Error (Brix)')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution ({args.mode.upper()} Mode)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 