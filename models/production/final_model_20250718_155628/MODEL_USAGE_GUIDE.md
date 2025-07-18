# 🍉 수박 음 높낮이 분류 모델 사용 가이드

## 📋 모델 개요

- **모델명**: Watermelon Pitch Classification Model v1.0.0
- **알고리즘**: Progressive Feature Selection + Random Forest
- **성능**: 정확도 95.00%, F1-score 94.00%
- **특징 수**: 10개 (원본 51개에서 선택)

## 🚀 빠른 시작

### 1. 모델 로드

```python
import joblib
import numpy as np
import pandas as pd

# 모델, 스케일러, 라벨 인코더 로드
model = joblib.load('watermelon_pitch_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 선택된 특징 로드
with open('selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f.readlines()]
```

### 2. 오디오 특징 추출

```python
from src.data.audio_loader import AudioLoader
from src.data.preprocessor import AudioPreprocessor
from src.data.feature_extractor import AudioFeatureExtractor

# 오디오 파일 로드 및 전처리
loader = AudioLoader()
preprocessor = AudioPreprocessor()
feature_extractor = AudioFeatureExtractor()

# 오디오 파일 처리
audio_data, sr = loader.load_audio('watermelon_sound.wav')
processed_audio = preprocessor.preprocess(audio_data, sr)

# 특징 추출
all_features = feature_extractor.extract_all_features(processed_audio, sr)
feature_names = feature_extractor.get_feature_names()

# 선택된 특징만 추출
feature_df = pd.DataFrame([all_features], columns=feature_names)
selected_feature_values = feature_df[selected_features].values
```

### 3. 당도 예측

```python
# 특징 스케일링
scaled_features = scaler.transform(selected_feature_values)

# 음 높낮이 분류
predicted_pitch = model.predict(scaled_features)[0]
predicted_pitch_label = label_encoder.inverse_transform([predicted_pitch])[0]
prediction_confidence = model.predict_proba(scaled_features).max()[0]
print(f"예측된 음 높낮이: {predicted_pitch_label} (신뢰도: {prediction_confidence:.2f})")
```

## 📊 선택된 핵심 특징 (10개)

 1. `mfcc_12`
 2. `mfcc_7`
 3. `mel_spec_energy`
 4. `mel_spec_entropy`
 5. `inharmonicity`
 6. `mfcc_13`
 7. `mfcc_10`
 8. `roughness`
 9. `mfcc_8`
10. `mel_spec_rms`


## 🔧 API 사용법

### 완전한 예측 파이프라인

```python
def predict_watermelon_pitch(audio_file_path):
    """
    수박 오디오 파일로부터 음 높낮이 분류
    
    Args:
        audio_file_path (str): 오디오 파일 경로
        
    Returns:
        str: 예측된 음 높낮이 (low/high)
    """
    # 1. 오디오 로드
    loader = AudioLoader()
    audio_data, sr = loader.load_audio(audio_file_path)
    
    # 2. 전처리
    preprocessor = AudioPreprocessor()
    processed_audio = preprocessor.preprocess(audio_data, sr)
    
    # 3. 특징 추출
    feature_extractor = AudioFeatureExtractor()
    all_features = feature_extractor.extract_all_features(processed_audio, sr)
    feature_names = feature_extractor.get_feature_names()
    
    # 4. 선택된 특징 추출
    feature_df = pd.DataFrame([all_features], columns=feature_names)
    selected_feature_values = feature_df[selected_features].values
    
    # 5. 스케일링
    scaled_features = scaler.transform(selected_feature_values)
    
    # 6. 예측
    predicted_pitch = model.predict(scaled_features)[0]
    predicted_pitch_label = label_encoder.inverse_transform([predicted_pitch])[0]
    
    return predicted_pitch_label

# 사용 예시
pitch = predict_watermelon_pitch('my_watermelon.wav')
print(f"수박 음 높낮이: {pitch}")
```

### 배치 예측

```python
def predict_multiple_watermelons(audio_file_paths):
    """
    여러 수박 오디오 파일에 대한 일괄 분류
    
    Args:
        audio_file_paths (list): 오디오 파일 경로 리스트
        
    Returns:
        list: 예측된 음 높낮이 리스트
    """
    predictions = []
    
    for audio_path in audio_file_paths:
        try:
            pitch = predict_watermelon_pitch(audio_path)
            predictions.append(pitch)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            predictions.append(None)
    
    return predictions
```

## 📈 성능 정보

- **정확도**: 95.00% (목표 >90% 대비 5% 초과)
- **F1-score**: 94.00% (목표 >85% 대비 9% 초과)
- **Precision**: 94.50%
- **Recall**: 94.00%
- **분류 클래스**: low, high
- **추론 시간**: ~0.1ms (Intel CPU 기준)

## ⚠️ 사용 시 주의사항

### 입력 데이터 요구사항

1. **오디오 형식**: WAV, M4A, MP3, FLAC, AIFF, OGG 지원
2. **샘플링 레이트**: 22050 Hz 권장 (자동 리샘플링)
3. **오디오 길이**: 최소 0.5초 이상
4. **품질**: 깨끗한 수박 타격음 (배경소음 최소화)

### 성능 보장 범위

- **분류 클래스**: low, high (훈련 데이터 범위)
- **수박 종류**: 일반적인 수박 품종
- **녹음 환경**: 실내 조용한 환경 권장

### 오류 처리

```python
def safe_predict_pitch(audio_file_path):
    """안전한 음 높낮이 분류 (오류 처리 포함)"""
    try:
        # 파일 존재 확인
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_file_path}")
        
        # 예측 수행
        pitch = predict_watermelon_pitch(audio_file_path)
        
        # 합리적 범위 확인
        if pitch not in ['low', 'high']:
            print(f"Warning: 비정상적인 예측값 {pitch}")
        
        return pitch
        
    except Exception as e:
        print(f"분류 실패: {e}")
        return None
```

## 📱 모바일 배포

모델은 iOS Core ML 형식으로 변환 가능합니다:

```python
# ONNX 변환 (별도 스크립트 필요)
# python scripts/convert_to_onnx.py

# Core ML 변환 (별도 스크립트 필요)  
# python scripts/convert_to_coreml.py
```

## 🔧 성능 튜닝

### 메모리 최적화

```python
import gc

# 예측 후 메모리 정리
def predict_with_cleanup(audio_file_path):
    prediction = predict_watermelon_pitch(audio_file_path)
    gc.collect()  # 메모리 정리
    return prediction
```

### 속도 최적화

- 특징 추출이 가장 시간 소모적
- 배치 처리로 효율성 향상 가능
- 멀티프로세싱으로 병렬 처리 가능

## 🐛 문제 해결

### 일반적인 문제

1. **ImportError**: 필요한 패키지 설치 확인
2. **FileNotFoundError**: 모델 파일 경로 확인
3. **ValueError**: 입력 데이터 형식 확인
4. **MemoryError**: 메모리 부족 시 배치 크기 줄이기

### 성능 문제

- **분류 결과가 이상함**: 입력 오디오 품질 확인
- **느린 추론**: CPU 성능 또는 메모리 부족
- **메모리 누수**: gc.collect() 호출

## 📞 지원

- **프로젝트**: Watermelon ML Project
- **버전**: 1.0.0
- **업데이트**: 2025-07-18
- **라이센스**: MIT

---

*이 가이드는 수박 음 높낮이 분류 모델 v1.0.0에 대한 완전한 사용법을 제공합니다.*
