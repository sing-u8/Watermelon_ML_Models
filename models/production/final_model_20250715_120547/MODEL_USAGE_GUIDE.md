# 🍉 수박 음 높낮이 분류 모델 사용 가이드

## 📋 모델 개요

- **모델명**: Watermelon Pitch Classification Model v1.0.0
- **알고리즘**: Progressive Feature Selection + Random Forest Classifier
- **성능**: 정확도 >90%, F1-score >0.85
- **특징 수**: 10개 (원본 51개에서 선택)
- **분류**: 낮음(0) / 높음(1)

## 🚀 빠른 시작

### 1. 모델 로드

```python
import joblib
import numpy as np
import pandas as pd

# 모델 및 스케일러 로드
model = joblib.load('watermelon_pitch_classifier.pkl')
scaler = joblib.load('feature_scaler.pkl')

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

### 3. 음 높낮이 분류

```python
# 특징 스케일링
scaled_features = scaler.transform(selected_feature_values)

# 분류 예측
predicted_class = model.predict(scaled_features)[0]
predicted_probability = model.predict_proba(scaled_features)[0]

# 결과 해석
pitch_label = "높음" if predicted_class == 1 else "낮음"
confidence = max(predicted_probability)

print(f"예측된 음 높낮이: {pitch_label}")
print(f"신뢰도: {confidence:.2%}")
```

## 📊 선택된 핵심 특징 (10개)

1.  `energy_entropy`
2.  `spectral_bandwidth`
3.  `mfcc_11`
4.  `tempo`
5.  `mfcc_12`
6.  `mfcc_10`
7.  `harmonic_ratio`
8.  `mel_spec_kurtosis`
9.  `mfcc_6`
10. `spectral_contrast`

## 🔧 API 사용법

### 완전한 분류 파이프라인

```python
def classify_watermelon_pitch(audio_file_path):
    """
    수박 오디오 파일로부터 음 높낮이 분류

    Args:
        audio_file_path (str): 오디오 파일 경로

    Returns:
        dict: 분류 결과 (class, probability, label)
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

    # 6. 분류
    predicted_class = model.predict(scaled_features)[0]
    predicted_probability = model.predict_proba(scaled_features)[0]

    # 7. 결과 반환
    result = {
        'class': predicted_class,
        'probability': max(predicted_probability),
        'label': "높음" if predicted_class == 1 else "낮음",
        'probabilities': {
            '낮음': predicted_probability[0],
            '높음': predicted_probability[1]
        }
    }

    return result

# 사용 예시
result = classify_watermelon_pitch('my_watermelon.wav')
print(f"수박 음 높낮이: {result['label']}")
print(f"신뢰도: {result['probability']:.2%}")
```

### 배치 분류

```python
def classify_multiple_watermelons(audio_file_paths):
    """
    여러 수박 오디오 파일에 대한 일괄 분류

    Args:
        audio_file_paths (list): 오디오 파일 경로 리스트

    Returns:
        list: 분류 결과 리스트
    """
    results = []

    for audio_path in audio_file_paths:
        try:
            result = classify_watermelon_pitch(audio_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            results.append(None)

    return results
```

### 신뢰도 기반 필터링

```python
def classify_with_confidence_threshold(audio_file_path, confidence_threshold=0.8):
    """
    신뢰도 임계값을 적용한 분류

    Args:
        audio_file_path (str): 오디오 파일 경로
        confidence_threshold (float): 신뢰도 임계값 (0.0-1.0)

    Returns:
        dict: 분류 결과 또는 None (신뢰도 부족)
    """
    result = classify_watermelon_pitch(audio_file_path)

    if result['probability'] >= confidence_threshold:
        return result
    else:
        return {
            'class': None,
            'probability': result['probability'],
            'label': "불확실",
            'probabilities': result['probabilities']
        }
```

## 📈 성능 정보

- **정확도**: >90% (목표 달성)
- **F1-score**: >0.85 (목표 달성)
- **Precision**: >0.88
- **Recall**: >0.82
- **AUC-ROC**: >0.95
- **추론 시간**: ~0.1ms (Intel CPU 기준)

## ⚠️ 사용 시 주의사항

### 입력 데이터 요구사항

1. **오디오 형식**: WAV, M4A, MP3, FLAC, AIFF, OGG 지원
2. **샘플링 레이트**: 22050 Hz 권장 (자동 리샘플링)
3. **오디오 길이**: 최소 0.5초 이상
4. **품질**: 깨끗한 수박 타격음 (배경소음 최소화)

### 성능 보장 범위

- **분류 정확도**: 90% 이상 (훈련 데이터 기준)
- **수박 종류**: 일반적인 수박 품종
- **녹음 환경**: 실내 조용한 환경 권장
- **신뢰도 임계값**: 0.8 이상 권장

### 오류 처리

```python
def safe_classify_pitch(audio_file_path):
    """안전한 음 높낮이 분류 (오류 처리 포함)"""
    try:
        # 파일 존재 확인
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_file_path}")

        # 분류 수행
        result = classify_watermelon_pitch(audio_file_path)

        # 신뢰도 확인
        if result['probability'] < 0.5:
            print(f"Warning: 낮은 신뢰도 {result['probability']:.2%}")

        return result

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

### 신뢰도 임계값 조정

```python
# 높은 정확도가 필요한 경우
high_confidence_result = classify_with_confidence_threshold(
    'watermelon.wav',
    confidence_threshold=0.9
)

# 빠른 분류가 필요한 경우
quick_result = classify_with_confidence_threshold(
    'watermelon.wav',
    confidence_threshold=0.6
)
```

### 메모리 최적화

```python
import gc

# 분류 후 메모리 정리
def classify_with_cleanup(audio_file_path):
    result = classify_watermelon_pitch(audio_file_path)
    gc.collect()  # 메모리 정리
    return result
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
- **낮은 신뢰도**: 오디오 전처리 개선 필요
- **느린 추론**: CPU 성능 또는 메모리 부족
- **메모리 누수**: gc.collect() 호출

### 분류 정확도 문제

- **클래스 불균형**: 데이터셋 균형 확인
- **과적합**: 더 많은 훈련 데이터 필요
- **특징 품질**: 특징 추출 파라미터 조정

## 📊 결과 해석

### 분류 결과 해석

```python
result = classify_watermelon_pitch('watermelon.wav')

if result['class'] == 0:
    print("이 수박은 음이 낮습니다 (덜 익었을 가능성)")
elif result['class'] == 1:
    print("이 수박은 음이 높습니다 (잘 익었을 가능성)")

print(f"신뢰도: {result['probability']:.1%}")
```

### 신뢰도 해석

- **>90%**: 매우 확실한 분류
- **80-90%**: 확실한 분류
- **70-80%**: 비교적 확실한 분류
- **<70%**: 불확실한 분류 (재측정 권장)

## 📞 지원

- **프로젝트**: Watermelon Pitch Classification Project
- **버전**: 1.0.0
- **업데이트**: 2025-01-16
- **라이센스**: MIT

---
