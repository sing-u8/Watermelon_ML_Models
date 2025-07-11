# 🍉 수박 당도 예측 모델 - 배포 패키지

## 📦 포함된 파일

### 핵심 모델 파일
- `watermelon_sweetness_model.pkl`: 훈련된 Random Forest 모델
- `feature_scaler.pkl`: StandardScaler 객체
- `selected_features.txt`: 선택된 10개 특징 리스트
- `selected_features.json`: 특징 정보 JSON 형식

### 메타데이터
- `model_metadata.json`: 모델 상세 정보
- `model_metadata.yaml`: 모델 정보 YAML 형식
- `test_predictions.csv`: 테스트 세트 예측 결과

### 문서
- `MODEL_USAGE_GUIDE.md`: 상세 사용 가이드
- `README.md`: 이 파일
- `model_finalization.log`: 모델 생성 로그

## 🚀 빠른 배포

### 1. 의존성 설치

```bash
pip install scikit-learn>=1.3.0 pandas>=1.5.0 numpy>=1.21.0
pip install librosa>=0.9.0 soundfile>=0.12.0
pip install joblib>=1.2.0
```

### 2. 모델 로드 테스트

```python
import joblib

# 모델 로드 테스트
model = joblib.load('watermelon_sweetness_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

print("모델 로드 성공!")
print(f"모델 타입: {type(model)}")
print(f"스케일러 타입: {type(scaler)}")
```

## 📊 모델 성능

- **MAE**: 0.0974 Brix
- **R²**: 0.9887  
- **특징 수**: 10개 (원본 51개에서 선택)
- **훈련 데이터**: 124개 샘플 (train + val)
- **테스트 데이터**: 22개 샘플

## 🔧 시스템 요구사항

### 최소 요구사항
- **Python**: 3.8+
- **RAM**: 512MB
- **CPU**: 1GHz (추론용)
- **저장공간**: 50MB

### 권장 요구사항
- **Python**: 3.9+
- **RAM**: 2GB
- **CPU**: 2GHz+ (특징 추출용)
- **저장공간**: 100MB

## 📱 배포 옵션

### 1. 서버 배포
- Flask/FastAPI 웹 서비스
- Docker 컨테이너
- 클라우드 배포 (AWS, GCP, Azure)

### 2. 모바일 배포
- iOS Core ML (변환 필요)
- Android TensorFlow Lite (변환 필요)

### 3. 엣지 배포
- Raspberry Pi
- NVIDIA Jetson
- 임베디드 시스템

## ⚠️ 중요 참고사항

1. **모델 버전**: v1.0.0 (2025-01-15)
2. **데이터 범위**: 8.1-12.9 Brix 당도 범위에서 훈련
3. **오디오 형식**: WAV, M4A, MP3 등 지원
4. **성능 보장**: 표준 수박 품종, 조용한 환경에서 녹음된 데이터

## 🔄 업데이트 및 재훈련

모델 재훈련이 필요한 경우:
1. 새로운 데이터 수집
2. 특징 추출 파이프라인 실행
3. 모델 재훈련 스크립트 실행
4. 성능 검증 후 배포

## 📞 연락처

- **프로젝트**: Watermelon ML Project Team
- **생성일**: 2025년 07월 12일
- **버전**: 1.0.0

---

*이 배포 패키지는 프로덕션 환경에서 바로 사용할 수 있도록 준비되었습니다.*
