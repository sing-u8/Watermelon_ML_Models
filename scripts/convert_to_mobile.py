#!/usr/bin/env python3
"""
Mobile Model Conversion Script for iOS Deployment

This script converts the trained scikit-learn model to ONNX and Core ML formats
for mobile deployment, specifically targeting iOS applications.

Author: Watermelon ML Project Team
Date: 2025-01-15
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
import joblib
import json
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Try to import conversion libraries
try:
    import skl2onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    print(f"✅ skl2onnx version: {skl2onnx.__version__}")
    ONNX_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ONNX conversion not available: {e}")
    ONNX_AVAILABLE = False

try:
    import onnx
    print(f"✅ ONNX version: {onnx.__version__}")
    ONNX_IMPORT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ONNX import not available: {e}")
    ONNX_IMPORT_AVAILABLE = False

try:
    import coremltools as ct
    print(f"✅ Core ML Tools version: {ct.__version__}")
    COREML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Core ML conversion not available: {e}")
    COREML_AVAILABLE = False


def setup_logging(output_dir: Path) -> None:
    """Setup logging configuration."""
    log_file = output_dir / 'mobile_conversion.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_production_model() -> tuple:
    """Load the latest production model and metadata."""
    logger = logging.getLogger(__name__)
    logger.info("=== 프로덕션 모델 로드 ===")
    
    # Find latest production model
    production_dir = PROJECT_ROOT / "models" / "production"
    latest_link = production_dir / "latest"
    
    if not latest_link.exists():
        raise FileNotFoundError("최신 프로덕션 모델 링크를 찾을 수 없습니다.")
    
    model_dir = production_dir / latest_link.readlink()
    logger.info(f"모델 디렉토리: {model_dir}")
    
    # Load model and scaler
    model_file = model_dir / "watermelon_sweetness_model.pkl"
    scaler_file = model_dir / "feature_scaler.pkl"
    
    if not model_file.exists() or not scaler_file.exists():
        raise FileNotFoundError("모델 또는 스케일러 파일을 찾을 수 없습니다.")
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    
    # Load metadata
    metadata_file = model_dir / "model_metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Load selected features
    features_file = model_dir / "selected_features.json"
    with open(features_file, 'r', encoding='utf-8') as f:
        features_info = json.load(f)
    
    selected_features = features_info['features']
    
    logger.info(f"모델 로드 완료: {len(selected_features)}개 특징")
    logger.info(f"모델 성능: MAE {metadata['performance']['test_mae']:.4f}, R² {metadata['performance']['test_r2']:.4f}")
    
    return model, scaler, selected_features, metadata, model_dir


def create_pipeline_model(model, scaler, selected_features: list):
    """Create a combined pipeline model for easier conversion."""
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    logger = logging.getLogger(__name__)
    logger.info("파이프라인 모델 생성 중...")
    
    # Create a simple pipeline with scaler and model
    # Note: We'll handle feature selection separately since we're working with selected features
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model.model)  # Access the underlying sklearn model
    ])
    
    logger.info("파이프라인 모델 생성 완료")
    return pipeline


def convert_to_onnx(pipeline, selected_features: list, output_dir: Path) -> Optional[Path]:
    """Convert scikit-learn model to ONNX format."""
    logger = logging.getLogger(__name__)
    
    if not ONNX_AVAILABLE:
        logger.warning("ONNX 변환 라이브러리가 없어 ONNX 변환을 건너뜁니다.")
        return None
    
    logger.info("=== ONNX 변환 시작 ===")
    
    try:
        # Define input type (10 features, float32)
        n_features = len(selected_features)
        initial_type = [('input_features', FloatTensorType([None, n_features]))]
        
        # Convert to ONNX
        logger.info("scikit-learn → ONNX 변환 중...")
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=12  # Compatible with most deployment environments
        )
        
        # Save ONNX model
        onnx_file = output_dir / "watermelon_sweetness_model.onnx"
        with open(onnx_file, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        logger.info(f"ONNX 모델 저장: {onnx_file}")
        
        # Verify ONNX model
        if ONNX_IMPORT_AVAILABLE:
            onnx_model_check = onnx.load(str(onnx_file))
            onnx.checker.check_model(onnx_model_check)
            logger.info("ONNX 모델 검증 완료")
        
        # Test ONNX model with sample data
        test_onnx_model(onnx_file, selected_features)
        
        return onnx_file
        
    except Exception as e:
        logger.error(f"ONNX 변환 실패: {str(e)}")
        return None


def test_onnx_model(onnx_file: Path, selected_features: list) -> None:
    """Test ONNX model with sample data."""
    logger = logging.getLogger(__name__)
    
    try:
        import onnxruntime as ort
        
        # Load ONNX model
        sess = ort.InferenceSession(str(onnx_file))
        
        # Create sample input (10 features)
        sample_input = np.random.randn(1, len(selected_features)).astype(np.float32)
        
        # Run inference
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: sample_input})
        
        predicted_sweetness = result[0][0][0]  # Extract scalar value
        logger.info(f"ONNX 모델 테스트 성공: 예측값 {predicted_sweetness:.2f} Brix")
        
    except ImportError:
        logger.warning("ONNX Runtime이 없어 ONNX 모델 테스트를 건너뜁니다.")
    except Exception as e:
        logger.warning(f"ONNX 모델 테스트 실패: {str(e)}")


def convert_to_coreml(onnx_file: Optional[Path], selected_features: list, 
                     metadata: dict, output_dir: Path) -> Optional[Path]:
    """Convert ONNX model to Core ML format for iOS deployment."""
    logger = logging.getLogger(__name__)
    
    if not COREML_AVAILABLE:
        logger.warning("Core ML Tools가 없어 Core ML 변환을 건너뜁니다.")
        return None
    
    if onnx_file is None or not onnx_file.exists():
        logger.warning("ONNX 파일이 없어 Core ML 변환을 건너뜁니다.")
        return None
    
    logger.info("=== Core ML 변환 시작 ===")
    
    try:
        # Convert ONNX to Core ML
        logger.info("ONNX → Core ML 변환 중...")
        
        # Load and convert
        coreml_model = ct.convert(
            str(onnx_file),
            minimum_deployment_target=ct.target.iOS14,  # iOS 14+ compatibility
            compute_precision=ct.precision.FLOAT32,
            convert_to="mlprogram"  # New Core ML format
        )
        
        # Set model metadata
        coreml_model.author = metadata['model_info']['author']
        coreml_model.short_description = "Watermelon Sweetness Prediction Model"
        coreml_model.version = metadata['model_info']['version']
        
        # Set input/output descriptions
        coreml_model.input_description['input_features'] = "Audio features extracted from watermelon sound (10 dimensions)"
        coreml_model.output_description['variable'] = "Predicted watermelon sweetness in Brix"
        
        # Create feature descriptions
        feature_descriptions = {}
        for i, feature in enumerate(selected_features):
            feature_descriptions[f"feature_{i}_{feature}"] = f"Audio feature: {feature}"
        
        # Save Core ML model
        coreml_file = output_dir / "WatermelonSweetness.mlpackage"
        coreml_model.save(str(coreml_file))
        
        logger.info(f"Core ML 모델 저장: {coreml_file}")
        
        # Test Core ML model
        test_coreml_model(coreml_model, selected_features)
        
        return coreml_file
        
    except Exception as e:
        logger.error(f"Core ML 변환 실패: {str(e)}")
        return None


def test_coreml_model(coreml_model, selected_features: list) -> None:
    """Test Core ML model with sample data."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create sample input
        sample_input = {
            'input_features': np.random.randn(len(selected_features)).astype(np.float32)
        }
        
        # Run prediction
        result = coreml_model.predict(sample_input)
        predicted_sweetness = result['variable']
        
        logger.info(f"Core ML 모델 테스트 성공: 예측값 {predicted_sweetness:.2f} Brix")
        
    except Exception as e:
        logger.warning(f"Core ML 모델 테스트 실패: {str(e)}")


def create_mobile_metadata(original_metadata: dict, selected_features: list, 
                          onnx_file: Optional[Path], coreml_file: Optional[Path],
                          output_dir: Path) -> None:
    """Create metadata for mobile deployment."""
    logger = logging.getLogger(__name__)
    logger.info("모바일 배포 메타데이터 생성 중...")
    
    mobile_metadata = {
        'model_info': {
            'name': 'WatermelonSweetnessMobile',
            'version': original_metadata['model_info']['version'],
            'conversion_date': datetime.now().isoformat(),
            'original_model': 'RandomForest + Progressive Feature Selection',
            'mobile_formats': []
        },
        'performance': original_metadata['performance'],
        'features': {
            'count': len(selected_features),
            'names': selected_features,
            'input_shape': [len(selected_features)],
            'input_type': 'float32',
            'preprocessing_required': True
        },
        'deployment': {
            'target_platforms': ['iOS 14+'],
            'model_size_mb': 0,
            'inference_time_ms': '<100',
            'memory_requirements_mb': '<50'
        },
        'usage': {
            'input_description': 'Scaled audio features (10 dimensions) extracted from watermelon sound',
            'output_description': 'Predicted sweetness value in Brix (float)',
            'preprocessing_steps': [
                '1. Extract 51 audio features using AudioFeatureExtractor',
                '2. Select 10 specific features using feature list',
                '3. Apply StandardScaler normalization',
                '4. Input to model as float32 array'
            ]
        }
    }
    
    # Add format information
    if onnx_file and onnx_file.exists():
        mobile_metadata['model_info']['mobile_formats'].append('ONNX')
        mobile_metadata['files'] = mobile_metadata.get('files', {})
        mobile_metadata['files']['onnx'] = {
            'filename': onnx_file.name,
            'size_mb': round(onnx_file.stat().st_size / (1024 * 1024), 2),
            'format': 'ONNX v1.12+',
            'opset_version': 12
        }
    
    if coreml_file and coreml_file.exists():
        mobile_metadata['model_info']['mobile_formats'].append('Core ML')
        mobile_metadata['files'] = mobile_metadata.get('files', {})
        mobile_metadata['files']['coreml'] = {
            'filename': coreml_file.name,
            'format': 'Core ML (iOS 14+)',
            'deployment_target': 'iOS 14.0+'
        }
    
    # Save mobile metadata
    metadata_file = output_dir / 'mobile_model_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(mobile_metadata, f, indent=2, ensure_ascii=False)
    
    metadata_yaml = output_dir / 'mobile_model_metadata.yaml'
    with open(metadata_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(mobile_metadata, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"모바일 메타데이터 저장: {metadata_file}")


def create_ios_integration_guide(selected_features: list, output_dir: Path) -> None:
    """Create iOS integration guide."""
    logger = logging.getLogger(__name__)
    logger.info("iOS 통합 가이드 생성 중...")
    
    guide_content = f"""# 🍉 iOS 수박 당도 예측 모델 통합 가이드

## 📱 개요

이 가이드는 Core ML 수박 당도 예측 모델을 iOS 앱에 통합하는 방법을 설명합니다.

## 📋 요구사항

### 시스템 요구사항
- **iOS**: 14.0+
- **Xcode**: 12.0+
- **Swift**: 5.3+
- **Core ML**: 4.0+

### 모델 파일
- `WatermelonSweetness.mlpackage`: Core ML 모델
- `selected_features.json`: 특징 정보

## 🚀 통합 단계

### 1. 프로젝트에 모델 추가

```swift
// 1. WatermelonSweetness.mlpackage를 Xcode 프로젝트에 드래그 앤 드롭
// 2. Target Membership 확인
// 3. 모델 클래스 자동 생성 확인
```

### 2. Core ML 프레임워크 임포트

```swift
import CoreML
import Foundation
```

### 3. 모델 로드 및 초기화

```swift
class WatermelonPredictor {{
    private var model: WatermelonSweetness?
    
    init() {{
        do {{
            self.model = try WatermelonSweetness(configuration: MLModelConfiguration())
        }} catch {{
            print("모델 로드 실패: \\(error)")
        }}
    }}
}}
```

### 4. 특징 추출 (가상 구현)

```swift
// 실제로는 오디오 처리 라이브러리 필요
func extractAudioFeatures(from audioData: Data) -> [Float]? {{
    // TODO: 실제 오디오 특징 추출 구현
    // 현재는 테스트용 랜덤 데이터
    
    let selectedFeatures = [
{', '.join([f'        "{feature}"' for feature in selected_features])}
    ]
    
    // 랜덤 테스트 데이터 (실제로는 오디오에서 추출)
    var features: [Float] = []
    for _ in 0..<{len(selected_features)} {{
        features.append(Float.random(in: -2.0...2.0))
    }}
    
    return features
}}
```

### 5. 당도 예측 함수

```swift
func predictSweetness(audioData: Data) -> Float? {{
    guard let model = self.model else {{
        print("모델이 로드되지 않았습니다")
        return nil
    }}
    
    // 1. 오디오에서 특징 추출
    guard let features = extractAudioFeatures(from: audioData) else {{
        print("특징 추출 실패")
        return nil
    }}
    
    // 2. MLMultiArray 생성
    guard let mlArray = try? MLMultiArray(shape: [1, {len(selected_features)}], dataType: .float32) else {{
        print("MLMultiArray 생성 실패")
        return nil
    }}
    
    // 3. 특징 데이터 복사
    for (index, value) in features.enumerated() {{
        mlArray[index] = NSNumber(value: value)
    }}
    
    // 4. 예측 수행
    do {{
        let input = WatermelonSweetnessInput(input_features: mlArray)
        let output = try model.prediction(input: input)
        
        // 5. 결과 반환
        let sweetness = output.variable.floatValue
        return sweetness
    }} catch {{
        print("예측 실패: \\(error)")
        return nil
    }}
}}
```

### 6. UI 연동 예제

```swift
class ViewController: UIViewController {{
    @IBOutlet weak var recordButton: UIButton!
    @IBOutlet weak var resultLabel: UILabel!
    
    private let predictor = WatermelonPredictor()
    
    @IBAction func recordButtonTapped(_ sender: UIButton) {{
        // TODO: 실제 오디오 녹음 구현
        let testAudioData = Data() // 테스트용 빈 데이터
        
        if let sweetness = predictor.predictSweetness(audioData: testAudioData) {{
            DispatchQueue.main.async {{
                self.resultLabel.text = String(format: "당도: %.1f Brix", sweetness)
            }}
        }} else {{
            DispatchQueue.main.async {{
                self.resultLabel.text = "예측 실패"
            }}
        }}
    }}
}}
```

## 📊 선택된 특징 ({len(selected_features)}개)

모델은 다음 {len(selected_features)}개 음향 특징을 사용합니다:

"""

    for i, feature in enumerate(selected_features, 1):
        guide_content += f"{i:2d}. `{feature}`\n"

    guide_content += f"""

## ⚠️ 주의사항

### 성능 최적화

1. **메모리 관리**
   ```swift
   // 모델을 싱글톤으로 관리하여 메모리 절약
   static let shared = WatermelonPredictor()
   ```

2. **백그라운드 처리**
   ```swift
   DispatchQueue.global(qos: .userInitiated).async {{
       let result = self.predictSweetness(audioData: audioData)
       DispatchQueue.main.async {{
           // UI 업데이트
       }}
   }}
   ```

3. **에러 처리**
   ```swift
   enum PredictionError: Error {{
       case modelNotLoaded
       case featureExtractionFailed
       case predictionFailed
   }}
   ```

### 오디오 처리 고려사항

1. **실제 구현 필요**
   - 현재 가이드는 특징 추출을 가상으로 구현
   - 실제로는 오디오 신호 처리 라이브러리 필요
   - Accelerate 프레임워크 또는 외부 라이브러리 활용

2. **특징 추출 순서**
   - 정확한 순서로 {len(selected_features)}개 특징 추출 필수
   - StandardScaler와 동일한 정규화 적용 필요

3. **품질 관리**
   - 녹음 품질이 예측 정확도에 직접 영향
   - 배경 소음 최소화 권장

## 🔧 문제 해결

### 일반적인 문제

1. **모델 로드 실패**
   - 모델 파일이 앱 번들에 포함되었는지 확인
   - Target Membership 설정 확인

2. **예측 실패**
   - 입력 데이터 형식 확인 (float32, shape: [1, {len(selected_features)}])
   - 특징 추출 결과 검증

3. **성능 저하**
   - 메인 스레드에서 예측 수행 피하기
   - 모델 재사용으로 초기화 비용 절약

### 디버깅 팁

```swift
// 입력 데이터 검증
func validateInput(_ features: [Float]) -> Bool {{
    guard features.count == {len(selected_features)} else {{
        print("특징 개수 불일치: 예상 {len(selected_features)}, 실제 \\(features.count)")
        return false
    }}
    
    for (index, value) in features.enumerated() {{
        if value.isNaN || value.isInfinite {{
            print("유효하지 않은 값 at index \\(index): \\(value)")
            return false
        }}
    }}
    
    return true
}}
```

## 📞 지원

- **모델 버전**: v1.0.0
- **지원 iOS**: 14.0+
- **업데이트**: {datetime.now().strftime('%Y-%m-%d')}

---

*이 가이드는 Core ML 수박 당도 예측 모델의 iOS 통합에 대한 완전한 가이드를 제공합니다.*
"""

    guide_file = output_dir / 'iOS_INTEGRATION_GUIDE.md'
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    logger.info(f"iOS 통합 가이드 저장: {guide_file}")


def main():
    """Main conversion function."""
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = PROJECT_ROOT / "models" / "mobile" / f"mobile_models_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("🎯 모바일 모델 변환 시작")
    logger.info(f"출력 디렉토리: {output_dir}")
    
    try:
        # Load production model
        model, scaler, selected_features, metadata, model_dir = load_production_model()
        
        # Create pipeline model
        pipeline = create_pipeline_model(model, scaler, selected_features)
        
        # Convert to ONNX
        onnx_file = convert_to_onnx(pipeline, selected_features, output_dir)
        
        # Convert to Core ML
        coreml_file = convert_to_coreml(onnx_file, selected_features, metadata, output_dir)
        
        # Create mobile metadata
        create_mobile_metadata(metadata, selected_features, onnx_file, coreml_file, output_dir)
        
        # Create iOS integration guide
        create_ios_integration_guide(selected_features, output_dir)
        
        # Copy selected features file
        import shutil
        shutil.copy2(model_dir / "selected_features.json", output_dir)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("🎉 모바일 모델 변환 완료!")
        logger.info("="*60)
        
        conversion_summary = []
        if onnx_file:
            conversion_summary.append(f"✅ ONNX: {onnx_file.name}")
        else:
            conversion_summary.append("❌ ONNX: 변환 실패")
            
        if coreml_file:
            conversion_summary.append(f"✅ Core ML: {coreml_file.name}")
        else:
            conversion_summary.append("❌ Core ML: 변환 실패")
        
        for summary in conversion_summary:
            logger.info(summary)
        
        logger.info(f"특징 수: {len(selected_features)}개")
        logger.info(f"iOS 배포 준비: {output_dir}")
        logger.info("="*60)
        
        # Create symlink to latest
        latest_dir = PROJECT_ROOT / "models" / "mobile" / "latest"
        if latest_dir.exists() or latest_dir.is_symlink():
            latest_dir.unlink()
        latest_dir.symlink_to(output_dir.name)
        logger.info(f"최신 모바일 모델 링크 생성: {latest_dir}")
        
    except Exception as e:
        logger.error(f"모바일 변환 중 오류 발생: {str(e)}")
        raise
    finally:
        # Cleanup
        import gc
        gc.collect()


if __name__ == "__main__":
    main() 