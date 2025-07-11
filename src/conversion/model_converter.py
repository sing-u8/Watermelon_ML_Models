"""
🍉 수박 당도 예측 모델 변환기
scikit-learn → ONNX → Core ML 변환 파이프라인

Author: Watermelon ML Team
Date: 2025-01-16 
Compatible with: scikit-learn 1.5.1
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# 변환 라이브러리들
import onnx
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import coremltools

# sklearn 모델들
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class WatermelonModelConverter:
    """
    수박 당도 예측 모델 변환 통합 클래스
    scikit-learn → ONNX → Core ML 전체 파이프라인 지원
    """
    
    def __init__(self, 
                 model_path: str = None,
                 scaler_path: str = None,
                 features_path: str = None,
                 output_dir: str = "models/mobile"):
        """
        변환기 초기화
        
        Args:
            model_path: 저장된 모델 파일 경로
            scaler_path: 저장된 스케일러 파일 경로  
            features_path: 선택된 특징 정보 파일 경로
            output_dir: 변환된 모델 저장 디렉토리
        """
        self.model_path = model_path or "models/production/latest/watermelon_sweetness_model.pkl"
        self.scaler_path = scaler_path or "models/production/latest/feature_scaler.pkl"
        self.features_path = features_path or "models/production/latest/selected_features.json"
        self.output_dir = Path(output_dir)
        
        # 출력 디렉토리 생성
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        self.logger = self._setup_logging()
        
        # 모델 관련 속성
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.feature_names = None
        self.n_features = None
        
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('WatermelonConverter')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_production_models(self) -> bool:
        """프로덕션 모델들을 로드합니다"""
        try:
            self.logger.info("🔄 Loading production models...")
            
            # 1. 모델 로드
            self.logger.info(f"   Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.logger.info(f"   ✅ Model loaded: {type(self.model).__name__}")
            
            # 2. 스케일러 로드
            self.logger.info(f"   Loading scaler from: {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            self.logger.info(f"   ✅ Scaler loaded: {type(self.scaler).__name__}")
            
            # 3. 선택된 특징 정보 로드
            self.logger.info(f"   Loading features from: {self.features_path}")
            with open(self.features_path, 'r') as f:
                features_info = json.load(f)
            
            # JSON 파일의 실제 키 이름 사용 ('features')
            self.selected_features = features_info['features']
            self.feature_names = self.selected_features
            self.n_features = len(self.selected_features)
            
            self.logger.info(f"   ✅ Features loaded: {self.n_features} features")
            self.logger.info(f"   Features: {self.feature_names}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load models: {str(e)}")
            return False
    
    def convert_to_onnx(self, 
                       output_name: str = "watermelon_predictor.onnx") -> bool:
        """scikit-learn 모델을 ONNX 형식으로 변환"""
        try:
            self.logger.info("🔄 Converting to ONNX format...")
            
            if self.model is None or self.scaler is None:
                raise ValueError("Models not loaded. Call load_production_models() first.")
            
            # 커스텀 Watermelon 클래스에서 실제 sklearn 모델 추출
            actual_model = self.model
            if hasattr(self.model, 'model') and self.model.model is not None:
                actual_model = self.model.model
                self.logger.info(f"   Extracted internal model: {type(actual_model).__name__}")
            else:
                self.logger.info(f"   Using model directly: {type(actual_model).__name__}")
            
            # ONNX 변환을 위한 초기 타입 정의
            initial_type = [('float_input', FloatTensorType([None, self.n_features]))]
            
            self.logger.info(f"   Input shape: [None, {self.n_features}]")
            self.logger.info(f"   Converting model type: {type(actual_model).__name__}")
            
            # 실제 sklearn 모델을 ONNX로 변환
            # 회귀 모델이므로 분류 관련 옵션 제거
            onnx_model = convert_sklearn(
                actual_model,  # 추출된 실제 sklearn 모델 사용
                initial_types=initial_type,
                target_opset=12  # 호환성을 위해 안정적인 opset 사용
                # RandomForestRegressor는 회귀 모델이므로 옵션 생략
            )
            
            # ONNX 모델 저장
            output_path = self.output_dir / output_name
            onnx.save_model(onnx_model, str(output_path))
            
            # ONNX 모델 검증
            onnx.checker.check_model(onnx_model)
            
            self.logger.info(f"   ✅ ONNX model saved: {output_path}")
            self.logger.info(f"   Model size: {output_path.stat().st_size / 1024:.1f} KB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ ONNX conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def convert_onnx_to_coreml(self,
                              onnx_path: str = None,
                              output_name: str = "watermelon_predictor.mlmodel") -> bool:
        """ONNX 모델을 Core ML 형식으로 변환"""
        try:
            self.logger.info("🔄 Converting ONNX to Core ML format...")
            
            # ONNX 모델 경로 설정
            if onnx_path is None:
                onnx_path = self.output_dir / "watermelon_predictor.onnx"
            
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
            
            self.logger.info(f"   Loading ONNX model: {onnx_path}")
            
            # ONNX → Core ML 변환 (최신 API 사용)
            try:
                # 최신 coremltools API 시도
                coreml_model = coremltools.convert(
                    model=str(onnx_path),
                    source='onnx',
                    minimum_deployment_target=coremltools.target.iOS14,
                    compute_precision=coremltools.precision.FLOAT32
                )
            except Exception as e:
                self.logger.warning(f"Modern API failed: {e}")
                # 대안: 구형 API 시도
                try:
                    import coremltools.converters
                    if hasattr(coremltools.converters, 'convert'):
                        coreml_model = coremltools.converters.convert(
                            model=str(onnx_path),
                            source='onnx'
                        )
                    else:
                        raise AttributeError("ONNX converter not available")
                except Exception as e2:
                    self.logger.error(f"Both APIs failed: {e2}")
                    raise e2
            
            # Core ML 모델 메타데이터 설정
            coreml_model.author = "Watermelon ML Team"
            coreml_model.license = "MIT"
            coreml_model.short_description = "수박 당도 예측 모델 (Traditional ML)"
            coreml_model.version = "1.0.0"
            
            # 입력/출력 설명 설정
            coreml_model.input_description['float_input'] = f"수박 오디오 특징 벡터 ({self.n_features}개 특징)"
            if hasattr(coreml_model, 'output_description'):
                output_keys = list(coreml_model.get_spec().description.output)
                if output_keys:
                    coreml_model.output_description[output_keys[0].name] = "예측된 당도 (Brix)"
            
            # Core ML 모델 저장
            output_path = self.output_dir / output_name
            coreml_model.save(str(output_path))
            
            self.logger.info(f"   ✅ Core ML model saved: {output_path}")
            self.logger.info(f"   Model size: {output_path.stat().st_size / 1024:.1f} KB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Core ML conversion failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def validate_conversion(self, 
                           num_test_samples: int = 10) -> Dict[str, Any]:
        """변환된 모델들의 정확도를 검증"""
        try:
            self.logger.info("🔄 Validating converted models...")
            
            # 테스트 데이터 생성 (랜덤)
            np.random.seed(42)
            test_data = np.random.randn(num_test_samples, self.n_features).astype(np.float32)
            
            results = {
                'original': [],
                'onnx': [],
                'coreml': []
            }
            
            # 1. 원본 모델 예측
            scaled_data = self.scaler.transform(test_data)
            original_predictions = self.model.predict(scaled_data)
            results['original'] = original_predictions.tolist()
            
            # 2. ONNX 모델 예측 (있는 경우)
            onnx_path = self.output_dir / "watermelon_predictor.onnx"
            if onnx_path.exists():
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                onnx_predictions = session.run(None, {'float_input': scaled_data})[0]
                results['onnx'] = onnx_predictions.flatten().tolist()
            
            # 3. Core ML 모델 예측 (있는 경우)
            coreml_path = self.output_dir / "watermelon_predictor.mlmodel"
            if coreml_path.exists():
                import coremltools
                coreml_model = coremltools.models.MLModel(str(coreml_path))
                coreml_predictions = []
                for i in range(num_test_samples):
                    input_dict = {'float_input': scaled_data[i:i+1]}
                    prediction = coreml_model.predict(input_dict)
                    # 출력 키 이름은 모델에 따라 다를 수 있음
                    pred_value = list(prediction.values())[0]
                    if isinstance(pred_value, np.ndarray):
                        pred_value = pred_value.item()
                    coreml_predictions.append(pred_value)
                results['coreml'] = coreml_predictions
            
            # 4. 정확도 비교
            validation_results = {
                'test_samples': num_test_samples,
                'predictions': results,
                'accuracy': {}
            }
            
            # 원본 vs ONNX 비교
            if results['onnx']:
                diff = np.abs(np.array(results['original']) - np.array(results['onnx']))
                validation_results['accuracy']['onnx_mae'] = float(np.mean(diff))
                validation_results['accuracy']['onnx_max_diff'] = float(np.max(diff))
            
            # 원본 vs Core ML 비교
            if results['coreml']:
                diff = np.abs(np.array(results['original']) - np.array(results['coreml']))
                validation_results['accuracy']['coreml_mae'] = float(np.mean(diff))
                validation_results['accuracy']['coreml_max_diff'] = float(np.max(diff))
            
            self.logger.info("✅ Validation completed")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Validation failed: {str(e)}")
            return {'error': str(e)}
    
    def convert_full_pipeline(self) -> Dict[str, bool]:
        """전체 변환 파이프라인 실행"""
        self.logger.info("🚀 Starting full conversion pipeline...")
        
        results = {
            'model_loaded': False,
            'onnx_converted': False,
            'coreml_converted': False
        }
        
        # 1. 모델 로드
        if self.load_production_models():
            results['model_loaded'] = True
            
            # 2. ONNX 변환
            if self.convert_to_onnx():
                results['onnx_converted'] = True
                
                # 3. Core ML 변환
                if self.convert_onnx_to_coreml():
                    results['coreml_converted'] = True
        
        # 결과 요약
        self.logger.info("📊 Conversion Pipeline Results:")
        self.logger.info(f"   Model Loading: {'✅' if results['model_loaded'] else '❌'}")
        self.logger.info(f"   ONNX Conversion: {'✅' if results['onnx_converted'] else '❌'}")
        self.logger.info(f"   Core ML Conversion: {'✅' if results['coreml_converted'] else '❌'}")
        
        return results
    
    def save_conversion_report(self, validation_results: Dict[str, Any]) -> str:
        """변환 결과 보고서 저장"""
        report_path = self.output_dir / "conversion_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 🍉 수박 당도 예측 모델 변환 보고서\n\n")
            f.write(f"**생성일**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**scikit-learn 버전**: 1.5.1\n")
            f.write(f"**특징 수**: {self.n_features}\n\n")
            
            f.write("## 📂 변환된 파일들\n\n")
            
            onnx_path = self.output_dir / "watermelon_predictor.onnx"
            coreml_path = self.output_dir / "watermelon_predictor.mlmodel"
            
            if onnx_path.exists():
                size_kb = onnx_path.stat().st_size / 1024
                f.write(f"- `watermelon_predictor.onnx` ({size_kb:.1f} KB)\n")
            
            if coreml_path.exists():
                size_kb = coreml_path.stat().st_size / 1024
                f.write(f"- `watermelon_predictor.mlmodel` ({size_kb:.1f} KB)\n")
            
            f.write("\n## 🎯 정확도 검증 결과\n\n")
            
            if 'accuracy' in validation_results:
                acc = validation_results['accuracy']
                if 'onnx_mae' in acc:
                    f.write(f"**ONNX 모델 정확도**:\n")
                    f.write(f"- 평균 절대 오차 (MAE): {acc['onnx_mae']:.6f}\n")
                    f.write(f"- 최대 오차: {acc['onnx_max_diff']:.6f}\n\n")
                
                if 'coreml_mae' in acc:
                    f.write(f"**Core ML 모델 정확도**:\n")
                    f.write(f"- 평균 절대 오차 (MAE): {acc['coreml_mae']:.6f}\n")
                    f.write(f"- 최대 오차: {acc['coreml_max_diff']:.6f}\n\n")
            
            f.write("## 🍉 선택된 특징들\n\n")
            for i, feature in enumerate(self.feature_names, 1):
                f.write(f"{i}. `{feature}`\n")
            
            f.write("\n---\n")
            f.write("*Generated by WatermelonModelConverter*")
        
        return str(report_path)


def main():
    """메인 실행 함수"""
    print("🍉 Watermelon Model Converter")
    print("=" * 50)
    
    # 변환기 초기화
    converter = WatermelonModelConverter()
    
    # 전체 파이프라인 실행
    results = converter.convert_full_pipeline()
    
    # 검증 실행
    if results['onnx_converted'] or results['coreml_converted']:
        validation_results = converter.validate_conversion()
        report_path = converter.save_conversion_report(validation_results)
        print(f"\n📋 Report saved: {report_path}")
    
    print("\n🎉 Conversion completed!")


if __name__ == "__main__":
    main() 