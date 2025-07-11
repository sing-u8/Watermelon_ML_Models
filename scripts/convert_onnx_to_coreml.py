#!/usr/bin/env python3
"""
ONNX 모델을 Core ML로 변환하는 스크립트
"""

import os
import sys
import logging
from pathlib import Path
from onnx_coreml import convert
import onnx

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def convert_onnx_to_coreml(onnx_path: str, output_dir: str) -> bool:
    """
    ONNX 모델을 Core ML로 변환
    
    Args:
        onnx_path: ONNX 모델 파일 경로
        output_dir: 출력 디렉토리
        
    Returns:
        bool: 변환 성공 여부
    """
    logger = setup_logging()
    
    try:
        # ONNX 모델 로드 및 검증
        logger.info(f"🔍 ONNX 모델 로드: {onnx_path}")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX 모델 검증 완료")
        
        # 출력 경로 설정
        output_path = os.path.join(output_dir, "watermelon_sweetness_model.mlmodel")
        
        # Core ML 변환
        logger.info("🔄 ONNX → Core ML 변환 중...")
        coreml_model = convert(
            onnx_model,
            minimum_ios_deployment_target='13.0'
        )
        
        # Core ML 모델 저장
        coreml_model.save(output_path)
        logger.info(f"✅ Core ML 모델 저장: {output_path}")
        
        # 모델 메타데이터 추가
        coreml_model.author = 'Watermelon ML Team'
        coreml_model.short_description = 'Watermelon Sweetness Prediction Model'
        coreml_model.version = '1.0'
        
        # 입력/출력 설명 추가
        coreml_model.input_description['input'] = 'Audio features extracted from watermelon sound (10 features)'
        coreml_model.output_description['output'] = 'Predicted sweetness in Brix scale'
        
        # 최종 저장
        coreml_model.save(output_path)
        logger.info("✅ 메타데이터 추가 완료")
        
        # 모델 정보 출력
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"📱 Core ML 모델 크기: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Core ML 변환 실패: {str(e)}")
        return False

def main():
    """메인 함수"""
    logger = setup_logging()
    
    # 경로 설정
    project_root = Path(__file__).parent.parent
    onnx_path = project_root / "models" / "mobile" / "latest" / "watermelon_sweetness_model.onnx"
    output_dir = project_root / "models" / "mobile" / "latest"
    
    # ONNX 파일 존재 확인
    if not onnx_path.exists():
        logger.error(f"❌ ONNX 파일을 찾을 수 없습니다: {onnx_path}")
        sys.exit(1)
    
    logger.info("🍉 ONNX → Core ML 변환 시작")
    logger.info(f"📂 ONNX 모델: {onnx_path}")
    logger.info(f"📂 출력 디렉토리: {output_dir}")
    
    # 변환 실행
    success = convert_onnx_to_coreml(str(onnx_path), str(output_dir))
    
    if success:
        logger.info("🎉 Core ML 변환 성공!")
        logger.info(f"📱 iOS 배포 준비 완료: {output_dir}")
    else:
        logger.error("💥 Core ML 변환 실패")
        sys.exit(1)

if __name__ == "__main__":
    main() 