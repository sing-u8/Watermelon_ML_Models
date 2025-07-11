#!/usr/bin/env python3
"""
🍉 수박 당도 예측 모델 변환 스크립트
scikit-learn 1.5.1 → ONNX → Core ML 자동 변환

Usage:
    python scripts/convert_to_mobile.py
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.conversion.model_converter import WatermelonModelConverter


def main():
    print("🍉 수박 당도 예측 모델 → 모바일 변환 시작!")
    print("=" * 60)
    print(f"🔧 scikit-learn 1.5.1 호환 버전")
    print(f"📱 Target: ONNX + Core ML")
    print()
    
    try:
        # 변환기 초기화
        converter = WatermelonModelConverter(
            output_dir="models/mobile"
        )
        
        # 전체 파이프라인 실행
        print("🚀 변환 파이프라인 시작...")
        results = converter.convert_full_pipeline()
        
        print()
        print("📊 변환 결과:")
        print(f"   모델 로딩: {'✅ 성공' if results['model_loaded'] else '❌ 실패'}")
        print(f"   ONNX 변환: {'✅ 성공' if results['onnx_converted'] else '❌ 실패'}")
        print(f"   Core ML 변환: {'✅ 성공' if results['coreml_converted'] else '❌ 실패'}")
        
        # 검증 및 보고서 생성
        if results['onnx_converted'] or results['coreml_converted']:
            print()
            print("🔍 변환 정확도 검증 중...")
            validation_results = converter.validate_conversion(num_test_samples=20)
            
            if 'error' not in validation_results:
                print("✅ 검증 완료!")
                
                # 정확도 결과 출력
                if 'accuracy' in validation_results:
                    acc = validation_results['accuracy']
                    if 'onnx_mae' in acc:
                        print(f"   📊 ONNX 정확도 - MAE: {acc['onnx_mae']:.6f}, 최대오차: {acc['onnx_max_diff']:.6f}")
                    if 'coreml_mae' in acc:
                        print(f"   📊 Core ML 정확도 - MAE: {acc['coreml_mae']:.6f}, 최대오차: {acc['coreml_max_diff']:.6f}")
                
                # 보고서 저장
                report_path = converter.save_conversion_report(validation_results)
                print(f"   📋 상세 보고서 저장: {report_path}")
            else:
                print(f"❌ 검증 실패: {validation_results['error']}")
        
        print()
        print("🎉 변환 프로세스 완료!")
        
        # 성공 상태 반환
        if results['onnx_converted'] and results['coreml_converted']:
            print("🏆 모든 변환 성공! iOS 배포 준비 완료!")
            return 0
        elif results['onnx_converted']:
            print("⚠️  ONNX 변환만 성공. Core ML은 ONNX 모델 사용 권장.")
            return 0
        else:
            print("❌ 변환 실패. 로그를 확인해주세요.")
            return 1
        
    except Exception as e:
        print(f"❌ 치명적 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 