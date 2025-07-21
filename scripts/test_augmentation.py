#!/usr/bin/env python3
"""
🍉 수박 당도 예측 ML 프로젝트 - 데이터 증강 테스트 스크립트
데이터 증강 기능을 테스트하고 결과를 확인합니다.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import time

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.audio_loader import AudioLoader
from src.data.preprocessor import AudioPreprocessor
from src.data.feature_extractor import AudioFeatureExtractor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_audio():
    """테스트용 오디오 생성"""
    logger.info("=== 테스트 오디오 생성 ===")
    
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 수박 소리 시뮬레이션 (440Hz 기본 주파수 + 하모닉스 + 노이즈)
    fundamental = 440
    test_signal = (
        0.5 * np.sin(2 * np.pi * fundamental * t) +          # 기본 주파수
        0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +      # 2차 하모닉
        0.2 * np.sin(2 * np.pi * fundamental * 3 * t) +      # 3차 하모닉
        0.1 * np.random.normal(0, 0.1, len(t))               # 노이즈
    )
    
    # 앞뒤에 무음 구간 추가
    silence_samples = int(0.2 * sample_rate)  # 0.2초 무음
    silence = np.zeros(silence_samples)
    test_signal_with_silence = np.concatenate([silence, test_signal, silence])
    
    logger.info(f"테스트 오디오 생성 완료: {test_signal_with_silence.shape}, 길이: {len(test_signal_with_silence)/sample_rate:.2f}초")
    
    return test_signal_with_silence, sample_rate


def test_augmentation_types():
    """각 증강 타입별 테스트"""
    logger.info("=== 증강 타입별 테스트 ===")
    
    # 테스트 오디오 생성
    test_audio, sample_rate = create_test_audio()
    
    # AudioPreprocessor 초기화
    config_path = project_root / 'configs' / 'preprocessing.yaml'
    preprocessor = AudioPreprocessor(config_path=config_path)
    
    # 각 증강 타입별로 테스트
    augmentation_types = ['pitch_shift', 'time_stretch', 'add_noise', 'time_shift', 'volume_change']
    
    results = {}
    
    for aug_type in augmentation_types:
        logger.info(f"\n--- {aug_type} 테스트 ---")
        
        # 해당 증강 타입만 활성화
        for aug_name in augmentation_types:
            preprocessor.enable_augmentation_type(aug_name, aug_name == aug_type)
        
        # 증강 실행
        augmented_samples = preprocessor.augment_audio(
            test_audio, 
            sample_rate,
            target_multiplier=3,
            force_enable=True
        )
        
        results[aug_type] = {
            'samples_count': len(augmented_samples),
            'samples': augmented_samples
        }
        
        logger.info(f"{aug_type}: {len(augmented_samples)}개 샘플 생성")
        
        # 각 샘플 정보 출력
        for i, (audio, info) in enumerate(augmented_samples):
            logger.info(f"  샘플 {i+1}: {info.get('type', 'unknown')} - 길이: {len(audio)/sample_rate:.2f}초")
    
    return results


def test_augmentation_multipliers():
    """다양한 배수로 증강 테스트"""
    logger.info("=== 배수별 증강 테스트 ===")
    
    # 테스트 오디오 생성
    test_audio, sample_rate = create_test_audio()
    
    # AudioPreprocessor 초기화
    config_path = project_root / 'configs' / 'preprocessing.yaml'
    preprocessor = AudioPreprocessor(config_path=config_path)
    
    # 모든 증강 타입 활성화
    augmentation_types = ['pitch_shift', 'time_stretch', 'add_noise', 'time_shift', 'volume_change']
    for aug_type in augmentation_types:
        preprocessor.enable_augmentation_type(aug_type, True)
    
    # 다양한 배수로 테스트
    multipliers = [1, 3, 5, 10]
    
    results = {}
    
    for multiplier in multipliers:
        logger.info(f"\n--- 배수 {multiplier} 테스트 ---")
        
        start_time = time.time()
        
        augmented_samples = preprocessor.augment_audio(
            test_audio, 
            sample_rate,
            target_multiplier=multiplier,
            force_enable=True
        )
        
        processing_time = time.time() - start_time
        
        results[multiplier] = {
            'samples_count': len(augmented_samples),
            'processing_time': processing_time,
            'samples': augmented_samples
        }
        
        logger.info(f"배수 {multiplier}: {len(augmented_samples)}개 샘플, {processing_time:.3f}초")
        
        # 증강 타입별 통계
        type_counts = {}
        for _, info in augmented_samples:
            aug_type = info.get('type', 'unknown')
            type_counts[aug_type] = type_counts.get(aug_type, 0) + 1
        
        logger.info(f"  증강 타입별 개수: {type_counts}")
    
    return results


def test_feature_extraction_with_augmentation():
    """증강된 오디오로 특징 추출 테스트"""
    logger.info("=== 증강 + 특징 추출 테스트 ===")
    
    # 테스트 오디오 생성
    test_audio, sample_rate = create_test_audio()
    
    # AudioPreprocessor 초기화
    config_path = project_root / 'configs' / 'preprocessing.yaml'
    preprocessor = AudioPreprocessor(config_path=config_path)
    
    # AudioFeatureExtractor 초기화
    feature_extractor = AudioFeatureExtractor(config_path=config_path)
    
    # 증강 설정
    preprocessor.enable_augmentation(True)
    preprocessor.set_augmentation_multiplier(5)
    
    # 모든 증강 타입 활성화
    augmentation_types = ['pitch_shift', 'time_stretch', 'add_noise', 'time_shift', 'volume_change']
    for aug_type in augmentation_types:
        preprocessor.enable_augmentation_type(aug_type, True)
    
    # 증강 실행
    augmented_samples = preprocessor.augment_audio(
        test_audio, 
        sample_rate,
        force_enable=True
    )
    
    logger.info(f"증강 완료: {len(augmented_samples)}개 샘플")
    
    # 각 샘플에서 특징 추출
    all_features = []
    feature_names = feature_extractor.get_feature_names()
    
    for i, (audio, info) in enumerate(augmented_samples):
        logger.info(f"특징 추출 중... 샘플 {i+1}/{len(augmented_samples)}")
        
        features = feature_extractor.extract_all_features(audio, sample_rate)
        all_features.append(features)
        
        logger.info(f"  샘플 {i+1} ({info.get('type', 'unknown')}): {len(features)}개 특징")
    
    # 특징 비교
    all_features = np.array(all_features)
    
    logger.info(f"\n특징 추출 완료: {all_features.shape}")
    logger.info(f"특징 이름: {len(feature_names)}개")
    
    # 특징 품질 확인
    nan_count = np.sum(np.isnan(all_features))
    inf_count = np.sum(np.isinf(all_features))
    
    logger.info(f"데이터 품질:")
    logger.info(f"  NaN 값: {nan_count}개")
    logger.info(f"  Inf 값: {inf_count}개")
    
    if nan_count == 0 and inf_count == 0:
        logger.info("✅ 모든 특징이 유효합니다!")
    else:
        logger.warning("⚠️ 일부 특징에 문제가 있습니다.")
    
    # 특징 통계
    feature_means = np.mean(all_features, axis=0)
    feature_stds = np.std(all_features, axis=0)
    
    logger.info(f"특징 통계:")
    logger.info(f"  평균 범위: {feature_means.min():.4f} ~ {feature_means.max():.4f}")
    logger.info(f"  표준편차 범위: {feature_stds.min():.4f} ~ {feature_stds.max():.4f}")
    
    return {
        'features': all_features,
        'feature_names': feature_names,
        'augmented_samples': augmented_samples
    }


def save_test_results(results, output_dir: Path):
    """테스트 결과 저장"""
    logger.info("=== 테스트 결과 저장 ===")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 증강된 오디오 샘플 저장
    audio_dir = output_dir / 'audio_samples'
    audio_dir.mkdir(exist_ok=True)
    
    for i, (audio, info) in enumerate(results['augmented_samples']):
        filename = f"sample_{i+1}_{info.get('type', 'unknown')}.wav"
        filepath = audio_dir / filename
        
        sf.write(filepath, audio, 16000)
        logger.info(f"오디오 저장: {filename}")
    
    # 특징 데이터 저장
    features_df = pd.DataFrame(results['features'], columns=results['feature_names'])
    features_path = output_dir / 'features.csv'
    features_df.to_csv(features_path, index=False)
    logger.info(f"특징 저장: {features_path}")
    
    # 메타데이터 저장
    metadata = []
    for i, (_, info) in enumerate(results['augmented_samples']):
        metadata.append({
            'sample_id': i + 1,
            'type': info.get('type', 'unknown'),
            'augmented': info.get('augmented', False),
            'multiplier_index': info.get('multiplier_index', 0)
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_dir / 'metadata.csv'
    metadata_df.to_csv(metadata_path, index=False)
    logger.info(f"메타데이터 저장: {metadata_path}")
    
    logger.info(f"모든 결과가 {output_dir}에 저장되었습니다.")


def main():
    """메인 함수"""
    logger.info("🍉 데이터 증강 테스트 시작")
    logger.info("=" * 60)
    
    try:
        # 1. 증강 타입별 테스트
        logger.info("\n1. 증강 타입별 테스트")
        type_results = test_augmentation_types()
        
        # 2. 배수별 테스트
        logger.info("\n2. 배수별 테스트")
        multiplier_results = test_augmentation_multipliers()
        
        # 3. 특징 추출 테스트
        logger.info("\n3. 특징 추출 테스트")
        feature_results = test_feature_extraction_with_augmentation()
        
        # 4. 결과 저장
        logger.info("\n4. 결과 저장")
        output_dir = project_root / 'experiments' / 'augmentation_test'
        save_test_results(feature_results, output_dir)
        
        # 5. 결과 요약
        logger.info("\n" + "=" * 60)
        logger.info("🎉 데이터 증강 테스트 완료!")
        logger.info("=" * 60)
        
        logger.info("📊 테스트 결과 요약:")
        logger.info(f"  • 증강 타입: {len(type_results)}개")
        logger.info(f"  • 테스트 배수: {len(multiplier_results)}개")
        logger.info(f"  • 생성된 샘플: {len(feature_results['augmented_samples'])}개")
        logger.info(f"  • 특징 개수: {len(feature_results['feature_names'])}개")
        logger.info(f"  • 결과 저장: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 