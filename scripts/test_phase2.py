#!/usr/bin/env python3
"""
🍉 수박 당도 예측 ML 프로젝트 - Phase 2 테스트 스크립트
전처리 및 특징 추출 모듈들의 기능을 테스트합니다.
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.audio_loader import AudioLoader
from src.data.preprocessor import AudioPreprocessor
from src.data.feature_extractor import AudioFeatureExtractor
from src.data.dataset_builder import DatasetBuilder
from src.data.data_splitter import DataSplitter

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_audio_loader():
    """AudioLoader 테스트"""
    logger.info("=== AudioLoader 테스트 시작 ===")
    
    loader = AudioLoader(sample_rate=16000, mono=True)
    logger.info(f"AudioLoader 생성: {loader}")
    
    # 테스트 오디오 생성 (사인파)
    duration = 2.0  # 2초
    sample_rate = 16000
    frequency = 440  # A4 음
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    test_audio = np.sin(2 * np.pi * frequency * t)
    
    # 임시 파일로 저장
    import soundfile as sf
    test_file = project_root / 'temp_test.wav'
    sf.write(test_file, test_audio, sample_rate)
    
    try:
        # 오디오 로드 테스트
        audio_data, sr = loader.load_audio(test_file)
        logger.info(f"오디오 로드 성공: shape={audio_data.shape}, sr={sr}")
        
        # 오디오 정보 추출 테스트
        info = loader.get_audio_info(test_file)
        logger.info(f"오디오 정보: {info}")
        
        # 통계 확인
        stats = loader.get_stats()
        logger.info(f"AudioLoader 통계: {stats}")
        
        logger.info("✅ AudioLoader 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AudioLoader 테스트 실패: {e}")
        return False
    finally:
        # 임시 파일 삭제
        if test_file.exists():
            test_file.unlink()


def test_audio_preprocessor():
    """AudioPreprocessor 테스트"""
    logger.info("=== AudioPreprocessor 테스트 시작 ===")
    
    try:
        config_path = project_root / 'configs' / 'preprocessing.yaml'
        preprocessor = AudioPreprocessor(config_path=config_path)
        logger.info(f"AudioPreprocessor 생성: {preprocessor}")
        
        # 테스트 오디오 생성 (노이즈 포함)
        duration = 3.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # 신호 + 노이즈
        signal = np.sin(2 * np.pi * 440 * t) * 0.5
        noise = np.random.normal(0, 0.1, signal.shape)
        test_audio = signal + noise
        
        # 전처리 테스트
        processed_audio, process_info = preprocessor.preprocess_audio(test_audio, sample_rate)
        logger.info(f"전처리 완료: 원본 shape={test_audio.shape}, "
                   f"처리 후 shape={processed_audio.shape}")
        
        # 통계 확인
        stats = preprocessor.get_stats()
        logger.info(f"AudioPreprocessor 통계: {stats}")
        
        logger.info("✅ AudioPreprocessor 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AudioPreprocessor 테스트 실패: {e}")
        return False


def test_feature_extractor():
    """AudioFeatureExtractor 테스트"""
    logger.info("=== AudioFeatureExtractor 테스트 시작 ===")
    
    try:
        config_path = project_root / 'configs' / 'preprocessing.yaml'
        extractor = AudioFeatureExtractor(config_path=config_path)
        logger.info(f"AudioFeatureExtractor 생성: {extractor}")
        
        # 특징 이름 확인
        feature_names = extractor.get_feature_names()
        logger.info(f"총 특징 개수: {len(feature_names)}")
        logger.info(f"특징 그룹: {extractor.get_feature_groups()}")
        
        # 테스트 오디오 생성 (좀 더 복잡한 신호)
        duration = 2.0
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # 복합 신호 (하모닉 + 노이즈)
        signal = (np.sin(2 * np.pi * 440 * t) +
                 0.5 * np.sin(2 * np.pi * 880 * t) +
                 0.25 * np.sin(2 * np.pi * 1320 * t))
        signal += 0.1 * np.random.normal(0, 1, signal.shape)
        
        # 특징 추출 테스트
        features = extractor.extract_all_features(signal, sample_rate)
        logger.info(f"특징 추출 완료: shape={features.shape}")
        logger.info(f"특징 벡터 요약: min={features.min():.3f}, "
                   f"max={features.max():.3f}, mean={features.mean():.3f}")
        
        # NaN/Inf 체크
        nan_count = np.isnan(features).sum()
        inf_count = np.isinf(features).sum()
        logger.info(f"품질 확인: NaN={nan_count}, Inf={inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            logger.warning("⚠️ NaN 또는 Inf 값이 발견되었습니다!")
        
        # 통계 확인
        stats = extractor.get_stats()
        logger.info(f"AudioFeatureExtractor 통계: {stats}")
        
        logger.info("✅ AudioFeatureExtractor 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AudioFeatureExtractor 테스트 실패: {e}")
        return False


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    logger.info("=== 전체 파이프라인 테스트 시작 ===")
    
    try:
        # 테스트 오디오 파일 생성
        import soundfile as sf
        test_dir = project_root / 'temp_test_data'
        test_dir.mkdir(exist_ok=True)
        
        # 여러 개의 테스트 오디오 파일 생성
        sample_rate = 16000
        duration = 1.5
        
        for i in range(3):
            # 다양한 주파수의 신호 생성
            freq = 200 + i * 100  # 200Hz, 300Hz, 400Hz
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            signal = np.sin(2 * np.pi * freq * t) * 0.7
            
            # 파일 저장
            file_path = test_dir / f'test_audio_{i+1}.wav'
            sf.write(file_path, signal, sample_rate)
        
        # 전체 파이프라인 실행
        logger.info("전체 파이프라인 실행 중...")
        
        # 1. AudioLoader
        loader = AudioLoader(sample_rate=16000, mono=True)
        
        # 2. AudioPreprocessor
        config_path = project_root / 'configs' / 'preprocessing.yaml'
        preprocessor = AudioPreprocessor(config_path=config_path)
        
        # 3. AudioFeatureExtractor
        extractor = AudioFeatureExtractor(config_path=config_path)
        
        # 파이프라인 실행
        all_features = []
        for audio_file in test_dir.glob('*.wav'):
            # 로드
            audio_data, sr = loader.load_audio(audio_file)
            
            # 전처리
            processed_audio, process_info = preprocessor.preprocess_audio(audio_data, sr)
            
            # 특징 추출
            features = extractor.extract_all_features(processed_audio, sr)
            all_features.append(features)
            
            logger.info(f"파일 {audio_file.name} 처리 완료: {features.shape}")
        
        # 결과 확인
        all_features = np.array(all_features)
        logger.info(f"전체 특징 행렬: {all_features.shape}")
        
        # 정리
        import shutil
        shutil.rmtree(test_dir)
        
        logger.info("✅ 전체 파이프라인 테스트 성공!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 전체 파이프라인 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    logger.info("🍉 Phase 2 모듈 테스트 시작")
    logger.info("=" * 60)
    
    test_results = []
    
    # 개별 모듈 테스트
    test_results.append(("AudioLoader", test_audio_loader()))
    test_results.append(("AudioPreprocessor", test_audio_preprocessor()))
    test_results.append(("AudioFeatureExtractor", test_feature_extractor()))
    test_results.append(("Full Pipeline", test_full_pipeline()))
    
    # 결과 요약
    logger.info("=" * 60)
    logger.info("🍉 Phase 2 테스트 결과 요약")
    logger.info("=" * 60)
    
    success_count = 0
    for test_name, result in test_results:
        status = "✅ 성공" if result else "❌ 실패"
        logger.info(f"{test_name}: {status}")
        if result:
            success_count += 1
    
    logger.info(f"\n총 {len(test_results)}개 테스트 중 {success_count}개 성공")
    
    if success_count == len(test_results):
        logger.info("🎉 모든 Phase 2 모듈이 정상적으로 작동합니다!")
        return True
    else:
        logger.error("⚠️ 일부 모듈에서 문제가 발견되었습니다.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 