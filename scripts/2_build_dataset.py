#!/usr/bin/env python3
"""
🍉 수박 당도 예측 ML 프로젝트 - 데이터셋 구축 스크립트
전체 수박 오디오 데이터에서 특징을 추출하고 데이터셋을 구축합니다.
데이터 증강 기능 포함.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import time
from typing import Optional

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_builder import DatasetBuilder
from src.data.data_splitter import DataSplitter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='수박 당도 예측 데이터셋 구축')
    
    parser.add_argument('--augmentation', action='store_true', 
                       help='데이터 증강 활성화')
    parser.add_argument('--multiplier', type=int, default=5,
                       help='증강 배수 (기본값: 5)')
    parser.add_argument('--aug-types', nargs='+', 
                       choices=['pitch_shift', 'time_stretch', 'add_noise', 'time_shift', 'volume_change'],
                       help='사용할 증강 타입들')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='출력 디렉토리 (기본값: data/processed/full_dataset)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='배치 크기 (기본값: 10)')
    
    return parser.parse_args()


def analyze_metadata():
    """메타데이터 분석"""
    logger.info("=== 메타데이터 분석 시작 ===")
    
    metadata_path = project_root / 'data' / 'watermelon_metadata.csv'
    if not metadata_path.exists():
        logger.error(f"메타데이터 파일이 없습니다: {metadata_path}")
        return None
    
    df = pd.read_csv(metadata_path)
    logger.info(f"총 데이터 포인트: {len(df)}개")
    logger.info(f"유니크 수박: {df['watermelon_id'].nunique()}개")
    logger.info(f"당도 범위: {df['sweetness'].min():.1f} ~ {df['sweetness'].max():.1f} Brix")
    logger.info(f"평균 당도: {df['sweetness'].mean():.2f} ± {df['sweetness'].std():.2f} Brix")
    
    # 당도 분포 확인
    sweetness_bins = pd.cut(df['sweetness'], bins=5)
    logger.info("당도 분포:")
    try:
        bin_counts = pd.Series(sweetness_bins).value_counts().sort_index()
        for bin_range, count in bin_counts.items():
            logger.info(f"  {bin_range}: {count}개")
    except Exception as e:
        logger.warning(f"당도 분포 분석 건너뜀: {e}")
    
    return df


def build_dataset_with_options(enable_augmentation: bool = False, 
                              multiplier: int = 5,
                              aug_types: Optional[list] = None,
                              output_dir: Optional[str] = None,
                              batch_size: int = 10):
    """옵션을 지정하여 데이터셋 구축"""
    logger.info(f"=== 데이터셋 구축 (증강: {enable_augmentation}, 배수: {multiplier}) ===")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        if enable_augmentation:
            output_dir = project_root / 'data' / 'processed' / f'augmented_dataset_{multiplier}x'
        else:
            output_dir = project_root / 'data' / 'processed' / 'full_dataset'
    else:
        output_dir = Path(output_dir)
    
    builder = DatasetBuilder(config_path=project_root / 'configs' / 'preprocessing.yaml')
    
    # 증강 설정
    if enable_augmentation:
        builder.enable_augmentation(True)
        builder.set_augmentation_multiplier(multiplier)
        
        # 특정 증강 타입만 활성화
        if aug_types:
            # 모든 증강 타입 비활성화
            all_types = ['pitch_shift', 'time_stretch', 'add_noise', 'time_shift', 'volume_change']
            for aug_type in all_types:
                builder.enable_augmentation_type(aug_type, False)
            
            # 지정된 타입만 활성화
            for aug_type in aug_types:
                builder.enable_augmentation_type(aug_type, True)
                logger.info(f"증강 타입 활성화: {aug_type}")
    else:
        builder.enable_augmentation(False)
    
    # 상태 출력
    status = builder.get_augmentation_status()
    logger.info(f"증강 상태: {status}")
    
    # 메타데이터 경로
    metadata_path = project_root / 'data' / 'watermelon_metadata.csv'
    
    # 데이터셋 구축
    build_result = builder.build_dataset(
        metadata_path=metadata_path,
        output_dir=output_dir,
        batch_size=batch_size,
        apply_augmentation=enable_augmentation
    )
    
    return build_result


def build_full_dataset():
    """전체 데이터셋 구축 (기존 방식)"""
    logger.info("=== 전체 데이터셋 구축 시작 ===")
    
    # 메타데이터 분석
    metadata_df = analyze_metadata()
    if metadata_df is None:
        return False
    
    # DatasetBuilder 초기화
    config_path = project_root / 'configs' / 'preprocessing.yaml'
    builder = DatasetBuilder(config_path=config_path)
    
    # 메타데이터 파일 경로
    metadata_path = project_root / 'data' / 'watermelon_metadata.csv'
    output_dir = project_root / 'data' / 'processed' / 'full_dataset'
    
    build_result = builder.build_dataset(
        metadata_path=metadata_path,
        output_dir=output_dir,
        batch_size=10
    )
    
    return build_result


def split_dataset():
    """데이터셋 분할"""
    logger.info("=== 데이터셋 분할 시작 ===")
    
    # 분할할 데이터셋 경로
    features_path = project_root / 'data' / 'processed' / 'full_dataset' / 'features.csv'
    labels_path = project_root / 'data' / 'processed' / 'full_dataset' / 'labels.csv'
    
    if not features_path.exists() or not labels_path.exists():
        logger.error("분할할 데이터셋이 없습니다. 먼저 데이터셋을 구축하세요.")
        return False
    
    # DataSplitter 초기화
    splitter = DataSplitter()
    
    # 분할 실행
    split_result = splitter.split_dataset(
        features_path=features_path,
        labels_path=labels_path,
        output_dir=project_root / 'data' / 'splits',
        test_size=0.2,
        val_size=0.2,
        random_state=42
    )
    
    return split_result


def verify_dataset():
    """데이터셋 검증"""
    logger.info("=== 데이터셋 검증 시작 ===")
    
    # 검증할 데이터셋 경로
    features_path = project_root / 'data' / 'processed' / 'full_dataset' / 'features.csv'
    
    if not features_path.exists():
        logger.error("검증할 데이터셋이 없습니다.")
        return False
    
    # DatasetBuilder 초기화
    config_path = project_root / 'configs' / 'preprocessing.yaml'
    builder = DatasetBuilder(config_path=config_path)
    
    # 검증 실행
    validation_result = builder.validate_dataset(features_path)
    
    logger.info("검증 결과:")
    for key, value in validation_result.items():
        logger.info(f"  {key}: {value}")
    
    return validation_result


def main():
    """메인 함수"""
    args = parse_arguments()
    
    logger.info(f"명령줄 인자: 증강={args.augmentation}, 배수={args.multiplier}, 타입={args.aug_types}")
    
    # 1. 메타데이터 분석
    metadata_df = analyze_metadata()
    if metadata_df is None:
        logger.error("메타데이터 분석 실패")
        return False
    
    # 2. 데이터셋 구축 (증강 옵션 포함)
    build_result = build_dataset_with_options(
        enable_augmentation=args.augmentation,
        multiplier=args.multiplier,
        aug_types=args.aug_types,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    if not build_result or not build_result.get('success', False):
        logger.error("데이터셋 구축 실패")
        return False
    
    # 3. 결과 출력
    logger.info("=== 데이터셋 구축 결과 ===")
    logger.info(f"성공: {build_result['success']}")
    logger.info(f"특징 형태: {build_result['features_shape']}")
    logger.info(f"라벨 형태: {build_result['labels_shape']}")
    logger.info(f"처리된 파일: {build_result['files_processed']}")
    logger.info(f"실패한 파일: {build_result['files_failed']}")
    logger.info(f"총 처리 시간: {build_result['total_processing_time']:.2f}초")
    
    if args.augmentation:
        aug_stats = build_result.get('augmentation_stats', {})
        logger.info(f"증강 활성화: {aug_stats.get('enabled', False)}")
        logger.info(f"증강된 샘플: {aug_stats.get('augmented_samples', 0)}")
        logger.info(f"사용된 증강 타입: {aug_stats.get('types_used', {})}")
    
    # 4. 데이터셋 분할 (선택사항)
    if not args.augmentation:  # 원본 데이터셋만 분할
        logger.info("\n=== 데이터셋 분할 시작 ===")
        split_result = split_dataset()
        if split_result:
            logger.info("데이터셋 분할 완료")
        else:
            logger.warning("데이터셋 분할 실패")
    
    # 5. 데이터셋 검증
    logger.info("\n=== 데이터셋 검증 시작 ===")
    validation_result = verify_dataset()
    if validation_result:
        logger.info("데이터셋 검증 완료")
    else:
        logger.warning("데이터셋 검증 실패")
    
    logger.info("=== 모든 작업 완료 ===")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 