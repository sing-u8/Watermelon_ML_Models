#!/usr/bin/env python3
"""
🍉 수박 당도 예측 ML 프로젝트 - 데이터셋 구축 스크립트
전체 수박 오디오 데이터에서 특징을 추출하고 데이터셋을 구축합니다.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import time

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


def build_full_dataset():
    """전체 데이터셋 구축"""
    logger.info("=== 전체 데이터셋 구축 시작 ===")
    
    # 메타데이터 분석
    metadata_df = analyze_metadata()
    if metadata_df is None:
        return False
    
    # DatasetBuilder 초기화
    config_path = project_root / 'configs' / 'preprocessing.yaml'
    builder = DatasetBuilder(config_path=config_path)
    
    # 데이터 루트 경로
    data_root = project_root / 'data' / 'raw'
    
    # 메타데이터에서 파일 경로 추출
    file_paths = []
    sweetness_values = []
    
    for _, row in metadata_df.iterrows():
        file_path = project_root / row['file_path']
        if file_path.exists():
            file_paths.append(file_path)
            sweetness_values.append(row['sweetness'])
        else:
            logger.warning(f"파일이 존재하지 않습니다: {file_path}")
    
    logger.info(f"처리할 파일 수: {len(file_paths)}개")
    
    # 특징 추출 실행
    start_time = time.time()
    
    # 메타데이터 파일 경로
    metadata_path = project_root / 'data' / 'watermelon_metadata.csv'
    output_dir = project_root / 'data' / 'processed' / 'full_dataset'
    
    build_result = builder.build_dataset(
        metadata_path=metadata_path,
        output_dir=output_dir,
        batch_size=10
    )
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    if build_result and build_result['processed_files'] > 0:
        logger.info(f"특징 추출 완료!")
        logger.info(f"총 처리 시간: {processing_time:.1f}초")
        logger.info(f"평균 파일당 처리 시간: {build_result['avg_processing_time']:.3f}초")
        logger.info(f"데이터셋 크기: {build_result['feature_shape']}")
        logger.info(f"특징 개수: {build_result['feature_shape'][1] - 1}")  # -1 for sweetness column
        
        # 통계 확인
        logger.info(f"DatasetBuilder 결과: {build_result}")
        
        return True
    else:
        logger.error("특징 추출 실패!")
        return False


def split_dataset():
    """데이터셋 분할"""
    logger.info("=== 데이터셋 분할 시작 ===")
    
    # 구축된 특징 데이터 로드
    features_path = project_root / 'data' / 'processed' / 'full_dataset' / 'features.csv'
    if not features_path.exists():
        logger.error(f"특징 데이터 파일이 없습니다: {features_path}")
        return False
    
    features_df = pd.read_csv(features_path)
    logger.info(f"특징 데이터 로드: {features_df.shape}")
    
    # DataSplitter 초기화
    splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42)
    
    # 데이터 분할 실행
    split_data = splitter.split_dataset(
        features_df=features_df,
        target_column='sweetness',
        stratify_bins=5
    )
    
    # 분할된 데이터 저장
    output_dir = project_root / 'data' / 'splits' / 'full_dataset'
    saved_files = splitter.save_splits(split_data, output_dir)
    
    # 분할 검증
    validation_result = splitter.validate_split(split_data, target_column='sweetness')
    
    split_result = True
    
    if split_result:
        logger.info("데이터 분할 완료!")
        
        # 분할 통계 확인
        split_stats = splitter.get_stats()
        logger.info(f"DataSplitter 통계: {split_stats}")
        
        return True
    else:
        logger.error("데이터 분할 실패!")
        return False


def verify_dataset():
    """데이터셋 검증"""
    logger.info("=== 데이터셋 검증 시작 ===")
    
    splits_dir = project_root / 'data' / 'splits' / 'full_dataset'
    
    split_files = {
        'train': splits_dir / 'train.csv',
        'val': splits_dir / 'val.csv',
        'test': splits_dir / 'test.csv'
    }
    
    total_samples = 0
    for split_name, file_path in split_files.items():
        if file_path.exists():
            split_df = pd.read_csv(file_path)
            logger.info(f"{split_name.upper()} 세트: {split_df.shape[0]}개 샘플")
            logger.info(f"  당도 범위: {split_df['sweetness'].min():.1f} ~ {split_df['sweetness'].max():.1f}")
            logger.info(f"  평균 당도: {split_df['sweetness'].mean():.2f} ± {split_df['sweetness'].std():.2f}")
            total_samples += split_df.shape[0]
        else:
            logger.warning(f"{split_name} 파일이 없습니다: {file_path}")
    
    logger.info(f"총 샘플 수: {total_samples}개")
    
    # 특징 품질 확인
    features_path = project_root / 'data' / 'processed' / 'full_dataset' / 'features.csv'
    if features_path.exists():
        features_df = pd.read_csv(features_path)
        
        # NaN/Inf 값 확인
        nan_count = features_df.isnull().sum().sum()
        inf_count = np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum()
        
        logger.info(f"데이터 품질 확인:")
        logger.info(f"  NaN 값: {nan_count}개")
        logger.info(f"  Inf 값: {inf_count}개")
        
        if nan_count == 0 and inf_count == 0:
            logger.info("✅ 데이터 품질: 우수")
        else:
            logger.warning("⚠️ 데이터 품질: 문제 발견")
    
    return True


def main():
    """메인 함수"""
    logger.info("🍉 수박 당도 예측 데이터셋 구축 시작")
    logger.info("=" * 60)
    
    try:
        # 1. 전체 데이터셋 구축
        if not build_full_dataset():
            logger.error("데이터셋 구축 실패")
            return False
        
        # 2. 데이터셋 분할
        if not split_dataset():
            logger.error("데이터셋 분할 실패")
            return False
        
        # 3. 데이터셋 검증
        if not verify_dataset():
            logger.error("데이터셋 검증 실패")
            return False
        
        logger.info("=" * 60)
        logger.info("🎉 데이터셋 구축이 성공적으로 완료되었습니다!")
        logger.info("=" * 60)
        
        # 결과 요약
        logger.info("📊 구축 결과 요약:")
        logger.info(f"  • 특징 데이터: data/processed/full_dataset/features.csv")
        logger.info(f"  • 훈련 세트: data/splits/full_dataset/train.csv")
        logger.info(f"  • 검증 세트: data/splits/full_dataset/val.csv")
        logger.info(f"  • 테스트 세트: data/splits/full_dataset/test.csv")
        
        return True
        
    except Exception as e:
        logger.error(f"데이터셋 구축 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 