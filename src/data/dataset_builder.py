"""
🍉 수박 당도 예측 ML 프로젝트 - 데이터셋 빌더 모듈
DatasetBuilder 클래스: 전체 데이터셋에 대한 특징 추출 및 데이터셋 구축
"""

import logging
import time
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import gc

from .audio_loader import AudioLoader
from .preprocessor import AudioPreprocessor
from .feature_extractor import AudioFeatureExtractor

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetBuilder:
    """
    전체 데이터셋의 특징 추출 및 구축을 담당하는 클래스
    
    기능:
    - 메타데이터 CSV 파일 로드
    - 오디오 파일 일괄 처리
    - 특징 추출 및 저장
    - 데이터 품질 검증
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        DatasetBuilder 초기화
        
        Args:
            config_path (Optional[Union[str, Path]]): 설정 파일 경로
        """
        self.config_path = config_path
        self.audio_loader = AudioLoader(sample_rate=16000, mono=True)
        self.preprocessor = AudioPreprocessor(config_path=config_path)
        self.feature_extractor = AudioFeatureExtractor(config_path=config_path)
        
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_processing_time': 0.0,
            'failed_file_list': []
        }
        
        logger.info("DatasetBuilder 초기화 완료")
    
    def load_metadata(self, metadata_path: Union[str, Path]) -> pd.DataFrame:
        """
        메타데이터 CSV 파일 로드
        
        Args:
            metadata_path (Union[str, Path]): 메타데이터 파일 경로
            
        Returns:
            pd.DataFrame: 메타데이터 DataFrame
        """
        metadata_path = Path(metadata_path)
        
        try:
            metadata_df = pd.read_csv(metadata_path)
            logger.info(f"메타데이터 로드 성공: {len(metadata_df)}개 파일")
            logger.info(f"컬럼: {list(metadata_df.columns)}")
            
            # 필수 컬럼 확인
            required_columns = ['file_path', 'sweetness']
            missing_columns = [col for col in required_columns if col not in metadata_df.columns]
            
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {missing_columns}")
            
            return metadata_df
            
        except Exception as e:
            logger.error(f"메타데이터 로드 실패: {e}")
            raise
    
    def process_single_file(self, file_path: Union[str, Path], 
                           sweetness: float) -> Tuple[Optional[np.ndarray], Dict]:
        """
        단일 오디오 파일 처리 (로딩 -> 전처리 -> 특징 추출)
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            sweetness (float): 당도값
            
        Returns:
            Tuple[Optional[np.ndarray], Dict]: (특징 벡터, 처리 정보)
        """
        processing_info = {
            'file_path': str(file_path),
            'sweetness': sweetness,
            'success': False,
            'error': None,
            'processing_time': 0.0,
            'audio_duration': 0.0,
            'feature_count': 0
        }
        
        start_time = time.time()
        
        try:
            # 1. 오디오 로딩
            audio_data, sample_rate = self.audio_loader.load_audio(file_path)
            processing_info['audio_duration'] = len(audio_data) / sample_rate
            
            # 2. 전처리
            processed_audio, preprocess_info = self.preprocessor.preprocess_audio(
                audio_data, sample_rate
            )
            
            # 3. 특징 추출
            features = self.feature_extractor.extract_all_features(
                processed_audio, sample_rate
            )
            
            # 4. 특징 검증
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                raise ValueError("NaN 또는 Inf 값이 포함된 특징 발견")
            
            processing_info['success'] = True
            processing_info['feature_count'] = len(features)
            processing_info['preprocess_info'] = preprocess_info
            
            # 메모리 정리
            del audio_data, processed_audio
            gc.collect()
            
            return features, processing_info
            
        except Exception as e:
            error_msg = f"파일 처리 실패: {str(e)}"
            logger.warning(f"{file_path} - {error_msg}")
            processing_info['error'] = error_msg
            return None, processing_info
            
        finally:
            processing_info['processing_time'] = time.time() - start_time
    
    def build_dataset(self, metadata_path: Union[str, Path],
                     output_dir: Union[str, Path],
                     batch_size: int = 10) -> Dict:
        """
        전체 데이터셋 구축
        
        Args:
            metadata_path (Union[str, Path]): 메타데이터 파일 경로
            output_dir (Union[str, Path]): 출력 디렉토리
            batch_size (int): 배치 크기 (메모리 관리용)
            
        Returns:
            Dict: 데이터셋 구축 결과 정보
        """
        logger.info("데이터셋 구축 시작")
        
        # 출력 디렉토리 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 메타데이터 로드
        metadata_df = self.load_metadata(metadata_path)
        self.stats['total_files'] = len(metadata_df)
        
        # 결과 저장용 리스트
        all_features = []
        all_labels = []
        processing_results = []
        
        # 특징 이름 가져오기
        feature_names = self.feature_extractor.get_feature_names()
        
        logger.info(f"처리할 파일 수: {len(metadata_df)}")
        logger.info(f"배치 크기: {batch_size}")
        
        # 배치별 처리
        for batch_start in tqdm(range(0, len(metadata_df), batch_size), 
                               desc="데이터셋 구축"):
            batch_end = min(batch_start + batch_size, len(metadata_df))
            batch_df = metadata_df.iloc[batch_start:batch_end]
            
            batch_features = []
            batch_labels = []
            
            # 배치 내 각 파일 처리
            for _, row in batch_df.iterrows():
                file_path = row['file_path']
                sweetness = row['sweetness']
                
                features, processing_info = self.process_single_file(file_path, sweetness)
                processing_results.append(processing_info)
                
                if features is not None:
                    batch_features.append(features)
                    batch_labels.append(sweetness)
                    self.stats['processed_files'] += 1
                else:
                    self.stats['failed_files'] += 1
                    self.stats['failed_file_list'].append(str(file_path))
            
            # 배치 결과 추가
            if batch_features:
                all_features.extend(batch_features)
                all_labels.extend(batch_labels)
            
            # 메모리 정리
            gc.collect()
            
            # 진행 상황 로그
            if batch_end % (batch_size * 5) == 0 or batch_end == len(metadata_df):
                success_rate = self.stats['processed_files'] / (batch_end) * 100
                logger.info(f"진행률: {batch_end}/{len(metadata_df)} "
                           f"({success_rate:.1f}% 성공)")
        
        # 결과 정리
        if all_features:
            # NumPy 배열로 변환
            feature_array = np.array(all_features)
            label_array = np.array(all_labels)
            
            logger.info(f"특징 배열 형태: {feature_array.shape}")
            logger.info(f"라벨 배열 형태: {label_array.shape}")
            
            # DataFrame 생성
            feature_df = pd.DataFrame(feature_array, columns=feature_names)
            feature_df['sweetness'] = label_array
            
            # CSV 파일 저장
            features_csv_path = output_dir / "features.csv"
            feature_df.to_csv(features_csv_path, index=False)
            logger.info(f"특징 데이터 저장: {features_csv_path}")
            
            # 특징 이름 저장
            feature_names_path = output_dir / "feature_names.txt"
            with open(feature_names_path, 'w', encoding='utf-8') as f:
                for name in feature_names:
                    f.write(f"{name}\n")
            logger.info(f"특징 이름 저장: {feature_names_path}")
            
            # 처리 결과 저장
            processing_df = pd.DataFrame(processing_results)
            processing_csv_path = output_dir / "processing_results.csv"
            processing_df.to_csv(processing_csv_path, index=False)
            logger.info(f"처리 결과 저장: {processing_csv_path}")
            
        else:
            logger.error("추출된 특징이 없습니다!")
            feature_array = np.array([])
            label_array = np.array([])
        
        # 최종 통계 계산
        total_time = sum(result['processing_time'] for result in processing_results)
        self.stats['total_processing_time'] = total_time
        
        build_summary = {
            'total_files': self.stats['total_files'],
            'processed_files': self.stats['processed_files'],
            'failed_files': self.stats['failed_files'],
            'success_rate': self.stats['processed_files'] / self.stats['total_files'] * 100,
            'total_processing_time': total_time,
            'avg_processing_time': total_time / self.stats['total_files'] if self.stats['total_files'] > 0 else 0,
            'feature_shape': feature_array.shape if all_features else (0, 0),
            'label_shape': label_array.shape if all_features else (0,),
            'output_files': {
                'features_csv': str(output_dir / "features.csv") if all_features else None,
                'feature_names_txt': str(output_dir / "feature_names.txt") if all_features else None,
                'processing_results_csv': str(output_dir / "processing_results.csv")
            },
            'failed_files': self.stats['failed_file_list']
        }
        
        logger.info("데이터셋 구축 완료")
        logger.info(f"성공률: {build_summary['success_rate']:.1f}%")
        logger.info(f"총 처리 시간: {build_summary['total_processing_time']:.1f}초")
        
        return build_summary
    
    def validate_dataset(self, features_csv_path: Union[str, Path]) -> Dict:
        """
        생성된 데이터셋의 품질 검증
        
        Args:
            features_csv_path (Union[str, Path]): 특징 CSV 파일 경로
            
        Returns:
            Dict: 검증 결과
        """
        logger.info("데이터셋 품질 검증 시작")
        
        try:
            # 데이터 로드
            df = pd.read_csv(features_csv_path)
            logger.info(f"데이터 로드: {df.shape}")
            
            # 기본 정보
            validation_result = {
                'shape': df.shape,
                'feature_count': df.shape[1] - 1,  # sweetness 컬럼 제외
                'sample_count': df.shape[0],
                'issues': []
            }
            
            # 결측값 확인
            missing_values = df.isnull().sum().sum()
            validation_result['missing_values'] = missing_values
            if missing_values > 0:
                validation_result['issues'].append(f"결측값 {missing_values}개 발견")
            
            # 무한값 확인
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            inf_values = np.isinf(df[numeric_columns]).sum().sum()
            validation_result['infinite_values'] = inf_values
            if inf_values > 0:
                validation_result['issues'].append(f"무한값 {inf_values}개 발견")
            
            # 당도값 검증
            if 'sweetness' in df.columns:
                sweetness_stats = {
                    'min': df['sweetness'].min(),
                    'max': df['sweetness'].max(),
                    'mean': df['sweetness'].mean(),
                    'std': df['sweetness'].std(),
                    'unique_count': df['sweetness'].nunique()
                }
                validation_result['sweetness_stats'] = sweetness_stats
                
                # 당도값 범위 확인 (일반적으로 8-13 Brix)
                if sweetness_stats['min'] < 5 or sweetness_stats['max'] > 15:
                    validation_result['issues'].append(
                        f"비정상적인 당도 범위: {sweetness_stats['min']:.1f} - {sweetness_stats['max']:.1f}"
                    )
            
            # 특징값 분포 확인
            feature_columns = [col for col in df.columns if col != 'sweetness']
            feature_stats = {
                'zero_variance_features': [],
                'high_variance_features': [],
                'skewed_features': []
            }
            
            for col in feature_columns:
                values = df[col]
                variance = values.var()
                
                # 분산이 0인 특징 (상수 특징)
                if variance == 0:
                    feature_stats['zero_variance_features'].append(col)
                
                # 분산이 매우 큰 특징
                elif variance > 1000:
                    feature_stats['high_variance_features'].append(col)
                
                # 왜도가 높은 특징
                skewness = abs(values.skew())
                if skewness > 3:
                    feature_stats['skewed_features'].append(col)
            
            validation_result['feature_stats'] = feature_stats
            
            # 상수 특징에 대한 경고
            if feature_stats['zero_variance_features']:
                validation_result['issues'].append(
                    f"상수 특징 {len(feature_stats['zero_variance_features'])}개 발견"
                )
            
            # 상관관계가 높은 특징 쌍 찾기
            correlation_matrix = df[feature_columns].corr()
            high_corr_pairs = []
            
            for i in range(len(feature_columns)):
                for j in range(i+1, len(feature_columns)):
                    corr = abs(correlation_matrix.iloc[i, j])
                    if corr > 0.95:  # 95% 이상 상관관계
                        high_corr_pairs.append((feature_columns[i], feature_columns[j], corr))
            
            validation_result['high_correlation_pairs'] = high_corr_pairs
            if high_corr_pairs:
                validation_result['issues'].append(
                    f"높은 상관관계 특징 쌍 {len(high_corr_pairs)}개 발견"
                )
            
            # 전체 품질 등급
            issue_count = len(validation_result['issues'])
            if issue_count == 0:
                validation_result['quality_grade'] = 'excellent'
            elif issue_count <= 2:
                validation_result['quality_grade'] = 'good'
            elif issue_count <= 4:
                validation_result['quality_grade'] = 'fair'
            else:
                validation_result['quality_grade'] = 'poor'
            
            logger.info(f"데이터셋 품질 등급: {validation_result['quality_grade']}")
            if validation_result['issues']:
                logger.warning(f"발견된 이슈: {validation_result['issues']}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"데이터셋 검증 실패: {e}")
            return {'error': str(e)}
    
    def get_stats(self) -> Dict:
        """통계 정보 반환"""
        return self.stats.copy()
    
    def reset_stats(self):
        """통계 정보 초기화"""
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_processing_time': 0.0,
            'failed_file_list': []
        }
        logger.info("DatasetBuilder 통계 정보가 초기화되었습니다.")
    
    def __repr__(self) -> str:
        return (f"DatasetBuilder(processed={self.stats['processed_files']}, "
                f"failed={self.stats['failed_files']})")


# 편의 함수들
def build_watermelon_dataset(metadata_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            config_path: Optional[Union[str, Path]] = None,
                            batch_size: int = 10) -> Dict:
    """
    수박 데이터셋 구축을 위한 편의 함수
    
    Args:
        metadata_path (Union[str, Path]): 메타데이터 파일 경로
        output_dir (Union[str, Path]): 출력 디렉토리
        config_path (Optional[Union[str, Path]]): 설정 파일 경로
        batch_size (int): 배치 크기
        
    Returns:
        Dict: 구축 결과 정보
    """
    builder = DatasetBuilder(config_path=config_path)
    return builder.build_dataset(metadata_path, output_dir, batch_size)


if __name__ == "__main__":
    # 사용 예제
    from pathlib import Path
    
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent.parent
    
    # 경로 설정
    metadata_path = project_root / "data" / "metadata.csv"
    output_dir = project_root / "data" / "processed"
    config_path = project_root / "configs" / "preprocessing.yaml"
    
    if metadata_path.exists():
        print(f"\n🏗️ DatasetBuilder 테스트")
        print(f"메타데이터: {metadata_path}")
        print(f"출력 디렉토리: {output_dir}")
        
        # DatasetBuilder 생성
        builder = DatasetBuilder(config_path=config_path)
        
        # 메타데이터 로드 테스트
        try:
            metadata_df = builder.load_metadata(metadata_path)
            print(f"메타데이터 로드 성공: {len(metadata_df)}개 파일")
            print(f"컬럼: {list(metadata_df.columns)}")
            
            # 처음 몇 개 파일만 테스트
            test_metadata = metadata_df.head(5)  # 처음 5개만
            test_output_dir = output_dir / "test"
            
            print(f"\n테스트 실행: {len(test_metadata)}개 파일")
            
            # 임시 메타데이터 저장
            test_metadata_path = test_output_dir / "test_metadata.csv"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            test_metadata.to_csv(test_metadata_path, index=False)
            
            # 데이터셋 구축 실행
            result = builder.build_dataset(
                metadata_path=test_metadata_path,
                output_dir=test_output_dir,
                batch_size=2
            )
            
            print(f"\n구축 결과:")
            print(f"  - 성공률: {result['success_rate']:.1f}%")
            print(f"  - 처리 시간: {result['total_processing_time']:.1f}초")
            print(f"  - 특징 형태: {result['feature_shape']}")
            
            if result['processed_files'] > 0:
                # 데이터셋 검증
                features_csv = test_output_dir / "features.csv"
                if features_csv.exists():
                    validation_result = builder.validate_dataset(features_csv)
                    print(f"  - 데이터셋 품질: {validation_result['quality_grade']}")
                    if validation_result['issues']:
                        print(f"  - 이슈: {validation_result['issues']}")
            
            # 통계 정보
            stats = builder.get_stats()
            print(f"\nBuilder 통계: {stats}")
            
        except Exception as e:
            print(f"테스트 실패: {e}")
    else:
        print(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}") 