"""
🍉 수박 당도 예측 ML 프로젝트 - 오디오 로더 모듈
AudioLoader 클래스: 다양한 형식의 오디오 파일 로딩 및 기본 처리
"""

import os
import logging
from typing import Tuple, Optional, Union, List
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioLoader:
    """
    오디오 파일 로딩 및 기본 처리를 담당하는 클래스
    
    지원 형식: .wav, .mp3, .m4a, .flac, .aiff
    """
    
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.m4a', '.flac', '.aiff', '.ogg'}
    
    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        """
        AudioLoader 초기화
        
        Args:
            sample_rate (int): 목표 샘플링 레이트 (기본값: 16000)
            mono (bool): 모노 변환 여부 (기본값: True)
        """
        self.sample_rate = sample_rate
        self.mono = mono
        self.stats = {
            'loaded_files': 0,
            'failed_files': 0,
            'total_duration': 0.0,
            'error_log': []
        }
        
        logger.info(f"AudioLoader 초기화: sample_rate={sample_rate}, mono={mono}")
    
    def load_audio(self, file_path: Union[str, Path], 
                   duration: Optional[float] = None,
                   offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """
        오디오 파일을 로드하고 기본 전처리를 수행
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            duration (Optional[float]): 로드할 길이 (초, None=전체)
            offset (float): 시작 오프셋 (초, 기본값: 0.0)
            
        Returns:
            Tuple[np.ndarray, int]: (오디오 데이터, 샘플링 레이트)
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            ValueError: 지원하지 않는 파일 형식인 경우
            RuntimeError: 오디오 로딩 실패
        """
        file_path = Path(file_path)
        
        # 파일 존재 확인
        if not file_path.exists():
            error_msg = f"파일을 찾을 수 없습니다: {file_path}"
            logger.error(error_msg)
            self.stats['failed_files'] += 1
            self.stats['error_log'].append(error_msg)
            raise FileNotFoundError(error_msg)
        
        # 파일 형식 확인
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            error_msg = f"지원하지 않는 파일 형식: {file_path.suffix}"
            logger.error(error_msg)
            self.stats['failed_files'] += 1
            self.stats['error_log'].append(error_msg)
            raise ValueError(error_msg)
        
        try:
            # librosa를 사용한 오디오 로딩
            audio_data, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=self.mono,
                duration=duration,
                offset=offset
            )
            
            # 샘플링 레이트를 정수로 변환
            sr = int(sr)
            
            # 통계 업데이트
            self.stats['loaded_files'] += 1
            self.stats['total_duration'] += len(audio_data) / sr
            
            logger.debug(f"오디오 로드 성공: {file_path} "
                        f"(shape: {audio_data.shape}, sr: {sr})")
            
            return audio_data, sr
            
        except Exception as e:
            error_msg = f"오디오 로딩 실패 - {file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['failed_files'] += 1
            self.stats['error_log'].append(error_msg)
            raise RuntimeError(error_msg) from e
    
    def load_multiple_files(self, file_paths: List[Union[str, Path]], 
                           duration: Optional[float] = None) -> List[Tuple[np.ndarray, int, str]]:
        """
        여러 오디오 파일을 일괄 로드
        
        Args:
            file_paths (List[Union[str, Path]]): 오디오 파일 경로 리스트
            duration (Optional[float]): 로드할 길이 (초, None=전체)
            
        Returns:
            List[Tuple[np.ndarray, int, str]]: (오디오 데이터, 샘플링 레이트, 파일 경로) 리스트
        """
        results = []
        failed_files = []
        
        logger.info(f"다중 파일 로딩 시작: {len(file_paths)}개 파일")
        
        for file_path in file_paths:
            try:
                audio_data, sr = self.load_audio(file_path, duration=duration)
                results.append((audio_data, sr, str(file_path)))
            except Exception as e:
                logger.warning(f"파일 로딩 실패 건너뜀: {file_path} - {str(e)}")
                failed_files.append(str(file_path))
        
        if failed_files:
            logger.warning(f"로딩 실패한 파일 {len(failed_files)}개: {failed_files[:3]}...")
        
        logger.info(f"다중 파일 로딩 완료: {len(results)}/{len(file_paths)}개 성공")
        return results
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """
        오디오 파일의 메타데이터 정보 추출
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            
        Returns:
            dict: 오디오 파일 정보 딕셔너리
        """
        file_path = Path(file_path)
        
        try:
            # soundfile로 빠른 정보 추출
            info = sf.info(file_path)
            
            # librosa로 추가 정보 (더 정확하지만 느림)
            duration = librosa.get_duration(path=file_path)
            
            return {
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'format': file_path.suffix.lower(),
                'channels': info.channels,
                'sample_rate': info.samplerate,
                'frames': info.frames,
                'duration': duration,
                'duration_sf': info.duration,  # soundfile 기준 길이
                'subtype': info.subtype,
                'endian': info.endian
            }
            
        except Exception as e:
            logger.error(f"오디오 정보 추출 실패: {file_path} - {str(e)}")
            return {
                'file_path': str(file_path),
                'error': str(e)
            }
    
    def validate_audio_files(self, file_paths: List[Union[str, Path]]) -> dict:
        """
        여러 오디오 파일의 유효성 검증
        
        Args:
            file_paths (List[Union[str, Path]]): 검증할 파일 경로 리스트
            
        Returns:
            dict: 검증 결과 요약
        """
        valid_files = []
        invalid_files = []
        file_info_list = []
        
        logger.info(f"오디오 파일 검증 시작: {len(file_paths)}개 파일")
        
        for file_path in file_paths:
            try:
                info = self.get_audio_info(file_path)
                if 'error' not in info:
                    valid_files.append(str(file_path))
                    file_info_list.append(info)
                else:
                    invalid_files.append(str(file_path))
            except Exception as e:
                logger.warning(f"파일 검증 중 오류: {file_path} - {str(e)}")
                invalid_files.append(str(file_path))
        
        # 통계 계산
        if file_info_list:
            total_duration = sum(info['duration'] for info in file_info_list)
            sample_rates = [info['sample_rate'] for info in file_info_list]
            channels = [info['channels'] for info in file_info_list]
            file_sizes = [info['file_size'] for info in file_info_list]
            
            validation_summary = {
                'total_files': len(file_paths),
                'valid_files': len(valid_files),
                'invalid_files': len(invalid_files),
                'validity_rate': len(valid_files) / len(file_paths) * 100,
                'total_duration': total_duration,
                'avg_duration': total_duration / len(file_info_list),
                'sample_rates': {
                    'unique': list(set(sample_rates)),
                    'most_common': max(set(sample_rates), key=sample_rates.count)
                },
                'channels': {
                    'unique': list(set(channels)),
                    'most_common': max(set(channels), key=channels.count)
                },
                'total_size': sum(file_sizes),
                'avg_size': sum(file_sizes) / len(file_info_list),
                'invalid_file_list': invalid_files
            }
        else:
            validation_summary = {
                'total_files': len(file_paths),
                'valid_files': 0,
                'invalid_files': len(invalid_files),
                'validity_rate': 0.0,
                'error': '유효한 오디오 파일이 없습니다.'
            }
        
        logger.info(f"파일 검증 완료: {validation_summary['valid_files']}/{validation_summary['total_files']}개 유효")
        return validation_summary
    
    def reset_stats(self):
        """통계 정보 초기화"""
        self.stats = {
            'loaded_files': 0,
            'failed_files': 0,
            'total_duration': 0.0,
            'error_log': []
        }
        logger.info("AudioLoader 통계 정보가 초기화되었습니다.")
    
    def get_stats(self) -> dict:
        """현재 통계 정보 반환"""
        return self.stats.copy()
    
    def __repr__(self) -> str:
        return (f"AudioLoader(sample_rate={self.sample_rate}, "
                f"mono={self.mono}, "
                f"loaded_files={self.stats['loaded_files']})")


# 편의 함수들
def load_audio_file(file_path: Union[str, Path], 
                   sample_rate: int = 16000, 
                   mono: bool = True,
                   duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    단일 오디오 파일 로딩을 위한 편의 함수
    
    Args:
        file_path (Union[str, Path]): 오디오 파일 경로
        sample_rate (int): 목표 샘플링 레이트
        mono (bool): 모노 변환 여부
        duration (Optional[float]): 로드할 길이 (초)
        
    Returns:
        Tuple[np.ndarray, int]: (오디오 데이터, 샘플링 레이트)
    """
    loader = AudioLoader(sample_rate=sample_rate, mono=mono)
    return loader.load_audio(file_path, duration=duration)


def get_supported_formats() -> set:
    """지원하는 오디오 파일 형식 반환"""
    return AudioLoader.SUPPORTED_FORMATS.copy()


if __name__ == "__main__":
    # 사용 예제
    import sys
    from pathlib import Path
    
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "raw"
    
    if data_dir.exists():
        # AudioLoader 테스트
        loader = AudioLoader(sample_rate=16000, mono=True)
        
        # 모든 오디오 파일 찾기
        audio_files = []
        for format_ext in get_supported_formats():
            audio_files.extend(data_dir.rglob(f"*{format_ext}"))
        
        if audio_files:
            print(f"\n🎵 발견된 오디오 파일: {len(audio_files)}개")
            
            # 파일 검증
            validation_result = loader.validate_audio_files(audio_files[:5])  # 처음 5개만 테스트
            print(f"검증 결과: {validation_result['valid_files']}/{validation_result['total_files']}개 유효")
            
            # 샘플 파일 로드 테스트
            if validation_result['valid_files'] > 0:
                sample_file = audio_files[0]
                try:
                    audio_data, sr = loader.load_audio(sample_file)
                    print(f"샘플 로드 성공: {sample_file.name}")
                    print(f"  - 형태: {audio_data.shape}")
                    print(f"  - 샘플링 레이트: {sr}")
                    print(f"  - 길이: {len(audio_data)/sr:.2f}초")
                    
                    # 통계 정보 출력
                    stats = loader.get_stats()
                    print(f"로더 통계: {stats}")
                    
                except Exception as e:
                    print(f"샘플 로드 실패: {e}")
        else:
            print("오디오 파일을 찾을 수 없습니다.")
    else:
        print(f"데이터 디렉토리를 찾을 수 없습니다: {data_dir}") 