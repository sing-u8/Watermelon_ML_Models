"""
🍉 수박 음 높낮이 분류 ML 프로젝트 - 오디오 로더 모듈
AudioLoader 클래스: 다양한 오디오 형식 로딩 및 전처리
"""

import logging
from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os

# pydub을 사용한 .m4a 파일 처리 (선택적)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logging.warning("pydub이 설치되지 않았습니다. .m4a 파일 처리가 제한될 수 있습니다.")

# ffmpeg 경고 억제
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub.utils")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioLoader:
    """
    다양한 오디오 형식을 로딩하고 전처리하는 클래스
    
    지원 형식:
    - .wav (librosa, soundfile)
    - .m4a (pydub + librosa)
    - .mp3 (librosa)
    
    기능:
    - 자동 형식 감지 및 적절한 로더 선택
    - 샘플링 레이트 변환
    - 모노/스테레오 변환
    - 오디오 길이 제한
    """
    
    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        """
        AudioLoader 초기화
        
        Args:
            sample_rate (int): 목표 샘플링 레이트 (기본값: 16000)
            mono (bool): 모노 채널로 변환 여부 (기본값: True)
        """
        self.sample_rate = sample_rate
        self.mono = mono
        
        # 지원하는 오디오 확장자
        self.supported_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.flac'}
        
        logger.info(f"AudioLoader 초기화: sample_rate={sample_rate}, mono={mono}")
    
    def load_audio(self, file_path: Union[str, Path], 
                  duration: Optional[float] = None,
                  offset: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        오디오 파일 로딩
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            duration (Optional[float]): 로딩할 오디오 길이 (초)
            offset (float): 시작 오프셋 (초)
            
        Returns:
            Tuple[Optional[np.ndarray], Optional[int]]: (오디오 데이터, 샘플링 레이트)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return None, None
        
        # 파일 확장자 확인
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_extensions:
            logger.error(f"지원하지 않는 오디오 형식: {file_extension}")
            return None, None
        
        try:
            # .m4a 파일 특별 처리
            if file_extension == '.m4a':
                return self._load_m4a_file(file_path, duration, offset)
            else:
                # 기타 형식은 librosa로 직접 로딩
                return self._load_with_librosa(file_path, duration, offset)
                
        except Exception as e:
            logger.error(f"오디오 로딩 실패 ({file_path}): {e}")
            return None, None
    
    def _load_m4a_file(self, file_path: Path, 
                      duration: Optional[float] = None,
                      offset: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        .m4a 파일 로딩 (librosa 우선, pydub은 백업)
        """
        # 먼저 librosa로 시도 (더 안정적)
        try:
            audio_data, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=self.mono,
                duration=duration,
                offset=offset
            )
            logger.info(f".m4a 파일 librosa 로딩 성공: {file_path.name}")
            return audio_data, sr
        except Exception as e:
            logger.warning(f"librosa로 .m4a 로딩 실패: {e}")
        
        # pydub이 있으면 백업으로 사용
        if PYDUB_AVAILABLE:
            try:
                # pydub으로 .m4a 파일 로딩
                audio_segment = AudioSegment.from_file(str(file_path), format="m4a")
                
                # 오프셋 및 길이 조정
                if offset > 0:
                    audio_segment = audio_segment[offset * 1000:]  # pydub은 밀리초 단위
                
                if duration is not None:
                    audio_segment = audio_segment[:duration * 1000]
                
                # 모노 변환
                if self.mono and audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # 샘플링 레이트 변환
                if audio_segment.frame_rate != self.sample_rate:
                    audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                
                # numpy 배열로 변환
                audio_array = np.array(audio_segment.get_array_of_samples())
                
                # 16비트 정수에서 float로 변환
                if audio_segment.sample_width == 2:  # 16비트
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif audio_segment.sample_width == 4:  # 32비트
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:  # 8비트
                    audio_array = audio_array.astype(np.float32) / 128.0
                
                logger.info(f".m4a 파일 pydub 로딩 성공: {file_path.name}")
                return audio_array, self.sample_rate
                
            except Exception as e:
                logger.error(f"pydub으로 .m4a 로딩도 실패: {e}")
        
        logger.error(f".m4a 파일 로딩 완전 실패: {file_path}")
        return None, None
    
    def _load_with_librosa(self, file_path: Path,
                          duration: Optional[float] = None,
                          offset: float = 0.0) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """
        librosa를 사용한 오디오 로딩
        """
        try:
            # librosa를 사용한 오디오 로딩 (Path 객체를 문자열로 변환)
            audio_data, sr = librosa.load(
                str(file_path),
                sr=self.sample_rate,
                mono=self.mono,
                duration=duration,
                offset=offset
            )
            
            logger.info(f"librosa 로딩 성공: {file_path.name}")
            return audio_data, sr
            
        except Exception as e:
            logger.error(f"librosa 로딩 실패 ({file_path}): {e}")
            return None, None
    
    def preprocess_audio(self, audio_data: np.ndarray, 
                        target_length: Optional[float] = None) -> np.ndarray:
        """
        오디오 데이터 전처리
        
        Args:
            audio_data (np.ndarray): 원본 오디오 데이터
            target_length (Optional[float]): 목표 길이 (초)
            
        Returns:
            np.ndarray: 전처리된 오디오 데이터
        """
        if audio_data is None or len(audio_data) == 0:
            return audio_data
        
        # 1. 묵음 구간 제거
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        
        # 2. 정규화 (-1 ~ 1 범위)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # 3. 길이 조정
        if target_length is not None:
            target_samples = int(target_length * self.sample_rate)
            
            if len(audio_data) > target_samples:
                # 길이가 길면 중앙 부분만 사용
                start = (len(audio_data) - target_samples) // 2
                audio_data = audio_data[start:start + target_samples]
            elif len(audio_data) < target_samples:
                # 길이가 짧으면 0으로 패딩
                padding = target_samples - len(audio_data)
                audio_data = np.pad(audio_data, (0, padding), mode='constant')
        
        return audio_data
    
    def get_audio_info(self, file_path: Union[str, Path]) -> dict:
        """
        오디오 파일 정보 반환
        
        Args:
            file_path (Union[str, Path]): 오디오 파일 경로
            
        Returns:
            dict: 오디오 파일 정보
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': '파일이 존재하지 않습니다'}
        
        try:
            # librosa로 기본 정보 로딩
            info = librosa.get_duration(filename=str(file_path))
            
            return {
                'file_path': str(file_path),
                'duration': info,
                'sample_rate': self.sample_rate,
                'channels': 1 if self.mono else 2,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            return {'error': f'정보 추출 실패: {e}'} 