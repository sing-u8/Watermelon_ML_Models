"""
🍉 수박 당도 예측 ML 프로젝트 - 오디오 전처리 모듈
AudioPreprocessor 클래스: 오디오 신호의 전처리 (묵음 제거, 정규화, 필터링 등)
"""

import logging
from typing import Tuple, Optional, Union, List
import numpy as np
import librosa
import scipy.signal
from pathlib import Path
import yaml

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """
    오디오 신호 전처리를 담당하는 클래스
    
    기능:
    - 묵음 구간 제거 (trimming)
    - 신호 정규화 (normalization)
    - 노이즈 필터링 (filtering)
    - 품질 검증 (quality validation)
    - 데이터 증강 (data augmentation)
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        AudioPreprocessor 초기화
        
        Args:
            config_path (Optional[Union[str, Path]]): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.augmentation_config = self.config.get('augmentation', {})
        self.random_seed = self.augmentation_config.get('advanced', {}).get('random_seed', 42)
        np.random.seed(self.random_seed)
        
        # 증강 상태 추적
        self.augmentation_enabled = self.augmentation_config.get('enabled', False)
        self.enabled_augmentation_types = self._get_enabled_augmentation_types()
        
        self.stats = {
            'processed_files': 0,
            'trim_applied': 0,
            'normalize_applied': 0,
            'filter_applied': 0,
            'quality_issues': 0,
            'augmented_samples': 0,
            'augmentation_types_used': {}
        }
        
        logger.info("AudioPreprocessor 초기화 완료")
    
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> dict:
        """설정 파일 로드"""
        if config_path is None:
            # 기본 설정 반환
            return self._get_default_config()
        
        try:
            config_path = Path(config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"설정 파일 로드 성공: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패, 기본 설정 사용: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            'audio': {
                'sample_rate': 16000,
                'trim': {
                    'enabled': True,
                    'top_db': 20,
                    'frame_length': 2048,
                    'hop_length': 512
                },
                'normalize': {
                    'enabled': True,
                    'method': 'peak',
                    'target_level': 0.9
                },
                'filter_noise': {
                    'enabled': False,
                    'low_freq': 80,
                    'high_freq': 8000
                }
            },
            'quality_check': {
                'min_duration': 0.1,
                'max_duration': 10.0,
                'check_clipping': True,
                'check_silence': True,
                'silence_threshold': -60,
                'max_silence_ratio': 0.8
            },
            'augmentation': {
                'enabled': False,
                'types': {
                    'pitch_shift': {
                        'enabled': False,
                        'count': 0,
                        'probability': 1.0,
                        'steps': [-2, -1, 1, 2]
                    },
                    'time_stretch': {
                        'enabled': False,
                        'count': 0,
                        'probability': 1.0,
                        'rate_range': [0.8, 1.2]
                    },
                    'add_noise': {
                        'enabled': False,
                        'count': 0,
                        'probability': 1.0,
                        'noise_level': [0.001, 0.01]
                    },
                    'time_shift': {
                        'enabled': False,
                        'count': 0,
                        'probability': 1.0,
                        'shift_range': [-0.1, 0.1]
                    },
                    'volume_change': {
                        'enabled': False,
                        'count': 0,
                        'probability': 1.0,
                        'gain_range': [0.5, 1.5]
                    }
                },
                'total_multiplier': 5,
                'max_augmented_samples': 10,
                'advanced': {
                    'random_seed': 42,
                    'preserve_original': True,
                    'quality_threshold': 0.8
                }
            }
        }
    
    def _get_enabled_augmentation_types(self) -> List[str]:
        """활성화된 증강 타입 목록 반환"""
        enabled_types = []
        for aug_type, config in self.augmentation_config.get('types', {}).items():
            if config.get('enabled', False):
                enabled_types.append(aug_type)
        return enabled_types
    
    def enable_augmentation(self, enabled: bool = True):
        """증강 활성화/비활성화"""
        self.augmentation_enabled = enabled
        logger.info(f"증강 {'활성화' if enabled else '비활성화'}")
    
    def enable_augmentation_type(self, aug_type: str, enabled: bool = True):
        """특정 증강 타입 활성화/비활성화"""
        if aug_type in self.augmentation_config.get('types', {}):
            self.augmentation_config['types'][aug_type]['enabled'] = enabled
            self.enabled_augmentation_types = self._get_enabled_augmentation_types()
            logger.info(f"증강 타입 '{aug_type}' {'활성화' if enabled else '비활성화'}")
        else:
            logger.warning(f"알 수 없는 증강 타입: {aug_type}")
    
    def set_augmentation_multiplier(self, multiplier: int):
        """증강 배수 설정"""
        self.augmentation_config['total_multiplier'] = multiplier
        logger.info(f"증강 배수 설정: {multiplier}")
    
    def get_augmentation_status(self) -> dict:
        """증강 상태 정보 반환"""
        return {
            'enabled': self.augmentation_enabled,
            'total_multiplier': self.augmentation_config.get('total_multiplier', 1),
            'enabled_types': self.enabled_augmentation_types,
            'config': self.augmentation_config
        }

    def trim_silence(self, audio_data: np.ndarray, 
                    sample_rate: int,
                    top_db: Optional[int] = None,
                    frame_length: Optional[int] = None,
                    hop_length: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        묵음 구간 제거
        
        Args:
            audio_data (np.ndarray): 입력 오디오 데이터
            sample_rate (int): 샘플링 레이트
            top_db (Optional[int]): dB 임계값
            frame_length (Optional[int]): 프레임 길이
            hop_length (Optional[int]): 홉 길이
            
        Returns:
            Tuple[np.ndarray, dict]: (처리된 오디오 데이터, 처리 정보)
        """
        if not self.config['audio']['trim']['enabled']:
            return audio_data, {'trimmed': False, 'original_length': len(audio_data)}
        
        # 파라미터 설정
        top_db = top_db or self.config['audio']['trim']['top_db']
        frame_length = frame_length or self.config['audio']['trim']['frame_length']
        hop_length = hop_length or self.config['audio']['trim']['hop_length']
        
        original_length = len(audio_data)
        
        try:
            # librosa의 trim 함수 사용
            trimmed_audio, indices = librosa.effects.trim(
                audio_data,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            trim_info = {
                'trimmed': True,
                'original_length': original_length,
                'trimmed_length': len(trimmed_audio),
                'removed_samples': original_length - len(trimmed_audio),
                'removed_ratio': (original_length - len(trimmed_audio)) / original_length,
                'start_index': indices[0],
                'end_index': indices[1],
                'top_db': top_db
            }
            
            self.stats['trim_applied'] += 1
            logger.debug(f"묵음 제거 완료: {original_length} -> {len(trimmed_audio)} samples "
                        f"({trim_info['removed_ratio']:.1%} 제거)")
            
            return trimmed_audio, trim_info
            
        except Exception as e:
            logger.warning(f"묵음 제거 실패, 원본 반환: {e}")
            return audio_data, {
                'trimmed': False,
                'original_length': original_length,
                'error': str(e)
            }
    
    def normalize_audio(self, audio_data: np.ndarray, 
                       method: Optional[str] = None,
                       target_level: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """
        오디오 신호 정규화
        
        Args:
            audio_data (np.ndarray): 입력 오디오 데이터
            method (Optional[str]): 정규화 방법 ('peak' 또는 'rms')
            target_level (Optional[float]): 목표 레벨 (0-1)
            
        Returns:
            Tuple[np.ndarray, dict]: (정규화된 오디오 데이터, 정규화 정보)
        """
        if not self.config['audio']['normalize']['enabled']:
            return audio_data, {'normalized': False}
        
        # 파라미터 설정
        method = method or self.config['audio']['normalize']['method']
        target_level = target_level or self.config['audio']['normalize']['target_level']
        
        original_peak = np.max(np.abs(audio_data))
        original_rms = np.sqrt(np.mean(audio_data ** 2))
        
        try:
            if method == 'peak':
                # 피크 정규화
                if original_peak > 0:
                    normalized_audio = audio_data * (target_level / original_peak)
                else:
                    normalized_audio = audio_data
                    
            elif method == 'rms':
                # RMS 정규화
                if original_rms > 0:
                    normalized_audio = audio_data * (target_level / original_rms)
                    # 클리핑 방지
                    max_val = np.max(np.abs(normalized_audio))
                    if max_val > 1.0:
                        normalized_audio = normalized_audio / max_val
                else:
                    normalized_audio = audio_data
            else:
                raise ValueError(f"지원하지 않는 정규화 방법: {method}")
            
            new_peak = np.max(np.abs(normalized_audio))
            new_rms = np.sqrt(np.mean(normalized_audio ** 2))
            
            normalize_info = {
                'normalized': True,
                'method': method,
                'target_level': target_level,
                'original_peak': original_peak,
                'original_rms': original_rms,
                'new_peak': new_peak,
                'new_rms': new_rms,
                'peak_ratio': new_peak / original_peak if original_peak > 0 else 1.0,
                'rms_ratio': new_rms / original_rms if original_rms > 0 else 1.0
            }
            
            self.stats['normalize_applied'] += 1
            logger.debug(f"정규화 완료 ({method}): peak {original_peak:.3f} -> {new_peak:.3f}, "
                        f"RMS {original_rms:.3f} -> {new_rms:.3f}")
            
            return normalized_audio, normalize_info
            
        except Exception as e:
            logger.warning(f"정규화 실패, 원본 반환: {e}")
            return audio_data, {
                'normalized': False,
                'error': str(e)
            }
    
    def apply_bandpass_filter(self, audio_data: np.ndarray, 
                             sample_rate: int,
                             low_freq: Optional[int] = None,
                             high_freq: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        대역통과 필터 적용
        
        Args:
            audio_data (np.ndarray): 입력 오디오 데이터
            sample_rate (int): 샘플링 레이트
            low_freq (Optional[int]): 저주파 컷오프
            high_freq (Optional[int]): 고주파 컷오프
            
        Returns:
            Tuple[np.ndarray, dict]: (필터링된 오디오 데이터, 필터 정보)
        """
        if not self.config['audio']['filter_noise']['enabled']:
            return audio_data, {'filtered': False}
        
        # 파라미터 설정
        low_freq = low_freq or self.config['audio']['filter_noise']['low_freq']
        high_freq = high_freq or self.config['audio']['filter_noise']['high_freq']
        
        try:
            # Nyquist 주파수
            nyquist = sample_rate / 2
            
            # 정규화된 컷오프 주파수
            low_norm = low_freq / nyquist
            high_norm = min(high_freq / nyquist, 0.99)  # Nyquist 미만으로 제한
            
            # Butterworth 필터 계수 계산
            filter_order = 4
            b, a = scipy.signal.butter(filter_order, [low_norm, high_norm], btype='band')
            
            # 필터 적용
            filtered_audio = scipy.signal.filtfilt(b, a, audio_data)
            
            filter_info = {
                'filtered': True,
                'filter_type': 'bandpass',
                'filter_order': filter_order,
                'low_freq': low_freq,
                'high_freq': high_freq,
                'sample_rate': sample_rate,
                'low_norm': low_norm,
                'high_norm': high_norm
            }
            
            self.stats['filter_applied'] += 1
            logger.debug(f"대역통과 필터 적용: {low_freq}-{high_freq} Hz")
            
            return filtered_audio, filter_info
            
        except Exception as e:
            logger.warning(f"필터링 실패, 원본 반환: {e}")
            return audio_data, {
                'filtered': False,
                'error': str(e)
            }
    
    def check_audio_quality(self, audio_data: np.ndarray, 
                           sample_rate: int) -> dict:
        """
        오디오 품질 검사
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            dict: 품질 검사 결과
        """
        quality_config = self.config['quality_check']
        
        duration = len(audio_data) / sample_rate
        peak_amplitude = np.max(np.abs(audio_data))
        rms_amplitude = np.sqrt(np.mean(audio_data ** 2))
        
        # 기본 품질 지표
        quality_result = {
            'duration': duration,
            'peak_amplitude': peak_amplitude,
            'rms_amplitude': rms_amplitude,
            'dynamic_range': 20 * np.log10(peak_amplitude / (rms_amplitude + 1e-10)),
            'issues': []
        }
        
        # 길이 검사
        if duration < quality_config['min_duration']:
            quality_result['issues'].append(f"너무 짧음: {duration:.2f}s < {quality_config['min_duration']}s")
        if duration > quality_config['max_duration']:
            quality_result['issues'].append(f"너무 김: {duration:.2f}s > {quality_config['max_duration']}s")
        
        # 클리핑 검사
        if quality_config['check_clipping']:
            clipping_ratio = np.sum(np.abs(audio_data) > 0.99) / len(audio_data)
            quality_result['clipping_ratio'] = clipping_ratio
            if clipping_ratio > 0.01:  # 1% 이상 클리핑
                quality_result['issues'].append(f"클리핑 감지: {clipping_ratio:.2%}")
        
        # 무음 검사
        if quality_config['check_silence']:
            silence_threshold_linear = 10 ** (quality_config['silence_threshold'] / 20)
            silence_ratio = np.sum(np.abs(audio_data) < silence_threshold_linear) / len(audio_data)
            quality_result['silence_ratio'] = silence_ratio
            if silence_ratio > quality_config['max_silence_ratio']:
                quality_result['issues'].append(f"과도한 무음: {silence_ratio:.2%}")
        
        # SNR 추정 (간단한 방법)
        if rms_amplitude > 0:
            noise_floor = np.percentile(np.abs(audio_data), 10)  # 하위 10%를 노이즈로 가정
            snr_estimate = 20 * np.log10(rms_amplitude / (noise_floor + 1e-10))
            quality_result['snr_estimate'] = snr_estimate
            
            if snr_estimate < 20:  # 20dB 미만
                quality_result['issues'].append(f"낮은 SNR: {snr_estimate:.1f}dB")
        
        # 품질 등급 결정
        if len(quality_result['issues']) == 0:
            quality_result['quality_grade'] = 'excellent'
        elif len(quality_result['issues']) <= 2:
            quality_result['quality_grade'] = 'good'
        elif len(quality_result['issues']) <= 4:
            quality_result['quality_grade'] = 'fair'
        else:
            quality_result['quality_grade'] = 'poor'
        
        if quality_result['issues']:
            self.stats['quality_issues'] += 1
        
        return quality_result
    
    def preprocess_audio(self, audio_data: np.ndarray, 
                        sample_rate: int) -> Tuple[np.ndarray, dict]:
        """
        전체 전처리 파이프라인 실행
        
        Args:
            audio_data (np.ndarray): 입력 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            Tuple[np.ndarray, dict]: (전처리된 오디오 데이터, 전처리 정보)
        """
        processing_info = {
            'original_shape': audio_data.shape,
            'sample_rate': sample_rate,
            'steps': []
        }
        
        processed_audio = audio_data.copy()
        
        # 1. 품질 검사 (전처리 전)
        pre_quality = self.check_audio_quality(processed_audio, sample_rate)
        processing_info['pre_quality'] = pre_quality
        
        # 2. 묵음 제거
        processed_audio, trim_info = self.trim_silence(processed_audio, sample_rate)
        processing_info['steps'].append(('trim', trim_info))
        
        # 3. 필터링 (노이즈 제거)
        processed_audio, filter_info = self.apply_bandpass_filter(processed_audio, sample_rate)
        processing_info['steps'].append(('filter', filter_info))
        
        # 4. 정규화
        processed_audio, normalize_info = self.normalize_audio(processed_audio)
        processing_info['steps'].append(('normalize', normalize_info))
        
        # 5. 품질 검사 (전처리 후)
        post_quality = self.check_audio_quality(processed_audio, sample_rate)
        processing_info['post_quality'] = post_quality
        
        # 최종 정보 업데이트
        processing_info['final_shape'] = processed_audio.shape
        processing_info['duration_change'] = (len(processed_audio) - len(audio_data)) / sample_rate
        processing_info['quality_improvement'] = len(pre_quality['issues']) - len(post_quality['issues'])
        
        self.stats['processed_files'] += 1
        
        logger.debug(f"전처리 완료: {audio_data.shape} -> {processed_audio.shape}, "
                    f"품질 개선: {processing_info['quality_improvement']}개 이슈 해결")
        
        return processed_audio, processing_info
    
    def get_stats(self) -> dict:
        """전처리 통계 정보 반환"""
        return self.stats.copy()
    
    def reset_stats(self):
        """통계 정보 초기화"""
        self.stats = {
            'processed_files': 0,
            'trim_applied': 0,
            'normalize_applied': 0,
            'filter_applied': 0,
            'quality_issues': 0,
            'augmented_samples': 0,
            'augmentation_types_used': {}
        }
        logger.info("AudioPreprocessor 통계 정보가 초기화되었습니다.")
    
    def __repr__(self) -> str:
        return f"AudioPreprocessor(processed_files={self.stats['processed_files']})"

    def augment_audio(self, audio_data: np.ndarray, 
                     sample_rate: int,
                     target_multiplier: Optional[int] = None,
                     force_enable: Optional[bool] = None) -> List[Tuple[np.ndarray, dict]]:
        """
        오디오 데이터 증강 (배수 조절 가능)
        
        Args:
            audio_data: 원본 오디오 데이터
            sample_rate: 샘플링 레이트
            target_multiplier: 목표 배수 (None=설정 파일 사용)
            force_enable: 강제 활성화/비활성화 (None=설정 파일 사용)
            
        Returns:
            List[Tuple[np.ndarray, dict]]: (증강된 오디오, 증강 정보) 튜플 리스트
        """
        # 증강 활성화 여부 결정
        should_augment = force_enable if force_enable is not None else self.augmentation_enabled
        
        if not should_augment:
            logger.debug("증강 비활성화됨 - 원본만 반환")
            return [(audio_data, {
                'augmented': False, 
                'type': 'original',
                'multiplier_index': 0,
                'reason': 'augmentation_disabled'
            })]
        
        # 활성화된 증강 타입이 없으면 원본만 반환
        if not self.enabled_augmentation_types:
            logger.warning("활성화된 증강 타입이 없음 - 원본만 반환")
            return [(audio_data, {
                'augmented': False, 
                'type': 'original',
                'multiplier_index': 0,
                'reason': 'no_enabled_augmentation_types'
            })]
        
        # 목표 배수 설정
        if target_multiplier is None:
            target_multiplier = self.augmentation_config.get('total_multiplier', 5)
        
        max_samples = self.augmentation_config.get('max_augmented_samples', 10)
        target_multiplier = min(target_multiplier, max_samples)
        
        preserve_original = self.augmentation_config.get('advanced', {}).get('preserve_original', True)
        
        augmented_samples = []
        
        # 원본 데이터 추가 (선택사항)
        if preserve_original:
            augmented_samples.append((audio_data, {
                'augmented': False, 
                'type': 'original',
                'multiplier_index': 0,
                'reason': 'preserve_original'
            }))
        
        # 증강 타입별로 샘플 생성
        current_multiplier = len(augmented_samples)
        multiplier_index = 1
        
        for aug_type in self.enabled_augmentation_types:
            config = self.augmentation_config['types'][aug_type]
            
            count = config.get('count', 0)
            probability = config.get('probability', 1.0)
            
            # 확률에 따라 적용 여부 결정
            if np.random.random() > probability:
                continue
            
            # 목표 배수에 도달하면 중단
            if current_multiplier >= target_multiplier:
                break
            
            # 실제 생성할 개수 계산
            actual_count = min(count, target_multiplier - current_multiplier)
            
            for i in range(actual_count):
                if aug_type == 'pitch_shift':
                    augmented_audio, aug_info = self._pitch_shift(audio_data, sample_rate, config)
                elif aug_type == 'time_stretch':
                    augmented_audio, aug_info = self._time_stretch(audio_data, sample_rate, config)
                elif aug_type == 'add_noise':
                    augmented_audio, aug_info = self._add_noise(audio_data, sample_rate, config)
                elif aug_type == 'time_shift':
                    augmented_audio, aug_info = self._time_shift(audio_data, sample_rate, config)
                elif aug_type == 'volume_change':
                    augmented_audio, aug_info = self._volume_change(audio_data, sample_rate, config)
                else:
                    continue
                
                # 품질 검사 (선택사항)
                if self._check_augmentation_quality(augmented_audio, aug_info):
                    aug_info['multiplier_index'] = multiplier_index
                    augmented_samples.append((augmented_audio, aug_info))
                    current_multiplier += 1
                    multiplier_index += 1
                    
                    # 통계 업데이트
                    self.stats['augmented_samples'] += 1
                    aug_type_count = self.stats['augmentation_types_used'].get(aug_type, 0)
                    self.stats['augmentation_types_used'][aug_type] = aug_type_count + 1
                
                # 목표 배수에 도달하면 중단
                if current_multiplier >= target_multiplier:
                    break
        
        # 목표 배수에 미치지 못하면 추가 증강
        while current_multiplier < target_multiplier:
            # 활성화된 증강 타입 중에서 랜덤 선택
            if not self.enabled_augmentation_types:
                break
            
            aug_type = np.random.choice(self.enabled_augmentation_types)
            config = self.augmentation_config['types'][aug_type]
            
            if aug_type == 'pitch_shift':
                augmented_audio, aug_info = self._pitch_shift(audio_data, sample_rate, config)
            elif aug_type == 'time_stretch':
                augmented_audio, aug_info = self._time_stretch(audio_data, sample_rate, config)
            elif aug_type == 'add_noise':
                augmented_audio, aug_info = self._add_noise(audio_data, sample_rate, config)
            elif aug_type == 'time_shift':
                augmented_audio, aug_info = self._time_shift(audio_data, sample_rate, config)
            elif aug_type == 'volume_change':
                augmented_audio, aug_info = self._volume_change(audio_data, sample_rate, config)
            else:
                continue
            
            if self._check_augmentation_quality(augmented_audio, aug_info):
                aug_info['multiplier_index'] = multiplier_index
                augmented_samples.append((augmented_audio, aug_info))
                current_multiplier += 1
                multiplier_index += 1
                
                # 통계 업데이트
                self.stats['augmented_samples'] += 1
                aug_type_count = self.stats['augmentation_types_used'].get(aug_type, 0)
                self.stats['augmentation_types_used'][aug_type] = aug_type_count + 1
        
        logger.info(f"증강 완료: 원본 1개 → 총 {len(augmented_samples)}개 (배수: {len(augmented_samples)})")
        return augmented_samples
    
    def _check_augmentation_quality(self, audio_data: np.ndarray, aug_info: dict) -> bool:
        """증강 품질 검사"""
        quality_threshold = self.augmentation_config.get('advanced', {}).get('quality_threshold', 0.8)
        
        # 기본 품질 검사
        if np.any(np.isnan(audio_data)) or np.any(np.isinf(audio_data)):
            return False
        
        # 신호 대비 노이즈 비율 검사 (노이즈 증강의 경우)
        if aug_info.get('type') == 'add_noise':
            signal_power = np.mean(audio_data ** 2)
            if signal_power < 1e-6:  # 너무 작은 신호
                return False
        
        # 클리핑 검사
        clipping_ratio = np.sum(np.abs(audio_data) > 0.99) / len(audio_data)
        if clipping_ratio > 0.05:  # 5% 이상 클리핑
            return False
        
        return True
    
    def _pitch_shift(self, audio_data: np.ndarray, sample_rate: int, config: dict) -> Tuple[np.ndarray, dict]:
        """피치 시프트 증강"""
        steps = config.get('steps', [-2, -1, 1, 2])
        
        # 랜덤하게 하나의 스텝 선택
        step = np.random.choice(steps)
        
        try:
            augmented_audio = librosa.effects.pitch_shift(
                audio_data, 
                sr=sample_rate, 
                n_steps=step
            )
            
            return augmented_audio, {
                'augmented': True,
                'type': 'pitch_shift',
                'step': step,
                'original_length': len(audio_data),
                'augmented_length': len(augmented_audio)
            }
        except Exception as e:
            logger.warning(f"피치 시프트 실패: {e}")
            return audio_data, {
                'augmented': False,
                'type': 'pitch_shift_failed',
                'error': str(e)
            }
    
    def _time_stretch(self, audio_data: np.ndarray, sample_rate: int, config: dict) -> Tuple[np.ndarray, dict]:
        """타임 스트레치 증강"""
        rate_range = config.get('rate_range', [0.8, 1.2])
        
        rate = np.random.uniform(rate_range[0], rate_range[1])
        
        try:
            augmented_audio = librosa.effects.time_stretch(
                audio_data, 
                rate=rate
            )
            
            return augmented_audio, {
                'augmented': True,
                'type': 'time_stretch',
                'rate': rate,
                'original_length': len(audio_data),
                'augmented_length': len(augmented_audio)
            }
        except Exception as e:
            logger.warning(f"타임 스트레치 실패: {e}")
            return audio_data, {
                'augmented': False,
                'type': 'time_stretch_failed',
                'error': str(e)
            }
    
    def _add_noise(self, audio_data: np.ndarray, sample_rate: int, config: dict) -> Tuple[np.ndarray, dict]:
        """노이즈 추가 증강"""
        noise_level = config.get('noise_level', [0.001, 0.01])
        
        level = np.random.uniform(noise_level[0], noise_level[1])
        noise = np.random.normal(0, level, audio_data.shape)
        
        augmented_audio = audio_data + noise
        
        return augmented_audio, {
            'augmented': True,
            'type': 'add_noise',
            'noise_level': level,
            'original_length': len(audio_data),
            'augmented_length': len(augmented_audio)
        }
    
    def _time_shift(self, audio_data: np.ndarray, sample_rate: int, config: dict) -> Tuple[np.ndarray, dict]:
        """타임 시프트 증강"""
        shift_range = config.get('shift_range', [-0.1, 0.1])
        
        shift_seconds = np.random.uniform(shift_range[0], shift_range[1])
        shift_samples = int(shift_seconds * sample_rate)
        
        if shift_samples > 0:
            augmented_audio = np.pad(audio_data, (shift_samples, 0), mode='constant')
            augmented_audio = augmented_audio[:-shift_samples]
        else:
            augmented_audio = np.pad(audio_data, (0, -shift_samples), mode='constant')
            augmented_audio = augmented_audio[-shift_samples:]
        
        return augmented_audio, {
            'augmented': True,
            'type': 'time_shift',
            'shift_seconds': shift_seconds,
            'original_length': len(audio_data),
            'augmented_length': len(augmented_audio)
        }
    
    def _volume_change(self, audio_data: np.ndarray, sample_rate: int, config: dict) -> Tuple[np.ndarray, dict]:
        """볼륨 변경 증강"""
        gain_range = config.get('gain_range', [0.5, 1.5])
        
        gain = np.random.uniform(gain_range[0], gain_range[1])
        
        augmented_audio = audio_data * gain
        
        return augmented_audio, {
            'augmented': True,
            'type': 'volume_change',
            'gain': gain,
            'original_length': len(audio_data),
            'augmented_length': len(augmented_audio)
        }


# 편의 함수들
def preprocess_audio_file(audio_data: np.ndarray, 
                         sample_rate: int,
                         config_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, dict]:
    """
    단일 오디오 전처리를 위한 편의 함수
    
    Args:
        audio_data (np.ndarray): 오디오 데이터
        sample_rate (int): 샘플링 레이트
        config_path (Optional[Union[str, Path]]): 설정 파일 경로
        
    Returns:
        Tuple[np.ndarray, dict]: (전처리된 오디오 데이터, 전처리 정보)
    """
    preprocessor = AudioPreprocessor(config_path=config_path)
    return preprocessor.preprocess_audio(audio_data, sample_rate)


if __name__ == "__main__":
    # 사용 예제
    from pathlib import Path
    import sys
    
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / "preprocessing.yaml"
    
    # AudioPreprocessor 테스트
    preprocessor = AudioPreprocessor(config_path=config_path)
    
    # 테스트용 신호 생성
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
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
    
    print(f"\n🔧 AudioPreprocessor 테스트")
    print(f"원본 신호: {test_signal_with_silence.shape}, 길이: {len(test_signal_with_silence)/sample_rate:.2f}초")
    
    # 전처리 실행
    processed_signal, processing_info = preprocessor.preprocess_audio(
        test_signal_with_silence, sample_rate
    )
    
    print(f"처리된 신호: {processed_signal.shape}, 길이: {len(processed_signal)/sample_rate:.2f}초")
    print(f"길이 변화: {processing_info['duration_change']:.3f}초")
    print(f"품질 개선: {processing_info['quality_improvement']}개 이슈 해결")
    
    # 각 단계별 정보 출력
    for step_name, step_info in processing_info['steps']:
        if step_info.get('trimmed', False):
            print(f"  - 묵음 제거: {step_info['removed_ratio']:.1%} 제거")
        elif step_info.get('filtered', False):
            print(f"  - 필터링: {step_info['low_freq']}-{step_info['high_freq']} Hz")
        elif step_info.get('normalized', False):
            print(f"  - 정규화: {step_info['method']}, peak {step_info['original_peak']:.3f} -> {step_info['new_peak']:.3f}")
    
    # 품질 정보 출력
    post_quality = processing_info['post_quality']
    print(f"최종 품질: {post_quality['quality_grade']}")
    if post_quality['issues']:
        print(f"  남은 이슈: {post_quality['issues']}")
    
    # 통계 정보 출력
    stats = preprocessor.get_stats()
    print(f"처리 통계: {stats}") 