"""
🍉 수박 당도 예측 ML 프로젝트 - 특징 추출 모듈
AudioFeatureExtractor 클래스: 51개 오디오 특징 추출 (MFCC, 스펙트럴, 에너지, 리듬, 수박 전용, 통계적 특징)
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import librosa
import librosa.feature
import scipy.stats
from pathlib import Path
import yaml

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """
    오디오 신호에서 51개 특징을 추출하는 클래스
    
    특징 구성:
    - MFCC 특성: 13개 (음성학적 특성)
    - 스펙트럴 특성: 7개 (주파수 도메인)
    - 에너지 특성: 4개 (강도 및 품질)
    - 리듬 특성: 3개 (타이밍과 비트)
    - 수박 전용 특성: 8개 (도메인 특화)
    - 통계적 특성: 16개 (멜-스펙트로그램 기반)
    총 51개 특징
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        AudioFeatureExtractor 초기화
        
        Args:
            config_path (Optional[Union[str, Path]]): 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.feature_names = self._generate_feature_names()
        self.stats = {
            'extracted_features': 0,
            'failed_extractions': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(f"AudioFeatureExtractor 초기화 완료: {len(self.feature_names)}개 특징")
    
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> dict:
        """설정 파일 로드"""
        if config_path is None:
            return self._get_default_config()
        
        try:
            config_path = Path(config_path)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"특징 추출 설정 로드: {config_path}")
            return config
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패, 기본 설정 사용: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """기본 설정 반환"""
        return {
            'features': {
                'mfcc': {
                    'n_mfcc': 13,
                    'n_fft': 2048,
                    'hop_length': 512,
                    'n_mels': 128,
                    'fmin': 0,
                    'fmax': None
                },
                'spectral': {
                    'n_fft': 2048,
                    'hop_length': 512,
                    'centroid': True,
                    'bandwidth': True,
                    'contrast': True,
                    'flatness': True,
                    'rolloff': True,
                    'zcr': True,
                    'rmse': True
                },
                'mel_spectrogram': {
                    'n_mels': 128,
                    'n_fft': 2048,
                    'hop_length': 512,
                    'fmin': 0,
                    'fmax': None,
                    'statistics': [
                        'mean', 'std', 'min', 'max', 'median',
                        'q25', 'q75', 'skewness', 'kurtosis', 'energy',
                        'entropy', 'rms', 'peak', 'crest_factor',
                        'spectral_slope', 'harmonic_mean'
                    ]
                },
                'rhythm': {
                    'tempo': True,
                    'beat_track': True,
                    'onset_strength': True,
                    'hop_length': 512
                },
                'watermelon_specific': {
                    'fundamental_freq': True,
                    'harmonic_ratio': True,
                    'attack_time': True,
                    'decay_rate': True,
                    'sustain_level': True,
                    'brightness': True,
                    'roughness': True,
                    'inharmonicity': True
                },
                'energy': {
                    'rms_energy': True,
                    'peak_energy': True,
                    'energy_entropy': True,
                    'dynamic_range': True
                }
            }
        }
    
    def _generate_feature_names(self) -> List[str]:
        """특징 이름 리스트 생성"""
        feature_names = []
        
        # MFCC 특성 (13개)
        n_mfcc = self.config['features']['mfcc']['n_mfcc']
        for i in range(n_mfcc):
            feature_names.append(f'mfcc_{i+1}')
        
        # 스펙트럴 특성 (7개)
        spectral_config = self.config['features']['spectral']
        if spectral_config['centroid']:
            feature_names.append('spectral_centroid')
        if spectral_config['bandwidth']:
            feature_names.append('spectral_bandwidth')
        if spectral_config['contrast']:
            feature_names.append('spectral_contrast')
        if spectral_config['flatness']:
            feature_names.append('spectral_flatness')
        if spectral_config['rolloff']:
            feature_names.append('spectral_rolloff')
        if spectral_config['zcr']:
            feature_names.append('zero_crossing_rate')
        if spectral_config['rmse']:
            feature_names.append('rmse_energy')
        
        # 에너지 특성 (4개)
        energy_config = self.config['features']['energy']
        if energy_config['rms_energy']:
            feature_names.append('rms_energy_mean')
        if energy_config['peak_energy']:
            feature_names.append('peak_energy')
        if energy_config['energy_entropy']:
            feature_names.append('energy_entropy')
        if energy_config['dynamic_range']:
            feature_names.append('dynamic_range')
        
        # 리듬 특성 (3개)
        rhythm_config = self.config['features']['rhythm']
        if rhythm_config['tempo']:
            feature_names.append('tempo')
        if rhythm_config['beat_track']:
            feature_names.append('beat_strength')
        if rhythm_config['onset_strength']:
            feature_names.append('onset_strength_mean')
        
        # 수박 전용 특성 (8개)
        watermelon_config = self.config['features']['watermelon_specific']
        if watermelon_config['fundamental_freq']:
            feature_names.append('fundamental_frequency')
        if watermelon_config['harmonic_ratio']:
            feature_names.append('harmonic_ratio')
        if watermelon_config['attack_time']:
            feature_names.append('attack_time')
        if watermelon_config['decay_rate']:
            feature_names.append('decay_rate')
        if watermelon_config['sustain_level']:
            feature_names.append('sustain_level')
        if watermelon_config['brightness']:
            feature_names.append('brightness')
        if watermelon_config['roughness']:
            feature_names.append('roughness')
        if watermelon_config['inharmonicity']:
            feature_names.append('inharmonicity')
        
        # 멜-스펙트로그램 통계적 특성 (16개)
        mel_stats = self.config['features']['mel_spectrogram']['statistics']
        for stat in mel_stats:
            feature_names.append(f'mel_spec_{stat}')
        
        return feature_names
    
    def extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        MFCC 특성 추출 (13개)
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: MFCC 특성 벡터 (13개)
        """
        config = self.config['features']['mfcc']
        
        try:
            # MFCC 계산
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=config['n_mfcc'],
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                n_mels=config['n_mels'],
                fmin=config['fmin'],
                fmax=config['fmax']
            )
            
            # 시간 축에 대한 평균 계산
            mfcc_features = np.mean(mfcc, axis=1)
            
            return mfcc_features
            
        except Exception as e:
            logger.warning(f"MFCC 특성 추출 실패: {e}")
            return np.zeros(config['n_mfcc'])
    
    def extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        스펙트럴 특성 추출 (7개)
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 스펙트럴 특성 벡터 (7개)
        """
        config = self.config['features']['spectral']
        features = []
        
        try:
            n_fft = config['n_fft']
            hop_length = config['hop_length']
            
            # 스펙트럴 중심 (Spectral Centroid)
            if config['centroid']:
                centroid = librosa.feature.spectral_centroid(
                    y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
                )
                features.append(np.mean(centroid))
            
            # 스펙트럴 대역폭 (Spectral Bandwidth)
            if config['bandwidth']:
                bandwidth = librosa.feature.spectral_bandwidth(
                    y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
                )
                features.append(np.mean(bandwidth))
            
            # 스펙트럴 대비 (Spectral Contrast)
            if config['contrast']:
                contrast = librosa.feature.spectral_contrast(
                    y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
                )
                features.append(np.mean(contrast))
            
            # 스펙트럴 평탄도 (Spectral Flatness)
            if config['flatness']:
                flatness = librosa.feature.spectral_flatness(
                    y=audio_data, n_fft=n_fft, hop_length=hop_length
                )
                features.append(np.mean(flatness))
            
            # 스펙트럴 롤오프 (Spectral Rolloff)
            if config['rolloff']:
                rolloff = librosa.feature.spectral_rolloff(
                    y=audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length
                )
                features.append(np.mean(rolloff))
            
            # 영교차율 (Zero Crossing Rate)
            if config['zcr']:
                zcr = librosa.feature.zero_crossing_rate(
                    y=audio_data, frame_length=n_fft, hop_length=hop_length
                )
                features.append(np.mean(zcr))
            
            # RMS 에너지
            if config['rmse']:
                rmse = librosa.feature.rms(
                    y=audio_data, frame_length=n_fft, hop_length=hop_length
                )
                features.append(np.mean(rmse))
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"스펙트럴 특성 추출 실패: {e}")
            expected_features = sum([
                config['centroid'], config['bandwidth'], config['contrast'],
                config['flatness'], config['rolloff'], config['zcr'], config['rmse']
            ])
            return np.zeros(expected_features)
    
    def extract_energy_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        에너지 특성 추출 (4개)
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 에너지 특성 벡터 (4개)
        """
        config = self.config['features']['energy']
        features = []
        
        try:
            # RMS 에너지 평균
            if config['rms_energy']:
                rms_energy = np.sqrt(np.mean(audio_data ** 2))
                features.append(rms_energy)
            
            # 피크 에너지
            if config['peak_energy']:
                peak_energy = np.max(np.abs(audio_data))
                features.append(peak_energy)
            
            # 에너지 엔트로피
            if config['energy_entropy']:
                # 프레임별 에너지 계산
                frame_length = 2048
                hop_length = 512
                
                frame_energies = []
                for i in range(0, len(audio_data) - frame_length, hop_length):
                    frame = audio_data[i:i + frame_length]
                    energy = np.sum(frame ** 2)
                    frame_energies.append(energy)
                
                frame_energies = np.array(frame_energies)
                
                # 에너지 정규화
                total_energy = np.sum(frame_energies)
                if total_energy > 0:
                    prob_energies = frame_energies / total_energy
                    # 0이 아닌 값들에 대해서만 엔트로피 계산
                    prob_energies = prob_energies[prob_energies > 0]
                    energy_entropy = -np.sum(prob_energies * np.log2(prob_energies + 1e-10))
                else:
                    energy_entropy = 0.0
                
                features.append(energy_entropy)
            
            # 다이나믹 레인지
            if config['dynamic_range']:
                max_amp = np.max(np.abs(audio_data))
                min_amp = np.percentile(np.abs(audio_data), 1)  # 하위 1%를 최소값으로
                
                if min_amp > 0:
                    dynamic_range = 20 * np.log10(max_amp / min_amp)
                else:
                    dynamic_range = 100.0  # 최대값 설정
                
                features.append(dynamic_range)
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"에너지 특성 추출 실패: {e}")
            expected_features = sum([
                config['rms_energy'], config['peak_energy'],
                config['energy_entropy'], config['dynamic_range']
            ])
            return np.zeros(expected_features)
    
    def extract_rhythm_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        리듬 특성 추출 (3개)
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 리듬 특성 벡터 (3개)
        """
        config = self.config['features']['rhythm']
        features = []
        
        try:
            hop_length = config['hop_length']
            
            # 템포 추출
            if config['tempo']:
                try:
                    tempo, _ = librosa.beat.beat_track(
                        y=audio_data, sr=sample_rate, hop_length=hop_length
                    )
                    features.append(float(tempo))
                except:
                    features.append(120.0)  # 기본값
            
            # 비트 강도
            if config['beat_track']:
                try:
                    onset_env = librosa.onset.onset_strength(
                        y=audio_data, sr=sample_rate, hop_length=hop_length
                    )
                    beat_strength = np.mean(onset_env)
                    features.append(beat_strength)
                except:
                    features.append(0.0)
            
            # 온셋 강도 평균
            if config['onset_strength']:
                try:
                    onset_env = librosa.onset.onset_strength(
                        y=audio_data, sr=sample_rate, hop_length=hop_length
                    )
                    onset_strength_mean = np.mean(onset_env)
                    features.append(onset_strength_mean)
                except:
                    features.append(0.0)
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"리듬 특성 추출 실패: {e}")
            expected_features = sum([
                config['tempo'], config['beat_track'], config['onset_strength']
            ])
            return np.zeros(expected_features)
    
    def extract_watermelon_specific_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        수박 전용 특성 추출 (8개)
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 수박 전용 특성 벡터 (8개)
        """
        config = self.config['features']['watermelon_specific']
        features = []
        
        try:
            # FFT 계산
            fft = np.fft.fft(audio_data)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft[:len(fft)//2])
            freqs = freqs[:len(freqs)//2]
            
            # 기본 주파수 (Fundamental Frequency)
            if config['fundamental_freq']:
                # 피크 주파수 찾기
                peak_idx = np.argmax(magnitude)
                fundamental_freq = freqs[peak_idx]
                features.append(fundamental_freq)
            
            # 하모닉 비율 (Harmonic Ratio)
            if config['harmonic_ratio']:
                # 간단한 하모닉 비율 계산
                if len(magnitude) > 10:
                    # 저주파 에너지 vs 고주파 에너지
                    low_freq_energy = np.sum(magnitude[:len(magnitude)//4])
                    high_freq_energy = np.sum(magnitude[len(magnitude)//4:])
                    
                    if high_freq_energy > 0:
                        harmonic_ratio = low_freq_energy / high_freq_energy
                    else:
                        harmonic_ratio = 1.0
                else:
                    harmonic_ratio = 1.0
                
                features.append(harmonic_ratio)
            
            # 어택 타임 (Attack Time)
            if config['attack_time']:
                # 신호의 초기 상승 시간 계산
                envelope = np.abs(audio_data)
                max_val = np.max(envelope)
                
                if max_val > 0:
                    # 10%에서 90%까지 상승하는 시간
                    threshold_10 = 0.1 * max_val
                    threshold_90 = 0.9 * max_val
                    
                    idx_10 = np.where(envelope >= threshold_10)[0]
                    idx_90 = np.where(envelope >= threshold_90)[0]
                    
                    if len(idx_10) > 0 and len(idx_90) > 0:
                        attack_time = (idx_90[0] - idx_10[0]) / sample_rate
                    else:
                        attack_time = 0.0
                else:
                    attack_time = 0.0
                
                features.append(attack_time)
            
            # 감쇠율 (Decay Rate)
            if config['decay_rate']:
                # 신호의 감쇠 특성 계산
                envelope = np.abs(audio_data)
                max_idx = np.argmax(envelope)
                
                if max_idx < len(envelope) - 100:  # 충분한 후행 데이터가 있는 경우
                    tail_signal = envelope[max_idx:]
                    
                    # 지수 감쇠 피팅 시도
                    if len(tail_signal) > 10 and np.max(tail_signal) > 0:
                        # 로그 스케일에서 선형 피팅
                        tail_signal_log = np.log(tail_signal + 1e-10)
                        t = np.arange(len(tail_signal))
                        
                        # 선형 회귀
                        slope, _ = np.polyfit(t, tail_signal_log, 1)
                        decay_rate = -slope * sample_rate  # 초당 감쇠율
                    else:
                        decay_rate = 0.0
                else:
                    decay_rate = 0.0
                
                features.append(decay_rate)
            
            # 서스테인 레벨 (Sustain Level)
            if config['sustain_level']:
                # 신호의 중간 부분 평균 레벨
                mid_start = len(audio_data) // 3
                mid_end = 2 * len(audio_data) // 3
                
                if mid_end > mid_start:
                    sustain_level = np.mean(np.abs(audio_data[mid_start:mid_end]))
                else:
                    sustain_level = np.mean(np.abs(audio_data))
                
                features.append(sustain_level)
            
            # 밝기 (Brightness) - 고주파 에너지 비율
            if config['brightness']:
                if len(magnitude) > 4:
                    high_freq_start = len(magnitude) // 2
                    high_freq_energy = np.sum(magnitude[high_freq_start:])
                    total_energy = np.sum(magnitude)
                    
                    if total_energy > 0:
                        brightness = high_freq_energy / total_energy
                    else:
                        brightness = 0.0
                else:
                    brightness = 0.0
                
                features.append(brightness)
            
            # 거칠기 (Roughness) - 주파수 변동성
            if config['roughness']:
                # 스펙트럴 불규칙성 측정
                if len(magnitude) > 3:
                    # 인접한 주파수 빈 간의 차이의 표준편차
                    spectral_diff = np.diff(magnitude)
                    roughness = np.std(spectral_diff)
                else:
                    roughness = 0.0
                
                features.append(roughness)
            
            # 비하모닉성 (Inharmonicity)
            if config['inharmonicity']:
                # 하모닉 구조에서 벗어난 정도 측정
                if len(magnitude) > 10:
                    # 주파수 스펙트럼의 불규칙성
                    # 이론적 하모닉 주파수와의 편차 측정 (간소화된 버전)
                    spectral_irregularity = np.std(magnitude) / (np.mean(magnitude) + 1e-10)
                    inharmonicity = spectral_irregularity
                else:
                    inharmonicity = 0.0
                
                features.append(inharmonicity)
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"수박 전용 특성 추출 실패: {e}")
            expected_features = sum([
                config['fundamental_freq'], config['harmonic_ratio'],
                config['attack_time'], config['decay_rate'],
                config['sustain_level'], config['brightness'],
                config['roughness'], config['inharmonicity']
            ])
            return np.zeros(expected_features)
    
    def extract_mel_spectrogram_statistics(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        멜-스펙트로그램 통계적 특성 추출 (16개)
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 멜-스펙트로그램 통계 특성 벡터 (16개)
        """
        config = self.config['features']['mel_spectrogram']
        features = []
        
        try:
            # 멜-스펙트로그램 계산
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=config['n_mels'],
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                fmin=config['fmin'],
                fmax=config['fmax']
            )
            
            # dB 스케일로 변환
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # 전체 스펙트로그램에 대한 통계량 계산
            flat_spec = mel_spec_db.flatten()
            
            for stat in config['statistics']:
                if stat == 'mean':
                    features.append(np.mean(flat_spec))
                elif stat == 'std':
                    features.append(np.std(flat_spec))
                elif stat == 'min':
                    features.append(np.min(flat_spec))
                elif stat == 'max':
                    features.append(np.max(flat_spec))
                elif stat == 'median':
                    features.append(np.median(flat_spec))
                elif stat == 'q25':
                    features.append(np.percentile(flat_spec, 25))
                elif stat == 'q75':
                    features.append(np.percentile(flat_spec, 75))
                elif stat == 'skewness':
                    features.append(scipy.stats.skew(flat_spec))
                elif stat == 'kurtosis':
                    features.append(scipy.stats.kurtosis(flat_spec))
                elif stat == 'energy':
                    features.append(np.sum(mel_spec ** 2))
                elif stat == 'entropy':
                    # 정규화된 스펙트로그램의 엔트로피
                    normalized_spec = mel_spec / (np.sum(mel_spec) + 1e-10)
                    entropy = -np.sum(normalized_spec * np.log2(normalized_spec + 1e-10))
                    features.append(entropy)
                elif stat == 'rms':
                    features.append(np.sqrt(np.mean(flat_spec ** 2)))
                elif stat == 'peak':
                    features.append(np.max(np.abs(flat_spec)))
                elif stat == 'crest_factor':
                    rms_val = np.sqrt(np.mean(flat_spec ** 2))
                    peak_val = np.max(np.abs(flat_spec))
                    if rms_val > 0:
                        features.append(peak_val / rms_val)
                    else:
                        features.append(0.0)
                elif stat == 'spectral_slope':
                    # 주파수 축에 대한 평균 기울기
                    freq_means = np.mean(mel_spec_db, axis=1)
                    freq_indices = np.arange(len(freq_means))
                    if len(freq_means) > 1:
                        slope, _ = np.polyfit(freq_indices, freq_means, 1)
                        features.append(slope)
                    else:
                        features.append(0.0)
                elif stat == 'harmonic_mean':
                    # 조화 평균 (양수 값들에 대해서만)
                    positive_values = flat_spec[flat_spec > 0]
                    if len(positive_values) > 0:
                        harmonic_mean = len(positive_values) / np.sum(1.0 / positive_values)
                        features.append(harmonic_mean)
                    else:
                        features.append(0.0)
                else:
                    features.append(0.0)  # 알 수 없는 통계량
            
            return np.array(features)
            
        except Exception as e:
            logger.warning(f"멜-스펙트로그램 통계 특성 추출 실패: {e}")
            return np.zeros(len(config['statistics']))
    
    def extract_all_features(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        모든 특징을 추출하여 1D 벡터로 반환
        
        Args:
            audio_data (np.ndarray): 오디오 데이터
            sample_rate (int): 샘플링 레이트
            
        Returns:
            np.ndarray: 모든 특징을 포함한 1D 벡터 (51개)
        """
        import time
        start_time = time.time()
        
        try:
            # 각 특징 그룹별 추출
            mfcc_features = self.extract_mfcc_features(audio_data, sample_rate)
            spectral_features = self.extract_spectral_features(audio_data, sample_rate)
            energy_features = self.extract_energy_features(audio_data, sample_rate)
            rhythm_features = self.extract_rhythm_features(audio_data, sample_rate)
            watermelon_features = self.extract_watermelon_specific_features(audio_data, sample_rate)
            mel_stat_features = self.extract_mel_spectrogram_statistics(audio_data, sample_rate)
            
            # 모든 특징 결합
            all_features = np.concatenate([
                mfcc_features,
                spectral_features,
                energy_features,
                rhythm_features,
                watermelon_features,
                mel_stat_features
            ])
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self.stats['extracted_features'] += 1
            self.stats['total_processing_time'] += processing_time
            
            logger.debug(f"특징 추출 완료: {len(all_features)}개 특징, {processing_time:.3f}초")
            
            return all_features
            
        except Exception as e:
            logger.error(f"특징 추출 실패: {e}")
            self.stats['failed_extractions'] += 1
            return np.zeros(len(self.feature_names))
    
    def get_feature_names(self) -> List[str]:
        """특징 이름 리스트 반환"""
        return self.feature_names.copy()
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """특징을 그룹별로 분류하여 반환"""
        groups = {
            'mfcc': [],
            'spectral': [],
            'energy': [],
            'rhythm': [],
            'watermelon_specific': [],
            'mel_statistics': []
        }
        
        for name in self.feature_names:
            if name.startswith('mfcc_'):
                groups['mfcc'].append(name)
            elif name.startswith('spectral_') or name in ['zero_crossing_rate', 'rmse_energy']:
                groups['spectral'].append(name)
            elif name in ['rms_energy_mean', 'peak_energy', 'energy_entropy', 'dynamic_range']:
                groups['energy'].append(name)
            elif name in ['tempo', 'beat_strength', 'onset_strength_mean']:
                groups['rhythm'].append(name)
            elif name in ['fundamental_frequency', 'harmonic_ratio', 'attack_time', 'decay_rate',
                         'sustain_level', 'brightness', 'roughness', 'inharmonicity']:
                groups['watermelon_specific'].append(name)
            elif name.startswith('mel_spec_'):
                groups['mel_statistics'].append(name)
        
        return groups
    
    def get_stats(self) -> dict:
        """특징 추출 통계 정보 반환"""
        stats = self.stats.copy()
        if stats['extracted_features'] > 0:
            stats['avg_processing_time'] = stats['total_processing_time'] / stats['extracted_features']
        else:
            stats['avg_processing_time'] = 0.0
        return stats
    
    def reset_stats(self):
        """통계 정보 초기화"""
        self.stats = {
            'extracted_features': 0,
            'failed_extractions': 0,
            'total_processing_time': 0.0
        }
        logger.info("AudioFeatureExtractor 통계 정보가 초기화되었습니다.")
    
    def __repr__(self) -> str:
        return (f"AudioFeatureExtractor(features={len(self.feature_names)}, "
                f"extracted={self.stats['extracted_features']})")


# 편의 함수들
def extract_audio_features(audio_data: np.ndarray, 
                          sample_rate: int,
                          config_path: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    단일 오디오 특징 추출을 위한 편의 함수
    
    Args:
        audio_data (np.ndarray): 오디오 데이터
        sample_rate (int): 샘플링 레이트
        config_path (Optional[Union[str, Path]]): 설정 파일 경로
        
    Returns:
        Tuple[np.ndarray, List[str]]: (특징 벡터, 특징 이름 리스트)
    """
    extractor = AudioFeatureExtractor(config_path=config_path)
    features = extractor.extract_all_features(audio_data, sample_rate)
    feature_names = extractor.get_feature_names()
    return features, feature_names


if __name__ == "__main__":
    # 사용 예제
    from pathlib import Path
    import pandas as pd
    
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "configs" / "preprocessing.yaml"
    
    # AudioFeatureExtractor 테스트
    extractor = AudioFeatureExtractor(config_path=config_path)
    
    print(f"\n🔍 AudioFeatureExtractor 테스트")
    print(f"총 특징 수: {len(extractor.get_feature_names())}")
    
    # 특징 그룹별 개수 출력
    feature_groups = extractor.get_feature_groups()
    for group_name, features in feature_groups.items():
        print(f"  - {group_name}: {len(features)}개")
    
    # 테스트용 신호 생성
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 수박 소리 시뮬레이션
    fundamental = 440
    test_signal = (
        0.5 * np.sin(2 * np.pi * fundamental * t) +
        0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +
        0.2 * np.sin(2 * np.pi * fundamental * 3 * t) +
        0.1 * np.random.normal(0, 0.1, len(t))
    )
    
    # 특징 추출 실행
    print(f"\n특징 추출 실행 중...")
    features = extractor.extract_all_features(test_signal, sample_rate)
    
    print(f"추출된 특징: {len(features)}개")
    print(f"특징 벡터 형태: {features.shape}")
    print(f"특징 값 범위: [{np.min(features):.3f}, {np.max(features):.3f}]")
    
    # NaN/Inf 확인
    nan_count = np.sum(np.isnan(features))
    inf_count = np.sum(np.isinf(features))
    print(f"NaN 값: {nan_count}개, Inf 값: {inf_count}개")
    
    # 통계 정보 출력
    stats = extractor.get_stats()
    print(f"추출 통계: {stats}")
    
    # 특징 이름과 값 출력 (처음 10개만)
    feature_names = extractor.get_feature_names()
    print(f"\n특징 샘플 (처음 10개):")
    for i in range(min(10, len(features))):
        print(f"  {feature_names[i]}: {features[i]:.6f}")
    
    # DataFrame으로 저장 예제
    feature_df = pd.DataFrame([features], columns=feature_names)
    print(f"\nDataFrame 생성 성공: {feature_df.shape}") 