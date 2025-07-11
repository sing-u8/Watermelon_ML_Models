#!/usr/bin/env python3
"""
수박 당도 예측 프로젝트를 위한 샘플 오디오 데이터 생성 스크립트

수박을 두드렸을 때 나는 소리를 시뮬레이션하여 다양한 당도의 샘플 데이터를 생성합니다.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path
import pandas as pd
from typing import Tuple, List
import random

# 설정 상수
SAMPLE_RATE = 22050  # 22kHz 샘플링 레이트
DURATION = 2.0  # 2초 길이
BASE_FREQUENCY = 100  # 기본 주파수 (Hz)
NUM_WATERMELONS = 50  # 생성할 수박 샘플 수
SWEETNESS_RANGE = (8.0, 13.0)  # 당도 범위 (Brix)

def generate_watermelon_sound(sweetness: float, 
                             duration: float = DURATION, 
                             sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    수박의 당도에 따른 소리를 시뮬레이션하여 생성
    
    Args:
        sweetness: 당도값 (Brix)
        duration: 오디오 길이 (초)
        sample_rate: 샘플링 레이트
        
    Returns:
        생성된 오디오 신호
    """
    # 시간 축 생성
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 당도에 따른 주파수 특성 계산
    # 익은 수박(높은 당도)은 낮은 주파수, 안 익은 수박(낮은 당도)은 높은 주파수
    primary_freq = BASE_FREQUENCY + (13.0 - sweetness) * 30  # 80-230Hz 범위
    
    # 주요 주파수 성분들 (배음 구조)
    harmonics = [
        (primary_freq, 1.0),  # 기본 주파수
        (primary_freq * 2.1, 0.6),  # 두 번째 하모닉
        (primary_freq * 3.2, 0.3),  # 세 번째 하모닉
        (primary_freq * 4.5, 0.15), # 네 번째 하모닉
    ]
    
    # 신호 생성
    signal = np.zeros_like(t)
    
    for freq, amplitude in harmonics:
        # 당도에 따른 진폭 조정
        sweetness_factor = 0.5 + (sweetness - 8.0) / 10.0  # 0.5-1.0 범위
        adjusted_amplitude = amplitude * sweetness_factor
        
        # 사인파 생성 with 위상 변조 (보다 자연스러운 소리)
        phase_mod = 0.1 * np.sin(2 * np.pi * freq * 0.1 * t)
        wave = adjusted_amplitude * np.sin(2 * np.pi * freq * t + phase_mod)
        
        # 지수적 감쇠 적용 (타격 후 소리가 줄어드는 효과)
        decay_rate = 3.0 + (sweetness - 8.0) * 0.5  # 당도가 높을수록 빠른 감쇠
        envelope = np.exp(-decay_rate * t)
        
        signal += wave * envelope
    
    # 타격 초기의 임팩트 소리 추가
    impact_duration = 0.05  # 50ms
    impact_samples = int(sample_rate * impact_duration)
    impact_noise = np.random.normal(0, 0.3, impact_samples)
    impact_envelope = np.exp(-50 * t[:impact_samples])
    signal[:impact_samples] += impact_noise * impact_envelope
    
    # 배경 노이즈 추가 (현실적인 녹음 환경 시뮬레이션)
    noise_amplitude = 0.02
    noise = np.random.normal(0, noise_amplitude, len(signal))
    signal += noise
    
    # 정규화 (-1 ~ 1 범위)
    max_amplitude = np.max(np.abs(signal))
    if max_amplitude > 0:
        signal = signal / max_amplitude * 0.8  # 약간의 헤드룸 확보
    
    return signal.astype(np.float32)

def create_sample_dataset(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    샘플 데이터셋을 생성하고 메타데이터를 반환
    
    Args:
        output_dir: 출력 디렉토리
        
    Returns:
        메타데이터 DataFrame
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    
    print(f"🍉 {NUM_WATERMELONS}개의 수박 샘플 데이터 생성 중...")
    
    for watermelon_id in range(1, NUM_WATERMELONS + 1):
        # 랜덤한 당도 생성 (현실적인 분포)
        # 보통 수박의 당도는 9-12 Brix 범위에 많이 분포
        if random.random() < 0.7:  # 70%는 정상 범위
            sweetness = random.uniform(9.0, 12.0)
        else:  # 30%는 극값
            sweetness = random.uniform(8.0, 13.0)
        
        sweetness = round(sweetness, 1)
        
        # 수박별 폴더 생성
        watermelon_dir = output_path / f"{watermelon_id:03d}_{sweetness}"
        watermelon_dir.mkdir(exist_ok=True)
        
        # 각 수박마다 2-4개의 녹음 파일 생성 (다양한 위치에서 두드린 것을 시뮬레이션)
        num_recordings = random.randint(2, 4)
        
        for recording_idx in range(1, num_recordings + 1):
            # 약간의 변화를 주어 같은 수박이라도 녹음마다 차이가 있도록 함
            sweetness_variation = sweetness + random.uniform(-0.2, 0.2)
            sweetness_variation = max(8.0, min(13.0, sweetness_variation))  # 범위 제한
            
            # 오디오 생성
            audio_data = generate_watermelon_sound(sweetness_variation)
            
            # 파일 저장
            filename = f"recording_{recording_idx:02d}.wav"
            file_path = watermelon_dir / filename
            sf.write(file_path, audio_data, SAMPLE_RATE)
            
            # 메타데이터 기록
            metadata.append({
                'file_path': str(file_path.relative_to(Path('.'))),
                'watermelon_id': f"WM_{watermelon_id:03d}",
                'sweetness': sweetness,
                'recording_session': recording_idx,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'duration_sec': DURATION,
                'sample_rate': SAMPLE_RATE
            })
        
        if watermelon_id % 10 == 0:
            print(f"   진행률: {watermelon_id}/{NUM_WATERMELONS} ({watermelon_id/NUM_WATERMELONS*100:.1f}%)")
    
    # 메타데이터 DataFrame 생성
    metadata_df = pd.DataFrame(metadata)
    
    print(f"✅ 샘플 데이터 생성 완료!")
    print(f"   - 총 수박 개수: {NUM_WATERMELONS}개")
    print(f"   - 총 오디오 파일: {len(metadata_df)}개")
    print(f"   - 당도 범위: {metadata_df['sweetness'].min():.1f} - {metadata_df['sweetness'].max():.1f} Brix")
    print(f"   - 평균 당도: {metadata_df['sweetness'].mean():.1f} Brix")
    print(f"   - 총 데이터 크기: {metadata_df['file_size_mb'].sum():.2f} MB")
    
    return metadata_df

def main():
    """메인 실행 함수"""
    print("🍉 수박 당도 예측 프로젝트 - 샘플 데이터 생성기")
    print("=" * 50)
    
    # 샘플 데이터 생성
    metadata_df = create_sample_dataset()
    
    # 메타데이터 저장
    metadata_path = "data/watermelon_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"   - 메타데이터 저장: {metadata_path}")
    
    # 간단한 통계 출력
    print("\n📊 데이터셋 통계:")
    print(f"   - 당도 분포:")
    sweetness_bins = pd.cut(metadata_df['sweetness'], bins=5)
    sweetness_counts = pd.Series(sweetness_bins).value_counts().sort_index()
    print(sweetness_counts.to_string(header=False))
    
    print(f"\n   - 수박별 녹음 수:")
    recordings_per_watermelon = metadata_df.groupby('watermelon_id').size()
    print(f"     최소: {recordings_per_watermelon.min()}개")
    print(f"     최대: {recordings_per_watermelon.max()}개") 
    print(f"     평균: {recordings_per_watermelon.mean():.1f}개")
    
    print(f"\n🎉 샘플 데이터 생성이 완료되었습니다!")
    print(f"   다음 단계: Phase 1.4 탐색적 데이터 분석 (EDA)")

if __name__ == "__main__":
    main() 