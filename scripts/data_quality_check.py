#!/usr/bin/env python3
"""
수박 당도 예측 프로젝트 - 데이터 품질 검사 스크립트

생성된 오디오 데이터와 메타데이터의 품질을 종합적으로 검증합니다.
"""

import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import os
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

def check_file_integrity(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    파일 무결성 검사
    
    Args:
        metadata_df: 메타데이터 DataFrame
        
    Returns:
        검사 결과 딕셔너리
    """
    print("🔍 파일 무결성 검사 중...")
    
    results = {
        'total_files': len(metadata_df),
        'existing_files': 0,
        'missing_files': [],
        'corrupted_files': [],
        'size_anomalies': []
    }
    
    for idx, row in metadata_df.iterrows():
        file_path = Path(str(row['file_path']))
        
        # 파일 존재 여부 확인
        if not file_path.exists():
            results['missing_files'].append(str(file_path))
            continue
            
        results['existing_files'] += 1
        
        try:
            # 오디오 파일 로딩 테스트
            data, sr = sf.read(file_path)
            
            # 기본 검증
            if len(data) == 0:
                results['corrupted_files'].append(str(file_path))
            
            # 샘플링 레이트 확인
            if sr != row['sample_rate']:
                results['corrupted_files'].append(f"{file_path} - sample rate mismatch")
            
            # 파일 크기 이상 확인
            actual_size = file_path.stat().st_size / (1024 * 1024)
            expected_size = row['file_size_mb']
            size_diff = abs(actual_size - expected_size) / expected_size
            
            if size_diff > 0.1:  # 10% 이상 차이
                results['size_anomalies'].append({
                    'file': str(file_path),
                    'expected': expected_size,
                    'actual': actual_size
                })
                
        except Exception as e:
            results['corrupted_files'].append(f"{file_path} - {str(e)}")
    
    return results

def check_audio_quality(metadata_df: pd.DataFrame, sample_size: int = 10) -> Dict[str, Any]:
    """
    오디오 품질 검사 (샘플 파일들)
    
    Args:
        metadata_df: 메타데이터 DataFrame
        sample_size: 검사할 샘플 파일 수
        
    Returns:
        음향 품질 검사 결과
    """
    print(f"🎵 오디오 품질 검사 중 ({sample_size}개 샘플)...")
    
    # 랜덤 샘플 선택
    sample_files = metadata_df.sample(min(sample_size, len(metadata_df)))
    
    results = {
        'sample_count': len(sample_files),
        'duration_consistency': True,
        'amplitude_stats': {},
        'frequency_analysis': {},
        'noise_levels': []
    }
    
    durations = []
    max_amplitudes = []
    rms_values = []
    
    for idx, row in sample_files.iterrows():
        try:
            file_path = Path(str(row['file_path']))
            data, sr = sf.read(file_path)
            
            # 길이 확인
            actual_duration = len(data) / sr
            durations.append(actual_duration)
            
            # 진폭 분석
            max_amp = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data**2))
            max_amplitudes.append(max_amp)
            rms_values.append(rms)
            
            # 노이즈 레벨 추정 (끝 부분의 조용한 구간)
            tail_samples = data[-int(0.1 * sr):]  # 마지막 0.1초
            noise_level = np.std(tail_samples)
            results['noise_levels'].append(noise_level)
            
        except Exception as e:
            print(f"   경고: {file_path} 분석 실패 - {e}")
    
    # 통계 계산
    results['amplitude_stats'] = {
        'max_amplitude_mean': np.mean(max_amplitudes),
        'max_amplitude_std': np.std(max_amplitudes),
        'rms_mean': np.mean(rms_values),
        'rms_std': np.std(rms_values)
    }
    
    # 길이 일관성 확인
    duration_std = np.std(durations)
    if duration_std > 0.01:  # 10ms 이상 차이
        results['duration_consistency'] = False
        
    results['duration_stats'] = {
        'mean': np.mean(durations),
        'std': duration_std,
        'min': np.min(durations),
        'max': np.max(durations)
    }
    
    return results

def check_metadata_quality(metadata_df: pd.DataFrame) -> Dict[str, Any]:
    """
    메타데이터 품질 검사
    
    Args:
        metadata_df: 메타데이터 DataFrame
        
    Returns:
        메타데이터 품질 검사 결과
    """
    print("📊 메타데이터 품질 검사 중...")
    
    results = {
        'total_records': len(metadata_df),
        'missing_values': {},
        'sweetness_distribution': {},
        'watermelon_balance': {},
        'data_coverage': {}
    }
    
    # 누락값 확인
    for column in metadata_df.columns:
        missing_count = metadata_df[column].isnull().sum()
        if missing_count > 0:
            results['missing_values'][column] = missing_count
    
    # 당도 분포 분석
    sweetness_values = metadata_df['sweetness']
    results['sweetness_distribution'] = {
        'min': sweetness_values.min(),
        'max': sweetness_values.max(),
        'mean': sweetness_values.mean(),
        'std': sweetness_values.std(),
        'unique_values': len(sweetness_values.unique())
    }
    
    # 수박별 녹음 균형 확인
    recordings_per_watermelon = metadata_df.groupby('watermelon_id').size()
    results['watermelon_balance'] = {
        'min_recordings': recordings_per_watermelon.min(),
        'max_recordings': recordings_per_watermelon.max(),
        'mean_recordings': recordings_per_watermelon.mean(),
        'imbalance_ratio': recordings_per_watermelon.max() / recordings_per_watermelon.min()
    }
    
    # 데이터 커버리지 분석
    sweetness_ranges = {
        'very_low': (8.0, 9.0),
        'low': (9.0, 10.0), 
        'medium': (10.0, 11.0),
        'high': (11.0, 12.0),
        'very_high': (12.0, 13.0)
    }
    
    for range_name, (min_val, max_val) in sweetness_ranges.items():
        count = len(metadata_df[
            (metadata_df['sweetness'] >= min_val) & 
            (metadata_df['sweetness'] < max_val)
        ])
        results['data_coverage'][range_name] = count
    
    return results

def generate_quality_report(integrity_results: Dict, 
                          audio_results: Dict, 
                          metadata_results: Dict) -> str:
    """
    품질 검사 결과 리포트 생성
    
    Args:
        integrity_results: 파일 무결성 검사 결과
        audio_results: 오디오 품질 검사 결과  
        metadata_results: 메타데이터 품질 검사 결과
        
    Returns:
        리포트 문자열
    """
    report = []
    report.append("🍉 수박 당도 예측 프로젝트 - 데이터 품질 검사 리포트")
    report.append("=" * 60)
    
    # 파일 무결성 결과
    report.append("\n📁 파일 무결성 검사 결과:")
    report.append(f"   총 파일 수: {integrity_results['total_files']}")
    report.append(f"   존재하는 파일: {integrity_results['existing_files']}")
    
    if integrity_results['missing_files']:
        report.append(f"   ❌ 누락된 파일: {len(integrity_results['missing_files'])}개")
        for file in integrity_results['missing_files'][:5]:  # 최대 5개만 표시
            report.append(f"      - {file}")
    else:
        report.append("   ✅ 누락된 파일 없음")
    
    if integrity_results['corrupted_files']:
        report.append(f"   ❌ 손상된 파일: {len(integrity_results['corrupted_files'])}개")
        for file in integrity_results['corrupted_files'][:5]:
            report.append(f"      - {file}")
    else:
        report.append("   ✅ 손상된 파일 없음")
    
    if integrity_results['size_anomalies']:
        report.append(f"   ⚠️  크기 이상 파일: {len(integrity_results['size_anomalies'])}개")
    else:
        report.append("   ✅ 파일 크기 정상")
    
    # 오디오 품질 결과
    report.append(f"\n🎵 오디오 품질 검사 결과 ({audio_results['sample_count']}개 샘플):")
    
    duration_stats = audio_results['duration_stats']
    report.append(f"   길이 일관성: {'✅ 일관됨' if audio_results['duration_consistency'] else '⚠️ 불일치'}")
    report.append(f"   평균 길이: {duration_stats['mean']:.3f}초 (±{duration_stats['std']:.3f})")
    
    amp_stats = audio_results['amplitude_stats']
    report.append(f"   진폭 정보:")
    report.append(f"      최대 진폭: {amp_stats['max_amplitude_mean']:.3f} (±{amp_stats['max_amplitude_std']:.3f})")
    report.append(f"      RMS 값: {amp_stats['rms_mean']:.3f} (±{amp_stats['rms_std']:.3f})")
    
    avg_noise = np.mean(audio_results['noise_levels'])
    report.append(f"   평균 노이즈 레벨: {avg_noise:.4f}")
    
    # 메타데이터 품질 결과
    report.append("\n📊 메타데이터 품질 검사 결과:")
    report.append(f"   총 레코드 수: {metadata_results['total_records']}")
    
    if metadata_results['missing_values']:
        report.append("   ❌ 누락값 발견:")
        for col, count in metadata_results['missing_values'].items():
            report.append(f"      {col}: {count}개")
    else:
        report.append("   ✅ 누락값 없음")
    
    sweet_dist = metadata_results['sweetness_distribution']
    report.append(f"   당도 분포:")
    report.append(f"      범위: {sweet_dist['min']:.1f} - {sweet_dist['max']:.1f} Brix")
    report.append(f"      평균: {sweet_dist['mean']:.1f} Brix (±{sweet_dist['std']:.1f})")
    report.append(f"      고유값: {sweet_dist['unique_values']}개")
    
    watermelon_bal = metadata_results['watermelon_balance'] 
    report.append(f"   수박별 녹음 균형:")
    report.append(f"      범위: {watermelon_bal['min_recordings']} - {watermelon_bal['max_recordings']}개")
    report.append(f"      평균: {watermelon_bal['mean_recordings']:.1f}개")
    report.append(f"      불균형 비율: {watermelon_bal['imbalance_ratio']:.1f}:1")
    
    report.append(f"   당도 구간별 데이터 분포:")
    for range_name, count in metadata_results['data_coverage'].items():
        report.append(f"      {range_name}: {count}개")
    
    # 전체 평가
    report.append("\n🎯 종합 평가:")
    
    issues = []
    if integrity_results['missing_files']:
        issues.append("누락된 파일 존재")
    if integrity_results['corrupted_files']:
        issues.append("손상된 파일 존재")
    if not audio_results['duration_consistency']:
        issues.append("오디오 길이 불일치")
    if metadata_results['missing_values']:
        issues.append("메타데이터 누락값 존재")
    if watermelon_bal['imbalance_ratio'] > 2.0:
        issues.append("수박별 녹음 수 불균형")
    
    if not issues:
        report.append("   ✅ 모든 품질 검사 통과! 데이터가 ML 훈련에 적합합니다.")
    else:
        report.append("   ⚠️  다음 이슈들이 발견되었습니다:")
        for issue in issues:
            report.append(f"      - {issue}")
    
    return "\n".join(report)

def main():
    """메인 실행 함수"""
    print("🔍 수박 당도 예측 프로젝트 - 데이터 품질 검사")
    print("=" * 50)
    
    # 메타데이터 로드
    metadata_path = "data/watermelon_metadata.csv"
    if not Path(metadata_path).exists():
        print(f"❌ 메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    print(f"📊 메타데이터 로드 완료: {len(metadata_df)}개 레코드")
    
    # 품질 검사 실행
    integrity_results = check_file_integrity(metadata_df)
    audio_results = check_audio_quality(metadata_df, sample_size=15)
    metadata_results = check_metadata_quality(metadata_df)
    
    # 리포트 생성 및 출력
    report = generate_quality_report(integrity_results, audio_results, metadata_results)
    print("\n" + report)
    
    # 리포트 파일 저장
    report_path = "experiments/data_quality_report.txt"
    Path("experiments").mkdir(exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n💾 품질 검사 리포트 저장: {report_path}")
    print("🎉 데이터 품질 검사 완료!")

if __name__ == "__main__":
    main() 