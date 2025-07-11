import pandas as pd
import soundfile as sf
from pathlib import Path
import re
import os

print('🍉 실제 수박 데이터 메타데이터 생성 시작...')

data = []
data_dir = Path('data/raw')

# 수박 폴더 스캔
watermelon_folders = []
for folder in data_dir.iterdir():
    if folder.is_dir() and re.match(r'\d+_\d+\.?\d*', folder.name):
        watermelon_folders.append(folder)

# 폴더명 기준 정렬
watermelon_folders.sort(key=lambda x: int(x.name.split('_')[0]))

print(f'발견된 수박 폴더: {len(watermelon_folders)}개')

total_files = 0
for folder in watermelon_folders:
    wm_num, sweetness = folder.name.split('_')
    sweetness = float(sweetness)
    
    print(f'  📁 {folder.name} (당도: {sweetness} Brix)')
    
    # audios 또는 audio 폴더에서 파일 찾기
    audio_files = []
    audio_folders = ['audios', 'audio']
    
    for af in audio_folders:
        audio_dir = folder / af
        if audio_dir.exists():
            audio_files = sorted(list(audio_dir.glob('*.wav')))
            if not audio_files:
                audio_files = sorted(list(audio_dir.glob('*.m4a')))
            if not audio_files:
                audio_files = sorted(list(audio_dir.glob('*.mp3')))
            if audio_files:
                break
    
    session_idx = 1
    for audio_file in audio_files:
        try:
            # 파일 크기 확인
            file_size_mb = audio_file.stat().st_size / (1024 * 1024)
            
            # 오디오 정보 확인 (soundfile 사용)
            try:
                with sf.SoundFile(str(audio_file)) as f:
                    duration_sec = len(f) / f.samplerate
                    sample_rate = f.samplerate
            except:
                # soundfile로 안되면 기본값 사용
                duration_sec = 2.0
                sample_rate = 22050
            
            # 메타데이터 추가
            data.append({
                'file_path': str(audio_file.relative_to(Path('.'))),
                'watermelon_id': f'WM_{int(wm_num):03d}',
                'sweetness': sweetness,
                'recording_session': session_idx,
                'file_size_mb': round(file_size_mb, 4),
                'duration_sec': round(duration_sec, 2),
                'sample_rate': int(sample_rate)
            })
            
            session_idx += 1
            total_files += 1
            
        except Exception as e:
            print(f'    ⚠️ 파일 처리 실패: {audio_file.name} - {e}')
    
    print(f'    ✅ {len(audio_files)}개 파일 처리 완료')

# DataFrame 생성
metadata_df = pd.DataFrame(data)

if len(metadata_df) > 0:
    # 메타데이터 저장
    output_path = 'data/watermelon_metadata.csv'
    metadata_df.to_csv(output_path, index=False)
    
    print(f'\n📊 메타데이터 생성 완료!')
    print(f'   - 총 수박: {metadata_df["watermelon_id"].nunique()}개')
    print(f'   - 총 오디오 파일: {len(metadata_df)}개')
    print(f'   - 당도 범위: {metadata_df["sweetness"].min():.1f} ~ {metadata_df["sweetness"].max():.1f} Brix')
    print(f'   - 평균 당도: {metadata_df["sweetness"].mean():.2f} ± {metadata_df["sweetness"].std():.2f} Brix')
    print(f'   - 저장 위치: {output_path}')
    
    # 샘플 데이터 미리보기
    print(f'\n📋 메타데이터 샘플:')
    print(metadata_df.head().to_string(index=False))
    
else:
    print('❌ 오디오 파일을 찾을 수 없습니다!')