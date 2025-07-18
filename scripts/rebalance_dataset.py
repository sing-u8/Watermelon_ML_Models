#!/usr/bin/env python3
"""
데이터셋 재균형 스크립트
각 브릭스별로 70:15:15 비율로 균일하게 분할
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

def rebalance_dataset():
    """각 브릭스별로 70:15:15 비율로 균일하게 분할"""
    
    print("=== 데이터셋 재균형 시작 ===")
    
    # 1. 원본 메타데이터 로드
    metadata_path = Path('data/watermelon_metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"원본 데이터: {len(metadata_df)}개 샘플")
    print(f"브릭스 종류: {len(metadata_df['sweetness'].unique())}개")
    
    # 2. 각 브릭스별 분포 확인
    sweetness_counts = metadata_df['sweetness'].value_counts().sort_index()
    print("\n=== 브릭스별 원본 분포 ===")
    for sweetness, count in sweetness_counts.items():
        print(f"{sweetness} Brix: {count}개")
    
    # 3. 각 브릭스별로 70:15:15 분할
    train_samples = []
    val_samples = []
    test_samples = []
    
    random.seed(42)  # 재현 가능성을 위한 시드 설정
    
    for sweetness in sorted(metadata_df['sweetness'].unique()):
        # 해당 브릭스의 모든 샘플 가져오기
        sweetness_samples = metadata_df[metadata_df['sweetness'] == sweetness].copy()
        total_count = len(sweetness_samples)
        
        # 70:15:15 비율로 분할 수 계산
        train_count = int(total_count * 0.7)
        val_count = int(total_count * 0.15)
        test_count = total_count - train_count - val_count
        
        print(f"\n{sweetness} Brix ({total_count}개): Train({train_count}) Val({val_count}) Test({test_count})")
        
        # 샘플을 랜덤하게 섞기
        indices = list(range(total_count))
        random.shuffle(indices)
        
        # 분할
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_count]
        test_indices = indices[train_count + val_count:]
        
        # 각 세트에 추가
        train_samples.append(sweetness_samples.iloc[train_indices])
        val_samples.append(sweetness_samples.iloc[val_indices])
        test_samples.append(sweetness_samples.iloc[test_indices])
    
    # 4. 새로운 데이터프레임 생성
    new_train_df = pd.concat(train_samples, ignore_index=True)
    new_val_df = pd.concat(val_samples, ignore_index=True)
    new_test_df = pd.concat(test_samples, ignore_index=True)
    
    print(f"\n=== 새로운 분할 결과 ===")
    print(f"훈련 세트: {len(new_train_df)}개 ({len(new_train_df)/len(metadata_df):.1%})")
    print(f"검증 세트: {len(new_val_df)}개 ({len(new_val_df)/len(metadata_df):.1%})")
    print(f"테스트 세트: {len(new_test_df)}개 ({len(new_test_df)/len(metadata_df):.1%})")
    
    # 5. 새로운 분할의 브릭스별 분포 확인
    print("\n=== 새로운 분할의 브릭스별 분포 ===")
    for sweetness in sorted(metadata_df['sweetness'].unique()):
        train_count = len(new_train_df[new_train_df['sweetness'] == sweetness])
        val_count = len(new_val_df[new_val_df['sweetness'] == sweetness])
        test_count = len(new_test_df[new_test_df['sweetness'] == sweetness])
        total_count = train_count + val_count + test_count
        
        train_ratio = train_count / total_count if total_count > 0 else 0
        val_ratio = val_count / total_count if total_count > 0 else 0
        test_ratio = test_count / total_count if total_count > 0 else 0
        
        print(f"{sweetness} Brix: Train({train_count}/{total_count} {train_ratio:.1%}) "
              f"Val({val_count}/{total_count} {val_ratio:.1%}) "
              f"Test({test_count}/{total_count} {test_ratio:.1%})")
    
    # 6. 균형성 점수 계산
    balance_scores = []
    for sweetness in sorted(metadata_df['sweetness'].unique()):
        train_count = len(new_train_df[new_train_df['sweetness'] == sweetness])
        val_count = len(new_val_df[new_val_df['sweetness'] == sweetness])
        test_count = len(new_test_df[new_test_df['sweetness'] == sweetness])
        total_count = train_count + val_count + test_count
        
        if total_count > 0:
            train_ratio = train_count / total_count
            val_ratio = val_count / total_count
            test_ratio = test_count / total_count
            
            # 이상적인 비율과의 차이 계산 (70%, 15%, 15%)
            ideal_train = 0.7
            ideal_val = 0.15
            ideal_test = 0.15
            
            deviation = abs(train_ratio - ideal_train) + abs(val_ratio - ideal_val) + abs(test_ratio - ideal_test)
            balance_scores.append(deviation)
    
    avg_balance_score = np.mean(balance_scores)
    print(f"\n=== 균형성 점수 ===")
    print(f"새로운 분할: {avg_balance_score:.3f} (낮을수록 균형적)")
    print("(0에 가까울수록 각 당도별로 70:15:15 비율에 가까움)")
    
    # 7. 파일 저장
    output_dir = Path('data/splits/full_dataset')
    
    new_train_df.to_csv(output_dir / 'train.csv', index=False)
    new_val_df.to_csv(output_dir / 'val.csv', index=False)
    new_test_df.to_csv(output_dir / 'test.csv', index=False)
    
    print(f"\n=== 파일 저장 완료 ===")
    print(f"훈련 세트: {output_dir / 'train.csv'}")
    print(f"검증 세트: {output_dir / 'val.csv'}")
    print(f"테스트 세트: {output_dir / 'test.csv'}")
    print(f"원본 백업: {output_dir / 'backup/'}")
    
    return new_train_df, new_val_df, new_test_df

if __name__ == "__main__":
    rebalance_dataset() 