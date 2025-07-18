import pandas as pd
import numpy as np

# 데이터 로드
metadata_df = pd.read_csv('data/watermelon_metadata.csv')
train_df = pd.read_csv('data/splits/full_dataset/train.csv')
val_df = pd.read_csv('data/splits/full_dataset/val.csv')
test_df = pd.read_csv('data/splits/full_dataset/test.csv')

print('=== 당도별 분포 비교 ===')
print('원본 데이터:')
orig_dist = metadata_df['sweetness'].value_counts().sort_index()
print(orig_dist)

print('\n훈련 세트:')
train_dist = train_df['sweetness'].value_counts().sort_index()
print(train_dist)

print('\n검증 세트:')
val_dist = val_df['sweetness'].value_counts().sort_index()
print(val_dist)

print('\n테스트 세트:')
test_dist = test_df['sweetness'].value_counts().sort_index()
print(test_dist)

print('\n=== 균형성 분석 ===')
print('원본 대비 각 세트 비율:')
for sweetness in sorted(metadata_df['sweetness'].unique()):
    orig_count = orig_dist.get(sweetness, 0)
    train_count = train_dist.get(sweetness, 0)
    val_count = val_dist.get(sweetness, 0)
    test_count = test_dist.get(sweetness, 0)
    
    train_ratio = train_count/orig_count if orig_count > 0 else 0
    val_ratio = val_count/orig_count if orig_count > 0 else 0
    test_ratio = test_count/orig_count if orig_count > 0 else 0
    
    print(f'{sweetness}: Train({train_count}/{orig_count} {train_ratio:.1%}) Val({val_count}/{orig_count} {val_ratio:.1%}) Test({test_count}/{orig_count} {test_ratio:.1%})')

print('\n=== 전체 통계 ===')
print(f'원본 데이터: {len(metadata_df)}개')
print(f'훈련 세트: {len(train_df)}개 ({len(train_df)/len(metadata_df):.1%})')
print(f'검증 세트: {len(val_df)}개 ({len(val_df)/len(metadata_df):.1%})')
print(f'테스트 세트: {len(test_df)}개 ({len(test_df)/len(metadata_df):.1%})')

# 균형성 점수 계산
balance_scores = []
for sweetness in sorted(metadata_df['sweetness'].unique()):
    orig_count = orig_dist.get(sweetness, 0)
    if orig_count > 0:
        train_ratio = train_dist.get(sweetness, 0) / orig_count
        val_ratio = val_dist.get(sweetness, 0) / orig_count
        test_ratio = test_dist.get(sweetness, 0) / orig_count
        
        # 이상적인 비율과의 차이 계산 (70%, 15%, 15%)
        ideal_train = 0.7
        ideal_val = 0.15
        ideal_test = 0.15
        
        deviation = abs(train_ratio - ideal_train) + abs(val_ratio - ideal_val) + abs(test_ratio - ideal_test)
        balance_scores.append(deviation)

avg_balance_score = np.mean(balance_scores)
print(f'\n균형성 점수 (낮을수록 균형적): {avg_balance_score:.3f}')
print('(0에 가까울수록 각 당도별로 70:15:15 비율에 가까움)')