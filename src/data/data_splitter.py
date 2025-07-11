"""
🍉 수박 당도 예측 ML 프로젝트 - 데이터 분할 모듈
DataSplitter 클래스: Train/Validation/Test 세트 분할 및 균형 확인
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    데이터셋을 Train/Validation/Test 세트로 분할하는 클래스
    
    기능:
    - 층화 샘플링 (당도 구간별 균등 분할)
    - 분할 비율 설정 가능
    - 분할 결과 검증 및 시각화
    - 재현 가능한 분할 (random seed)
    """
    
    def __init__(self, train_ratio: float = 0.7, 
                 val_ratio: float = 0.15, 
                 test_ratio: float = 0.15,
                 random_state: int = 42):
        """
        DataSplitter 초기화
        
        Args:
            train_ratio (float): 훈련 세트 비율 (기본값: 0.7)
            val_ratio (float): 검증 세트 비율 (기본값: 0.15)
            test_ratio (float): 테스트 세트 비율 (기본값: 0.15)
            random_state (int): 재현성을 위한 랜덤 시드 (기본값: 42)
        """
        # 비율 검증
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"분할 비율의 합이 1.0이 아닙니다: {total_ratio}")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        
        self.stats = {
            'original_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'sweetness_bins': 0,
            'split_time': 0.0
        }
        
        logger.info(f"DataSplitter 초기화: Train({train_ratio:.1%}), "
                   f"Val({val_ratio:.1%}), Test({test_ratio:.1%})")
    
    def _create_sweetness_bins(self, sweetness_values: np.ndarray, 
                              n_bins: Optional[int] = None) -> np.ndarray:
        """
        당도 값을 구간별로 분류
        
        Args:
            sweetness_values (np.ndarray): 당도 값 배열
            n_bins (Optional[int]): 구간 수 (None이면 자동 결정)
            
        Returns:
            np.ndarray: 구간 레이블 배열
        """
        if n_bins is None:
            # 샘플 수에 따라 구간 수 자동 결정
            n_samples = len(sweetness_values)
            if n_samples < 50:
                n_bins = 3
            elif n_samples < 100:
                n_bins = 4
            else:
                n_bins = 5
        
        # 당도 범위에 따른 구간 분할
        min_sweetness = np.min(sweetness_values)
        max_sweetness = np.max(sweetness_values)
        
        # 구간 경계 생성
        bin_edges = np.linspace(min_sweetness, max_sweetness, n_bins + 1)
        
        # 구간 레이블 할당
        bin_labels = np.digitize(sweetness_values, bin_edges) - 1
        
        # 마지막 구간 조정 (최대값이 포함되도록)
        bin_labels[bin_labels >= n_bins] = n_bins - 1
        
        self.stats['sweetness_bins'] = n_bins
        
        logger.debug(f"당도 구간 생성: {n_bins}개 구간, 범위 [{min_sweetness:.1f}, {max_sweetness:.1f}]")
        
        return bin_labels, bin_edges
    
    def split_dataset(self, features_df: pd.DataFrame, 
                     target_column: str = 'sweetness',
                     stratify_bins: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        데이터셋을 Train/Validation/Test로 분할
        
        Args:
            features_df (pd.DataFrame): 특징과 타겟을 포함한 DataFrame
            target_column (str): 타겟 컬럼명 (기본값: 'sweetness')
            stratify_bins (Optional[int]): 층화 샘플링용 구간 수
            
        Returns:
            Dict[str, pd.DataFrame]: {'train': train_df, 'val': val_df, 'test': test_df}
        """
        import time
        start_time = time.time()
        
        logger.info(f"데이터셋 분할 시작: {len(features_df)}개 샘플")
        
        # 타겟 컬럼 확인
        if target_column not in features_df.columns:
            raise ValueError(f"타겟 컬럼을 찾을 수 없습니다: {target_column}")
        
        self.stats['original_samples'] = len(features_df)
        
        # 특징과 타겟 분리
        X = features_df.drop(columns=[target_column])
        y = features_df[target_column]
        
        # 층화 샘플링을 위한 구간 생성
        stratify_labels, bin_edges = self._create_sweetness_bins(
            y.values, n_bins=stratify_bins
        )
        
        # 1단계: Train과 (Val+Test) 분할
        train_val_ratio = self.val_ratio + self.test_ratio
        
        X_train, X_temp, y_train, y_temp, stratify_train, stratify_temp = train_test_split(
            X, y, stratify_labels,
            test_size=train_val_ratio,
            stratify=stratify_labels,
            random_state=self.random_state
        )
        
        # 2단계: (Val+Test)를 Val과 Test로 분할
        val_test_ratio = self.val_ratio / train_val_ratio
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_test_ratio),
            stratify=stratify_temp,
            random_state=self.random_state
        )
        
        # DataFrame 재구성
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        # 인덱스 재설정
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # 통계 업데이트
        self.stats['train_samples'] = len(train_df)
        self.stats['val_samples'] = len(val_df)
        self.stats['test_samples'] = len(test_df)
        self.stats['split_time'] = time.time() - start_time
        
        split_result = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        # 분할 결과 로그
        logger.info(f"분할 완료:")
        logger.info(f"  - Train: {len(train_df)}개 ({len(train_df)/len(features_df):.1%})")
        logger.info(f"  - Val: {len(val_df)}개 ({len(val_df)/len(features_df):.1%})")
        logger.info(f"  - Test: {len(test_df)}개 ({len(test_df)/len(features_df):.1%})")
        
        return split_result
    
    def validate_split(self, split_data: Dict[str, pd.DataFrame], 
                      target_column: str = 'sweetness') -> Dict:
        """
        분할 결과 검증
        
        Args:
            split_data (Dict[str, pd.DataFrame]): 분할된 데이터셋
            target_column (str): 타겟 컬럼명
            
        Returns:
            Dict: 검증 결과
        """
        logger.info("데이터 분할 검증 시작")
        
        validation_result = {
            'split_ratios': {},
            'sweetness_distributions': {},
            'statistical_tests': {},
            'issues': []
        }
        
        total_samples = sum(len(df) for df in split_data.values())
        
        # 분할 비율 확인
        for split_name, df in split_data.items():
            actual_ratio = len(df) / total_samples
            validation_result['split_ratios'][split_name] = {
                'actual': actual_ratio,
                'target': getattr(self, f'{split_name}_ratio'),
                'samples': len(df)
            }
        
        # 타겟 분포 비교
        for split_name, df in split_data.items():
            sweetness_values = df[target_column]
            validation_result['sweetness_distributions'][split_name] = {
                'mean': float(sweetness_values.mean()),
                'std': float(sweetness_values.std()),
                'min': float(sweetness_values.min()),
                'max': float(sweetness_values.max()),
                'median': float(sweetness_values.median()),
                'q25': float(sweetness_values.quantile(0.25)),
                'q75': float(sweetness_values.quantile(0.75))
            }
        
        # 분포 균형성 검사
        train_mean = validation_result['sweetness_distributions']['train']['mean']
        train_std = validation_result['sweetness_distributions']['train']['std']
        
        for split_name in ['val', 'test']:
            split_mean = validation_result['sweetness_distributions'][split_name]['mean']
            split_std = validation_result['sweetness_distributions'][split_name]['std']
            
            # 평균 차이 검사
            mean_diff = abs(split_mean - train_mean)
            if mean_diff > 0.5:  # 0.5 Brix 이상 차이
                validation_result['issues'].append(
                    f"{split_name} 세트의 평균 당도가 train과 {mean_diff:.2f} Brix 차이"
                )
            
            # 표준편차 차이 검사
            std_ratio = split_std / train_std if train_std > 0 else 1.0
            if std_ratio < 0.7 or std_ratio > 1.3:  # 30% 이상 차이
                validation_result['issues'].append(
                    f"{split_name} 세트의 표준편차가 train과 {abs(1-std_ratio):.1%} 차이"
                )
        
        # 분할 비율 검사
        for split_name, ratio_info in validation_result['split_ratios'].items():
            ratio_diff = abs(ratio_info['actual'] - ratio_info['target'])
            if ratio_diff > 0.02:  # 2% 이상 차이
                validation_result['issues'].append(
                    f"{split_name} 세트 비율이 목표와 {ratio_diff:.1%} 차이"
                )
        
        # 최소 샘플 수 검사
        min_samples_required = 10  # 최소 10개 샘플
        for split_name, df in split_data.items():
            if len(df) < min_samples_required:
                validation_result['issues'].append(
                    f"{split_name} 세트의 샘플 수가 부족: {len(df)}개 < {min_samples_required}개"
                )
        
        # 전체 검증 결과
        if len(validation_result['issues']) == 0:
            validation_result['overall_quality'] = 'excellent'
        elif len(validation_result['issues']) <= 2:
            validation_result['overall_quality'] = 'good'
        else:
            validation_result['overall_quality'] = 'poor'
        
        logger.info(f"분할 검증 완료: {validation_result['overall_quality']}")
        if validation_result['issues']:
            logger.warning(f"발견된 이슈: {validation_result['issues']}")
        
        return validation_result
    
    def save_splits(self, split_data: Dict[str, pd.DataFrame], 
                   output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        분할된 데이터를 CSV 파일로 저장
        
        Args:
            split_data (Dict[str, pd.DataFrame]): 분할된 데이터셋
            output_dir (Union[str, Path]): 출력 디렉토리
            
        Returns:
            Dict[str, str]: 저장된 파일 경로들
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        for split_name, df in split_data.items():
            file_path = output_dir / f"{split_name}.csv"
            df.to_csv(file_path, index=False)
            saved_files[split_name] = str(file_path)
            logger.info(f"{split_name} 세트 저장: {file_path} ({len(df)}개 샘플)")
        
        # 분할 정보 저장
        split_info = {
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'random_state': self.random_state,
            'total_samples': sum(len(df) for df in split_data.values()),
            'train_samples': len(split_data['train']),
            'val_samples': len(split_data['val']),
            'test_samples': len(split_data['test'])
        }
        
        info_path = output_dir / "split_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("🍉 수박 데이터셋 분할 정보\n")
            f.write("=" * 40 + "\n\n")
            for key, value in split_info.items():
                f.write(f"{key}: {value}\n")
        
        saved_files['split_info'] = str(info_path)
        logger.info(f"분할 정보 저장: {info_path}")
        
        return saved_files
    
    def visualize_split_distribution(self, split_data: Dict[str, pd.DataFrame], 
                                    target_column: str = 'sweetness',
                                    output_path: Optional[Union[str, Path]] = None) -> None:
        """
        분할된 데이터의 당도 분포 시각화
        
        Args:
            split_data (Dict[str, pd.DataFrame]): 분할된 데이터셋
            target_column (str): 타겟 컬럼명
            output_path (Optional[Union[str, Path]]): 저장할 이미지 경로
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🍉 수박 데이터셋 분할 결과', fontsize=16, fontweight='bold')
        
        # 색상 설정
        colors = {'train': '#2E8B57', 'val': '#FF6347', 'test': '#4682B4'}
        
        # 1. 히스토그램 비교
        ax1 = axes[0, 0]
        for split_name, df in split_data.items():
            sweetness_values = df[target_column]
            ax1.hist(sweetness_values, bins=15, alpha=0.7, 
                    label=f'{split_name.title()} (n={len(df)})',
                    color=colors[split_name])
        
        ax1.set_xlabel('당도 (Brix)')
        ax1.set_ylabel('샘플 수')
        ax1.set_title('당도 분포 히스토그램')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 박스 플롯
        ax2 = axes[0, 1]
        box_data = [split_data[name][target_column] for name in ['train', 'val', 'test']]
        box_plot = ax2.boxplot(box_data, labels=['Train', 'Val', 'Test'], 
                              patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], [colors['train'], colors['val'], colors['test']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('당도 (Brix)')
        ax2.set_title('당도 분포 박스 플롯')
        ax2.grid(True, alpha=0.3)
        
        # 3. 샘플 수 비교
        ax3 = axes[1, 0]
        split_names = list(split_data.keys())
        sample_counts = [len(split_data[name]) for name in split_names]
        bars = ax3.bar(split_names, sample_counts, 
                      color=[colors[name] for name in split_names], alpha=0.8)
        
        # 바 위에 숫자 표시
        for bar, count in zip(bars, sample_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom', fontweight='bold')
        
        ax3.set_ylabel('샘플 수')
        ax3.set_title('세트별 샘플 수')
        ax3.grid(True, alpha=0.3)
        
        # 4. 통계 정보 테이블
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # 통계 데이터 준비
        stats_data = []
        for split_name, df in split_data.items():
            sweetness_values = df[target_column]
            stats_data.append([
                split_name.title(),
                len(df),
                f"{sweetness_values.mean():.2f}",
                f"{sweetness_values.std():.2f}",
                f"{sweetness_values.min():.1f}-{sweetness_values.max():.1f}"
            ])
        
        # 테이블 생성
        table = ax4.table(cellText=stats_data,
                         colLabels=['세트', '샘플 수', '평균', '표준편차', '범위'],
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 헤더 스타일링
        for i in range(5):
            table[(0, i)].set_facecolor('#E6E6FA')
            table[(0, i)].set_text_props(weight='bold')
        
        # 행 색상 설정
        for i, split_name in enumerate(['train', 'val', 'test']):
            for j in range(5):
                table[(i+1, j)].set_facecolor(colors[split_name])
                table[(i+1, j)].set_alpha(0.3)
        
        ax4.set_title('분할 통계 요약')
        
        plt.tight_layout()
        
        # 이미지 저장
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"분할 시각화 저장: {output_path}")
        
        plt.show()
    
    def get_stats(self) -> Dict:
        """통계 정보 반환"""
        return self.stats.copy()
    
    def reset_stats(self):
        """통계 정보 초기화"""
        self.stats = {
            'original_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'sweetness_bins': 0,
            'split_time': 0.0
        }
        logger.info("DataSplitter 통계 정보가 초기화되었습니다.")
    
    def __repr__(self) -> str:
        return (f"DataSplitter(train={self.train_ratio:.1%}, "
                f"val={self.val_ratio:.1%}, test={self.test_ratio:.1%}, "
                f"random_state={self.random_state})")


# 편의 함수들
def split_watermelon_dataset(features_csv_path: Union[str, Path],
                            output_dir: Union[str, Path],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            test_ratio: float = 0.15,
                            random_state: int = 42) -> Dict:
    """
    수박 데이터셋 분할을 위한 편의 함수
    
    Args:
        features_csv_path (Union[str, Path]): 특징 CSV 파일 경로
        output_dir (Union[str, Path]): 출력 디렉토리
        train_ratio (float): 훈련 세트 비율
        val_ratio (float): 검증 세트 비율
        test_ratio (float): 테스트 세트 비율
        random_state (int): 랜덤 시드
        
    Returns:
        Dict: 분할 결과 정보
    """
    # 데이터 로드
    features_df = pd.read_csv(features_csv_path)
    
    # DataSplitter 생성 및 분할
    splitter = DataSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    split_data = splitter.split_dataset(features_df)
    
    # 분할 검증
    validation_result = splitter.validate_split(split_data)
    
    # 파일 저장
    saved_files = splitter.save_splits(split_data, output_dir)
    
    # 시각화 저장
    viz_path = Path(output_dir) / "split_distribution.png"
    splitter.visualize_split_distribution(split_data, output_path=viz_path)
    
    return {
        'split_data': split_data,
        'validation_result': validation_result,
        'saved_files': saved_files,
        'stats': splitter.get_stats()
    }


if __name__ == "__main__":
    # 사용 예제
    from pathlib import Path
    import numpy as np
    
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).parent.parent.parent
    
    print(f"\n📊 DataSplitter 테스트")
    
    # 테스트용 데이터 생성
    np.random.seed(42)
    n_samples = 100
    n_features = 51
    
    # 가짜 특징 데이터 생성
    features = np.random.randn(n_samples, n_features)
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    
    # 가짜 당도 데이터 생성 (9-12 Brix 범위)
    sweetness = np.random.normal(10.5, 1.0, n_samples)
    sweetness = np.clip(sweetness, 9.0, 12.0)
    
    # DataFrame 생성
    test_df = pd.DataFrame(features, columns=feature_names)
    test_df['sweetness'] = sweetness
    
    print(f"테스트 데이터 생성: {test_df.shape}")
    print(f"당도 범위: {sweetness.min():.1f} - {sweetness.max():.1f} Brix")
    
    # DataSplitter 생성
    splitter = DataSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    # 데이터 분할
    split_data = splitter.split_dataset(test_df)
    
    print(f"\n분할 결과:")
    for split_name, df in split_data.items():
        sweetness_stats = df['sweetness']
        print(f"  - {split_name.title()}: {len(df)}개 샘플, "
              f"당도 평균 {sweetness_stats.mean():.2f}±{sweetness_stats.std():.2f}")
    
    # 분할 검증
    validation_result = splitter.validate_split(split_data)
    print(f"\n검증 결과: {validation_result['overall_quality']}")
    if validation_result['issues']:
        print(f"이슈: {validation_result['issues']}")
    
    # 임시 디렉토리에 저장
    test_output_dir = project_root / "data" / "splits" / "test"
    saved_files = splitter.save_splits(split_data, test_output_dir)
    
    print(f"\n저장된 파일:")
    for split_name, file_path in saved_files.items():
        print(f"  - {split_name}: {file_path}")
    
    # 시각화 (선택사항)
    try:
        viz_path = test_output_dir / "test_split_distribution.png"
        splitter.visualize_split_distribution(split_data, output_path=viz_path)
        print(f"시각화 저장: {viz_path}")
    except Exception as e:
        print(f"시각화 실패 (matplotlib 환경 이슈일 수 있음): {e}")
    
    # 통계 정보
    stats = splitter.get_stats()
    print(f"\nSplitter 통계: {stats}") 