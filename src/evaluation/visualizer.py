"""
🍉 결과 시각화 모듈
수박 당도 예측 모델들의 포괄적인 시각화 및 결과 분석

Author: Watermelon ML Team
Date: 2025-01-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import platform
from matplotlib import font_manager, rc

if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
else:  # Linux 등
    rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultVisualizer:
    """
    수박 당도 예측 모델 결과의 포괄적인 시각화 클래스
    
    기능:
    - 성능 비교 차트
    - 예측 정확도 시각화
    - 잔차 분석 플롯
    - 특징 중요도 시각화
    - 인터랙티브 대시보드
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (12, 8)):
        """
        시각화 도구 초기화
        
        Args:
            style: matplotlib 스타일
            figsize: 기본 그림 크기
        """
        self.style = style
        self.figsize = figsize
        self.colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        # matplotlib 설정
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
            logger.warning(f"스타일 '{style}'을 찾을 수 없어 기본 스타일을 사용합니다.")
        
        self.logger = logging.getLogger(f"{__name__}.ResultVisualizer")
        
    def plot_performance_comparison(self, evaluations: Dict[str, Dict[str, Any]], 
                                  metrics: List[str] = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        모델별 성능 비교 차트
        
        Args:
            evaluations: 모델별 평가 결과
            metrics: 비교할 메트릭 리스트
            save_path: 저장 경로
            
        Returns:
            matplotlib Figure 객체
        """
        if metrics is None:
            metrics = ['mae', 'mse', 'r2', 'mape']
        
        n_metrics = len(metrics)
        n_models = len(evaluations)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🍉 수박 당도 예측 모델 성능 비교', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # 최대 4개 메트릭
            if i >= len(axes):
                break
                
            models = list(evaluations.keys())
            scores = [evaluations[model].get(metric, 0) for model in models]
            
            # 바 차트
            bars = axes[i].bar(models, scores, color=self.colors[:n_models], alpha=0.7)
            axes[i].set_title(f'{metric.upper()} 비교', fontweight='bold')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{score:.3f}', ha='center', va='bottom')
            
            # 최고 성능 강조
            if metric in ['mae', 'mse', 'mape']:  # 낮을수록 좋음
                best_idx = np.argmin(scores)
            else:  # r2: 높을수록 좋음
                best_idx = np.argmax(scores)
            
            bars[best_idx].set_color('#FF6B6B')
            bars[best_idx].set_alpha(1.0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"성능 비교 차트 저장: {save_path}")
        
        return fig
    
    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        예측값 vs 실제값 산점도
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            model_name: 모델 이름
            save_path: 저장 경로
            
        Returns:
            matplotlib Figure 객체
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'🍉 {model_name} 예측 정확도 분석', fontsize=14, fontweight='bold')
        
        # 산점도
        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors[0], s=50)
        
        # 완벽한 예측선 (y=x)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8, label='완벽한 예측')
        
        # ±1 Brix 허용 구간
        ax1.fill_between([min_val, max_val], [min_val-1, max_val-1], [min_val+1, max_val+1], 
                        alpha=0.2, color='green', label='±1 Brix 허용구간')
        
        ax1.set_xlabel('실제 당도 (Brix)')
        ax1.set_ylabel('예측 당도 (Brix)')
        ax1.set_title('예측값 vs 실제값')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 오차 히스토그램
        errors = y_pred - y_true
        ax2.hist(errors, bins=20, alpha=0.7, color=self.colors[1], edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='완벽한 예측')
        ax2.axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                   label=f'평균 오차: {np.mean(errors):.3f}')
        
        ax2.set_xlabel('예측 오차 (Brix)')
        ax2.set_ylabel('빈도')
        ax2.set_title('예측 오차 분포')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"예측 산점도 저장: {save_path}")
        
        return fig
    
    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Model",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        잔차 분석 플롯
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            model_name: 모델 이름
            save_path: 저장 경로
            
        Returns:
            matplotlib Figure 객체
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'🍉 {model_name} 잔차 분석', fontsize=14, fontweight='bold')
        
        # 1. 잔차 vs 예측값
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, color=self.colors[2])
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('예측값')
        axes[0, 0].set_ylabel('잔차')
        axes[0, 0].set_title('잔차 vs 예측값')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 히스토그램
        axes[0, 1].hist(residuals, bins=20, alpha=0.7, color=self.colors[3], edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('잔차')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].set_title('잔차 분포')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q 플롯 (정규성 검정)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q 플롯 (정규성 검정)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 잔차 vs 실제값
        axes[1, 1].scatter(y_true, residuals, alpha=0.6, color=self.colors[4])
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('실제값')
        axes[1, 1].set_ylabel('잔차')
        axes[1, 1].set_title('잔차 vs 실제값')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"잔차 분석 플롯 저장: {save_path}")
        
        return fig
    
    def plot_feature_importance(self, importance_dict: Dict[str, Optional[List[Tuple[str, float]]]],
                               top_k: int = 15,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        특징 중요도 시각화
        
        Args:
            importance_dict: 모델별 특징 중요도 딕셔너리
            top_k: 상위 k개 특징만 표시
            save_path: 저장 경로
            
        Returns:
            matplotlib Figure 객체
        """
        # None이 아닌 특징 중요도만 필터링
        valid_models = {k: v for k, v in importance_dict.items() if v is not None}
        
        if not valid_models:
            self.logger.warning("표시할 특징 중요도가 없습니다.")
            return plt.figure()
        
        n_models = len(valid_models)
        
        if n_models == 1:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            axes = [ax]
        else:
            n_cols = min(n_models, 2)
            n_rows = (n_models + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
            if n_models == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_models > 1 else [axes]
        
        fig.suptitle('🍉 모델별 특징 중요도 분석', fontsize=14, fontweight='bold')
        
        for i, (model_name, importance_list) in enumerate(valid_models.items()):
            if i >= len(axes):
                break
                
            # 상위 k개 특징 선택
            top_features = importance_list[:top_k]
            features, scores = zip(*top_features)
            
            # 수평 바 차트
            y_pos = np.arange(len(features))
            bars = axes[i].barh(y_pos, scores, color=self.colors[i % len(self.colors)], alpha=0.7)
            
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(features)
            axes[i].invert_yaxis()  # 상위 특징이 위에 오도록
            axes[i].set_xlabel('중요도')
            axes[i].set_title(f'{model_name} 특징 중요도')
            axes[i].grid(True, alpha=0.3)
            
            # 값 표시
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                axes[i].text(width, bar.get_y() + bar.get_height()/2., 
                           f'{score:.3f}', ha='left', va='center')
        
        # 빈 서브플롯 숨기기
        for i in range(len(valid_models), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"특징 중요도 플롯 저장: {save_path}")
        
        return fig
    
    def plot_training_history(self, models_history: Dict[str, Dict[str, Any]],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        모델별 훈련 기록 시각화
        
        Args:
            models_history: 모델별 훈련 기록
            save_path: 저장 경로
            
        Returns:
            matplotlib Figure 객체
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🍉 모델 훈련 기록 분석', fontsize=14, fontweight='bold')
        
        models = list(models_history.keys())
        
        # 1. 훈련 시간 비교
        training_times = [models_history[model].get('training_time', 0) for model in models]
        bars1 = axes[0, 0].bar(models, training_times, color=self.colors[:len(models)], alpha=0.7)
        axes[0, 0].set_title('모델별 훈련 시간')
        axes[0, 0].set_ylabel('시간 (초)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars1, training_times):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                          f'{time_val:.1f}s', ha='center', va='bottom')
        
        # 2. 훈련 vs 검증 MAE
        train_maes = [models_history[model].get('train_mae', 0) for model in models]
        val_maes = [models_history[model].get('val_mae', 0) for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, train_maes, width, label='훈련 MAE', alpha=0.7, color=self.colors[0])
        axes[0, 1].bar(x + width/2, val_maes, width, label='검증 MAE', alpha=0.7, color=self.colors[1])
        axes[0, 1].set_title('훈련 vs 검증 성능 (MAE)')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models, rotation=45)
        axes[0, 1].legend()
        
        # 3. R² 점수 비교
        train_r2s = [models_history[model].get('train_r2', 0) for model in models]
        val_r2s = [models_history[model].get('val_r2', 0) for model in models]
        
        axes[1, 0].bar(x - width/2, train_r2s, width, label='훈련 R²', alpha=0.7, color=self.colors[2])
        axes[1, 0].bar(x + width/2, val_r2s, width, label='검증 R²', alpha=0.7, color=self.colors[3])
        axes[1, 0].set_title('훈련 vs 검증 성능 (R²)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models, rotation=45)
        axes[1, 0].legend()
        
        # 4. 데이터셋 정보
        n_samples = [models_history[model].get('n_samples', 0) for model in models]
        n_features = [models_history[model].get('n_features', 0) for model in models]
        
        ax_twin = axes[1, 1].twinx()
        bars2 = axes[1, 1].bar(x - width/2, n_samples, width, label='샘플 수', alpha=0.7, color=self.colors[4])
        bars3 = ax_twin.bar(x + width/2, n_features, width, label='특징 수', alpha=0.7, color=self.colors[5])
        
        axes[1, 1].set_title('데이터셋 정보')
        axes[1, 1].set_ylabel('샘플 수')
        ax_twin.set_ylabel('특징 수')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(models, rotation=45)
        
        # 범례 결합
        lines1, labels1 = axes[1, 1].get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"훈련 기록 플롯 저장: {save_path}")
        
        return fig
    
    def create_interactive_dashboard(self, evaluations: Dict[str, Dict[str, Any]],
                                   y_true_dict: Dict[str, np.ndarray],
                                   y_pred_dict: Dict[str, np.ndarray],
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        인터랙티브 대시보드 생성 (Plotly)
        
        Args:
            evaluations: 모델별 평가 결과
            y_true_dict: 모델별 실제값
            y_pred_dict: 모델별 예측값
            save_path: 저장 경로
            
        Returns:
            plotly Figure 객체
        """
        models = list(evaluations.keys())
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('성능 비교', '예측 정확도', '오차 분포', '모델 순위'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 성능 비교 (MAE, R²)
        mae_scores = [evaluations[model]['mae'] for model in models]
        r2_scores = [evaluations[model]['r2'] for model in models]
        
        fig.add_trace(
            go.Bar(name='MAE', x=models, y=mae_scores, marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. 예측 정확도 (첫 번째 모델)
        if models:
            first_model = models[0]
            y_true = y_true_dict.get(first_model, np.array([]))
            y_pred = y_pred_dict.get(first_model, np.array([]))
            
            fig.add_trace(
                go.Scatter(
                    x=y_true, y=y_pred,
                    mode='markers',
                    name=f'{first_model} 예측',
                    marker=dict(color='green', opacity=0.6)
                ),
                row=1, col=2
            )
            
            # 완벽한 예측선
            if len(y_true) > 0:
                min_val = min(np.min(y_true), np.min(y_pred))
                max_val = max(np.max(y_true), np.max(y_pred))
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines',
                        name='완벽한 예측',
                        line=dict(color='red', dash='dash')
                    ),
                    row=1, col=2
                )
        
        # 3. 오차 분포
        if models:
            errors = y_pred_dict.get(first_model, np.array([])) - y_true_dict.get(first_model, np.array([]))
            if len(errors) > 0:
                fig.add_trace(
                    go.Histogram(x=errors, name='오차 분포', marker_color='orange'),
                    row=2, col=1
                )
        
        # 4. 모델 순위 (R² 기준)
        sorted_models = sorted(models, key=lambda x: evaluations[x]['r2'], reverse=True)
        sorted_r2 = [evaluations[model]['r2'] for model in sorted_models]
        
        fig.add_trace(
            go.Bar(
                name='R² 순위', 
                x=sorted_models, 
                y=sorted_r2,
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title_text="🍉 수박 당도 예측 모델 인터랙티브 대시보드",
            title_x=0.5,
            showlegend=True,
            height=700
        )
        
        # 축 라벨 업데이트
        fig.update_xaxes(title_text="모델", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=1)
        fig.update_xaxes(title_text="실제 당도 (Brix)", row=1, col=2)
        fig.update_yaxes(title_text="예측 당도 (Brix)", row=1, col=2)
        fig.update_xaxes(title_text="예측 오차", row=2, col=1)
        fig.update_yaxes(title_text="빈도", row=2, col=1)
        fig.update_xaxes(title_text="모델", row=2, col=2)
        fig.update_yaxes(title_text="R²", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"인터랙티브 대시보드 저장: {save_path}")
        
        return fig
    
    def create_comprehensive_report(self, evaluations: Dict[str, Dict[str, Any]],
                                  y_true_dict: Dict[str, np.ndarray],
                                  y_pred_dict: Dict[str, np.ndarray],
                                  importance_dict: Dict[str, Optional[List[Tuple[str, float]]]],
                                  models_history: Dict[str, Dict[str, Any]],
                                  save_dir: Optional[str] = None) -> Dict[str, str]:
        """
        종합 리포트 생성 (모든 시각화 포함)
        
        Args:
            evaluations: 모델별 평가 결과
            y_true_dict: 모델별 실제값
            y_pred_dict: 모델별 예측값
            importance_dict: 모델별 특징 중요도
            models_history: 모델별 훈련 기록
            save_dir: 저장 디렉토리
            
        Returns:
            생성된 파일 경로 딕셔너리
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        try:
            # 1. 성능 비교 차트
            if save_dir:
                perf_path = save_dir / "performance_comparison.png"
                self.plot_performance_comparison(evaluations, save_path=str(perf_path))
                file_paths['performance_comparison'] = str(perf_path)
            
            # 2. 각 모델별 예측 산점도
            for model_name in evaluations.keys():
                if model_name in y_true_dict and model_name in y_pred_dict:
                    if save_dir:
                        scatter_path = save_dir / f"prediction_scatter_{model_name}.png"
                        self.plot_prediction_scatter(
                            y_true_dict[model_name], 
                            y_pred_dict[model_name],
                            model_name,
                            save_path=str(scatter_path)
                        )
                        file_paths[f'scatter_{model_name}'] = str(scatter_path)
            
            # 3. 각 모델별 잔차 분석
            for model_name in evaluations.keys():
                if model_name in y_true_dict and model_name in y_pred_dict:
                    if save_dir:
                        residual_path = save_dir / f"residual_analysis_{model_name}.png"
                        self.plot_residual_analysis(
                            y_true_dict[model_name],
                            y_pred_dict[model_name],
                            model_name,
                            save_path=str(residual_path)
                        )
                        file_paths[f'residual_{model_name}'] = str(residual_path)
            
            # 4. 특징 중요도
            if save_dir:
                importance_path = save_dir / "feature_importance.png"
                self.plot_feature_importance(importance_dict, save_path=str(importance_path))
                file_paths['feature_importance'] = str(importance_path)
            
            # 5. 훈련 기록
            if save_dir:
                history_path = save_dir / "training_history.png"
                self.plot_training_history(models_history, save_path=str(history_path))
                file_paths['training_history'] = str(history_path)
            
            # 6. 인터랙티브 대시보드
            if save_dir:
                dashboard_path = save_dir / "interactive_dashboard.html"
                self.create_interactive_dashboard(
                    evaluations, y_true_dict, y_pred_dict,
                    save_path=str(dashboard_path)
                )
                file_paths['interactive_dashboard'] = str(dashboard_path)
            
            self.logger.info(f"종합 리포트 생성 완료: {len(file_paths)}개 파일")
            
        except Exception as e:
            self.logger.error(f"종합 리포트 생성 실패: {e}")
        
        return file_paths


# 테스트 함수
def test_visualizer():
    """시각화 모듈 기본 기능 테스트"""
    print("🍉 ResultVisualizer 테스트 시작")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    
    # 가짜 평가 결과
    evaluations = {
        'GBT': {'mae': 0.8, 'mse': 1.2, 'r2': 0.85, 'mape': 0.08, 'performance_grade': 'GOOD'},
        'SVM': {'mae': 1.1, 'mse': 1.8, 'r2': 0.75, 'mape': 0.12, 'performance_grade': 'FAIR'},
        'RF': {'mae': 0.9, 'mse': 1.4, 'r2': 0.82, 'mape': 0.09, 'performance_grade': 'GOOD'}
    }
    
    # 가짜 예측 데이터
    y_true = np.random.randn(100) * 2 + 10
    y_pred_dict = {
        'GBT': y_true + np.random.randn(100) * 0.8,
        'SVM': y_true + np.random.randn(100) * 1.1,
        'RF': y_true + np.random.randn(100) * 0.9
    }
    y_true_dict = {model: y_true for model in evaluations.keys()}
    
    # 가짜 특징 중요도
    feature_names = [f'특징_{i}' for i in range(10)]
    importance_dict = {
        'GBT': [(name, np.random.random()) for name in feature_names],
        'SVM': None,  # SVM은 특징 중요도 없음
        'RF': [(name, np.random.random()) for name in feature_names]
    }
    
    # 가짜 훈련 기록
    models_history = {
        'GBT': {'training_time': 25.3, 'train_mae': 0.7, 'val_mae': 0.8, 'train_r2': 0.9, 'val_r2': 0.85, 'n_samples': 100, 'n_features': 51},
        'SVM': {'training_time': 45.1, 'train_mae': 0.9, 'val_mae': 1.1, 'train_r2': 0.8, 'val_r2': 0.75, 'n_samples': 100, 'n_features': 51},
        'RF': {'training_time': 18.7, 'train_mae': 0.6, 'val_mae': 0.9, 'train_r2': 0.95, 'val_r2': 0.82, 'n_samples': 100, 'n_features': 51}
    }
    
    try:
        # 시각화 도구 생성
        visualizer = ResultVisualizer()
        print("✅ ResultVisualizer 초기화 성공")
        
        # 성능 비교 차트
        fig1 = visualizer.plot_performance_comparison(evaluations)
        print("✅ 성능 비교 차트 생성 완료")
        plt.close(fig1)
        
        # 예측 산점도
        fig2 = visualizer.plot_prediction_scatter(y_true, y_pred_dict['GBT'], "GBT")
        print("✅ 예측 산점도 생성 완료")
        plt.close(fig2)
        
        # 잔차 분석
        fig3 = visualizer.plot_residual_analysis(y_true, y_pred_dict['GBT'], "GBT")
        print("✅ 잔차 분석 플롯 생성 완료")
        plt.close(fig3)
        
        # 특징 중요도
        fig4 = visualizer.plot_feature_importance(importance_dict)
        print("✅ 특징 중요도 시각화 완료")
        plt.close(fig4)
        
        # 훈련 기록
        fig5 = visualizer.plot_training_history(models_history)
        print("✅ 훈련 기록 시각화 완료")
        plt.close(fig5)
        
        # 인터랙티브 대시보드
        plotly_fig = visualizer.create_interactive_dashboard(evaluations, y_true_dict, y_pred_dict)
        print("✅ 인터랙티브 대시보드 생성 완료")
        
        print("\n🎉 ResultVisualizer 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 시각화 모듈 테스트 실패: {e}")
        raise


if __name__ == "__main__":
    test_visualizer() 