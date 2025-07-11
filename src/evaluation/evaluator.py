"""
🍉 수박 당도 예측 - 성능 평가 모듈

전통적인 ML 모델들의 포괄적 성능 평가
- 회귀 메트릭 계산 및 분석
- 통계적 유의성 검정
- 모델 간 비교 분석
- 성능 보고서 생성
- 잔차 분석 및 진단
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    max_error, explained_variance_score
)
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    ML 모델 성능 평가를 위한 종합 클래스
    """
    
    def __init__(self, target_range: Tuple[float, float] = (8.0, 13.0)):
        """
        평가자 초기화
        
        Args:
            target_range: 당도 예측 범위 (min, max)
        """
        self.target_range = target_range
        self.evaluation_results = {}
        
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   model_name: str = "model") -> Dict[str, float]:
        """
        회귀 메트릭 계산
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            model_name: 모델 이름
            
        Returns:
            메트릭 딕셔너리
        """
        try:
            metrics = {
                # 기본 회귀 메트릭
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'r2': float(r2_score(y_true, y_pred)),
                'mape': float(mean_absolute_percentage_error(y_true, y_pred)),
                'median_ae': float(median_absolute_error(y_true, y_pred)),
                'max_error': float(max_error(y_true, y_pred)),
                'explained_variance': float(explained_variance_score(y_true, y_pred)),
                
                # 추가 메트릭
                'relative_mae': float(mean_absolute_error(y_true, y_pred) / np.mean(y_true)),
                'std_residuals': float(np.std(y_true - y_pred)),
                'mean_residuals': float(np.mean(y_true - y_pred)),
                
                # 정확도 메트릭 (±0.5, ±1.0 Brix 내)
                'accuracy_0_5': self._accuracy_within_threshold(y_true, y_pred, 0.5),
                'accuracy_1_0': self._accuracy_within_threshold(y_true, y_pred, 1.0),
                
                # 상관계수
                'pearson_corr': float(stats.pearsonr(y_true, y_pred)[0]),
                'spearman_corr': float(stats.spearmanr(y_true, y_pred)[0]),
                
                # 범위 적합성
                'predictions_in_range': self._predictions_in_valid_range(y_pred),
                'range_coverage': self._range_coverage(y_pred)
            }
            
            # 잔차 분석
            residuals = y_true - y_pred
            metrics.update(self._analyze_residuals(residuals))
            
            logger.info(f"Calculated {len(metrics)} metrics for {model_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {model_name}: {e}")
            return {}
    
    def _accuracy_within_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  threshold: float) -> float:
        """임계값 내 정확도 계산"""
        return float(np.mean(np.abs(y_true - y_pred) <= threshold))
    
    def _predictions_in_valid_range(self, y_pred: np.ndarray) -> float:
        """유효 범위 내 예측 비율"""
        min_val, max_val = self.target_range
        in_range = np.logical_and(y_pred >= min_val, y_pred <= max_val)
        return float(np.mean(in_range))
    
    def _range_coverage(self, y_pred: np.ndarray) -> float:
        """예측 범위 커버리지"""
        pred_range = np.max(y_pred) - np.min(y_pred)
        target_range = self.target_range[1] - self.target_range[0]
        return float(min(pred_range / target_range, 1.0))
    
    def _analyze_residuals(self, residuals: np.ndarray) -> Dict[str, float]:
        """잔차 분석"""
        residuals_analysis = {
            'residuals_skewness': float(stats.skew(residuals)),
            'residuals_kurtosis': float(stats.kurtosis(residuals)),
            'residuals_normality_pvalue': float(stats.normaltest(residuals)[1]),
            'residuals_q25': float(np.percentile(residuals, 25)),
            'residuals_q75': float(np.percentile(residuals, 75)),
            'residuals_iqr': float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
        }
        
        return residuals_analysis
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str, dataset_name: str = "test") -> Dict[str, Any]:
        """
        단일 모델 성능 평가
        
        Args:
            y_true: 실제 값
            y_pred: 예측 값
            model_name: 모델 이름
            dataset_name: 데이터셋 이름
            
        Returns:
            평가 결과 딕셔너리
        """
        metrics = self.calculate_regression_metrics(y_true, y_pred, model_name)
        
        # 성능 등급 결정
        performance_grade = self._determine_performance_grade(metrics)
        
        # 강점/약점 분석
        strengths, weaknesses = self._analyze_strengths_weaknesses(metrics)
        
        result = {
            'model_name': model_name,
            'dataset': dataset_name,
            'n_samples': len(y_true),
            'metrics': metrics,
            'performance_grade': performance_grade,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'summary': self._generate_summary(model_name, metrics, performance_grade)
        }
        
        # 결과 저장
        if model_name not in self.evaluation_results:
            self.evaluation_results[model_name] = {}
        self.evaluation_results[model_name][dataset_name] = result
        
        return result
    
    def _determine_performance_grade(self, metrics: Dict[str, float]) -> str:
        """성능 등급 결정"""
        mae = metrics.get('mae', float('inf'))
        r2 = metrics.get('r2', 0)
        accuracy_1_0 = metrics.get('accuracy_1_0', 0)
        
        # 등급 기준
        if mae < 0.5 and r2 > 0.9 and accuracy_1_0 > 0.95:
            return "EXCELLENT"
        elif mae < 0.8 and r2 > 0.8 and accuracy_1_0 > 0.9:
            return "VERY_GOOD"
        elif mae < 1.0 and r2 > 0.7 and accuracy_1_0 > 0.85:
            return "GOOD"
        elif mae < 1.5 and r2 > 0.5 and accuracy_1_0 > 0.75:
            return "FAIR"
        else:
            return "POOR"
    
    def _analyze_strengths_weaknesses(self, metrics: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """강점/약점 분석"""
        strengths = []
        weaknesses = []
        
        # MAE 분석
        mae = metrics.get('mae', float('inf'))
        if mae < 0.5:
            strengths.append("Very low prediction error (MAE < 0.5)")
        elif mae < 1.0:
            strengths.append("Low prediction error (MAE < 1.0)")
        elif mae > 2.0:
            weaknesses.append("High prediction error (MAE > 2.0)")
        
        # R² 분석
        r2 = metrics.get('r2', 0)
        if r2 > 0.9:
            strengths.append("Excellent variance explanation (R² > 0.9)")
        elif r2 > 0.7:
            strengths.append("Good variance explanation (R² > 0.7)")
        elif r2 < 0.5:
            weaknesses.append("Poor variance explanation (R² < 0.5)")
        
        # 정확도 분석
        acc_1_0 = metrics.get('accuracy_1_0', 0)
        if acc_1_0 > 0.9:
            strengths.append("High accuracy within ±1.0 Brix")
        elif acc_1_0 < 0.7:
            weaknesses.append("Low accuracy within ±1.0 Brix")
        
        # 잔차 분석
        residuals_normality = metrics.get('residuals_normality_pvalue', 0)
        if residuals_normality > 0.05:
            strengths.append("Normally distributed residuals")
        else:
            weaknesses.append("Non-normal residual distribution")
        
        # 예측 범위 분석
        in_range = metrics.get('predictions_in_range', 0)
        if in_range > 0.95:
            strengths.append("Predictions within valid range")
        elif in_range < 0.9:
            weaknesses.append("Some predictions outside valid range")
        
        return strengths, weaknesses
    
    def _generate_summary(self, model_name: str, metrics: Dict[str, float], 
                         grade: str) -> str:
        """성능 요약 생성"""
        mae = metrics.get('mae', 0)
        r2 = metrics.get('r2', 0)
        acc_1_0 = metrics.get('accuracy_1_0', 0)
        
        summary = (
            f"{model_name} achieved {grade} performance with "
            f"MAE of {mae:.3f} Brix, R² of {r2:.3f}, and "
            f"{acc_1_0:.1%} accuracy within ±1.0 Brix."
        )
        
        return summary
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]], 
                      dataset_name: str = "test") -> pd.DataFrame:
        """
        여러 모델 성능 비교
        
        Args:
            model_results: 모델별 평가 결과
            dataset_name: 비교할 데이터셋
            
        Returns:
            비교 결과 DataFrame
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            if dataset_name in results:
                result = results[dataset_name]
                metrics = result['metrics']
                
                comparison_data.append({
                    'model': model_name,
                    'mae': metrics.get('mae', np.nan),
                    'rmse': metrics.get('rmse', np.nan),
                    'r2': metrics.get('r2', np.nan),
                    'mape': metrics.get('mape', np.nan),
                    'accuracy_0_5': metrics.get('accuracy_0_5', np.nan),
                    'accuracy_1_0': metrics.get('accuracy_1_0', np.nan),
                    'pearson_corr': metrics.get('pearson_corr', np.nan),
                    'performance_grade': result.get('performance_grade', 'UNKNOWN'),
                    'n_samples': result.get('n_samples', 0)
                })
        
        if not comparison_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(comparison_data)
        
        # 순위 추가
        df['mae_rank'] = df['mae'].rank()
        df['r2_rank'] = df['r2'].rank(ascending=False)
        df['overall_rank'] = (df['mae_rank'] + df['r2_rank']) / 2
        
        # 정렬
        df = df.sort_values('overall_rank').reset_index(drop=True)
        
        return df
    
    def get_performance_summary(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """전체 성능 요약"""
        summary = {
            'total_models': len(model_results),
            'datasets_evaluated': set(),
            'best_performers': {},
            'performance_distribution': {},
            'key_insights': []
        }
        
        # 데이터셋별 최고 성능 모델 찾기
        all_metrics = {}
        for model_name, results in model_results.items():
            for dataset_name, result in results.items():
                summary['datasets_evaluated'].add(dataset_name)
                
                if dataset_name not in all_metrics:
                    all_metrics[dataset_name] = []
                
                metrics = result['metrics']
                all_metrics[dataset_name].append({
                    'model': model_name,
                    'mae': metrics.get('mae', float('inf')),
                    'r2': metrics.get('r2', 0),
                    'grade': result.get('performance_grade', 'UNKNOWN')
                })
        
        # 최고 성능 모델 결정
        for dataset, metrics_list in all_metrics.items():
            best_mae = min(metrics_list, key=lambda x: x['mae'])
            best_r2 = max(metrics_list, key=lambda x: x['r2'])
            
            summary['best_performers'][dataset] = {
                'best_mae': best_mae['model'],
                'best_r2': best_r2['model'],
                'mae_value': best_mae['mae'],
                'r2_value': best_r2['r2']
            }
        
        # 성능 분포
        grades = [result['performance_grade'] 
                 for results in model_results.values() 
                 for result in results.values()]
        
        grade_counts = pd.Series(grades).value_counts().to_dict()
        summary['performance_distribution'] = grade_counts
        
        # 주요 인사이트
        summary['key_insights'] = self._generate_key_insights(model_results, summary)
        
        return summary
    
    def _generate_key_insights(self, model_results: Dict[str, Dict[str, Any]], 
                              summary: Dict[str, Any]) -> List[str]:
        """주요 인사이트 생성"""
        insights = []
        
        # 최고 성능 분석
        test_best = summary['best_performers'].get('test', {})
        if test_best:
            mae_value = test_best.get('mae_value', 0)
            if mae_value < 0.5:
                insights.append("Achieved excellent prediction accuracy (MAE < 0.5 Brix)")
            elif mae_value < 1.0:
                insights.append("Achieved target prediction accuracy (MAE < 1.0 Brix)")
            else:
                insights.append("Did not achieve target prediction accuracy (MAE ≥ 1.0 Brix)")
        
        # 모델 일관성 분석
        model_grades = {}
        for model_name, results in model_results.items():
            grades = [result['performance_grade'] for result in results.values()]
            model_grades[model_name] = grades
        
        consistent_models = [model for model, grades in model_grades.items() 
                           if len(set(grades)) == 1]
        
        if consistent_models:
            insights.append(f"Models with consistent performance: {', '.join(consistent_models)}")
        
        # 성능 분포 분석
        grade_dist = summary['performance_distribution']
        excellent_count = grade_dist.get('EXCELLENT', 0)
        poor_count = grade_dist.get('POOR', 0)
        
        if excellent_count > 0:
            insights.append(f"{excellent_count} model(s) achieved excellent performance")
        if poor_count > 0:
            insights.append(f"{poor_count} model(s) showed poor performance")
        
        return insights


class ComparisonAnalyzer:
    """
    모델 간 통계적 비교 분석
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        비교 분석자 초기화
        
        Args:
            alpha: 유의수준
        """
        self.alpha = alpha
    
    def statistical_comparison(self, results1: np.ndarray, results2: np.ndarray, 
                             model1_name: str, model2_name: str) -> Dict[str, Any]:
        """
        두 모델의 통계적 비교
        
        Args:
            results1: 모델1 예측 결과 (잔차 또는 에러)
            results2: 모델2 예측 결과 (잔차 또는 에러)
            model1_name: 모델1 이름
            model2_name: 모델2 이름
            
        Returns:
            통계적 비교 결과
        """
        comparison = {
            'models': f"{model1_name} vs {model2_name}",
            'n_samples': len(results1)
        }
        
        # 평균 비교 (t-test)
        t_stat, t_pvalue = stats.ttest_rel(results1, results2)
        comparison['paired_ttest'] = {
            't_statistic': float(t_stat),
            'p_value': float(t_pvalue),
            'significant': t_pvalue < self.alpha,
            'interpretation': self._interpret_ttest(t_pvalue, model1_name, model2_name)
        }
        
        # 분포 비교 (Wilcoxon signed-rank test)
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(results1, results2)
        comparison['wilcoxon_test'] = {
            'statistic': float(wilcoxon_stat),
            'p_value': float(wilcoxon_pvalue),
            'significant': wilcoxon_pvalue < self.alpha,
            'interpretation': self._interpret_wilcoxon(wilcoxon_pvalue, model1_name, model2_name)
        }
        
        # 효과 크기 (Cohen's d)
        cohens_d = self._calculate_cohens_d(results1, results2)
        comparison['effect_size'] = {
            'cohens_d': cohens_d,
            'magnitude': self._interpret_effect_size(cohens_d)
        }
        
        # 실용적 차이
        mean_diff = np.mean(results1) - np.mean(results2)
        comparison['practical_difference'] = {
            'mean_difference': float(mean_diff),
            'relative_difference': float(mean_diff / np.mean(np.abs(results1)) * 100),
            'practically_significant': abs(mean_diff) > 0.1  # 0.1 Brix 이상 차이
        }
        
        return comparison
    
    def _interpret_ttest(self, p_value: float, model1: str, model2: str) -> str:
        """t-test 결과 해석"""
        if p_value < self.alpha:
            return f"Significant difference between {model1} and {model2} (p < {self.alpha})"
        else:
            return f"No significant difference between {model1} and {model2} (p ≥ {self.alpha})"
    
    def _interpret_wilcoxon(self, p_value: float, model1: str, model2: str) -> str:
        """Wilcoxon test 결과 해석"""
        if p_value < self.alpha:
            return f"Significant difference in distributions between {model1} and {model2}"
        else:
            return f"No significant difference in distributions between {model1} and {model2}"
    
    def _calculate_cohens_d(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Cohen's d 효과 크기 계산"""
        pooled_std = np.sqrt(((len(x1) - 1) * np.var(x1, ddof=1) + 
                             (len(x2) - 1) * np.var(x2, ddof=1)) / 
                            (len(x1) + len(x2) - 2))
        return float((np.mean(x1) - np.mean(x2)) / pooled_std)
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """효과 크기 해석"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_model_comparison(self, model_errors: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        여러 모델 간 종합 비교
        
        Args:
            model_errors: 모델별 에러 배열
            
        Returns:
            종합 비교 결과
        """
        models = list(model_errors.keys())
        n_models = len(models)
        
        if n_models < 2:
            return {"error": "At least 2 models required for comparison"}
        
        # 일원 분산분석 (ANOVA)
        error_arrays = [model_errors[model] for model in models]
        f_stat, anova_pvalue = stats.f_oneway(*error_arrays)
        
        comparison = {
            'models': models,
            'n_models': n_models,
            'anova': {
                'f_statistic': float(f_stat),
                'p_value': float(anova_pvalue),
                'significant': anova_pvalue < self.alpha,
                'interpretation': self._interpret_anova(anova_pvalue)
            }
        }
        
        # 사후 분석 (pairwise comparisons)
        if anova_pvalue < self.alpha:
            pairwise_results = []
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    pair_result = self.statistical_comparison(
                        model_errors[models[i]], 
                        model_errors[models[j]],
                        models[i], 
                        models[j]
                    )
                    pairwise_results.append(pair_result)
            
            comparison['pairwise_comparisons'] = pairwise_results
        
        return comparison
    
    def _interpret_anova(self, p_value: float) -> str:
        """ANOVA 결과 해석"""
        if p_value < self.alpha:
            return f"Significant differences exist among models (p < {self.alpha})"
        else:
            return f"No significant differences among models (p ≥ {self.alpha})"
    
    def performance_degradation_analysis(self, train_metrics: Dict[str, float], 
                                       val_metrics: Dict[str, float],
                                       test_metrics: Dict[str, float],
                                       model_name: str) -> Dict[str, Any]:
        """
        성능 저하 분석 (과적합 탐지)
        
        Args:
            train_metrics: 훈련 성능
            val_metrics: 검증 성능  
            test_metrics: 테스트 성능
            model_name: 모델 이름
            
        Returns:
            성능 저하 분석 결과
        """
        analysis = {
            'model_name': model_name,
            'overfitting_indicators': {},
            'generalization_gap': {},
            'recommendations': []
        }
        
        # MAE 기반 과적합 분석
        train_mae = train_metrics.get('mae', 0)
        val_mae = val_metrics.get('mae', 0)
        test_mae = test_metrics.get('mae', 0)
        
        train_val_gap = val_mae - train_mae
        val_test_gap = test_mae - val_mae
        
        analysis['generalization_gap'] = {
            'train_val_gap': float(train_val_gap),
            'val_test_gap': float(val_test_gap),
            'train_val_ratio': float(val_mae / train_mae) if train_mae > 0 else np.inf,
            'val_test_ratio': float(test_mae / val_mae) if val_mae > 0 else np.inf
        }
        
        # 과적합 지표
        analysis['overfitting_indicators'] = {
            'significant_train_val_gap': train_val_gap > 0.2,  # 0.2 Brix 이상
            'large_performance_ratio': (val_mae / train_mae) > 1.5 if train_mae > 0 else False,
            'degraded_test_performance': test_mae > val_mae * 1.2
        }
        
        # 권장사항
        if any(analysis['overfitting_indicators'].values()):
            analysis['recommendations'].extend([
                "Consider regularization techniques",
                "Increase training data size",
                "Reduce model complexity",
                "Apply cross-validation"
            ])
        
        if train_val_gap > 0.5:
            analysis['recommendations'].append("Strong overfitting detected - review model architecture")
        
        if val_test_gap > 0.3:
            analysis['recommendations'].append("Poor generalization - validate data distribution")
        
        return analysis


def test_evaluator():
    """평가자 클래스 테스트"""
    logger.info("Testing ModelEvaluator...")
    
    # 가상 데이터 생성
    np.random.seed(42)
    n_samples = 100
    
    # 실제 값
    y_true = np.random.uniform(8.0, 13.0, n_samples)
    
    # 여러 모델의 예측 값 생성
    models_predictions = {
        'GBT': y_true + np.random.normal(0, 0.3, n_samples),  # 좋은 성능
        'SVM': y_true + np.random.normal(0, 0.8, n_samples),  # 보통 성능
        'RF': y_true + np.random.normal(0, 0.5, n_samples)    # 중간 성능
    }
    
    # 평가자 생성
    evaluator = ModelEvaluator()
    
    # 각 모델 평가
    model_results = {}
    for model_name, y_pred in models_predictions.items():
        result = evaluator.evaluate_model_performance(
            y_true, y_pred, model_name, "test"
        )
        model_results[model_name] = {'test': result}
        
        logger.info(f"{model_name} - Grade: {result['performance_grade']}, "
                   f"MAE: {result['metrics']['mae']:.3f}")
    
    # 모델 비교
    comparison_df = evaluator.compare_models(model_results, "test")
    logger.info(f"Model comparison completed. Best model: {comparison_df.iloc[0]['model']}")
    
    # 통계적 비교
    analyzer = ComparisonAnalyzer()
    
    # GBT vs SVM 비교
    gbt_errors = np.abs(y_true - models_predictions['GBT'])
    svm_errors = np.abs(y_true - models_predictions['SVM'])
    
    stat_comparison = analyzer.statistical_comparison(
        gbt_errors, svm_errors, "GBT", "SVM"
    )
    
    logger.info(f"Statistical comparison: {stat_comparison['paired_ttest']['interpretation']}")
    
    # 성능 요약
    summary = evaluator.get_performance_summary(model_results)
    logger.info(f"Performance summary: {len(summary['key_insights'])} key insights generated")
    
    logger.info("ModelEvaluator test completed successfully! ✅")


if __name__ == "__main__":
    test_evaluator() 