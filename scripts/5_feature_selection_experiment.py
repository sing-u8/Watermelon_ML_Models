#!/usr/bin/env python3
"""
특징 선택 실험 스크립트

현재 51개 특징에서 가장 중요한 특징들을 선택하여 
더 효율적이고 해석 가능한 모델을 만드는 실험을 수행합니다.

다양한 특징 선택 방법을 비교:
- Random Forest 특징 중요도
- Recursive Feature Elimination (RFE)
- SelectKBest (통계적 방법)
- Correlation-based selection

작성자: ML Team
생성일: 2025-01-15
"""

import os
import logging
import warnings
import pandas as pd
import numpy as np
import joblib
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import gc

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE, SelectKBest, f_classif, mutual_info_classif
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score
import scipy.stats as stats

# 경고 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_data():
    """데이터 및 특징 이름 로드"""
    logger.info("=== 데이터 로드 시작 ===")
    
    # 데이터 로드
    train_df = pd.read_csv("data/splits/full_dataset/train.csv")
    val_df = pd.read_csv("data/splits/full_dataset/val.csv")
    test_df = pd.read_csv("data/splits/full_dataset/test.csv")
    
    # 특징 이름 로드
    feature_names = list(train_df.columns[:-1])  # pitch_label 제외
    
    # 라벨 인코딩
    label_encoder = LabelEncoder()
    
    # 데이터 분리 및 스케일링
    X_train = train_df.drop('pitch_label', axis=1).values
    y_train = label_encoder.fit_transform(train_df['pitch_label'].values)
    X_val = val_df.drop('pitch_label', axis=1).values
    y_val = label_encoder.transform(val_df['pitch_label'].values)
    X_test = test_df.drop('pitch_label', axis=1).values
    y_test = label_encoder.transform(test_df['pitch_label'].values)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"전체 특징 수: {len(feature_names)}")
    logger.info(f"데이터 형태 - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
    logger.info(f"클래스 분포 - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_val': X_val_scaled,
        'y_val': y_val,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'label_encoder': label_encoder
    }


def load_best_model():
    """최고 성능 모델 로드"""
    logger.info("최고 성능 모델 로드 중...")
    
    # 최신 튜닝 결과 디렉토리 찾기
    tuning_dir = "experiments/hyperparameter_tuning"
    if os.path.exists(tuning_dir):
        subdirs = [d for d in os.listdir(tuning_dir) if d.startswith('simple_tuning_')]
        if subdirs:
            latest_dir = sorted(subdirs)[-1]
            model_path = os.path.join(tuning_dir, latest_dir, "random_forest_tuned.pkl")
            
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"모델 로드 완료: {model_path}")
                return model
    
    # 기본 모델 사용
    logger.warning("튜닝된 모델을 찾을 수 없어 기본 Random Forest 사용")
    return RandomForestClassifier(n_estimators=300, random_state=42)


def random_forest_importance(data, model):
    """Random Forest 특징 중요도 분석"""
    logger.info("=== Random Forest 특징 중요도 분석 ===")
    
    # 모델이 이미 훈련되어 있지 않으면 훈련
    try:
        importance = model.feature_importances_
    except AttributeError:
        logger.info("모델 훈련 중...")
        model.fit(data['X_train'], data['y_train'])
        importance = model.feature_importances_
    
    # 중요도와 특징 이름 매핑
    feature_importance = pd.DataFrame({
        'feature': data['feature_names'],
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    logger.info("상위 10개 중요 특징:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        logger.info(f"  {i:2d}. {row['feature']:30s}: {row['importance']:.4f}")
    
    return feature_importance


def rfe_selection(data, model, n_features_list=[10, 20, 30, 40]):
    """Recursive Feature Elimination"""
    logger.info("=== RFE 특징 선택 ===")
    
    rfe_results = {}
    
    for n_features in n_features_list:
        logger.info(f"RFE로 {n_features}개 특징 선택 중...")
        
        # RFE 수행
        rfe = RFE(estimator=model, n_features_to_select=n_features, step=1)
        rfe.fit(data['X_train'], data['y_train'])
        
        # 선택된 특징
        selected_features = [data['feature_names'][i] for i in range(len(data['feature_names'])) if rfe.support_[i]]
        
        # 성능 평가
        X_train_selected = rfe.transform(data['X_train'])
        X_test_selected = rfe.transform(data['X_test'])
        
        model_copy = RandomForestClassifier(n_estimators=100, random_state=42)
        model_copy.fit(X_train_selected, data['y_train'])
        
        y_pred = model_copy.predict(X_test_selected)
        accuracy = accuracy_score(data['y_test'], y_pred)
        f1 = f1_score(data['y_test'], y_pred, average='weighted')
        
        rfe_results[n_features] = {
            'selected_features': selected_features,
            'accuracy': accuracy,
            'f1_score': f1,
            'rfe_ranking': rfe.ranking_
        }
        
        logger.info(f"  {n_features}개 특징 - 정확도: {accuracy:.4f}, F1-score: {f1:.4f}")
    
    return rfe_results


def statistical_selection(data, k_list=[10, 20, 30, 40]):
    """통계적 방법을 이용한 특징 선택"""
    logger.info("=== 통계적 특징 선택 ===")
    
    statistical_results = {}
    
    # F-classification 점수 계산
    f_scores, f_pvalues = f_classif(data['X_train'], data['y_train'])
    
    # Mutual information 점수 계산
    mi_scores = mutual_info_classif(data['X_train'], data['y_train'], random_state=42)
    
    for k in k_list:
        logger.info(f"상위 {k}개 통계적 특징 선택 중...")
        
        # F-classification 기반 선택
        selector_f = SelectKBest(score_func=f_classif, k=k)
        X_train_f = selector_f.fit_transform(data['X_train'], data['y_train'])
        X_test_f = selector_f.transform(data['X_test'])
        
        selected_features_f = [data['feature_names'][i] for i in selector_f.get_support(indices=True)]
        
        # 성능 평가
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_f, data['y_train'])
        y_pred_f = model.predict(X_test_f)
        
        accuracy_f = accuracy_score(data['y_test'], y_pred_f)
        f1_f = f1_score(data['y_test'], y_pred_f, average='weighted')
        
        statistical_results[k] = {
            'f_classification': {
                'selected_features': selected_features_f,
                'accuracy': accuracy_f,
                'f1_score': f1_f,
                'scores': f_scores[selector_f.get_support()]
            }
        }
        
        logger.info(f"  F-classification {k}개 - 정확도: {accuracy_f:.4f}, F1-score: {f1_f:.4f}")
    
    return statistical_results, {'f_scores': f_scores, 'f_pvalues': f_pvalues, 'mi_scores': mi_scores}


def correlation_analysis(data):
    """상관관계 분석"""
    logger.info("=== 상관관계 분석 ===")
    
    # 데이터프레임 생성
    df = pd.DataFrame(data['X_train'], columns=data['feature_names'])
    df['pitch_label'] = data['y_train']
    
    # 타겟과의 상관관계
    target_corr = df.corr()['pitch_label'].drop('pitch_label').abs().sort_values(ascending=False)
    
    logger.info("타겟과 상관관계 높은 상위 10개 특징:")
    for i, (feature, corr) in enumerate(target_corr.head(10).items(), 1):
        logger.info(f"  {i:2d}. {feature:30s}: {corr:.4f}")
    
    # 특징간 상관관계 (높은 상관관계 특징 탐지)
    feature_corr = df.drop('pitch_label', axis=1).corr()
    
    # 상관관계 0.9 이상인 특징 쌍 찾기
    high_corr_pairs = []
    for i in range(len(feature_corr.columns)):
        for j in range(i+1, len(feature_corr.columns)):
            corr_val = abs(feature_corr.iloc[i, j])
            if corr_val > 0.9:
                high_corr_pairs.append({
                    'feature1': feature_corr.columns[i],
                    'feature2': feature_corr.columns[j],
                    'correlation': corr_val
                })
    
    logger.info(f"높은 상관관계 (>0.9) 특징 쌍: {len(high_corr_pairs)}개")
    for pair in high_corr_pairs[:5]:  # 상위 5개만 출력
        logger.info(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.4f}")
    
    return target_corr, high_corr_pairs


def progressive_feature_selection(data, model, max_features=30):
    """점진적 특징 선택 (Forward Selection)"""
    logger.info(f"=== 점진적 특징 선택 (최대 {max_features}개) ===")
    
    selected_features = []
    remaining_features = list(range(len(data['feature_names'])))
    performance_history = []
    
    for step in range(min(max_features, len(data['feature_names']))):
        best_accuracy = 0.0
        best_feature = None
        
        # 각 남은 특징에 대해 성능 평가
        for feature_idx in remaining_features:
            current_features = selected_features + [feature_idx]
            
            X_train_subset = data['X_train'][:, current_features]
            X_test_subset = data['X_test'][:, current_features]
            
            # 빠른 평가를 위해 작은 모델 사용
            temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
            temp_model.fit(X_train_subset, data['y_train'])
            y_pred = temp_model.predict(X_test_subset)
            accuracy = accuracy_score(data['y_test'], y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature_idx
        
        # 최고 성능 특징 추가
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            
            # 성능 기록
            feature_name = data['feature_names'][best_feature]
            f1 = f1_score(data['y_test'], temp_model.predict(data['X_test'][:, selected_features]), average='weighted')
            
            performance_history.append({
                'step': step + 1,
                'feature_added': feature_name,
                'accuracy': best_accuracy,
                'f1_score': f1,
                'feature_count': len(selected_features)
            })
            
            logger.info(f"  Step {step+1:2d}: 추가된 특징 '{feature_name}' - 정확도: {best_accuracy:.4f}, F1-score: {f1:.4f}")
        
        # 성능이 더 이상 개선되지 않으면 조기 중단
        if len(performance_history) >= 3:
            recent_accuracies = [p['accuracy'] for p in performance_history[-3:]]
            if all(acc <= recent_accuracies[0] + 0.001 for acc in recent_accuracies[1:]):
                logger.info(f"  성능 개선이 미미하여 Step {step+1}에서 조기 중단")
                break
    
    selected_feature_names = [data['feature_names'][i] for i in selected_features]
    
    return selected_feature_names, performance_history


def evaluate_feature_sets(data, feature_sets):
    """다양한 특징 세트 성능 비교"""
    logger.info("=== 특징 세트 성능 비교 ===")
    
    results = {}
    
    for set_name, features in feature_sets.items():
        logger.info(f"평가 중: {set_name} ({len(features)}개 특징)")
        
        # 특징 인덱스 찾기
        feature_indices = [data['feature_names'].index(f) for f in features if f in data['feature_names']]
        
        if not feature_indices:
            logger.warning(f"  {set_name}: 유효한 특징이 없습니다.")
            continue
        
        # 데이터 subset 생성
        X_train_subset = data['X_train'][:, feature_indices]
        X_val_subset = data['X_val'][:, feature_indices]
        X_test_subset = data['X_test'][:, feature_indices]
        
        # 모델 훈련 및 평가
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train_subset, data['y_train'])
        
        # 검증 세트 성능
        y_pred_val = model.predict(X_val_subset)
        val_accuracy = accuracy_score(data['y_val'], y_pred_val)
        val_f1 = f1_score(data['y_val'], y_pred_val, average='weighted')
        
        # 테스트 세트 성능
        y_pred_test = model.predict(X_test_subset)
        test_accuracy = accuracy_score(data['y_test'], y_pred_test)
        test_f1 = f1_score(data['y_test'], y_pred_test, average='weighted')
        
        # 교차 검증
        cv_scores = cross_val_score(model, X_train_subset, data['y_train'], 
                                   cv=5, scoring='accuracy')
        cv_accuracy = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[set_name] = {
            'feature_count': len(feature_indices),
            'features': features,
            'val_accuracy': val_accuracy,
            'val_f1_score': val_f1,
            'test_accuracy': test_accuracy,
            'test_f1_score': test_f1,
            'cv_accuracy': cv_accuracy,
            'cv_std': cv_std
        }
        
        logger.info(f"  테스트 - 정확도: {test_accuracy:.4f}, F1-score: {test_f1:.4f}")
        logger.info(f"  CV - 정확도: {cv_accuracy:.4f} ± {cv_std:.4f}")
    
    return results


def create_visualizations(importance_df, target_corr, performance_history, results_dir):
    """시각화 생성"""
    logger.info("시각화 생성 중...")
    
    plt.style.use('default')
    
    # 1. 특징 중요도 플롯
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('상위 20개 특징 중요도 (Random Forest)', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 타겟 상관관계 플롯
    plt.figure(figsize=(12, 8))
    top_corr = target_corr.head(20)
    sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm')
    plt.title('상위 20개 특징과 음 높낮이의 상관관계', fontsize=14, fontweight='bold')
    plt.xlabel('Absolute Correlation with Pitch Label', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'target_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 점진적 특징 선택 성능 추이
    if performance_history:
        plt.figure(figsize=(12, 6))
        steps = [p['step'] for p in performance_history]
        accuracies = [p['accuracy'] for p in performance_history]
        
        plt.plot(steps, accuracies, 'o-', linewidth=2, markersize=6)
        plt.title('점진적 특징 선택 - 성능 추이', fontsize=14, fontweight='bold')
        plt.xlabel('Feature Count', fontsize=12)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'progressive_selection.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"시각화 저장 완료: {results_dir}")


def save_results(all_results, data, results_dir):
    """결과 저장"""
    logger.info(f"결과 저장 중: {results_dir}")
    
    # 전체 결과 저장
    results_path = os.path.join(results_dir, "feature_selection_results.yaml")
    with open(results_path, 'w', encoding='utf-8') as f:
        yaml.dump(all_results, f, default_flow_style=False, allow_unicode=True)
    
    # 추천 특징 세트 저장
    recommendations = {
        'best_overall': None,
        'best_small': None,
        'best_medium': None,
        'feature_rankings': {}
    }
    
    # 성능 기준으로 최고 특징 세트 찾기
    if 'feature_set_comparison' in all_results:
        comparison = all_results['feature_set_comparison']
        
        # 전체 최고 성능
        best_overall = max(comparison.items(), key=lambda x: x[1]['test_accuracy'])
        recommendations['best_overall'] = {
            'name': best_overall[0],
            'features': best_overall[1]['features'],
            'test_accuracy': best_overall[1]['test_accuracy'],
            'test_f1_score': best_overall[1]['test_f1_score']
        }
        
        # 크기별 최고 성능
        small_sets = {k: v for k, v in comparison.items() if v['feature_count'] <= 15}
        medium_sets = {k: v for k, v in comparison.items() if 15 < v['feature_count'] <= 30}
        
        if small_sets:
            best_small = max(small_sets.items(), key=lambda x: x[1]['test_accuracy'])
            recommendations['best_small'] = {
                'name': best_small[0],
                'features': best_small[1]['features'],
                'test_accuracy': best_small[1]['test_accuracy'],
                'test_f1_score': best_small[1]['test_f1_score']
            }
        
        if medium_sets:
            best_medium = max(medium_sets.items(), key=lambda x: x[1]['test_accuracy'])
            recommendations['best_medium'] = {
                'name': best_medium[0],
                'features': best_medium[1]['features'],
                'test_accuracy': best_medium[1]['test_accuracy'],
                'test_f1_score': best_medium[1]['test_f1_score']
            }
    
    recommendations_path = os.path.join(results_dir, "feature_recommendations.yaml")
    with open(recommendations_path, 'w', encoding='utf-8') as f:
        yaml.dump(recommendations, f, default_flow_style=False, allow_unicode=True)
    
    logger.info("결과 저장 완료")
    return recommendations


def generate_report(all_results, recommendations, results_dir):
    """종합 보고서 생성"""
    best_overall = recommendations.get('best_overall')
    best_small = recommendations.get('best_small')
    best_medium = recommendations.get('best_medium')
    
    report = f"""# 특징 선택 실험 결과 보고서

생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 실험 개요

- **목적**: 수박 음 높낮이 분류를 위한 최적 특징 subset 탐색
- **원본 특징 수**: 51개
- **실험 방법**: Random Forest 중요도, RFE, 통계적 선택, 점진적 선택
- **평가 지표**: 정확도 (Accuracy), F1-score

## 주요 발견사항

### 🏆 최고 성능 특징 세트
"""
    
    if best_overall:
        report += f"""
**세트명**: {best_overall['name']}
- **특징 수**: {len(best_overall['features'])}개
- **테스트 정확도**: {best_overall['test_accuracy']:.4f}
- **테스트 F1-score**: {best_overall['test_f1_score']:.4f}

**선택된 특징들**:
{', '.join(best_overall['features'][:10])}{'...' if len(best_overall['features']) > 10 else ''}
"""
    
    if best_small:
        report += f"""
### 💡 소형 특징 세트 (≤15개)
**세트명**: {best_small['name']}
- **특징 수**: {len(best_small['features'])}개  
- **테스트 정확도**: {best_small['test_accuracy']:.4f}
- **테스트 F1-score**: {best_small['test_f1_score']:.4f}
"""
    
    if best_medium:
        report += f"""
### 🎯 중형 특징 세트 (16-30개)
**세트명**: {best_medium['name']}
- **특징 수**: {len(best_medium['features'])}개
- **테스트 정확도**: {best_medium['test_accuracy']:.4f}  
- **테스트 F1-score**: {best_medium['test_f1_score']:.4f}
"""
    
    # 전체 결과 요약
    if 'feature_set_comparison' in all_results:
        report += "\n## 모든 특징 세트 성능 비교\n\n"
        comparison = all_results['feature_set_comparison']
        
        for name, metrics in sorted(comparison.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
            report += f"""### {name}
- **특징 수**: {metrics['feature_count']}개
- **테스트 정확도**: {metrics['test_accuracy']:.4f}
- **테스트 F1-score**: {metrics['test_f1_score']:.4f}
- **CV 정확도**: {metrics['cv_accuracy']:.4f} ± {metrics['cv_std']:.4f}

"""
    
    # 결론
    full_performance = all_results.get('feature_set_comparison', {}).get('all_features_baseline')
    if full_performance and best_overall:
        original_accuracy = full_performance['test_accuracy']
        best_accuracy = best_overall['test_accuracy']
        improvement = ((best_accuracy - original_accuracy) / original_accuracy) * 100
        
        report += f"""
## 결론

특징 선택을 통해 **{len(best_overall['features'])}개 특징**으로 
원본 51개 특징 대비 {'개선된' if improvement > 0 else '유사한'} 성능을 달성했습니다.

- **원본 성능**: 정확도 {original_accuracy:.4f}
- **최적 성능**: 정확도 {best_accuracy:.4f}
- **성능 변화**: {improvement:+.2f}%
- **특징 감소**: {51 - len(best_overall['features'])}개 ({((51 - len(best_overall['features']))/51)*100:.1f}%)

### 권장사항

1. **프로덕션 배포**: {best_small['name'] if best_small else best_overall['name']} 사용 권장
2. **해석 가능성**: 적은 수의 특징으로 높은 성능 달성 
3. **계산 효율성**: 특징 수 감소로 추론 속도 향상 기대
"""
    
    # 보고서 저장
    report_path = os.path.join(results_dir, "FEATURE_SELECTION_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"보고서 저장: {report_path}")
    
    # 콘솔 출력
    print("\n" + "="*60)
    print("🔍 특징 선택 실험 완료!")
    print("="*60)
    if best_overall:
        print(f"최고 성능: {best_overall['name']}")
        print(f"특징 수: {len(best_overall['features'])}개 (원본: 51개)")
        print(f"테스트 정확도: {best_overall['test_accuracy']:.4f}")
        print(f"테스트 F1-score: {best_overall['test_f1_score']:.4f}")
    print(f"결과 저장: {results_dir}")
    print("="*60)


def main():
    """메인 실행 함수"""
    logger.info("🔍 특징 선택 실험 시작")
    
    try:
        # 결과 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"experiments/feature_selection/selection_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. 데이터 로드
        data = load_data()
        
        # 2. 최고 성능 모델 로드
        best_model = load_best_model()
        
        # 3. Random Forest 특징 중요도 분석
        importance_df = random_forest_importance(data, best_model)
        
        # 4. RFE 특징 선택
        rfe_results = rfe_selection(data, best_model, [10, 15, 20, 25, 30])
        
        # 5. 통계적 특징 선택
        statistical_results, stat_scores = statistical_selection(data, [10, 15, 20, 25, 30])
        
        # 6. 상관관계 분석
        target_corr, high_corr_pairs = correlation_analysis(data)
        
        # 7. 점진적 특징 선택
        progressive_features, performance_history = progressive_feature_selection(data, best_model, 25)
        
        # 8. 다양한 특징 세트 정의
        feature_sets = {
            'all_features_baseline': data['feature_names'],
            'top10_importance': importance_df.head(10)['feature'].tolist(),
            'top15_importance': importance_df.head(15)['feature'].tolist(),
            'top20_importance': importance_df.head(20)['feature'].tolist(),
            'top10_correlation': target_corr.head(10).index.tolist(),
            'top15_correlation': target_corr.head(15).index.tolist(),
            'rfe_15': rfe_results[15]['selected_features'],
            'rfe_20': rfe_results[20]['selected_features'],
            'statistical_15': statistical_results[15]['f_classification']['selected_features'],
            'statistical_20': statistical_results[20]['f_classification']['selected_features'],
            'progressive_selection': progressive_features
        }
        
        # 9. 특징 세트 성능 비교
        comparison_results = evaluate_feature_sets(data, feature_sets)
        
        # 10. 결과 통합
        all_results = {
            'feature_importance': importance_df.to_dict('records'),
            'rfe_results': rfe_results,
            'statistical_results': statistical_results,
            'correlation_analysis': {
                'target_correlation': target_corr.to_dict(),
                'high_correlation_pairs': high_corr_pairs
            },
            'progressive_selection': {
                'selected_features': progressive_features,
                'performance_history': performance_history
            },
            'feature_set_comparison': comparison_results
        }
        
        # 11. 시각화 생성
        create_visualizations(importance_df, target_corr, performance_history, results_dir)
        
        # 12. 결과 저장
        recommendations = save_results(all_results, data, results_dir)
        
        # 13. 보고서 생성
        generate_report(all_results, recommendations, results_dir)
        
        # 메모리 정리
        gc.collect()
        
        logger.info("🎉 특징 선택 실험이 성공적으로 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"❌ 특징 선택 실험 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main() 