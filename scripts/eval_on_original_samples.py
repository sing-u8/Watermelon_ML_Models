import sys
import os
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.append(str(Path(__file__).parent.parent))

from src.data.audio_loader import AudioLoader
from src.data.preprocessor import AudioPreprocessor
from src.data.feature_extractor import AudioFeatureExtractor

# 1. Find latest production model directory
def get_latest_model_dir():
    production_dir = Path('models/production')
    latest_link = production_dir / 'latest'
    if latest_link.exists() and latest_link.is_symlink():
        return latest_link.resolve()
    model_dirs = [d for d in production_dir.iterdir() if d.is_dir() and d.name.startswith('final_model_')]
    if not model_dirs:
        raise FileNotFoundError("No models found in production folder.")
    model_dirs = sorted(model_dirs, key=lambda d: d.stat().st_mtime, reverse=True)
    return model_dirs[0]

# 2. Load model, scaler, feature list, and label encoder
def load_model_assets(model_dir):
    model = joblib.load(model_dir / 'watermelon_pitch_model.pkl')
    scaler = joblib.load(model_dir / 'feature_scaler.pkl')
    label_encoder = joblib.load(model_dir / 'label_encoder.pkl')
    with open(model_dir / 'selected_features.txt', 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    return model, scaler, label_encoder, selected_features

# 3. Load original sample metadata (adjust sample count by mode)
def load_metadata(mode='all'):
    meta_path = Path('data/watermelon_metadata.csv')
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
    
    metadata = pd.read_csv(meta_path)
    
    if mode == 'all':
        # All samples (173)
        return metadata
    elif mode == 'train':
        # Only samples used in training (121)
        # Assume first 121 samples in metadata were used for training
        return metadata.head(121)
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'all' or 'train'.")

# 4. Prediction and evaluation
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Watermelon Pitch Classification Model Evaluation')
    parser.add_argument('--mode', choices=['all', 'train'], default='all',
                       help='Evaluation sample mode: all(173 samples) or train(121 samples)')
    args = parser.parse_args()
    
    print(f"=== Watermelon Pitch Classification Model Evaluation ({args.mode} mode) ===")
    print(f"Evaluation mode: {args.mode}")
    if args.mode == 'all':
        print("Evaluation target: All original samples (173)")
    else:
        print("Evaluation target: Samples used in training (121)")
    
    print("\nLoading latest production model...")
    model_dir = get_latest_model_dir()
    print(f"Model directory: {model_dir}")
    model, scaler, label_encoder, selected_features = load_model_assets(model_dir)

    metadata = load_metadata(args.mode)
    print(f"Loaded samples: {len(metadata)}")
    
    loader = AudioLoader(sample_rate=16000, mono=True)
    preprocessor = AudioPreprocessor()
    feature_extractor = AudioFeatureExtractor()

    actual = []
    predicted = []
    watermelon_ids = []
    failed = 0

    for idx, row in metadata.iterrows():
        file_path = str(row['file_path'])
        true_pitch = row['pitch_label']
        watermelon_id = row['watermelon_id']
        try:
            audio_data, sr = loader.load_audio(file_path)
            processed_audio, _ = preprocessor.preprocess_audio(audio_data, sr)
            all_features = feature_extractor.extract_all_features(processed_audio, sr)
            feature_names = pd.Index(feature_extractor.get_feature_names())
            feature_df = pd.DataFrame([all_features], columns=feature_names)
            selected_feature_values = feature_df[selected_features].values
            scaled_features = scaler.transform(selected_feature_values)
            pred = model.predict(scaled_features)[0]
            actual.append(true_pitch)
            predicted.append(pred)
            watermelon_ids.append(watermelon_id)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            failed += 1

    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Use label encoder to convert actual labels to numbers
    actual_encoded = label_encoder.transform(actual)
    
    # Calculate classification metrics
    accuracy = accuracy_score(actual_encoded, predicted)
    f1 = f1_score(actual_encoded, predicted, average='weighted')
    
    print(f"\n=== {args.mode.upper()} Mode Evaluation Results ===")
    print(f"Evaluation samples: {len(actual)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    if failed > 0:
        print(f"Failed samples: {failed}")

    # Print classification report (using original labels)
    print(f"\n=== Classification Report ===")
    print(classification_report(actual_encoded, predicted))

    # Save results
    result_df = pd.DataFrame({
        'watermelon_id': watermelon_ids,
        'actual_pitch': actual,
        'predicted_pitch': predicted,
        'correct': (actual == predicted)
    })
    result_csv = model_dir / f'{args.mode}_mode_predictions.csv'
    result_df.to_csv(result_csv, index=False)
    print(f"Prediction results saved: {result_csv}")

    # Visualization
    plt.figure(figsize=(15, 5))
    
    # (1) Confusion Matrix
    plt.subplot(1, 3, 1)
    cm = confusion_matrix(actual_encoded, predicted)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'High'], 
                yticklabels=['Low', 'High'])
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.title(f'Confusion Matrix ({args.mode.upper()} Mode)')
    
    # (2) Class Distribution
    plt.subplot(1, 3, 2)
    actual_counts = pd.Series(actual_encoded).value_counts()
    predicted_counts = pd.Series(predicted).value_counts()
    
    x = np.arange(len(actual_counts))
    width = 0.35
    
    plt.bar(x - width/2, actual_counts.values, width, label='Actual', alpha=0.8)
    plt.bar(x + width/2, predicted_counts.values, width, label='Predicted', alpha=0.8)
    plt.xlabel('Pitch Level')
    plt.ylabel('Sample Count')
    plt.title(f'Class Distribution ({args.mode.upper()} Mode)')
    plt.xticks(x, ['Low', 'High'])
    plt.legend()
    
    # (3) Accuracy by Class
    plt.subplot(1, 3, 3)
    class_accuracy = []
    class_names = ['Low', 'High']
    
    for i in range(2):  # 0: low, 1: high
        mask = actual_encoded == i
        if mask.sum() > 0:
            class_acc = (actual_encoded[mask] == predicted[mask]).mean()
            class_accuracy.append(class_acc)
        else:
            class_accuracy.append(0)
    
    plt.bar(class_names, class_accuracy, color=['skyblue', 'lightcoral'])
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy by Class ({args.mode.upper()} Mode)')
    plt.ylim(0, 1)
    
    # Display accuracy values
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 