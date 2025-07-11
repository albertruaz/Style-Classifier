#!/usr/bin/env python3
# test_trained_model.py

import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, accuracy_score,
    precision_recall_fscore_support
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Local modules
import sys
sys.path.append('..')
from prepare_training_data import MorigirlDataProcessor, MorigirlDataset
from morigirl.morigirl_model import MoriGirlVectorClassifier

class MoriGirlModelTester:
    """Trained Morigirl vector classification model test class"""
    
    def __init__(self, checkpoint_path: str = None, config_path: str = "config.json", data_path: str = None):
        self.config = self.load_config(config_path)
        
        # Checkpoint path setting (priority: parameter > config > auto-find)
        if checkpoint_path is None:
            self.checkpoint_path = self._get_checkpoint_path()
        else:
            self.checkpoint_path = checkpoint_path
        
        # Data path setting (priority: parameter > config > default)
        if data_path is None:
            self.data_path = self._get_data_path()
        else:
            self.data_path = data_path
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Results directory (save under the same experiment folder as checkpoint)
        self.results_dir = self._get_test_results_dir()
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🧪 Starting Morigirl model test")
        print(f"  - Checkpoint: {checkpoint_path}")
        print(f"  - Data path: {self.data_path}")
        print(f"  - Device: {self.device}")
        print(f"  - Results: {self.results_dir}")

    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded: {config_path}")
            return config
        except Exception as e:
            print(f"❌ Configuration load failed: {e}")
            # Return default configuration
            return {
                "data": {"max_products_per_type": 5000},
                "model": {"input_vector_dim": 1024, "hidden_dim": 128, "dropout_rate": 0.1, "batch_size": 64}
            }

    def _get_data_path(self) -> str:
        """Get test data path from config"""
        data_config = self.config["data"]
        data_paths = data_config.get("data_paths", {})
        
        # 1. Use test_data_dir if set
        if data_paths.get("test_data_dir"):
            print(f"📁 User-defined test data path: {data_paths['test_data_dir']}")
            return data_paths["test_data_dir"]
        
        # 2. Use base_data_dir (automatic path generation)
        if data_paths.get("auto_generate_path", True):
            max_products = data_config["max_products_per_type"]
            base_path = data_paths.get("base_data_dir", "../data/morigirl_{max_products}")
            final_path = base_path.format(max_products=max_products)
            print(f"📁 Auto-generated data path: {final_path}")
            return final_path
        
        # 3. Default value
        max_products = data_config["max_products_per_type"]
        default_path = f"../data/morigirl_{max_products}"
        print(f"📁 Default data path: {default_path}")
        return default_path

    def _get_checkpoint_path(self) -> str:
        """Get checkpoint path from config"""
        data_config = self.config["data"]
        test_paths = data_config.get("test_paths", {})
        
        # 1. Use checkpoint_path if directly set
        if test_paths.get("checkpoint_path"):
            checkpoint_path = test_paths["checkpoint_path"]
            if os.path.exists(checkpoint_path):
                print(f"🔍 User-defined checkpoint: {checkpoint_path}")
                return checkpoint_path
            else:
                print(f"⚠️  Specified checkpoint does not exist: {checkpoint_path}")
        
        # 2. Auto-search if auto_find_best_model is enabled
        if test_paths.get("auto_find_best_model", True):
            target_experiment = test_paths.get("target_experiment")
            
            if target_experiment:
                # Find best_model.pth of specific experiment
                checkpoint_path = f"result/{target_experiment}/checkpoints/best_model.pth"
                if os.path.exists(checkpoint_path):
                    print(f"🔍 Auto-found checkpoint: {checkpoint_path}")
                    return checkpoint_path
                else:
                    print(f"⚠️  Target experiment checkpoint does not exist: {checkpoint_path}")
            
            # Find latest experiment's best_model.pth
            result_dir = "result"
            if os.path.exists(result_dir):
                experiments = [d for d in os.listdir(result_dir) 
                             if os.path.isdir(os.path.join(result_dir, d))]
                if experiments:
                    # Sort by experiment name (date-time based, latest first)
                    experiments.sort(reverse=True)
                    for exp in experiments:
                        checkpoint_path = f"{result_dir}/{exp}/checkpoints/best_model.pth"
                        if os.path.exists(checkpoint_path):
                            print(f"🔍 Latest experiment checkpoint: {checkpoint_path}")
                            return checkpoint_path
        
        # 3. Error if not found
        raise FileNotFoundError(
            "Could not find checkpoint. Please set one of the following:\n"
            "1. Specify with --checkpoint argument\n"
            "2. Set test_paths.checkpoint_path in config.json\n"
            "3. Set test_paths.target_experiment in config.json\n"
            "4. Check if experiment results exist in result/ folder"
        )

    def _get_test_results_dir(self) -> str:
        """Get test results directory based on config and checkpoint location"""
        data_config = self.config["data"]
        result_paths = data_config.get("result_paths", {})
        
        # 1. Check if specific test result directory is configured
        if result_paths.get("test_result_dir"):
            test_dir = result_paths["test_result_dir"]
            print(f"📁 User-defined test results directory: {test_dir}")
            return test_dir
        
        # 2. Use target_experiment if specified
        test_paths = data_config.get("test_paths", {})
        target_experiment = test_paths.get("target_experiment")
        if target_experiment:
            base_result_dir = result_paths.get("base_result_dir", "result")
            test_results_dir = f"{base_result_dir}/{target_experiment}/test_results"
            print(f"📁 Target experiment test results: {test_results_dir}")
            return test_results_dir
        
        # 3. Extract from checkpoint path (existing logic)
        checkpoint_path = self.checkpoint_path
        if checkpoint_path.startswith("result/") and "/checkpoints/" in checkpoint_path:
            # Extract experiment name from path like "result/12345678_90/checkpoints/best_model.pth"
            parts = checkpoint_path.split("/")
            if len(parts) >= 3:
                experiment_name = parts[1]
                base_result_dir = result_paths.get("base_result_dir", "result")
                test_results_dir = f"{base_result_dir}/{experiment_name}/test_results"
                print(f"📁 Checkpoint-based test results: {test_results_dir}")
                return test_results_dir
        
        # 4. Fallback: create new directory with timestamp
        import random
        date_str = datetime.now().strftime('%m%d%H%M')  # MMDDHHMM
        random_num = random.randint(10, 99)  # random 2 digits
        base_result_dir = result_paths.get("base_result_dir", "result")
        result_name = f"test_{date_str}_{random_num:02d}"
        fallback_dir = f"{base_result_dir}/{result_name}"
        print(f"📁 Creating new test results directory: {fallback_dir}")
        return fallback_dir

    def load_model(self) -> nn.Module:
        """Load model from checkpoint"""
        print(f"\n📦 Loading model...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Read model parameters from config
        model_config = self.config["model"]
        model_kwargs = {
            "input_dim": model_config["input_vector_dim"],
            "hidden_dim": model_config["hidden_dim"],
            "hidden_dim2": model_config["hidden_dim2"],
            "dropout_rate": model_config["dropout_rate"]
        }
        
        # Create model
        model = MoriGirlVectorClassifier(**model_kwargs)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Validation accuracy: {checkpoint.get('metrics', {}).get('accuracy', 'N/A')}")
        
        return model

    def setup_test_dataset(self) -> DataLoader:
        """Setup test dataset"""
        print(f"\n📊 Setting up test dataset")
        
        # Read settings from config
        data_config = self.config["data"]
        model_config = self.config["model"]
        
        test_size = 1 - data_config["train_test_split"]
        batch_size = model_config["batch_size"]
        
        # Create data processor and load (test files only)
        processor = MorigirlDataProcessor(self.data_path)
        if not processor.load_npy_files(split_type="test"):
            raise RuntimeError("Failed to load test data.")
        
        # Create test dataset (using pre-split test files)
        test_dataset = MorigirlDataset(
            processor.vectors, 
            processor.labels, 
            processor.product_ids
        )
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"📦 Test DataLoader created:")
        print(f"  - Test batches: {len(test_loader)}")
        print(f"  - Batch size: {batch_size}")
        
        return test_loader

    def predict_all(self, model: nn.Module, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run predictions on entire test set"""
        print(f"\n🔮 Running model predictions")
        
        all_probs = []
        all_preds = []
        all_labels = []
        all_product_ids = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                vectors = batch['vector'].to(self.device)
                labels = batch['label']
                product_ids = batch['product_id']
                
                # Run predictions
                outputs = model(vectors)  # sigmoid already applied
                probs = outputs.cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_product_ids.extend(product_ids.numpy())
        
        return np.array(all_probs), np.array(all_preds), np.array(all_labels), np.array(all_product_ids)

    def compute_metrics(self, probs: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics"""
        print(f"\n📊 Calculating performance metrics")
        
        # Basic metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        
        # AUC
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # ROC curve data
        fpr, tpr, _ = roc_curve(labels, probs)
        
        # Precision-Recall curve data
        pr_precision, pr_recall, _ = precision_recall_curve(labels, probs)
        
        # Classification report by class
        report = classification_report(labels, preds, target_names=['Non-Morigirl', 'Morigirl'], output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'roc_curve': (fpr, tpr),
            'pr_curve': (pr_precision, pr_recall),
            'classification_report': report
        }
        
        # Print results
        print(f"✅ Performance results:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - Precision: {precision:.4f}")
        print(f"  - Recall: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        return metrics

    def create_visualizations(self, metrics: Dict[str, Any], probs: np.ndarray, labels: np.ndarray):
        """Create visualizations"""
        print(f"\n📈 Creating visualizations")
        
        # Use default font (avoid Korean font issues)
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Morigirl Classification Model Test Results', fontsize=16)
        
        # 1. Confusion Matrix
        ax1 = axes[0, 0]
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Non-Morigirl', 'Morigirl'], yticklabels=['Non-Morigirl', 'Morigirl'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. ROC Curve
        ax2 = axes[0, 1]
        fpr, tpr = metrics['roc_curve']
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {metrics["auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True)
        
        # 3. Precision-Recall Curve
        ax3 = axes[0, 2]
        precision, recall = metrics['pr_curve']
        ax3.plot(recall, precision, color='blue', lw=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.grid(True)
        
        # 4. Probability Distribution
        ax4 = axes[1, 0]
        morigirl_probs = probs[labels == 1]
        non_morigirl_probs = probs[labels == 0]
        
        ax4.hist(non_morigirl_probs, bins=30, alpha=0.7, label='Non-Morigirl', color='red', density=True)
        ax4.hist(morigirl_probs, bins=30, alpha=0.7, label='Morigirl', color='blue', density=True)
        ax4.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Probability Distribution by Class')
        ax4.legend()
        ax4.grid(True)
        
        # 5. Performance Metrics Bar Chart
        ax5 = axes[1, 1]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1'], metrics['auc']]
        
        bars = ax5.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax5.set_ylim(0, 1.0)
        ax5.set_title('Performance Metrics')
        ax5.set_ylabel('Score')
        
        # Add values on top of bars
        for bar, value in zip(bars, metric_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. Class-wise Performance
        ax6 = axes[1, 2]
        report = metrics['classification_report']
        
        class_names = ['Non-Morigirl', 'Morigirl']
        # Safely access classification report with fallback
        try:
            precision_scores = [report['0']['precision'], report['1']['precision']]
            recall_scores = [report['0']['recall'], report['1']['recall']]
            f1_scores = [report['0']['f1-score'], report['1']['f1-score']]
        except (KeyError, TypeError):
            # Fallback to binary metrics if detailed report is not available
            precision_scores = [metrics['precision'], metrics['precision']]
            recall_scores = [metrics['recall'], metrics['recall']]
            f1_scores = [metrics['f1'], metrics['f1']]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax6.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax6.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax6.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax6.set_xlabel('Class')
        ax6.set_ylabel('Score')
        ax6.set_title('Class-wise Performance')
        ax6.set_xticks(x)
        ax6.set_xticklabels(class_names)
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(self.results_dir, 'test_results_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Visualization saved: {save_path}")
        
        # Show (optional)
        # plt.show()
        plt.close()

    def save_detailed_results(self, metrics: Dict[str, Any], probs: np.ndarray, 
                            preds: np.ndarray, labels: np.ndarray, product_ids: np.ndarray):
        """Save detailed results"""
        print(f"\n💾 Saving detailed results")
        
        # 1. Save Metrics JSON
        metrics_to_save = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc']),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'classification_report': metrics['classification_report']
        }
        
        metrics_path = os.path.join(self.results_dir, 'metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
        
        # 2. Save Prediction Results CSV
        results_df = pd.DataFrame({
            'product_id': product_ids,
            'actual_label': labels,
            'predicted_label': preds,
            'predicted_probability': probs,
            'correct': labels == preds
        })
        
        csv_path = os.path.join(self.results_dir, 'predictions.csv')
        results_df.to_csv(csv_path, index=False)
        
        # 3. Classification Report Text Save
        report_text = classification_report(labels, preds, target_names=['Non-Morigirl', 'Morigirl'])
        report_path = os.path.join(self.results_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        # 4. Save Top/Bottom 10 samples for each case
        self.save_case_analysis(results_df)
        
        print(f"💾 Results saved successfully:")
        print(f"  - Metrics: {metrics_path}")
        print(f"  - Predictions: {csv_path}")
        print(f"  - Classification Report: {report_path}")

    def save_case_analysis(self, results_df: pd.DataFrame):
        """Save analysis of top/bottom 10 samples for each case"""
        print(f"\n📊 Analyzing cases...")
        
        # Split data by actual labels
        actual_morigirl = results_df[results_df['actual_label'] == 1].copy()
        actual_non_morigirl = results_df[results_df['actual_label'] == 0].copy()
        
        analysis_results = {}
        
        # Case 1: Actual Morigirl (1) - High probability (True Positives with high confidence)
        if len(actual_morigirl) > 0:
            morigirl_high = actual_morigirl.nlargest(10, 'predicted_probability')
            analysis_results['actual_morigirl_high_prob'] = {
                'description': 'Actual Morigirl - Highest predicted probabilities',
                'samples': morigirl_high[['product_id', 'predicted_probability', 'predicted_label', 'correct']].to_dict('records')
            }
            
            # Case 2: Actual Morigirl (1) - Low probability (False Negatives and low confidence TPs)
            morigirl_low = actual_morigirl.nsmallest(10, 'predicted_probability')
            analysis_results['actual_morigirl_low_prob'] = {
                'description': 'Actual Morigirl - Lowest predicted probabilities',
                'samples': morigirl_low[['product_id', 'predicted_probability', 'predicted_label', 'correct']].to_dict('records')
            }
        
        # Case 3: Actual Non-Morigirl (0) - High probability (False Positives)
        if len(actual_non_morigirl) > 0:
            non_morigirl_high = actual_non_morigirl.nlargest(10, 'predicted_probability')
            analysis_results['actual_non_morigirl_high_prob'] = {
                'description': 'Actual Non-Morigirl - Highest predicted probabilities (False Positives)',
                'samples': non_morigirl_high[['product_id', 'predicted_probability', 'predicted_label', 'correct']].to_dict('records')
            }
            
            # Case 4: Actual Non-Morigirl (0) - Low probability (True Negatives with high confidence)
            non_morigirl_low = actual_non_morigirl.nsmallest(10, 'predicted_probability')
            analysis_results['actual_non_morigirl_low_prob'] = {
                'description': 'Actual Non-Morigirl - Lowest predicted probabilities',
                'samples': non_morigirl_low[['product_id', 'predicted_probability', 'predicted_label', 'correct']].to_dict('records')
            }
        
        # Save to JSON
        analysis_path = os.path.join(self.results_dir, 'case_analysis.json')
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        # Save to CSV for easy viewing
        all_cases = []
        for case_name, case_data in analysis_results.items():
            for sample in case_data['samples']:
                sample['case'] = case_name
                sample['description'] = case_data['description']
                all_cases.append(sample)
        
        if all_cases:
            cases_df = pd.DataFrame(all_cases)
            cases_csv_path = os.path.join(self.results_dir, 'case_analysis.csv')
            cases_df.to_csv(cases_csv_path, index=False)
            
            print(f"📊 Case analysis saved:")
            print(f"  - JSON: {analysis_path}")
            print(f"  - CSV: {cases_csv_path}")
            
            # Print summary
            print(f"\n📈 Case Analysis Summary:")
            for case_name, case_data in analysis_results.items():
                print(f"  - {case_data['description']}: {len(case_data['samples'])} samples")
                if case_data['samples']:
                    probs = [s['predicted_probability'] for s in case_data['samples']]
                    print(f"    Probability range: {min(probs):.4f} - {max(probs):.4f}")
        else:
            print(f"⚠️  No samples found for case analysis")

    def run_comprehensive_test(self):
        """Run comprehensive test (config.json based)"""
        print(f"🚀 Starting comprehensive test")
        
        # 1. Load model
        model = self.load_model()
        
        # 2. Setup test dataset
        test_loader = self.setup_test_dataset()
        
        # 3. Run predictions
        probs, preds, labels, product_ids = self.predict_all(model, test_loader)
        
        # 4. Calculate performance metrics
        metrics = self.compute_metrics(probs, preds, labels)
        
        # 5. Create visualizations
        self.create_visualizations(metrics, probs, labels)
        
        # 6. Save detailed results
        self.save_detailed_results(metrics, probs, preds, labels, product_ids)
        
        print(f"\n✅ Comprehensive test completed!")
        print(f"📁 Results saved to: {self.results_dir}")
        
        return metrics

    def quick_test(self, num_samples: int = 10):
        """Quick test (few samples only)"""
        print(f"\n⚡ Quick test ({num_samples} samples)")
        
        # Load model
        model = self.load_model()
        
        # Read settings from config
        data_config = self.config["data"]
        test_size = 1 - data_config["train_test_split"]
        
        # Load data with processor (test files only)
        processor = MorigirlDataProcessor(self.data_path)
        if not processor.load_npy_files(split_type="test"):
            raise RuntimeError("Failed to load test data.")
        
        test_dataset = MorigirlDataset(
            processor.vectors, 
            processor.labels, 
            processor.product_ids
        )
        
        # Select random samples
        indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
        
        print(f"\n🔮 Sample prediction results:")
        print(f"{'Index':<8} {'Actual':<8} {'Pred':<8} {'Prob':<10} {'Product ID':<12} {'Correct'}")
        print("-" * 70)
        
        correct = 0
        for i, idx in enumerate(indices):
            sample = test_dataset[idx]
            vector = sample['vector'].unsqueeze(0).to(self.device)
            label = sample['label']
            product_id = sample['product_id']
            
            with torch.no_grad():
                prob = model(vector).item()
                pred = int(prob > 0.5)
            
            is_correct = pred == int(label.item())
            correct += is_correct
            
            print(f"{idx:<8} {int(label.item()):<8} {pred:<8} {prob:<10.4f} {product_id:<12} {'✓' if is_correct else '✗'}")
        
        accuracy = correct / len(indices)
        print(f"\n📊 Quick test accuracy: {accuracy:.4f} ({correct}/{len(indices)})")

def main():
    parser = argparse.ArgumentParser(description='Morigirl vector classification model test')
    parser.add_argument('--checkpoint', default=None, help='Model checkpoint path (auto-find from config possible)')
    parser.add_argument('--config-path', default='config.json', help='Configuration file path')
    parser.add_argument('--data-path', default=None, help='Test data path (config file priority)')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test only')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples for quick test')
    
    args = parser.parse_args()
    
    try:
        # Create tester (including checkpoint auto-search)
        tester = MoriGirlModelTester(
            checkpoint_path=args.checkpoint,
            config_path=args.config_path,
            data_path=args.data_path
        )
        
        if args.quick_test:
            # Quick test
            tester.quick_test(args.num_samples)
        else:
            # Comprehensive test
            tester.run_comprehensive_test()
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 