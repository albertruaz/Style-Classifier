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

# 로컬 모듈
from prepare_training_data import MorigirlDataProcessor, MorigirlDataset
from model.morigirl_model import MoriGirlVectorClassifier

class MoriGirlModelTester:
    """학습된 모리걸 벡터 분류 모델 테스트 클래스"""
    
    def __init__(self, checkpoint_path: str, data_path: str = "data/morigirl_50"):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 결과 저장 디렉토리 (월일시분_랜덤2자리)
        import random
        date_str = datetime.now().strftime('%m%d%H%M')  # 월일시분
        random_num = random.randint(10, 99)  # 랜덤 2자리
        result_name = f"{date_str}_{random_num:02d}"
        self.results_dir = f"result/{result_name}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"🧪 모리걸 모델 테스트 시작")
        print(f"  - 체크포인트: {checkpoint_path}")
        print(f"  - 데이터 경로: {data_path}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - 결과 저장: {self.results_dir}")

    def load_model(self) -> nn.Module:
        """체크포인트에서 모델 로드"""
        print(f"\n📦 모델 로드 중...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 모델 생성
        model = MoriGirlVectorClassifier()
        
        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ 모델 로드 완료")
        print(f"  - 에포크: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - 검증 정확도: {checkpoint.get('metrics', {}).get('accuracy', 'N/A')}")
        
        return model

    def setup_test_dataset(self, batch_size: int = 32) -> DataLoader:
        """테스트 데이터셋 설정"""
        print(f"\n📊 테스트 데이터셋 설정")
        
        # 데이터 처리기 생성 및 로딩
        processor = MorigirlDataProcessor(self.data_path)
        if not processor.load_npy_files():
            raise RuntimeError("데이터 로딩에 실패했습니다.")
        
        # Train/Test 분할 (테스트 셋만 사용)
        _, test_dataset = processor.create_train_test_split(
            test_size=0.2, random_state=42
        )
        
        # 데이터로더 생성
        _, test_loader = processor.create_dataloaders(test_dataset, test_dataset, batch_size)
        
        return test_loader

    def predict_all(self, model: nn.Module, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """전체 테스트 셋에 대해 예측 수행"""
        print(f"\n🔮 모델 예측 수행")
        
        all_probs = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="예측 중"):
                vectors = batch['vector'].to(self.device)
                labels = batch['label']
                
                # 예측 수행
                outputs = model(vectors)  # 이미 sigmoid 적용됨
                probs = outputs.cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        return np.array(all_probs), np.array(all_preds), np.array(all_labels)

    def compute_metrics(self, probs: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """성능 메트릭 계산"""
        print(f"\n📊 성능 메트릭 계산")
        
        # 기본 메트릭
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', zero_division=0
        )
        
        # AUC
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.0
        
        # 혼동 행렬
        cm = confusion_matrix(labels, preds)
        
        # ROC 곡선 데이터
        fpr, tpr, _ = roc_curve(labels, probs)
        
        # Precision-Recall 곡선 데이터
        pr_precision, pr_recall, _ = precision_recall_curve(labels, probs)
        
        # 클래스별 분류 리포트
        report = classification_report(labels, preds, target_names=['비모리걸', '모리걸'], output_dict=True)
        
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
        
        # 결과 출력
        print(f"✅ 성능 결과:")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - 정밀도: {precision:.4f}")
        print(f"  - 재현율: {recall:.4f}")
        print(f"  - F1 점수: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        return metrics

    def create_visualizations(self, metrics: Dict[str, Any], probs: np.ndarray, labels: np.ndarray):
        """시각화 생성"""
        print(f"\n📈 시각화 생성")
        
        # 한글 폰트 설정 (시스템에 따라 조정 필요)
        plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('모리걸 분류 모델 테스트 결과', fontsize=16)
        
        # 1. 혼동 행렬
        ax1 = axes[0, 0]
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['비모리걸', '모리걸'], yticklabels=['비모리걸', '모리걸'])
        ax1.set_title('혼동 행렬')
        ax1.set_xlabel('예측')
        ax1.set_ylabel('실제')
        
        # 2. ROC 곡선
        ax2 = axes[0, 1]
        fpr, tpr = metrics['roc_curve']
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {metrics["auc"]:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC 곡선')
        ax2.legend(loc="lower right")
        ax2.grid(True)
        
        # 3. Precision-Recall 곡선
        ax3 = axes[0, 2]
        precision, recall = metrics['pr_curve']
        ax3.plot(recall, precision, color='blue', lw=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall 곡선')
        ax3.grid(True)
        
        # 4. 확률 분포
        ax4 = axes[1, 0]
        morigirl_probs = probs[labels == 1]
        non_morigirl_probs = probs[labels == 0]
        
        ax4.hist(non_morigirl_probs, bins=30, alpha=0.7, label='비모리걸', color='red', density=True)
        ax4.hist(morigirl_probs, bins=30, alpha=0.7, label='모리걸', color='blue', density=True)
        ax4.axvline(x=0.5, color='black', linestyle='--', label='임계값 (0.5)')
        ax4.set_xlabel('예측 확률')
        ax4.set_ylabel('밀도')
        ax4.set_title('클래스별 확률 분포')
        ax4.legend()
        ax4.grid(True)
        
        # 5. 성능 메트릭 바 차트
        ax5 = axes[1, 1]
        metric_names = ['정확도', '정밀도', '재현율', 'F1 점수', 'AUC']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1'], metrics['auc']]
        
        bars = ax5.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
        ax5.set_ylim(0, 1.0)
        ax5.set_title('성능 메트릭')
        ax5.set_ylabel('점수')
        
        # 바 위에 수치 표시
        for bar, value in zip(bars, metric_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 6. 클래스별 성능
        ax6 = axes[1, 2]
        report = metrics['classification_report']
        
        class_names = ['비모리걸', '모리걸']
        precision_scores = [report['0']['precision'], report['1']['precision']]
        recall_scores = [report['0']['recall'], report['1']['recall']]
        f1_scores = [report['0']['f1-score'], report['1']['f1-score']]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax6.bar(x - width, precision_scores, width, label='정밀도', alpha=0.8)
        ax6.bar(x, recall_scores, width, label='재현율', alpha=0.8)
        ax6.bar(x + width, f1_scores, width, label='F1 점수', alpha=0.8)
        
        ax6.set_xlabel('클래스')
        ax6.set_ylabel('점수')
        ax6.set_title('클래스별 성능')
        ax6.set_xticks(x)
        ax6.set_xticklabels(class_names)
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        
        # 저장
        save_path = os.path.join(self.results_dir, 'test_results_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 시각화 저장: {save_path}")
        
        # 보여주기 (선택사항)
        # plt.show()
        plt.close()

    def save_detailed_results(self, metrics: Dict[str, Any], probs: np.ndarray, 
                            preds: np.ndarray, labels: np.ndarray):
        """상세 결과 저장"""
        print(f"\n💾 상세 결과 저장")
        
        # 1. 메트릭 JSON 저장
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
        
        # 2. 예측 결과 CSV 저장
        results_df = pd.DataFrame({
            'actual_label': labels,
            'predicted_label': preds,
            'predicted_probability': probs,
            'correct': labels == preds
        })
        
        csv_path = os.path.join(self.results_dir, 'predictions.csv')
        results_df.to_csv(csv_path, index=False)
        
        # 3. 분류 리포트 텍스트 저장
        report_text = classification_report(labels, preds, target_names=['비모리걸', '모리걸'])
        report_path = os.path.join(self.results_dir, 'classification_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"💾 결과 저장 완료:")
        print(f"  - 메트릭: {metrics_path}")
        print(f"  - 예측 결과: {csv_path}")
        print(f"  - 분류 리포트: {report_path}")

    def run_comprehensive_test(self, batch_size: int = 32):
        """종합 테스트 수행"""
        print(f"🚀 종합 테스트 시작")
        
        # 1. 모델 로드
        model = self.load_model()
        
        # 2. 테스트 데이터셋 설정
        test_loader = self.setup_test_dataset(batch_size)
        
        # 3. 예측 수행
        probs, preds, labels = self.predict_all(model, test_loader)
        
        # 4. 성능 메트릭 계산
        metrics = self.compute_metrics(probs, preds, labels)
        
        # 5. 시각화 생성
        self.create_visualizations(metrics, probs, labels)
        
        # 6. 상세 결과 저장
        self.save_detailed_results(metrics, probs, preds, labels)
        
        print(f"\n✅ 종합 테스트 완료!")
        print(f"📁 결과 저장 위치: {self.results_dir}")
        
        return metrics

    def quick_test(self, num_samples: int = 10):
        """빠른 테스트 (몇 개 샘플만)"""
        print(f"\n⚡ 빠른 테스트 ({num_samples}개 샘플)")
        
        # 모델 로드
        model = self.load_model()
        
        # 데이터 처리기로 데이터 로드
        processor = MorigirlDataProcessor(self.data_path)
        if not processor.load_npy_files():
            raise RuntimeError("데이터 로딩에 실패했습니다.")
        
        _, test_dataset = processor.create_train_test_split(test_size=0.2, random_state=42)
        
        # 랜덤 샘플 선택
        indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)
        
        print(f"\n🔮 샘플 예측 결과:")
        print(f"{'인덱스':<8} {'실제':<8} {'예측':<8} {'확률':<10} {'상품ID':<12} {'정답'}")
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
        print(f"\n📊 빠른 테스트 정확도: {accuracy:.4f} ({correct}/{len(indices)})")

def main():
    parser = argparse.ArgumentParser(description='모리걸 벡터 분류 모델 테스트')
    parser.add_argument('--checkpoint', required=True, help='모델 체크포인트 경로')
    parser.add_argument('--data-path', default='data/morigirl_50', help='테스트 데이터 경로 (예: data/morigirl_50)')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    parser.add_argument('--quick-test', action='store_true', help='빠른 테스트만 수행')
    parser.add_argument('--num-samples', type=int, default=10, help='빠른 테스트 샘플 수')
    
    args = parser.parse_args()
    
    # 체크포인트 파일 존재 확인
    if not os.path.exists(args.checkpoint):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        return
    
    # 테스터 생성
    tester = MoriGirlModelTester(args.checkpoint, args.data_path)
    
    if args.quick_test:
        # 빠른 테스트
        tester.quick_test(args.num_samples)
    else:
        # 종합 테스트
        tester.run_comprehensive_test(args.batch_size)

if __name__ == "__main__":
    main() 