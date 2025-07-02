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
    precision_recall_curve, roc_curve, mean_squared_error, 
    mean_absolute_error, r2_score
)
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Any
from datetime import datetime

# 로컬 모듈
from database import DatabaseManager
from dataset.morigirl_dataset import MorigirlDataset
from dataset.product_score_dataset import ProductScoreDataset
from model.morigirl_model import MorigirlModel
from model.score_prediction_model import ScorePredictionModel
from utils.train_utils import load_checkpoint

class ModelTester:
    """학습된 모델 테스트 클래스"""
    
    def __init__(self, config_path: str = "./config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ 테스트 환경: {self.device}")
        
        # 결과 저장 디렉토리
        self.results_dir = f"./results/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"📁 결과 저장 디렉토리: {self.results_dir}")

    def load_model(self, checkpoint_path: str, task_type: str) -> nn.Module:
        """체크포인트에서 모델 로드"""
        print(f"📦 모델 로딩: {checkpoint_path}")
        
        checkpoint = load_checkpoint(checkpoint_path)
        
        # 모델 생성
        if task_type == "morigirl":
            model = MorigirlModel(
                input_dim=1024,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 512),
                num_classes=2,
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.3)
            )
        elif task_type == "score":
            model = ScorePredictionModel(
                image_dim=1024,
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 512),
                dropout_rate=self.config.get('model', {}).get('dropout_rate', 0.3)
            )
        else:
            raise ValueError(f"지원하지 않는 태스크 타입: {task_type}")
        
        # 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"✅ 모델 로드 완료 (Epoch: {checkpoint.get('epoch', 'N/A')})")
        return model

    def setup_dataset(self, task_type: str, mode: str = "test") -> DataLoader:
        """테스트 데이터셋 설정"""
        print(f"📊 테스트 데이터셋 설정 중... (Task: {task_type})")
        
        if task_type == "morigirl":
            dataset = MorigirlDataset(
                config=self.config,
                mode=mode
            )
        elif task_type == "score":
            dataset = ProductScoreDataset(
                config=self.config,
                mode=mode
            )
        else:
            raise ValueError(f"지원하지 않는 태스크 타입: {task_type}")
        
        # 전체 데이터셋을 테스트용으로 사용 (실제로는 학습/검증/테스트 분할 필요)
        test_loader = DataLoader(
            dataset,
            batch_size=self.config.get('training', {}).get('batch_size', 32),
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ 테스트 데이터: {len(dataset):,}개")
        return test_loader

    def predict_batch(self, model: nn.Module, data_loader: DataLoader, 
                     task_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """배치 예측"""
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="예측 중"):
                if task_type == "morigirl":
                    image_vectors, labels = batch
                    image_vectors = image_vectors.to(self.device)
                    
                    outputs = model(image_vectors)
                    probs = torch.softmax(outputs, dim=1)
                    
                    predictions.extend(probs.cpu().numpy())
                    targets.extend(labels.numpy())
                    
                elif task_type == "score":
                    image_vectors, scores = batch
                    image_vectors = image_vectors.to(self.device)
                    
                    outputs = model(image_vectors)
                    
                    predictions.extend(outputs.cpu().numpy().flatten())
                    targets.extend(scores.numpy().flatten())
        
        return np.array(predictions), np.array(targets)

    def evaluate_classification(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """분류 모델 평가"""
        print(f"🎯 분류 성능 평가 중...")
        
        # 예측 클래스 (확률 -> 클래스)
        pred_classes = np.argmax(predictions, axis=1)
        pred_probs = predictions[:, 1]  # 모리걸 확률
        
        # 기본 메트릭
        accuracy = np.mean(pred_classes == targets)
        
        # 분류 리포트
        report = classification_report(targets, pred_classes, output_dict=True)
        
        # 혼동 행렬
        cm = confusion_matrix(targets, pred_classes)
        
        # ROC AUC
        try:
            auc = roc_auc_score(targets, pred_probs)
        except:
            auc = None
        
        # ROC 곡선
        fpr, tpr, _ = roc_curve(targets, pred_probs)
        
        # Precision-Recall 곡선
        precision, recall, _ = precision_recall_curve(targets, pred_probs)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'auc': auc,
            'roc_curve': (fpr, tpr),
            'precision_recall_curve': (precision, recall)
        }
        
        # 결과 출력
        print(f"  📊 정확도: {accuracy:.4f}")
        if auc:
            print(f"  📊 AUC: {auc:.4f}")
        print(f"  📊 정밀도 (Class 1): {report['1']['precision']:.4f}")
        print(f"  📊 재현율 (Class 1): {report['1']['recall']:.4f}")
        print(f"  📊 F1-Score (Class 1): {report['1']['f1-score']:.4f}")
        
        return results

    def evaluate_regression(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """회귀 모델 평가"""
        print(f"🎯 회귀 성능 평가 중...")
        
        # 기본 메트릭
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        
        # 오차 분석
        errors = predictions - targets
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        
        results = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'error_mean': error_mean,
            'error_std': error_std,
            'predictions': predictions,
            'targets': targets,
            'errors': errors
        }
        
        # 결과 출력
        print(f"  📊 MSE: {mse:.4f}")
        print(f"  📊 MAE: {mae:.4f}")
        print(f"  📊 RMSE: {rmse:.4f}")
        print(f"  📊 R²: {r2:.4f}")
        print(f"  📊 오차 평균: {error_mean:.4f}")
        print(f"  📊 오차 표준편차: {error_std:.4f}")
        
        return results

    def visualize_classification_results(self, results: Dict[str, Any], save_prefix: str):
        """분류 결과 시각화"""
        print("📈 분류 결과 시각화 중...")
        
        # 1. 혼동 행렬
        plt.figure(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{save_prefix}_confusion_matrix.png'), dpi=300)
        plt.close()
        
        # 2. ROC 곡선
        if results['auc']:
            plt.figure(figsize=(8, 6))
            fpr, tpr = results['roc_curve']
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["auc"]:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'{save_prefix}_roc_curve.png'), dpi=300)
            plt.close()
        
        # 3. Precision-Recall 곡선
        plt.figure(figsize=(8, 6))
        precision, recall = results['precision_recall_curve']
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{save_prefix}_pr_curve.png'), dpi=300)
        plt.close()

    def visualize_regression_results(self, results: Dict[str, Any], save_prefix: str):
        """회귀 결과 시각화"""
        print("📈 회귀 결과 시각화 중...")
        
        predictions = results['predictions']
        targets = results['targets']
        errors = results['errors']
        
        # 1. 예측 vs 실제값 산점도
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.scatter(targets, predictions, alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual (R² = {results["r2"]:.3f})')
        plt.grid(True)
        
        # 2. 오차 히스토그램
        plt.subplot(1, 2, 2)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution (μ = {results["error_mean"]:.3f})')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{save_prefix}_regression_analysis.png'), dpi=300)
        plt.close()
        
        # 3. 잔차 플롯
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, errors, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{save_prefix}_residual_plot.png'), dpi=300)
        plt.close()

    def save_detailed_results(self, results: Dict[str, Any], task_type: str, save_prefix: str):
        """상세 결과 저장"""
        print("💾 결과 저장 중...")
        
        # JSON으로 메트릭 저장
        metrics = {}
        
        if task_type == "morigirl":
            metrics = {
                'accuracy': float(results['accuracy']),
                'auc': float(results['auc']) if results['auc'] else None,
                'classification_report': results['classification_report']
            }
        elif task_type == "score":
            metrics = {
                'mse': float(results['mse']),
                'mae': float(results['mae']),
                'rmse': float(results['rmse']),
                'r2': float(results['r2']),
                'error_mean': float(results['error_mean']),
                'error_std': float(results['error_std'])
            }
        
        # 메트릭 저장
        with open(os.path.join(self.results_dir, f'{save_prefix}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        # 예측 결과 저장
        if task_type == "score":
            results_df = pd.DataFrame({
                'actual': results['targets'],
                'predicted': results['predictions'],
                'error': results['errors']
            })
            results_df.to_csv(os.path.join(self.results_dir, f'{save_prefix}_predictions.csv'), index=False)

    def run_comprehensive_test(self, checkpoint_path: str, task_type: str):
        """종합 테스트 실행"""
        print(f"🚀 종합 테스트 시작 (Task: {task_type})")
        
        # 모델 로드
        model = self.load_model(checkpoint_path, task_type)
        
        # 데이터셋 설정
        test_loader = self.setup_dataset(task_type, mode="test")
        
        # 예측 수행
        predictions, targets = self.predict_batch(model, test_loader, task_type)
        
        # 성능 평가
        if task_type == "morigirl":
            results = self.evaluate_classification(predictions, targets)
            self.visualize_classification_results(results, "classification")
        elif task_type == "score":
            results = self.evaluate_regression(predictions, targets)
            self.visualize_regression_results(results, "regression")
        
        # 결과 저장
        self.save_detailed_results(results, task_type, task_type)
        
        print(f"🎉 테스트 완료! 결과 저장 위치: {self.results_dir}")
        
        return results

    def run_single_inference(self, checkpoint_path: str, task_type: str, 
                           product_ids: List[int] = None):
        """개별 상품 추론 테스트"""
        print(f"🔍 개별 상품 추론 테스트 (Task: {task_type})")
        
        # 모델 로드
        model = self.load_model(checkpoint_path, task_type)
        
        # 데이터베이스에서 상품 정보 가져오기
        db_manager = DatabaseManager()
        
        try:
            if not product_ids:
                # 랜덤하게 10개 상품 선택
                session = db_manager.mysql.Session()
                from sqlalchemy import text
                
                sql = text("""
                    SELECT p.id, p.name, p.main_image
                    FROM vingle.product p
                    JOIN product_vectors pv ON p.id = pv.id
                    WHERE p.status = 'SALE'
                    ORDER BY RAND()
                    LIMIT 10
                """)
                
                result = session.execute(sql)
                product_ids = [row[0] for row in result.fetchall()]
                session.close()
            
            print(f"📦 테스트 상품: {len(product_ids)}개")
            
            # 벡터 조회
            vectors_data = db_manager.vector_db.query_product_vectors(product_ids)
            
            results = []
            
            for product_id in product_ids:
                if product_id not in vectors_data:
                    print(f"⚠️ 상품 {product_id}: 벡터 없음")
                    continue
                
                # 벡터를 텐서로 변환
                vector = torch.FloatTensor(vectors_data[product_id]).unsqueeze(0).to(self.device)
                
                # 추론
                with torch.no_grad():
                    output = model(vector)
                    
                    if task_type == "morigirl":
                        probs = torch.softmax(output, dim=1)
                        morigirl_prob = probs[0][1].item()
                        prediction = "모리걸" if morigirl_prob > 0.5 else "일반"
                        confidence = morigirl_prob if morigirl_prob > 0.5 else 1 - morigirl_prob
                        
                        result = {
                            'product_id': product_id,
                            'prediction': prediction,
                            'morigirl_probability': morigirl_prob,
                            'confidence': confidence
                        }
                        
                    elif task_type == "score":
                        score = output[0].item()
                        result = {
                            'product_id': product_id,
                            'predicted_score': score
                        }
                    
                    results.append(result)
                    print(f"  상품 {product_id}: {result}")
            
            # 결과 저장
            results_df = pd.DataFrame(results)
            results_df.to_csv(os.path.join(self.results_dir, 'single_inference_results.csv'), index=False)
            
            return results
            
        finally:
            db_manager.dispose_all()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='학습된 모델 테스트')
    parser.add_argument('--checkpoint', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--task', choices=['morigirl', 'score'], required=True, 
                       help='테스트할 태스크')
    parser.add_argument('--config', type=str, default='./config.json', help='설정 파일 경로')
    parser.add_argument('--mode', choices=['comprehensive', 'single'], default='comprehensive',
                       help='테스트 모드 (comprehensive: 전체 평가, single: 개별 상품)')
    parser.add_argument('--product-ids', nargs='+', type=int, help='테스트할 상품 ID들 (single 모드)')
    
    args = parser.parse_args()
    
    try:
        tester = ModelTester(args.config)
        
        if args.mode == 'comprehensive':
            results = tester.run_comprehensive_test(args.checkpoint, args.task)
            print(f"\n✅ 종합 테스트 완료!")
            
        elif args.mode == 'single':
            results = tester.run_single_inference(args.checkpoint, args.task, args.product_ids)
            print(f"\n✅ 개별 추론 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        raise

if __name__ == "__main__":
    main() 