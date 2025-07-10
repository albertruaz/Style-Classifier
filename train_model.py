#!/usr/bin/env python3
# train_model.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report

# 로컬 모듈
from prepare_training_data import MorigirlDataProcessor, MorigirlDataset
from model.morigirl_model import MoriGirlVectorClassifier, get_model_info

class MoriGirlTrainer:
    """모리걸 벡터 분류 모델 학습 클래스"""
    
    def __init__(self, 
                 data_path: str = "data/morigirl_50",
                 experiment_name: str = None):
        
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 실험 이름 설정 (월일시분_랜덤2자리)
        if experiment_name is None:
            import random
            date_str = datetime.now().strftime('%m%d%H%M')  # 월일시분
            random_num = random.randint(10, 99)  # 랜덤 2자리
            self.experiment_name = f"{date_str}_{random_num:02d}"
        else:
            self.experiment_name = experiment_name
        
        # result 폴더에 실험별 디렉토리 생성
        self.result_dir = f"result/{self.experiment_name}"
        self.checkpoint_dir = f"{self.result_dir}/checkpoints"
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"🚀 모리걸 모델 학습 시작")
        print(f"  - 실험명: {self.experiment_name}")
        print(f"  - 데이터 경로: {self.data_path}")
        print(f"  - 결과 폴더: {self.result_dir}")
        print(f"  - 디바이스: {self.device}")

    def setup_datasets(self, 
                      test_size: float = 0.2,
                      val_size: float = 0.1,
                      batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터셋 설정"""
        print(f"\n📊 데이터셋 설정")
        
        # 데이터 처리기 생성 및 로딩
        processor = MorigirlDataProcessor(self.data_path)
        if not processor.load_npy_files():
            raise RuntimeError("데이터 로딩에 실패했습니다.")
        
        # Train/Test 분할
        train_dataset, test_dataset = processor.create_train_test_split(
            test_size=test_size, random_state=42
        )
        
        # Train에서 Validation 분할
        val_size_adjusted = val_size / (1 - test_size)  # 전체 데이터 기준으로 조정
        train_vectors = train_dataset.vectors.numpy()
        train_labels = train_dataset.labels.numpy()
        train_ids = train_dataset.product_ids
        
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            train_vectors, train_labels, train_ids,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=train_labels
        )
        
        # 새로운 Dataset 객체 생성
        train_dataset = MorigirlDataset(X_train, y_train, ids_train)
        val_dataset = MorigirlDataset(X_val, y_val, ids_val)
        
        # 클래스 가중치 계산 (불균형 해결)
        pos_count = torch.sum(train_dataset.labels).item()
        neg_count = len(train_dataset) - pos_count
        self.pos_weight = torch.tensor(neg_count / pos_count).to(self.device)
        print(f"📊 클래스 가중치: {self.pos_weight.item():.2f}")
        
        # 데이터로더 생성
        train_loader, _ = processor.create_dataloaders(train_dataset, val_dataset, batch_size)
        _, val_loader = processor.create_dataloaders(val_dataset, test_dataset, batch_size)
        _, test_loader = processor.create_dataloaders(test_dataset, test_dataset, batch_size)
        
        return train_loader, val_loader, test_loader

    def setup_model(self, **model_kwargs) -> nn.Module:
        """모델 설정"""
        print(f"\n🧠 모델 설정")
        
        model = MoriGirlVectorClassifier(**model_kwargs)
        model.to(self.device)
        
        # 모델 정보 출력
        get_model_info(model)
        
        return model

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer) -> Dict[str, float]:
        """한 에포크 학습"""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        progress_bar = tqdm(train_loader, desc="학습")
        
        for batch_idx, batch in enumerate(progress_bar):
            vectors = batch['vector'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)  # (batch_size, 1)
            
            # Forward pass
            outputs = model(vectors)  # 이미 sigmoid 적용됨
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 메트릭 수집
            total_loss += loss.item()
            probs = outputs.detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Progress bar 업데이트
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        # 최종 메트릭 계산
        accuracy = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy,
            'auc': auc
        }

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module) -> Dict[str, float]:
        """한 에포크 검증"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="검증"):
                vectors = batch['vector'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)
                
                outputs = model(vectors)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # 메트릭 계산
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'experiment_name': self.experiment_name
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"💾 최고 성능 모델 저장: {best_path}")

    def train(self, 
              epochs: int = 50,
              learning_rate: float = 0.001,
              weight_decay: float = 0.01,
              patience: int = 10,
              **model_kwargs):
        """모델 학습"""
        
        # 데이터셋 설정
        train_loader, val_loader, test_loader = self.setup_datasets()
        
        # 모델 설정
        model = self.setup_model(**model_kwargs)
        
        # 손실함수 및 옵티마이저 설정
        criterion = nn.BCELoss(weight=self.pos_weight if self.pos_weight.item() > 1 else None)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        
        # 학습 기록
        best_val_acc = 0.0
        patience_counter = 0
        train_history = []
        
        print(f"\n🎯 학습 시작 (총 {epochs} 에포크)")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # 학습
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # 검증
            val_metrics = self.validate_epoch(model, val_loader, criterion)
            
            # 스케줄러 업데이트
            scheduler.step(val_metrics['accuracy'])
            
            # 결과 출력
            print(f"학습 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
            print(f"검증 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # 최고 성능 확인
            is_best = val_metrics['accuracy'] > best_val_acc
            if is_best:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 체크포인트 저장
            self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)
            
            # 기록 저장
            train_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Early stopping
            if patience_counter >= patience:
                print(f"⏰ Early stopping at epoch {epoch+1} (patience: {patience})")
                break
        
        # 최고 모델로 테스트
        print(f"\n🎯 최고 성능 모델로 테스트 평가")
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        self.evaluate_model(test_loader, best_model_path)
        
        # 학습 기록 저장
        history_path = os.path.join(self.result_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)
        
        print(f"✅ 학습 완료! 결과 폴더: {self.result_dir}")

    def evaluate_model(self, test_loader: DataLoader, model_path: str):
        """모델 평가"""
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = MoriGirlVectorClassifier()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="테스트 평가"):
                vectors = batch['vector'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(vectors)
                probs = outputs.cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # 최종 성능 출력
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )
        auc = roc_auc_score(all_labels, all_probs)
        
        print(f"\n📊 최종 테스트 성능:")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - 정밀도: {precision:.4f}")
        print(f"  - 재현율: {recall:.4f}")
        print(f"  - F1 점수: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        # 분류 리포트
        print(f"\n📋 분류 리포트:")
        print(classification_report(all_labels, all_preds, target_names=['비모리걸', '모리걸']))

def main():
    parser = argparse.ArgumentParser(description='모리걸 벡터 분류 모델 학습')
    parser.add_argument('--data-path', default='data/morigirl_50', help='학습 데이터 경로 (예: data/morigirl_50)')
    parser.add_argument('--epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--batch-size', type=int, default=32, help='배치 크기')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='학습률')
    parser.add_argument('--experiment-name', help='실험 이름')
    
    args = parser.parse_args()
    
    # 학습 시작
    trainer = MoriGirlTrainer(
        data_path=args.data_path,
        experiment_name=args.experiment_name
    )
    
    trainer.train(
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main() 