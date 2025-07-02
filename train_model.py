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
import wandb  # 학습 모니터링용 (선택사항)
from tqdm import tqdm
from typing import Dict, Tuple, List
import argparse

# 로컬 모듈
from database import DatabaseManager
from dataset.morigirl_dataset import MorigirlDataset
from dataset.product_score_dataset import ProductScoreDataset
from model.morigirl_model import MorigirlModel
from model.score_prediction_model import ScorePredictionModel
from utils.train_utils import EarlyStopping, compute_metrics, save_checkpoint, load_checkpoint

class ModelTrainer:
    """모델 학습 클래스"""
    
    def __init__(self, config_path: str = "./config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✅ 학습 환경: {self.device}")
        
        # 실험 ID 생성
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 체크포인트 디렉토리 생성
        self.checkpoint_dir = f"./checkpoints/{self.experiment_id}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        print(f"📁 체크포인트 디렉토리: {self.checkpoint_dir}")

    def setup_datasets(self, task_type: str, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터셋 설정"""
        print(f"📊 데이터셋 설정 중... (Task: {task_type})")
        
        if task_type == "morigirl":
            # 모리걸 분류 데이터셋
            dataset = MorigirlDataset(
                config=self.config,
                mode="train"
            )
            
            # 학습/검증/테스트 분할
            total_size = len(dataset)
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * 0.1)  # 10% 검증
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
            
        elif task_type == "score":
            # 인기도 점수 예측 데이터셋
            dataset = ProductScoreDataset(
                config=self.config,
                mode="train"
            )
            
            # 학습/검증/테스트 분할
            total_size = len(dataset)
            train_size = int(total_size * train_ratio)
            val_size = int(total_size * 0.1)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
        else:
            raise ValueError(f"지원하지 않는 태스크 타입: {task_type}")
        
        # 데이터로더 생성
        batch_size = self.config.get('training', {}).get('batch_size', 32)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ 데이터셋 분할 완료:")
        print(f"  - 학습: {len(train_dataset):,}개")
        print(f"  - 검증: {len(val_dataset):,}개")
        print(f"  - 테스트: {len(test_dataset):,}개")
        
        return train_loader, val_loader, test_loader

    def setup_model(self, task_type: str) -> nn.Module:
        """모델 설정"""
        print(f"🧠 모델 설정 중... (Task: {task_type})")
        
        if task_type == "morigirl":
            model = MorigirlModel(
                input_dim=1024,  # 이미지 벡터 차원
                hidden_dim=self.config.get('model', {}).get('hidden_dim', 512),
                num_classes=2,   # 모리걸 vs 일반
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
        
        model.to(self.device)
        
        # 모델 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ 모델 로드 완료:")
        print(f"  - 총 파라미터: {total_params:,}개")
        print(f"  - 학습 파라미터: {trainable_params:,}개")
        
        return model

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer, 
                   task_type: str) -> Dict[str, float]:
        """한 에포크 학습"""
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="학습 중")
        
        for batch_idx, batch in enumerate(progress_bar):
            if task_type == "morigirl":
                image_vectors, labels = batch
                image_vectors = image_vectors.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = model(image_vectors)
                loss = criterion(outputs, labels)
                
                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == labels).sum().item()
                
            elif task_type == "score":
                image_vectors, scores = batch
                image_vectors = image_vectors.to(self.device)
                scores = scores.to(self.device)
                
                # Forward pass
                outputs = model(image_vectors)
                loss = criterion(outputs.squeeze(), scores)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (선택사항)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_predictions += labels.size(0) if task_type == "morigirl" else scores.size(0)
            
            # Progress bar 업데이트
            avg_loss = total_loss / (batch_idx + 1)
            if task_type == "morigirl":
                accuracy = correct_predictions / total_predictions
                progress_bar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.4f}'
                })
            else:
                progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        
        metrics = {'train_loss': total_loss / len(train_loader)}
        if task_type == "morigirl":
            metrics['train_accuracy'] = correct_predictions / total_predictions
        
        return metrics

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, task_type: str) -> Dict[str, float]:
        """한 에포크 검증"""
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="검증 중"):
                if task_type == "morigirl":
                    image_vectors, labels = batch
                    image_vectors = image_vectors.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = model(image_vectors)
                    loss = criterion(outputs, labels)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    correct_predictions += (predicted == labels).sum().item()
                    
                elif task_type == "score":
                    image_vectors, scores = batch
                    image_vectors = image_vectors.to(self.device)
                    scores = scores.to(self.device)
                    
                    outputs = model(image_vectors)
                    loss = criterion(outputs.squeeze(), scores)
                
                total_loss += loss.item()
                total_predictions += labels.size(0) if task_type == "morigirl" else scores.size(0)
        
        metrics = {'val_loss': total_loss / len(val_loader)}
        if task_type == "morigirl":
            metrics['val_accuracy'] = correct_predictions / total_predictions
        
        return metrics

    def train_model(self, task_type: str, epochs: int = 100, use_wandb: bool = False):
        """모델 학습 메인 함수"""
        print(f"🚀 모델 학습 시작 (Task: {task_type})")
        
        # WandB 초기화 (선택사항)
        if use_wandb:
            wandb.init(
                project="mori-look",
                name=f"{task_type}_{self.experiment_id}",
                config=self.config
            )
        
        # 데이터셋 설정
        train_loader, val_loader, test_loader = self.setup_datasets(task_type)
        
        # 모델 설정
        model = self.setup_model(task_type)
        
        # 손실 함수 및 옵티마이저 설정
        if task_type == "morigirl":
            criterion = nn.CrossEntropyLoss()
            monitor_metric = 'val_accuracy'
            mode = 'max'
        else:  # score
            criterion = nn.MSELoss()
            monitor_metric = 'val_loss'
            mode = 'min'
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.get('training', {}).get('learning_rate', 1e-3),
            weight_decay=self.config.get('training', {}).get('weight_decay', 1e-4)
        )
        
        # 학습률 스케줄러
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.5, patience=5, verbose=True
        )
        
        # Early Stopping
        early_stopping = EarlyStopping(
            patience=10,
            mode=mode,
            min_delta=1e-4
        )
        
        # 학습 루프
        best_metric = float('-inf') if mode == 'max' else float('inf')
        
        for epoch in range(epochs):
            print(f"\n📅 Epoch {epoch+1}/{epochs}")
            
            # 학습
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, task_type)
            
            # 검증
            val_metrics = self.validate_epoch(model, val_loader, criterion, task_type)
            
            # 메트릭 출력
            print(f"학습 손실: {train_metrics['train_loss']:.4f}")
            print(f"검증 손실: {val_metrics['val_loss']:.4f}")
            
            if task_type == "morigirl":
                print(f"학습 정확도: {train_metrics['train_accuracy']:.4f}")
                print(f"검증 정확도: {val_metrics['val_accuracy']:.4f}")
            
            # WandB 로깅
            if use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    **train_metrics,
                    **val_metrics,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # 학습률 조정
            scheduler.step(val_metrics[monitor_metric])
            
            # 모델 저장 (최고 성능)
            current_metric = val_metrics[monitor_metric]
            is_best = (mode == 'max' and current_metric > best_metric) or \
                     (mode == 'min' and current_metric < best_metric)
            
            if is_best:
                best_metric = current_metric
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_metric': best_metric,
                    'config': self.config
                }, os.path.join(self.checkpoint_dir, 'best_model.pth'))
                
                print(f"✅ 최고 성능 모델 저장! ({monitor_metric}: {best_metric:.4f})")
            
            # Early Stopping 체크
            if early_stopping(val_metrics[monitor_metric]):
                print(f"⏹️ Early Stopping 발동! (Epoch {epoch+1})")
                break
        
        # 테스트 평가
        print(f"\n🎯 최종 테스트 평가")
        checkpoint = load_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_metrics = self.validate_epoch(model, test_loader, criterion, task_type)
        
        print(f"테스트 결과:")
        for key, value in test_metrics.items():
            print(f"  - {key}: {value:.4f}")
        
        if use_wandb:
            wandb.log(test_metrics)
            wandb.finish()
        
        print(f"🎉 학습 완료! 모델 저장 위치: {self.checkpoint_dir}")
        
        return model, test_metrics

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='모델 학습')
    parser.add_argument('--task', choices=['morigirl', 'score'], required=True, 
                       help='학습할 태스크 (morigirl: 분류, score: 점수 예측)')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에포크 수')
    parser.add_argument('--config', type=str, default='./config.json', help='설정 파일 경로')
    parser.add_argument('--use-wandb', action='store_true', help='WandB 사용 여부')
    
    args = parser.parse_args()
    
    try:
        trainer = ModelTrainer(args.config)
        model, metrics = trainer.train_model(
            task_type=args.task,
            epochs=args.epochs,
            use_wandb=args.use_wandb
        )
        
        print(f"\n✅ 학습 성공!")
        print(f"최종 성능: {metrics}")
        
    except Exception as e:
        print(f"❌ 학습 실패: {e}")
        raise

if __name__ == "__main__":
    main() 