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

# wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  wandb가 설치되지 않았습니다. pip install wandb 로 설치하세요.")

# 로컬 모듈
import sys
sys.path.append('..')
from prepare_training_data import MorigirlDataProcessor, MorigirlDataset
from morigirl.morigirl_model import MoriGirlVectorClassifier, get_model_info

class MoriGirlTrainer:
    """모리걸 벡터 분류 모델 학습 클래스"""
    
    def __init__(self, 
                 config_path: str = "config.json",
                 data_path: str = None,
                 experiment_name: str = None):
        
        # 설정 파일 로드
        self.config = self.load_config(config_path)
        
        # 데이터 경로 설정 (우선순위: 파라미터 > config > 기본값)
        if data_path is None:
            self.data_path = self._get_data_path()
        else:
            self.data_path = data_path
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 실험 이름과 결과 폴더 설정
        self.experiment_name = self._get_experiment_name(experiment_name)
        self.result_dir = self._get_result_dir()
        self.checkpoint_dir = f"{self.result_dir}/checkpoints"
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # config.json을 실험 폴더에 복사하여 저장
        self._save_experiment_config(config_path)
        
        # wandb 초기화
        self.use_wandb = WANDB_AVAILABLE and self.config["wandb"]["enabled"]
        if self.use_wandb:
            self.init_wandb()
        
        print(f"🚀 모리걸 모델 학습 시작")
        print(f"  - 실험명: {self.experiment_name}")
        print(f"  - 데이터 경로: {self.data_path}")
        print(f"  - 결과 폴더: {self.result_dir}")
        print(f"  - 디바이스: {self.device}")
        print(f"  - wandb: {'활성화' if self.use_wandb else '비활성화'}")

    def load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 설정 파일 로드: {config_path}")
            return config
        except Exception as e:
            print(f"❌ 설정 파일 로드 실패: {e}")
            # 기본 설정 반환
            return {
                "data": {"max_products_per_type": 5000, "train_test_split": 0.8, "val_split": 0.1},
                "model": {"input_vector_dim": 1024, "hidden_dim": 128, "dropout_rate": 0.1, 
                         "learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": 64, 
                         "epochs": 50, "patience": 10},
                "wandb": {"enabled": False, "project": "morigirl-classification"}
            }

    def _get_data_path(self) -> str:
        """config에서 데이터 경로 가져오기"""
        data_config = self.config["data"]
        data_paths = data_config.get("data_paths", {})
        
        # 1. train_data_dir이 설정되어 있으면 사용
        if data_paths.get("train_data_dir"):
            print(f"📁 사용자 지정 train 데이터 경로: {data_paths['train_data_dir']}")
            return data_paths["train_data_dir"]
        
        # 2. base_data_dir 사용 (자동 경로 생성)
        if data_paths.get("auto_generate_path", True):
            max_products = data_config["max_products_per_type"]
            base_path = data_paths.get("base_data_dir", "../data/morigirl_{max_products}")
            final_path = base_path.format(max_products=max_products)
            print(f"📁 자동 생성 데이터 경로: {final_path}")
            return final_path
        
        # 3. 기본값
        max_products = data_config["max_products_per_type"]
        default_path = f"../data/morigirl_{max_products}"
        print(f"📁 기본 데이터 경로: {default_path}")
        return default_path

    def _get_experiment_name(self, experiment_name: str = None) -> str:
        """config에서 실험 이름 가져오기"""
        data_config = self.config["data"]
        result_paths = data_config.get("result_paths", {})
        
        # 1. 파라미터로 받은 실험 이름 우선
        if experiment_name:
            print(f"🔬 사용자 지정 실험 이름: {experiment_name}")
            return experiment_name
            
        # 2. config에서 실험 이름 사용
        if result_paths.get("experiment_name"):
            print(f"🔬 config 실험 이름: {result_paths['experiment_name']}")
            return result_paths["experiment_name"]
        
        # 3. 자동 생성 (월일시분_랜덤2자리)
        import random
        date_str = datetime.now().strftime('%m%d%H%M')  # 월일시분
        random_num = random.randint(10, 99)  # 랜덤 2자리
        auto_name = f"{date_str}_{random_num:02d}"
        print(f"🔬 자동 생성 실험 이름: {auto_name}")
        return auto_name

    def _get_result_dir(self) -> str:
        """config에서 결과 폴더 경로 가져오기"""
        data_config = self.config["data"]
        result_paths = data_config.get("result_paths", {})
        
        # 1. 자동 경로 생성 (기본값)
        if result_paths.get("auto_generate_result_path", True):
            base_result_dir = result_paths.get("base_result_dir", "result")
            final_path = f"{base_result_dir}/{self.experiment_name}"
            print(f"📁 자동 생성 결과 경로: {final_path}")
            return final_path
        
        # 2. 기본값
        default_path = f"result/{self.experiment_name}"
        print(f"📁 기본 결과 경로: {default_path}")
        return default_path

    def _save_experiment_config(self, config_path: str):
        """실험 설정을 결과 폴더에 저장"""
        import shutil
        import json
        from datetime import datetime
        
        try:
            # 원본 config 복사
            experiment_config_path = f"{self.result_dir}/config.json"
            shutil.copy2(config_path, experiment_config_path)
            
            # 실험 메타데이터 추가 저장
            experiment_metadata = {
                "experiment_name": self.experiment_name,
                "start_time": datetime.now().isoformat(),
                "data_path": self.data_path,
                "result_dir": self.result_dir,
                "device": str(self.device),
                "wandb_enabled": self.use_wandb,
                "original_config_path": config_path
            }
            
            with open(f"{self.result_dir}/experiment_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(experiment_metadata, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 실험 설정 저장:")
            print(f"  - {experiment_config_path}")
            print(f"  - {self.result_dir}/experiment_metadata.json")
            
        except Exception as e:
            print(f"⚠️  실험 설정 저장 실패: {e}")

    def init_wandb(self):
        """wandb 초기화"""
        try:
            wandb_config = self.config["wandb"]
            
            # wandb 설정
            wandb.init(
                project=wandb_config["project"],
                entity=wandb_config.get("entity"),
                name=self.experiment_name,
                config={
                    "model": self.config["model"],
                    "data": self.config["data"],
                    "experiment_name": self.experiment_name,
                    "data_path": self.data_path
                }
            )
            
            print(f"✅ wandb 초기화 완료: {wandb_config['project']}")
            
        except Exception as e:
            print(f"⚠️  wandb 초기화 실패: {e}")
            self.use_wandb = False

    def setup_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """데이터셋 설정"""
        print(f"\n📊 데이터셋 설정")
        
        # config에서 설정 읽기
        data_config = self.config["data"]
        model_config = self.config["model"]
        
        test_size = 1 - data_config["train_test_split"]
        val_size = data_config["val_split"] 
        batch_size = model_config["batch_size"]
        
        # 데이터 처리기 생성 및 로딩 (train 파일만)
        train_processor = MorigirlDataProcessor(self.data_path)
        if not train_processor.load_npy_files(split_type="train"):
            raise RuntimeError("Train 데이터 로딩에 실패했습니다.")
        
        # Test 데이터 로딩
        test_processor = MorigirlDataProcessor(self.data_path)
        if not test_processor.load_npy_files(split_type="test"):
            raise RuntimeError("Test 데이터 로딩에 실패했습니다.")
        
        # Test 데이터셋 생성 (미리 분할된 test 파일 사용)
        test_dataset = MorigirlDataset(
            test_processor.vectors, 
            test_processor.labels, 
            test_processor.product_ids
        )
        
        # Train에서 Validation 분할
        # val_size는 전체 데이터 기준이므로, train 데이터 내에서의 비율로 조정
        # 예: 전체에서 train 80%, val 10%, test 20%라면
        # train 파일에서 val을 분할할 때는 10/80 = 0.125 비율로 분할
        val_size_adjusted = val_size / data_config["train_test_split"]
        
        train_vectors = train_processor.vectors
        train_labels = train_processor.labels
        train_ids = train_processor.product_ids
        
        from sklearn.model_selection import train_test_split
        
        # 연속값 라벨을 이진값으로 변환 (stratify용)
        # 0.5 기준으로 이진화: 0.5 이상이면 1(모리걸), 미만이면 0(비모리걸)
        binary_labels_for_stratify = (train_labels >= 0.5).astype(int)
        
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            train_vectors, train_labels, train_ids,
            test_size=val_size_adjusted,
            random_state=42,
            stratify=binary_labels_for_stratify  # 이진 라벨로 stratify
        )
        
        # 새로운 Dataset 객체 생성
        train_dataset = MorigirlDataset(X_train, y_train, ids_train)
        val_dataset = MorigirlDataset(X_val, y_val, ids_val)
        
        # 클래스 가중치 계산 (불균형 해결)
        pos_count = torch.sum(train_dataset.labels).item()
        neg_count = len(train_dataset) - pos_count
        self.pos_weight = torch.tensor(neg_count / pos_count).to(self.device)
        print(f"📊 클래스 가중치: {self.pos_weight.item():.2f}")
        
        # wandb에 데이터 정보 로그
        if self.use_wandb:
            total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
            train_pos = torch.sum(train_dataset.labels).item()
            val_pos = torch.sum(val_dataset.labels).item()
            test_pos = torch.sum(test_dataset.labels).item()
            
            wandb.log({
                "data/train_size": len(train_dataset),
                "data/val_size": len(val_dataset), 
                "data/test_size": len(test_dataset),
                "data/total_size": total_samples,
                "data/pos_weight": self.pos_weight.item(),
                "data/train_pos_ratio": train_pos / len(train_dataset),
                "data/val_pos_ratio": val_pos / len(val_dataset),
                "data/test_pos_ratio": test_pos / len(test_dataset),
                "data/train_pos_count": train_pos,
                "data/val_pos_count": val_pos,
                "data/test_pos_count": test_pos,
                "data/vector_dim": train_dataset.vectors.shape[1],
                "data/batch_size": batch_size
            })
        
        # 데이터로더 생성
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"📦 DataLoader 생성 완료:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        print(f"  - Batch size: {batch_size}")
        
        return train_loader, val_loader, test_loader

    def setup_model(self) -> nn.Module:
        """모델 설정"""
        print(f"\n🧠 모델 설정")
        
        # config에서 모델 파라미터 읽기
        model_config = self.config["model"]
        model_kwargs = {
            "input_dim": model_config["input_vector_dim"],
            "hidden_dim": model_config["hidden_dim"],
            "hidden_dim2": model_config["hidden_dim2"],
            "dropout_rate": model_config["dropout_rate"]
        }
        
        model = MoriGirlVectorClassifier(**model_kwargs)
        model.to(self.device)
        
        # 모델 정보 출력
        get_model_info(model)
        
        # wandb에 모델 정보 로그
        if self.use_wandb:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 * 1024)  # float32 기준
            
            wandb.log({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/model_size_mb": model_size_mb,
                "model/input_dim": model_config["input_vector_dim"],
                "model/hidden_dim": model_config["hidden_dim"],
                "model/hidden_dim2": model_config["hidden_dim2"],
                "model/dropout_rate": model_config["dropout_rate"],
                "model/architecture": "2-layer-mlp"
            })
            
            # 모델 감시 (gradients, parameters)
            if self.config["wandb"]["watch_model"]:
                wandb.watch(model, log="all", log_freq=self.config["wandb"]["log_frequency"])
        
        return model

    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   criterion: nn.Module, optimizer: optim.Optimizer, epoch: int, show_progress: bool = True) -> Dict[str, float]:
        """한 에포크 학습"""
        model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        progress_bar = tqdm(train_loader, desc="학습", disable=not show_progress)
        
        # 배치 단위 로깅을 위한 변수
        log_frequency = self.config["wandb"]["log_frequency"]
        
        for batch_idx, batch in enumerate(progress_bar):
            vectors = batch['vector'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)  # (batch_size, 1)
            
            # Forward pass
            outputs = model(vectors)  # 이미 sigmoid 적용됨
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient norm 계산
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            optimizer.step()
            
            # 메트릭 수집
            total_loss += loss.item()
            probs = outputs.detach().cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten().astype(int))  # 명시적으로 int 타입 보장
            all_labels.extend(labels.cpu().numpy().flatten())
            
            # Progress bar 업데이트
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'GradNorm': f'{grad_norm:.4f}'})
            
            # 배치 단위 로깅 제거 - 에포크 단위로만 로깅
        
        # 최종 메트릭 계산
        # 연속값 라벨을 이진값으로 변환 (메트릭 계산용)
        binary_labels = (np.array(all_labels) >= 0.5).astype(int)
        binary_preds = np.array(all_preds).astype(int)  # 예측값도 명시적으로 이진값으로 변환
        
        accuracy = accuracy_score(binary_labels, binary_preds)
        try:
            auc = roc_auc_score(binary_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy,
            'auc': auc,
            'grad_norm': grad_norm
        }

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                      criterion: nn.Module, show_progress: bool = True) -> Dict[str, float]:
        """한 에포크 검증"""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="검증", disable=not show_progress):
                vectors = batch['vector'].to(self.device)
                labels = batch['label'].to(self.device).unsqueeze(1)
                
                outputs = model(vectors)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                probs = outputs.cpu().numpy()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten().astype(int))  # 명시적으로 int 타입 보장
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # 메트릭 계산
        # 연속값 라벨을 이진값으로 변환 (메트릭 계산용)
        binary_labels = (np.array(all_labels) >= 0.5).astype(int)
        binary_preds = np.array(all_preds).astype(int)  # 예측값도 명시적으로 이진값으로 변환
        
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels, binary_preds, average='binary', zero_division=0
        )
        try:
            auc = roc_auc_score(binary_labels, all_probs)
        except:
            auc = 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'num_samples': len(all_labels),
            'pos_samples': sum(all_labels),
            'neg_samples': len(all_labels) - sum(all_labels)
        }

    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                       epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """체크포인트 저장 (최고 성능 모델만)"""
        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'experiment_name': self.experiment_name
            }
            
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def train(self):
        """모델 학습 (config.json 기반)"""
        
        # config에서 학습 파라미터 읽기
        model_config = self.config["model"]
        epochs = model_config["epochs"]
        learning_rate = model_config["learning_rate"]
        weight_decay = model_config["weight_decay"]
        patience = model_config["patience"]
        min_delta = model_config.get("min_delta", 0.001)  # 최소 성능 향상 임계값
        early_stopping_enabled = model_config.get("early_stopping_enabled", True)  # Early stopping 활성화
        
        # 데이터셋 설정
        train_loader, val_loader, test_loader = self.setup_datasets()
        
        # 모델 설정
        model = self.setup_model()
        
        # 손실함수 및 옵티마이저 설정
        criterion = nn.BCELoss(weight=self.pos_weight if self.pos_weight.item() > 1 else None)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)  # 스케줄러도 더 빠르게
        
        # 엄격한 Early Stopping을 위한 추적 변수들
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_composite_score = 0.0  # accuracy + f1의 조합 점수
        patience_counter = 0
        no_improvement_counter = 0  # 연속 성능 저하 추적
        train_history = []
        
        # 출력 간격 설정 (전체 에포크의 10분의 1)
        print_interval = max(1, epochs // 10)  # 최소 1 에포크마다는 출력
        print(f"\n🎯 학습 시작 (총 {epochs} 에포크, {print_interval} 에포크마다 출력)")
        print(f"⏰ Early stopping: {'활성화' if early_stopping_enabled else '비활성화'}")
        
        for epoch in range(epochs):
            # 출력 여부 결정 (첫 에포크, 마지막 에포크, 또는 설정된 간격)
            should_print = (epoch == 0 or 
                          epoch == epochs - 1 or 
                          (epoch + 1) % print_interval == 0)
            
            if should_print:
                print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # 학습
            train_metrics = self.train_epoch(model, train_loader, criterion, optimizer, epoch, show_progress=should_print)
            
            # 검증
            val_metrics = self.validate_epoch(model, val_loader, criterion, show_progress=should_print)
            
            # 스케줄러 업데이트
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_metrics['accuracy'])
            
            # 결과 출력 (조건부)
            if should_print:
                print(f"학습 - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f}")
                print(f"검증 - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # 엄격한 성능 평가 (accuracy + f1 조합 점수)
            current_composite_score = (val_metrics['accuracy'] + val_metrics['f1']) / 2
            
            # wandb 로깅 (에포크 단위) - 정수 step 사용
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_metrics['loss'],
                    "train/accuracy": train_metrics['accuracy'],
                    "train/auc": train_metrics['auc'],
                    "train/grad_norm": train_metrics['grad_norm'],
                    "val/loss": val_metrics['loss'],
                    "val/accuracy": val_metrics['accuracy'],
                    "val/precision": val_metrics['precision'],
                    "val/recall": val_metrics['recall'],
                    "val/f1": val_metrics['f1'],
                    "val/auc": val_metrics['auc'],
                    "learning_rate": current_lr,
                    "optimizer/lr": current_lr,
                    "metrics/loss_diff": train_metrics['loss'] - val_metrics['loss'],
                    "metrics/accuracy_diff": train_metrics['accuracy'] - val_metrics['accuracy'],
                    "metrics/composite_score": current_composite_score,
                    "patience/counter": patience_counter,
                    "patience/no_improvement": no_improvement_counter
                }, step=epoch + 1)
            
            # 성능 향상이 충분한지 확인 (min_delta 임계값 적용)
            acc_improvement = val_metrics['accuracy'] - best_val_acc
            f1_improvement = val_metrics['f1'] - best_val_f1
            composite_improvement = current_composite_score - best_composite_score
            
            # 여러 조건을 모두 만족해야 "진짜 개선"으로 인정
            significant_acc_improvement = acc_improvement > min_delta
            significant_f1_improvement = f1_improvement > min_delta  
            significant_composite_improvement = composite_improvement > min_delta
            
            # 최고 성능 확인 (더 엄격한 조건)
            is_best = (significant_acc_improvement and significant_f1_improvement) or significant_composite_improvement
            
            if is_best:
                best_val_acc = val_metrics['accuracy']
                best_val_f1 = val_metrics['f1']
                best_composite_score = current_composite_score
                patience_counter = 0
                no_improvement_counter = 0
                
                if should_print:
                    print(f"🎯 새로운 최고 성능! Acc: {best_val_acc:.4f} (+{acc_improvement:.4f}), F1: {best_val_f1:.4f} (+{f1_improvement:.4f})")
                
                # wandb에 최고 성능 기록
                if self.use_wandb:
                    wandb.log({
                        "best/epoch": epoch + 1,
                        "best/val_accuracy": best_val_acc,
                        "best/val_f1": best_val_f1,
                        "best/val_precision": val_metrics['precision'],
                        "best/val_recall": val_metrics['recall'],
                        "best/val_auc": val_metrics['auc'],
                        "best/val_loss": val_metrics['loss'],
                        "best/composite_score": best_composite_score,
                        "best/train_accuracy": train_metrics['accuracy'],
                        "best/train_loss": train_metrics['loss'],
                        "best/accuracy_improvement": acc_improvement,
                        "best/f1_improvement": f1_improvement,
                        "best/learning_rate": current_lr
                    })
            else:
                patience_counter += 1
                no_improvement_counter += 1
                
                # 성능 저하 정도 체크 (조건부 출력)
                if acc_improvement < -min_delta and f1_improvement < -min_delta and should_print:
                    print(f"⚠️  성능 저하 감지: Acc {acc_improvement:+.4f}, F1 {f1_improvement:+.4f}")
            
            # 체크포인트 저장
            self.save_checkpoint(model, optimizer, epoch, val_metrics, is_best)
            
            # 기록 저장
            train_history.append({
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'is_best': is_best,
                'improvements': {
                    'accuracy': acc_improvement,
                    'f1': f1_improvement,
                    'composite': composite_improvement
                }
            })
            
            # Early Stopping 체크 (활성화된 경우에만)
            if early_stopping_enabled:
                # 엄격한 Early Stopping 조건들
                early_stop_reasons = []
                
                # 1. 기본 patience 초과
                if patience_counter >= patience:
                    early_stop_reasons.append(f"patience {patience} 초과")
                
                # 2. 연속 성능 저하가 너무 많음 (patience의 1.5배)
                if no_improvement_counter >= int(patience * 1.5):
                    early_stop_reasons.append(f"연속 {no_improvement_counter}회 성능 향상 없음")
                
                # 3. 성능이 심각하게 저하되고 있음
                if (acc_improvement < -min_delta * 3 and f1_improvement < -min_delta * 3 and 
                    no_improvement_counter >= 3):
                    early_stop_reasons.append("심각한 성능 저하 감지")
                
                if early_stop_reasons:
                    print(f"⏰ Early stopping at epoch {epoch+1}")
                    print(f"   이유: {', '.join(early_stop_reasons)}")
                    print(f"   최고 성능: Acc {best_val_acc:.4f}, F1 {best_val_f1:.4f}")
                    
                    if self.use_wandb:
                        wandb.log({
                            "early_stopped": True, 
                            "stopped_epoch": epoch + 1,
                            "stop_reasons": early_stop_reasons,
                            "final_patience_counter": patience_counter,
                            "final_no_improvement_counter": no_improvement_counter,
                            "final_best_accuracy": best_val_acc,
                            "final_best_f1": best_val_f1,
                            "final_composite_score": best_composite_score,
                            "final_val_accuracy": val_metrics['accuracy'],
                            "final_val_f1": val_metrics['f1'],
                            "final_accuracy_drop": best_val_acc - val_metrics['accuracy'],
                            "final_f1_drop": best_val_f1 - val_metrics['f1']
                        })
                    break
        
        # 최고 모델로 validation 재평가 (상세 리포트)
        print(f"\n🎯 최고 성능 모델로 validation 상세 평가")
        best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        val_detailed_metrics = self.evaluate_model_detailed(val_loader, best_model_path, "Validation")
        
        # 최고 모델로 테스트
        print(f"\n🎯 최고 성능 모델로 테스트 평가")
        test_metrics = self.evaluate_model_detailed(test_loader, best_model_path, "Test")
        
        # wandb에 최종 결과 로그
        if self.use_wandb:
            wandb.log({
                "final_validation/accuracy": val_detailed_metrics['accuracy'],
                "final_validation/precision": val_detailed_metrics['precision'],
                "final_validation/recall": val_detailed_metrics['recall'],
                "final_validation/f1": val_detailed_metrics['f1'],
                "final_validation/auc": val_detailed_metrics['auc'],
                "final_test/accuracy": test_metrics['accuracy'],
                "final_test/precision": test_metrics['precision'],
                "final_test/recall": test_metrics['recall'],
                "final_test/f1": test_metrics['f1'],
                "final_test/auc": test_metrics['auc']
            })
            
            # 모델 아티팩트 저장
            if self.config["wandb"]["save_model"]:
                model_artifact = wandb.Artifact(
                    f"morigirl-model-{self.experiment_name}", 
                    type="model",
                    description=f"Best model from experiment {self.experiment_name}"
                )
                model_artifact.add_file(best_model_path)
                wandb.log_artifact(model_artifact)
        
        # 학습 기록 저장
        history_path = os.path.join(self.result_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(train_history, f, indent=2)
        
        # wandb 종료
        if self.use_wandb:
            wandb.finish()
        
        print(f"✅ 학습 완료! 결과 폴더: {self.result_dir}")

    def evaluate_model_detailed(self, data_loader: DataLoader, model_path: str, dataset_name: str) -> Dict[str, float]:
        """모델 상세 평가"""
        # 모델 로드
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # config에서 모델 파라미터 읽기
        model_config = self.config["model"]
        model_kwargs = {
            "input_dim": model_config["input_vector_dim"],
            "hidden_dim": model_config["hidden_dim"],
            "hidden_dim2": model_config["hidden_dim2"],
            "dropout_rate": model_config["dropout_rate"]
        }
        
        model = MoriGirlVectorClassifier(**model_kwargs)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"{dataset_name} 평가"):
                vectors = batch['vector'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(vectors)
                probs = outputs.cpu().numpy().flatten()
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds.astype(int))  # 명시적으로 int 타입 보장
                all_labels.extend(labels.cpu().numpy())
        
        # 최종 성능 계산
        # 연속값 라벨을 이진값으로 변환 (메트릭 계산용)
        binary_labels = (np.array(all_labels) >= 0.5).astype(int)
        binary_preds = np.array(all_preds).astype(int)  # 예측값도 명시적으로 이진값으로 변환
        
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_labels, binary_preds, average='binary', zero_division=0
        )
        auc = roc_auc_score(binary_labels, all_probs)
        
        print(f"\n📊 최종 {dataset_name} 성능:")
        print(f"  - 정확도: {accuracy:.4f}")
        print(f"  - 정밀도: {precision:.4f}")
        print(f"  - 재현율: {recall:.4f}")
        print(f"  - F1 점수: {f1:.4f}")
        print(f"  - AUC: {auc:.4f}")
        
        # 분류 리포트
        print(f"\n📋 {dataset_name} 분류 리포트:")
        print(classification_report(binary_labels, binary_preds, target_names=['비모리걸', '모리걸']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

def main():
    parser = argparse.ArgumentParser(description='모리걸 벡터 분류 모델 학습')
    parser.add_argument('--config-path', default='config.json', help='설정 파일 경로')
    parser.add_argument('--data-path', default=None, help='데이터 경로 (설정 파일 우선)')
    parser.add_argument('--experiment-name', help='실험 이름')
    
    args = parser.parse_args()
    
    try:
        # 학습 시작
        trainer = MoriGirlTrainer(
            config_path=args.config_path,
            data_path=args.data_path,
            experiment_name=args.experiment_name
        )
        
        # config.json 기반 학습
        trainer.train()
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 