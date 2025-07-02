# main.py - WandB 추가

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
import wandb
from dotenv import load_dotenv

# 같은 디렉토리에 있는 모듈들 import
from model.morigirl_model import MoriGirlClassifier, LightMorigirlCNN, get_model_info
from dataset.morigirl_dataset import MoriGirlDataset
from utils.train_utils import train_one_epoch, evaluate, save_checkpoint, load_checkpoint, calculate_class_weights, EarlyStopping

# .env 파일 로드
load_dotenv()

def initialize_wandb(config: dict) -> bool:
    """WandB 초기화"""
    try:
        # WandB 프로젝트 설정
        project_name = os.getenv('WANDB_PROJECT', 'mori-look-classification')
        entity_name = os.getenv('WANDB_ENTITY', 'albertruaz')
        
        wandb.init(
            project=project_name,
            entity=entity_name,
            config=config,
            name=f"morigirl_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=['morigirl', 'classification', 'efficientnet', 'fashion'],
            notes="모리걸 스타일 이진 분류 모델"
        )
        
        print("✅ WandB 초기화 성공")
        print(f"📊 프로젝트: {project_name}")
        print(f"🔗 실험 URL: {wandb.run.url}")
        
        return True
        
    except ImportError:
        print("⚠️ WandB 라이브러리가 설치되지 않았습니다.")
        return False
    except Exception as e:
        print(f"⚠️ WandB 초기화 실패: {e}")
        return False

def create_data_transforms():
    """
    데이터 전처리 파이프라인 생성
    """
    # 훈련용 transform (데이터 증강 포함)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 검증/테스트용 transform (증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """
    학습 과정 시각화
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss 그래프
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy 그래프
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # 하이퍼파라미터 설정
    config = {
        'batch_size': 32,
        'epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'patience': 7,  # early stopping
        'data_root': './data',  # 데이터 폴더 경로
        'checkpoint_dir': './checkpoints',
        'num_workers': 4,
        'model_type': 'efficientnet',  # 'efficientnet' 또는 'light_cnn'
        'image_size': 224,
    }
    
    # WandB 초기화
    use_wandb = initialize_wandb(config)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 체크포인트 디렉토리 생성
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 데이터 전처리 파이프라인
    train_transform, val_transform = create_data_transforms()
    
    # 데이터셋 생성
    train_dataset = MoriGirlDataset(
        root_dir=config['data_root'],
        split='train',
        transform=train_transform
    )
    
    val_dataset = MoriGirlDataset(
        root_dir=config['data_root'],
        split='val',
        transform=val_transform
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"훈련 데이터: {len(train_dataset)}개")
    print(f"검증 데이터: {len(val_dataset)}개")
    
    # 클래스 가중치 계산 (불균형 데이터 대응)
    class_weights = train_dataset.get_class_weights()
    if class_weights is not None:
        class_weights = torch.tensor(class_weights).to(device)
        print(f"클래스 가중치: {class_weights}")
    
    # 모델 초기화
    if config['model_type'] == 'efficientnet':
        model = MoriGirlClassifier(pretrained=True).to(device)
    else:
        model = LightMorigirlCNN().to(device)
    
    # 모델 정보 출력
    model_info = get_model_info(model)
    print(f"📊 모델 정보:")
    print(f"  - 타입: {config['model_type']}")
    print(f"  - 총 파라미터: {model_info['total_params']:,}")
    print(f"  - 모델 크기: {model_info['model_size_mb']:.2f} MB")
    
    # WandB에 모델 감시 설정
    if use_wandb:
        wandb.watch(model, log='all', log_freq=10)
        wandb.log({"model_info": model_info})
    
    # 손실함수 및 옵티마이저
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'], 
                          weight_decay=config['weight_decay'])
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 학습 기록용 리스트
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 학습 루프
    for epoch in range(config['epochs']):
        print(f"\n[Epoch {epoch+1}/{config['epochs']}]")
        
        # 훈련
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # 검증
        val_loss, val_acc, val_auc = evaluate(
            model, val_loader, criterion, device
        )
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 기록 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # 체크포인트 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, 
                os.path.join(config['checkpoint_dir'], 'best_model.pth')
            )
            print("✅ 최고 모델 저장됨")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['patience']:
            print(f"Early stopping! (patience: {config['patience']})")
            break
    
    # 학습 과정 시각화
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # 최고 모델 로드 후 최종 평가
    print("\n=== 최종 평가 ===")
    best_model = MoriGirlClassifier()
    load_checkpoint(best_model, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
    best_model.to(device)
    
    final_val_loss, final_val_acc, final_val_auc = evaluate(
        best_model, val_loader, criterion, device, detailed=True
    )
    
    print(f"최종 검증 Loss: {final_val_loss:.4f}")
    print(f"최종 검증 Accuracy: {final_val_acc:.4f}")
    print(f"최종 검증 AUC: {final_val_auc:.4f}")
    
    # 모델을 TorchScript로 내보내기 (배포용)
    model.eval()
    example_input = torch.randn(1, 3, 224, 224).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("morigirl_model_traced.pt")
    print("✅ TorchScript 모델 저장 완료: morigirl_model_traced.pt")

    # 최종 모델 저장
    final_path = os.path.join(config['checkpoint_dir'], f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
    save_checkpoint(model, optimizer, epoch + 1, val_loss, final_path)
    
    # WandB 아티팩트 저장
    if use_wandb:
        try:
            # 모델 아티팩트
            model_artifact = wandb.Artifact(
                name="morigirl_classifier_model",
                type="model",
                description="모리걸 스타일 분류 모델"
            )
            
            best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            if os.path.exists(best_model_path):
                model_artifact.add_file(best_model_path)
            
            wandb.log_artifact(model_artifact)
            print("💾 WandB 아티팩트 저장 완료")
            
        except Exception as e:
            print(f"⚠️ WandB 아티팩트 저장 실패: {e}")
        
        wandb.finish()
        print("🎯 WandB 실험 종료")

if __name__ == "__main__":
    main() 