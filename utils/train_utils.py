# utils/train_utils.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    한 epoch 동안 학습을 진행하는 함수
    
    Returns:
        avg_loss: 평균 손실
        avg_acc: 평균 정확도
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (batch_size, 1)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 예측 및 정확도 계산
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        # 기록
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        
        # Progress bar 업데이트
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy_score(all_labels, all_preds):.4f}'
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, avg_acc

def evaluate(model, dataloader, criterion, device, detailed=False):
    """
    모델 평가 함수
    
    Returns:
        avg_loss: 평균 손실
        avg_acc: 평균 정확도  
        auc_score: AUC 점수
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)
    
    if detailed:
        print("\n=== 상세 평가 결과 ===")
        print(classification_report(all_labels, all_preds, 
                                    target_names=['일반', '모리걸']))
    
    return avg_loss, avg_acc, auc_score

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """체크포인트 저장"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, filepath, optimizer=None, device='cpu'):
    """체크포인트 로드"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'] 