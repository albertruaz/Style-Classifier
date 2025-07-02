# utils/visualization.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from torchvision import transforms
import cv2
import os
from typing import List, Dict, Any
import pandas as pd

def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """정규화된 이미지 텐서를 원본으로 복원"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

def visualize_predictions(model, dataset, device, num_samples=8, save_path='prediction_samples.png'):
    """예측 결과 시각화"""
    model.eval()
    
    # 랜덤 샘플 선택
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[idx]
            
            # 예측
            image_batch = image.unsqueeze(0).to(device)
            logit = model(image_batch)
            prob = torch.sigmoid(logit).item()
            pred_label = 1 if prob > 0.5 else 0
            
            # 이미지 표시
            img_display = denormalize_image(image)
            img_display = torch.clamp(img_display, 0, 1)
            img_display = img_display.permute(1, 2, 0).numpy()
            
            axes[i].imshow(img_display)
            axes[i].axis('off')
            
            # 제목 설정
            true_class = '모리걸' if true_label == 1 else '일반'
            pred_class = '모리걸' if pred_label == 1 else '일반'
            color = 'green' if true_label == pred_label else 'red'
            
            axes[i].set_title(
                f'실제: {true_class}\n예측: {pred_class} ({prob:.3f})',
                color=color, fontsize=10
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 예측 결과 저장: {save_path}")
    plt.show()

def visualize_class_distribution(dataset, save_path='class_distribution.png'):
    """클래스 분포 시각화"""
    labels = [dataset.labels[i] for i in range(len(dataset))]
    classes = ['일반', '모리걸']
    
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
    plt.title('클래스 분포', fontsize=16)
    plt.ylabel('이미지 수', fontsize=12)
    
    # 막대 위에 숫자 표시
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', fontsize=12)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 클래스 분포 저장: {save_path}")
    plt.show()

def plot_confidence_distribution(model, dataset, device, save_path='confidence_distribution.png'):
    """예측 신뢰도 분포 시각화"""
    model.eval()
    
    confidences_correct = []
    confidences_wrong = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            image, true_label = dataset[i]
            image_batch = image.unsqueeze(0).to(device)
            
            logit = model(image_batch)
            prob = torch.sigmoid(logit).item()
            pred_label = 1 if prob > 0.5 else 0
            
            confidence = prob if pred_label == 1 else (1 - prob)
            
            if pred_label == true_label:
                confidences_correct.append(confidence)
            else:
                confidences_wrong.append(confidence)
    
    plt.figure(figsize=(10, 6))
    plt.hist(confidences_correct, bins=20, alpha=0.7, label='정답', color='green')
    plt.hist(confidences_wrong, bins=20, alpha=0.7, label='오답', color='red')
    plt.xlabel('예측 신뢰도', fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.title('예측 신뢰도 분포', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 신뢰도 분포 저장: {save_path}")
    plt.show()

def plot_training_history(history: Dict[str, List], save_path='training_history.png'):
    """학습 히스토리 시각화"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 손실 그래프
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    if 'val_loss' in history:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 정확도 그래프 (있는 경우)
    if 'train_acc' in history:
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
        if 'val_acc' in history:
            axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 학습 히스토리 저장: {save_path}")
    plt.show()

def plot_score_distribution(predictions_df: pd.DataFrame, save_path='score_distribution.png'):
    """점수 예측 결과 분포 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 모리걸 확률 분포
    axes[0].hist(predictions_df['morigirl_prob'], bins=50, alpha=0.7, color='pink')
    axes[0].set_title('모리걸 확률 분포')
    axes[0].set_xlabel('모리걸 확률')
    axes[0].set_ylabel('상품 수')
    axes[0].grid(True, alpha=0.3)
    
    # 인기도 확률 분포
    axes[1].hist(predictions_df['popularity_prob'], bins=50, alpha=0.7, color='skyblue')
    axes[1].set_title('인기도 확률 분포')
    axes[1].set_xlabel('인기도 확률')
    axes[1].set_ylabel('상품 수')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 점수 분포 저장: {save_path}")
    plt.show()

def generate_grad_cam(model, image_tensor, target_layer_name='features'):
    """Grad-CAM을 사용한 주목 영역 시각화"""
    model.eval()
    
    # 그래디언트 저장용 변수
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # 훅 등록
    target_layer = model.backbone.features[-1]  # 마지막 conv layer
    backward_handle = target_layer.register_backward_hook(backward_hook)
    forward_handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        # Forward pass
        output = model(image_tensor)
        
        # Backward pass
        output.backward()
        
        # Grad-CAM 계산
        gradient = gradients[0][0]  # (C, H, W)
        activation = activations[0][0]  # (C, H, W)
        
        weights = torch.mean(gradient, dim=(1, 2))  # (C,)
        grad_cam = torch.sum(weights.unsqueeze(1).unsqueeze(2) * activation, dim=0)
        grad_cam = torch.relu(grad_cam)
        grad_cam = grad_cam / torch.max(grad_cam)
        
        return grad_cam.detach().cpu().numpy()
        
    finally:
        # 훅 제거
        backward_handle.remove()
        forward_handle.remove()

def visualize_grad_cam(model, dataset, device, num_samples=4, save_path='grad_cam_visualization.png'):
    """Grad-CAM 시각화"""
    model.eval()
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image_batch = image.unsqueeze(0).to(device)
        image_batch.requires_grad_()
        
        # 원본 이미지
        img_display = denormalize_image(image)
        img_display = torch.clamp(img_display, 0, 1)
        img_display = img_display.permute(1, 2, 0).numpy()
        
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f'{"모리걸" if label == 1 else "일반"}')
        axes[0, i].axis('off')
        
        # Grad-CAM
        grad_cam = generate_grad_cam(model, image_batch)
        grad_cam_resized = cv2.resize(grad_cam, (224, 224))
        
        # 히트맵 오버레이
        heatmap = plt.cm.jet(grad_cam_resized)[:, :, :3]
        superimposed = heatmap * 0.4 + img_display * 0.6
        
        axes[1, i].imshow(superimposed)
        axes[1, i].set_title('주목 영역')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Grad-CAM 시각화 저장: {save_path}")
    plt.show()

class ModelVisualizer:
    """모델 시각화 도구 클래스"""
    
    def __init__(self, model, device, save_dir='./results'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def analyze_predictions(self, dataset, num_samples=8):
        """예측 결과 종합 분석"""
        print("📈 예측 결과 분석 중...")
        
        # 예측 샘플 시각화
        visualize_predictions(
            self.model, dataset, self.device, num_samples,
            save_path=os.path.join(self.save_dir, 'prediction_samples.png')
        )
        
        # 클래스 분포
        visualize_class_distribution(
            dataset,
            save_path=os.path.join(self.save_dir, 'class_distribution.png')
        )
        
        # 신뢰도 분포
        plot_confidence_distribution(
            self.model, dataset, self.device,
            save_path=os.path.join(self.save_dir, 'confidence_distribution.png')
        )
        
        print(f"✅ 분석 완료! 결과 저장: {self.save_dir}") 