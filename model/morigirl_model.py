# model/morigirl_model.py
"""
모리걸 스타일 분류를 위한 신경망 모델

임베딩 벡터 기반 2레이어 분류기:
- MoriGirlVectorClassifier: 2-layer 분류기
- 구조: 1024 → 512 → 128 → 1 (약 590K 파라미터)
- 임베딩 벡터에 최적화된 2단계 특징 추출 구조

입력: EfficientNet에서 추출한 1024차원 임베딩 벡터
출력: 모리걸일 확률 (0~1)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

class MoriGirlVectorClassifier(nn.Module):
    """
    임베딩 벡터 기반 모리걸 스타일 이진 분류기 (2레이어 버전)
    
    입력: 이미지 임베딩 벡터 (1024차원)
    출력: 모리걸일 확률 (0~1 사이)
    
    구조: 1024 → 512 → 128 → 1 (약 590K 파라미터)
    """
    
    def __init__(self, 
                 input_dim: int = 1024,
                 hidden_dim: int = 512,
                 hidden_dim2: int = 128,
                 dropout_rate: float = 0.1):
        super(MoriGirlVectorClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        
        # 2레이어 분류기 (더 복잡한 패턴 학습 가능)
        self.classifier = nn.Sequential(
            # 첫 번째 히든 레이어: 1024 → 512
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # 두 번째 히든 레이어: 512 → 128  
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # 출력 레이어: 128 → 1
            nn.Linear(hidden_dim2, 1),
            nn.Sigmoid()
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 (임베딩 벡터에 적합하게)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # 임베딩 벡터에 더 적합
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: (batch_size, 1024) 형태의 임베딩 벡터
            
        Returns:
            probs: (batch_size, 1) 형태의 모리걸 확률 (0~1)
        """
        return self.classifier(x)
    
    def predict(self, x, threshold=0.5):
        """
        예측 수행 (확률과 클래스 반환)
        
        Args:
            x: 임베딩 벡터 텐서
            threshold: 분류 임계값
            
        Returns:
            probs: 모리걸일 확률
            preds: 예측 클래스 (0: 비모리걸, 1: 모리걸)
        """
        self.eval()
        with torch.no_grad():
            probs = self.forward(x)
            preds = (probs > threshold).float()
        return probs, preds



def get_model_info(model):
    """
    모델 정보 출력 (파라미터 수, 모델 크기 등)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 크기 추정 (MB)
    param_size = total_params * 4 / (1024 * 1024)  # float32 기준
    
    print(f"🧠 모델 정보:")
    print(f"  - 총 파라미터 수: {total_params:,}")
    print(f"  - 학습 가능 파라미터 수: {trainable_params:,}")
    print(f"  - 예상 모델 크기: {param_size:.2f} MB")
    print(f"  - 모델 타입: {model.__class__.__name__}")
    
    # 모델별 상세 정보
    if hasattr(model, 'input_dim'):
        print(f"  - 입력 차원: {model.input_dim}")
    if hasattr(model, 'hidden_dim'):
        print(f"  - 히든 차원 1: {model.hidden_dim}")
    if hasattr(model, 'hidden_dim2'):
        print(f"  - 히든 차원 2: {model.hidden_dim2}")
    
    return total_params, trainable_params, param_size

def create_morigirl_model(**kwargs) -> nn.Module:
    """
    모리걸 모델 생성 함수
    
    Args:
        **kwargs: 모델 파라미터 (input_dim, hidden_dim, hidden_dim2, dropout_rate)
    
    Returns:
        모리걸 분류 모델 (1024 → 512 → 128 → 1)
    """
    model = MoriGirlVectorClassifier(**kwargs)
    print(f"✅ 모리걸 분류기 생성 (2-layer)")
    return model

# 호환성을 위한 별칭
MoriGirlClassifier = MoriGirlVectorClassifier 