# model/score_prediction_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Tuple, Dict, Any

class ProductScorePredictor(nn.Module):
    """
    상품의 모리걸 확률과 인기도 확률을 동시에 예측하는 멀티태스크 모델
    
    Input: [image_vector(1024), price(1)] -> shape: (1025,)
    Output: [morigirl_prob, popularity_prob] -> shape: (2,)
    """
    
    def __init__(self, 
                 input_dim: int = 1025,
                 hidden_dim: int = 512,
                 dropout: float = 0.3):
        super(ProductScorePredictor, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 공통 특징 추출 레이어
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 모리걸 예측 헤드
        self.morigirl_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 인기도 예측 헤드
        self.popularity_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            x: (batch_size, 1025) - [image_vector(1024), price(1)]
            
        Returns:
            morigirl_prob: (batch_size, 1) - 모리걸일 확률
            popularity_prob: (batch_size, 1) - 인기도 확률
        """
        # 공통 특징 추출
        shared_features = self.shared_layers(x)
        
        # 각 태스크별 예측
        morigirl_prob = self.morigirl_head(shared_features)
        popularity_prob = self.popularity_head(shared_features)
        
        return morigirl_prob, popularity_prob
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        예측 수행 (평가 모드)
        
        Returns:
            dict: {'morigirl_prob': tensor, 'popularity_prob': tensor}
        """
        self.eval()
        with torch.no_grad():
            morigirl_prob, popularity_prob = self.forward(x)
            return {
                'morigirl_prob': morigirl_prob,
                'popularity_prob': popularity_prob
            }

class ProductScoreLoss(nn.Module):
    """
    멀티태스크 손실함수
    모리걸 분류와 인기도 회귀를 동시에 최적화
    """
    
    def __init__(self, 
                 morigirl_weight: float = 1.0,
                 popularity_weight: float = 1.0,
                 class_balance: bool = True):
        super(ProductScoreLoss, self).__init__()
        
        self.morigirl_weight = morigirl_weight
        self.popularity_weight = popularity_weight
        self.class_balance = class_balance
        
        # 모리걸 분류 손실 (이진 분류)
        self.bce_loss = nn.BCELoss()
        
        # 인기도 회귀 손실
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                morigirl_pred: torch.Tensor,
                popularity_pred: torch.Tensor,
                morigirl_target: torch.Tensor,
                popularity_target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        손실 계산
        
        Args:
            morigirl_pred: (batch_size, 1) - 모리걸 예측값
            popularity_pred: (batch_size, 1) - 인기도 예측값
            morigirl_target: (batch_size, 1) - 모리걸 정답
            popularity_target: (batch_size, 1) - 인기도 정답
            
        Returns:
            dict: 각 손실값들
        """
        # 모리걸 분류 손실
        morigirl_loss = self.bce_loss(morigirl_pred, morigirl_target)
        
        # 인기도 회귀 손실
        popularity_loss = self.mse_loss(popularity_pred, popularity_target)
        
        # 총 손실
        total_loss = (self.morigirl_weight * morigirl_loss + 
                     self.popularity_weight * popularity_loss)
        
        return {
            'total_loss': total_loss,
            'morigirl_loss': morigirl_loss,
            'popularity_loss': popularity_loss
        }

class LightProductScorePredictor(nn.Module):
    """
    경량화된 상품 점수 예측 모델 (모바일용)
    """
    
    def __init__(self, 
                 input_dim: int = 1025,
                 hidden_dim: int = 256,
                 dropout: float = 0.2):
        super(LightProductScorePredictor, self).__init__()
        
        # 더 작은 네트워크
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # 공유 출력 레이어
        self.output_layer = nn.Linear(hidden_dim // 2, 2)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        outputs = torch.sigmoid(self.output_layer(features))
        
        morigirl_prob = outputs[:, 0:1]
        popularity_prob = outputs[:, 1:2]
        
        return morigirl_prob, popularity_prob

def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """모델 정보 반환"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 크기 추정 (MB)
    param_size = total_params * 4 / (1024 * 1024)  # float32 기준
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': param_size
    }

def create_model_from_config(config_path: str = "./config.json") -> ProductScorePredictor:
    """설정 파일에서 모델 생성"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_config = config['model']
    
    model = ProductScorePredictor(
        input_dim=model_config['input_vector_dim'] + 1,  # +1 for price
        hidden_dim=model_config['hidden_dim'],
        dropout=model_config['dropout']
    )
    
    return model

# 사용 예시
if __name__ == "__main__":
    # 모델 생성 및 테스트
    print("=== 상품 점수 예측 모델 테스트 ===")
    
    # 일반 모델
    model = ProductScorePredictor()
    info = get_model_info(model)
    
    print(f"📊 모델 정보:")
    print(f"  - 총 파라미터: {info['total_params']:,}")
    print(f"  - 학습 가능 파라미터: {info['trainable_params']:,}")
    print(f"  - 모델 크기: {info['model_size_mb']:.2f} MB")
    
    # 테스트 입력
    batch_size = 4
    input_dim = 1025  # 1024(vector) + 1(price)
    test_input = torch.randn(batch_size, input_dim)
    
    # 순전파 테스트
    model.eval()
    with torch.no_grad():
        morigirl_prob, popularity_prob = model(test_input)
        
        print(f"\n🧪 순전파 테스트:")
        print(f"  - 입력 크기: {test_input.shape}")
        print(f"  - 모리걸 확률 출력: {morigirl_prob.shape}")
        print(f"  - 인기도 확률 출력: {popularity_prob.shape}")
        print(f"  - 모리걸 확률 범위: {morigirl_prob.min():.3f} ~ {morigirl_prob.max():.3f}")
        print(f"  - 인기도 확률 범위: {popularity_prob.min():.3f} ~ {popularity_prob.max():.3f}")
    
    # 경량 모델 테스트
    print(f"\n=== 경량 모델 ===")
    light_model = LightProductScorePredictor()
    light_info = get_model_info(light_model)
    
    print(f"📊 경량 모델 정보:")
    print(f"  - 총 파라미터: {light_info['total_params']:,}")
    print(f"  - 모델 크기: {light_info['model_size_mb']:.2f} MB")
    
    # 손실함수 테스트
    print(f"\n=== 손실함수 테스트 ===")
    criterion = ProductScoreLoss()
    
    # 더미 타겟
    morigirl_target = torch.randint(0, 2, (batch_size, 1)).float()
    popularity_target = torch.rand(batch_size, 1)
    
    losses = criterion(morigirl_prob, popularity_prob, morigirl_target, popularity_target)
    
    print(f"  - 총 손실: {losses['total_loss']:.4f}")
    print(f"  - 모리걸 손실: {losses['morigirl_loss']:.4f}")
    print(f"  - 인기도 손실: {losses['popularity_loss']:.4f}")
    
    print("\n✅ 모든 테스트 통과!") 