# model/morigirl_model.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class MoriGirlClassifier(nn.Module):
    """
    EfficientNet-B0 기반 모리걸 스타일 이진 분류기
    
    입력: RGB 이미지 (3, 224, 224)
    출력: 모리걸일 확률 (sigmoid 적용 전 logit)
    """
    
    def __init__(self, pretrained=True, dropout_rate=0.3):
        super(MoriGirlClassifier, self).__init__()
        
        # EfficientNet-B0 백본 로드
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)
        
        # 분류기 헤드 교체 (이진 분류용)
        num_features = self.backbone.classifier[1].in_features  # 1280
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(512, 1)  # 이진 분류: 출력 1개
        )
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        분류기 헤드의 가중치 초기화
        """
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: (batch_size, 3, 224, 224) 형태의 이미지 텐서
            
        Returns:
            logits: (batch_size, 1) 형태의 로짓 (sigmoid 적용 전)
        """
        logits = self.backbone(x)
        return logits
    
    def predict(self, x, threshold=0.5):
        """
        예측 수행 (확률과 클래스 반환)
        
        Args:
            x: 이미지 텐서
            threshold: 분류 임계값
            
        Returns:
            probs: 모리걸일 확률
            preds: 예측 클래스 (0: 일반, 1: 모리걸)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()
        return probs, preds
    
    def get_feature_maps(self, x, layer_name='features'):
        """
        특정 레이어의 특징맵 추출 (시각화용)
        
        Args:
            x: 이미지 텐서
            layer_name: 추출할 레이어 이름
            
        Returns:
            features: 특징맵
        """
        features = {}
        
        def hook(name):
            def hook_fn(module, input, output):
                features[name] = output.detach()
            return hook_fn
        
        # 훅 등록
        if layer_name == 'features':
            handle = self.backbone.features.register_forward_hook(hook('features'))
        
        # 순전파
        self.eval()
        with torch.no_grad():
            _ = self.forward(x)
        
        # 훅 제거
        handle.remove()
        
        return features.get(layer_name, None)

class LightMoriGirlClassifier(nn.Module):
    """
    더욱 경량화된 모리걸 분류기 (모바일용)
    파라미터 수를 대폭 줄인 버전
    """
    
    def __init__(self, dropout_rate=0.2):
        super(LightMoriGirlClassifier, self).__init__()
        
        # 간단한 CNN 아키텍처
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
    def predict(self, x, threshold=0.5):
        """예측 수행"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
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
    
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능 파라미터 수: {trainable_params:,}")
    print(f"예상 모델 크기: {param_size:.2f} MB")
    
    return total_params, trainable_params, param_size

# 사용 예시
if __name__ == "__main__":
    # 모델 생성 및 정보 출력
    print("=== EfficientNet-B0 기반 모리걸 분류기 ===")
    model = MoriGirlClassifier()
    get_model_info(model)
    
    print("\n=== 경량 모리걸 분류기 ===")
    light_model = LightMoriGirlClassifier()
    get_model_info(light_model)
    
    # 테스트 입력
    x = torch.randn(1, 3, 224, 224)
    
    # 순전파 테스트
    with torch.no_grad():
        logits = model(x)
        probs, preds = model.predict(x)
        print(f"\n모리걸 확률: {probs.item():.4f}")
        print(f"예측 클래스: {'모리걸' if preds.item() == 1 else '일반'}") 