# test_model.py

import torch
from model.morigirl_model import MoriGirlClassifier, LightMoriGirlClassifier, get_model_info

def test_models():
    """
    모델들의 기본 동작을 테스트
    """
    print("🧪 모델 테스트 시작")
    print("="*50)
    
    # 테스트 입력 생성
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 224, 224)
    print(f"테스트 입력 크기: {test_input.shape}")
    
    # EfficientNet-B0 기반 모델 테스트
    print("\n1️⃣ EfficientNet-B0 기반 모리걸 분류기")
    print("-" * 40)
    
    model = MoriGirlClassifier()
    get_model_info(model)
    
    # 순전파 테스트
    model.eval()
    with torch.no_grad():
        outputs = model(test_input)
        probs, preds = model.predict(test_input)
    
    print(f"\n출력 로짓 크기: {outputs.shape}")
    print(f"확률 크기: {probs.shape}")
    print(f"예측 크기: {preds.shape}")
    print(f"확률 범위: {probs.min().item():.4f} ~ {probs.max().item():.4f}")
    
    # 경량 모델 테스트
    print("\n2️⃣ 경량 모리걸 분류기")
    print("-" * 40)
    
    light_model = LightMoriGirlClassifier()
    get_model_info(light_model)
    
    # 순전파 테스트
    light_model.eval()
    with torch.no_grad():
        light_outputs = light_model(test_input)
        light_probs, light_preds = light_model.predict(test_input)
    
    print(f"\n출력 로짓 크기: {light_outputs.shape}")
    print(f"확률 크기: {light_probs.shape}")
    print(f"예측 크기: {light_preds.shape}")
    print(f"확률 범위: {light_probs.min().item():.4f} ~ {light_probs.max().item():.4f}")
    
    # GPU 테스트 (가능한 경우)
    if torch.cuda.is_available():
        print("\n3️⃣ GPU 테스트")
        print("-" * 40)
        
        device = torch.device("cuda")
        model_gpu = model.to(device)
        test_input_gpu = test_input.to(device)
        
        with torch.no_grad():
            gpu_outputs = model_gpu(test_input_gpu)
        
        print(f"GPU 출력 크기: {gpu_outputs.shape}")
        print(f"GPU 디바이스: {gpu_outputs.device}")
        print("✅ GPU 테스트 성공")
    else:
        print("\n⚠️ GPU가 사용 불가능합니다")
    
    print("\n🎉 모든 테스트 통과!")
    print("\n다음 단계:")
    print("1. python setup_data.py 로 데이터 준비")
    print("2. python main.py 로 학습 시작")

if __name__ == "__main__":
    test_models() 