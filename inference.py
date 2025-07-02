# inference.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse

from model.morigirl_model import MoriGirlClassifier

def create_inference_transform():
    """추론용 이미지 전처리 파이프라인"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

def load_model(checkpoint_path, device):
    """학습된 모델 로드"""
    model = MoriGirlClassifier()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_single_image(model, image_path, transform, device, threshold=0.5):
    """단일 이미지 예측"""
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    
    # 전처리
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob > threshold else 0
    
    return prob, pred

def batch_inference(model, image_dir, transform, device, threshold=0.5):
    """폴더 내 모든 이미지 일괄 예측"""
    results = []
    
    # 이미지 파일들 찾기
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"총 {len(image_files)}개 이미지 처리 중...")
    
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        try:
            prob, pred = predict_single_image(
                model, image_path, transform, device, threshold
            )
            
            result = {
                'filename': filename,
                'morigirl_prob': prob,
                'prediction': '모리걸' if pred == 1 else '일반',
                'confidence': prob if pred == 1 else (1 - prob)
            }
            results.append(result)
            
            print(f"{filename}: {result['prediction']} ({result['confidence']:.3f})")
            
        except Exception as e:
            print(f"❌ {filename} 처리 실패: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='모리걸 스타일 분류기 추론')
    parser.add_argument('--checkpoint', required=True, 
                       help='모델 체크포인트 경로')
    parser.add_argument('--image', type=str, 
                       help='단일 이미지 파일 경로')
    parser.add_argument('--image_dir', type=str, 
                       help='이미지 폴더 경로 (일괄 처리)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='분류 임계값 (기본값: 0.5)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 모델 로드
    print("모델 로딩 중...")
    model = load_model(args.checkpoint, device)
    transform = create_inference_transform()
    
    if args.image:
        # 단일 이미지 예측
        prob, pred = predict_single_image(
            model, args.image, transform, device, args.threshold
        )
        
        print(f"\n=== 예측 결과 ===")
        print(f"이미지: {args.image}")
        print(f"모리걸 확률: {prob:.4f}")
        print(f"예측: {'모리걸' if pred == 1 else '일반'}")
        print(f"신뢰도: {prob if pred == 1 else (1-prob):.4f}")
        
    elif args.image_dir:
        # 폴더 일괄 처리
        results = batch_inference(
            model, args.image_dir, transform, device, args.threshold
        )
        
        # 결과 요약
        morigirl_count = sum(1 for r in results if r['prediction'] == '모리걸')
        print(f"\n=== 결과 요약 ===")
        print(f"총 이미지: {len(results)}개")
        print(f"모리걸: {morigirl_count}개")
        print(f"일반: {len(results) - morigirl_count}개")
        
    else:
        print("--image 또는 --image_dir 중 하나를 지정해주세요.")

if __name__ == "__main__":
    main() 