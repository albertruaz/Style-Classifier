# run_db_inference.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from tqdm import tqdm
import os

from model.morigirl_model import MoriGirlClassifier
from dataset.db_dataset import DBMorigirlInferenceDataset, save_morigirl_predictions_to_db

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

def run_batch_inference(model, dataloader, device, threshold=0.5):
    """배치 단위로 추론 수행"""
    predictions = {}
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, product_ids) in enumerate(tqdm(dataloader, desc="추론 중")):
            images = images.to(device)
            
            # 모델 추론
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # 결과 저장
            for prob, product_id in zip(probs, product_ids):
                predictions[str(product_id.item())] = {
                    'is_morigirl': prob > threshold,
                    'confidence': float(prob)
                }
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='데이터베이스 상품들에 대한 모리걸 분류 추론')
    parser.add_argument('--checkpoint', required=True,
                       help='모델 체크포인트 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='분류 임계값')
    parser.add_argument('--where_condition', type=str, 
                       default="status = 'SALE' AND main_image IS NOT NULL",
                       help='상품 필터링 조건')
    parser.add_argument('--save_to_db', action='store_true',
                       help='결과를 데이터베이스에 저장할지 여부')
    parser.add_argument('--max_products', type=int, default=None,
                       help='최대 처리할 상품 수 (테스트용)')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    
    # 체크포인트 파일 확인
    if not os.path.exists(args.checkpoint):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {args.checkpoint}")
        return
    
    try:
        # 모델 로드
        print("🤖 모델 로딩 중...")
        model = load_model(args.checkpoint, device)
        print("✅ 모델 로드 완료")
        
        # 데이터셋 생성
        print("📊 데이터베이스에서 상품 데이터 로딩 중...")
        transform = create_inference_transform()
        
        dataset = DBMorigirlInferenceDataset(
            where_condition=args.where_condition,
            transform=transform,
            batch_size=1000  # DB에서 배치 단위로 로드
        )
        
        # 최대 상품 수 제한 (테스트용)
        if args.max_products:
            dataset.total_count = min(dataset.total_count, args.max_products)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"📈 추론 시작 - 총 {dataset.total_count:,}개 상품")
        print(f"🔧 설정: 배치크기={args.batch_size}, 임계값={args.threshold}")
        
        # 추론 수행
        all_predictions = {}
        batch_count = 0
        
        try:
            while True:
                try:
                    # 현재 배치 추론
                    predictions = run_batch_inference(model, dataloader, device, args.threshold)
                    all_predictions.update(predictions)
                    batch_count += 1
                    
                    # 진행 상황 출력
                    progress = dataset.get_progress()
                    print(f"📊 진행률: {progress['progress_pct']:.1f}% "
                          f"({progress['processed']:,}/{progress['total']:,})")
                    
                    # 중간 저장 (1000개마다)
                    if args.save_to_db and len(all_predictions) % 1000 == 0:
                        print("💾 중간 저장 중...")
                        save_morigirl_predictions_to_db(predictions)
                        predictions.clear()  # 메모리 절약
                    
                    # 다음 배치 로드
                    dataset._load_next_batch()
                    
                except StopIteration:
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 중단됨")
        
        # 최종 결과 처리
        print(f"\n✅ 추론 완료 - 총 {len(all_predictions):,}개 상품 처리")
        
        # 결과 요약
        morigirl_count = sum(1 for pred in all_predictions.values() if pred['is_morigirl'])
        avg_confidence = sum(pred['confidence'] for pred in all_predictions.values()) / len(all_predictions)
        
        print(f"📊 결과 요약:")
        print(f"  - 모리걸로 분류된 상품: {morigirl_count:,}개 ({morigirl_count/len(all_predictions)*100:.1f}%)")
        print(f"  - 평균 신뢰도: {avg_confidence:.3f}")
        
        # 데이터베이스에 저장
        if args.save_to_db and all_predictions:
            print("💾 최종 결과를 데이터베이스에 저장 중...")
            save_morigirl_predictions_to_db(all_predictions)
            print("✅ 데이터베이스 저장 완료")
        
        # 상위 신뢰도 모리걸 상품들 출력
        morigirl_products = [(pid, pred) for pid, pred in all_predictions.items() 
                            if pred['is_morigirl']]
        morigirl_products.sort(key=lambda x: x[1]['confidence'], reverse=True)
        
        print(f"\n🏆 상위 10개 모리걸 상품:")
        for i, (product_id, pred) in enumerate(morigirl_products[:10], 1):
            print(f"  {i}. 상품 ID: {product_id}, 신뢰도: {pred['confidence']:.3f}")
        
    except Exception as e:
        print(f"❌ 추론 중 오류 발생: {e}")
        raise
    
    finally:
        print("🧹 리소스 정리 중...")
        if 'dataset' in locals():
            dataset.close()

if __name__ == "__main__":
    main() 