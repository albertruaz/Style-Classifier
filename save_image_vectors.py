#!/usr/bin/env python3
# save_image_vectors.py

import os
import sys
import json
import requests
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
import io
from typing import List, Dict, Any
from database import DatabaseManager

class ImageVectorExtractor:
    """이미지를 다운로드하고 벡터로 변환하여 DB에 저장"""
    
    def __init__(self, config_path: str = "./config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db_manager = DatabaseManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # EfficientNet 모델 로드 (특징 추출용)
        self.model = efficientnet_b0(pretrained=True)
        self.model.classifier = torch.nn.Identity()  # 분류층 제거, 특징만 추출
        self.model.eval()
        self.model.to(self.device)
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ 이미지 벡터 추출기 초기화 완료 (Device: {self.device})")

    def download_image(self, url: str, timeout: int = 10) -> Image.Image:
        """URL에서 이미지 다운로드"""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            raise ValueError(f"이미지 다운로드 실패: {e}")

    def extract_features(self, image: Image.Image) -> np.ndarray:
        """이미지에서 1024차원 특징 벡터 추출"""
        try:
            # 전처리
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 특징 추출
            with torch.no_grad():
                features = self.model(input_tensor)
                features = features.cpu().numpy().flatten()
            
            # L2 정규화
            features = features / np.linalg.norm(features)
            
            return features
        except Exception as e:
            raise ValueError(f"특징 추출 실패: {e}")

    def get_products_to_process(self, limit: int = 1000, offset: int = 0) -> List[Dict]:
        """처리할 상품 목록 조회"""
        session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT 
                    p.id,
                    p.main_image,
                    p.status,
                    p.primary_category_id,
                    p.secondary_category_id
                FROM vingle.product p
                LEFT JOIN product_vectors pv ON p.id = pv.id
                WHERE p.main_image IS NOT NULL
                  AND p.status IN ('SALE', 'SOLD_OUT')
                  AND pv.id IS NULL  -- 아직 벡터가 없는 상품만
                ORDER BY p.id
                LIMIT :limit OFFSET :offset
            """)
            
            result = session.execute(sql, {"limit": limit, "offset": offset})
            
            products = []
            for row in result.fetchall():
                # S3 URL 생성
                cloudfront_domain = os.getenv('S3_CLOUDFRONT_DOMAIN')
                image_url = f"https://{cloudfront_domain}/{row[1]}" if cloudfront_domain and row[1] else None
                
                if image_url:
                    products.append({
                        'product_id': row[0],
                        'image_url': image_url,
                        'status': row[2],
                        'primary_category_id': row[3],
                        'secondary_category_id': row[4]
                    })
            
            return products
            
        finally:
            session.close()

    def save_vectors_to_db(self, vectors_data: List[Dict[str, Any]]):
        """벡터 데이터를 DB에 저장"""
        if not vectors_data:
            return
        
        try:
            self.db_manager.vector_db.upsert_product_vectors(vectors_data)
            print(f"✅ {len(vectors_data)}개 벡터 저장 완료")
        except Exception as e:
            print(f"❌ 벡터 저장 실패: {e}")

    def process_products(self, limit: int = 1000, batch_size: int = 50):
        """상품들을 배치로 처리"""
        offset = 0
        total_processed = 0
        total_failed = 0
        
        while True:
            # 상품 조회
            products = self.get_products_to_process(limit, offset)
            
            if not products:
                print("처리할 상품이 더 이상 없습니다.")
                break
            
            print(f"\n📦 {len(products)}개 상품 처리 중... (Offset: {offset})")
            
            # 배치 단위로 처리
            vectors_batch = []
            
            for product in tqdm(products, desc="이미지 처리"):
                try:
                    # 이미지 다운로드
                    image = self.download_image(product['image_url'])
                    
                    # 특징 벡터 추출
                    features = self.extract_features(image)
                    
                    # 벡터 데이터 준비
                    vectors_batch.append({
                        'product_id': product['product_id'],
                        'image_vector': features.tolist(),
                        'status': product['status'],
                        'primary_category_id': product['primary_category_id'],
                        'secondary_category_id': product['secondary_category_id']
                    })
                    
                    total_processed += 1
                    
                    # 배치 크기에 도달하면 저장
                    if len(vectors_batch) >= batch_size:
                        self.save_vectors_to_db(vectors_batch)
                        vectors_batch = []
                    
                except Exception as e:
                    print(f"⚠️ 상품 {product['product_id']} 처리 실패: {e}")
                    total_failed += 1
                    continue
            
            # 남은 배치 저장
            if vectors_batch:
                self.save_vectors_to_db(vectors_batch)
            
            offset += limit
            
            print(f"📊 진행 상황: 성공 {total_processed}개, 실패 {total_failed}개")
        
        print(f"\n🎉 전체 처리 완료!")
        print(f"  - 총 처리 성공: {total_processed}개")
        print(f"  - 총 처리 실패: {total_failed}개")

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'db_manager'):
            self.db_manager.dispose_all()

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='상품 이미지 벡터 생성 및 저장')
    parser.add_argument('--limit', type=int, default=1000, help='한 번에 처리할 상품 수')
    parser.add_argument('--batch-size', type=int, default=50, help='DB 저장 배치 크기')
    parser.add_argument('--max-products', type=int, default=10000, help='총 처리할 최대 상품 수')
    
    args = parser.parse_args()
    
    try:
        extractor = ImageVectorExtractor()
        
        print(f"🚀 이미지 벡터 생성 시작")
        print(f"  - 한 번에 처리: {args.limit}개")
        print(f"  - 배치 크기: {args.batch_size}개")
        print(f"  - 최대 처리: {args.max_products}개")
        
        # Vector DB 테이블 생성
        extractor.db_manager.vector_db.create_product_table(dimension=1024)
        
        # 상품 처리
        extractor.process_products(
            limit=args.limit,
            batch_size=args.batch_size
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        try:
            extractor.close()
        except:
            pass

if __name__ == "__main__":
    main() 