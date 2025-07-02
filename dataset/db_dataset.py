# dataset/db_dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import os
from tqdm import tqdm

from database import DatabaseManager

class DBProductDataset(Dataset):
    """
    데이터베이스에서 상품 이미지를 가져와서 모리걸 분류용 데이터셋으로 사용하는 클래스
    """
    
    def __init__(self, 
                 where_condition: str = "status = 'SALE'", 
                 transform=None,
                 limit: Optional[int] = None,
                 cache_images: bool = False):
        """
        Args:
            where_condition: MySQL WHERE 조건문
            transform: 이미지 전처리 함수
            limit: 최대 로드할 이미지 수
            cache_images: 이미지를 메모리에 캐싱할지 여부
        """
        self.where_condition = where_condition
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {}
        
        # 데이터베이스 매니저 초기화
        self.db_manager = DatabaseManager()
        
        # 상품 데이터 로드
        print(f"📥 데이터베이스에서 상품 데이터 로딩 중...")
        self.products = self._load_products_from_db(limit)
        
        print(f"✅ {len(self.products)}개 상품 로드 완료")
        
        if self.cache_images:
            print("🖼️ 이미지 캐싱 중... (시간이 걸릴 수 있습니다)")
            self._preload_images()
    
    def _load_products_from_db(self, limit: Optional[int]) -> List[Tuple[int, str]]:
        """데이터베이스에서 상품 이미지 URL 목록 로드"""
        products = []
        batch_size = 1000
        batch_no = 0
        
        while True:
            # 배치 단위로 데이터 로드
            batch_products = self.db_manager.mysql.get_product_images(
                where_condition=self.where_condition,
                limit=batch_size,
                batch_no=batch_no
            )
            
            if not batch_products:
                break
                
            products.extend(batch_products)
            batch_no += 1
            
            # limit 체크
            if limit and len(products) >= limit:
                products = products[:limit]
                break
                
            print(f"  배치 {batch_no}: {len(batch_products)}개 추가 (총 {len(products)}개)")
        
        return products
    
    def _preload_images(self):
        """모든 이미지를 미리 로드해서 캐싱"""
        for idx in tqdm(range(len(self.products)), desc="이미지 캐싱"):
            try:
                image = self._load_image_from_url(self.products[idx][1])
                if image:
                    self.image_cache[idx] = image
            except Exception as e:
                print(f"⚠️ 이미지 로드 실패 (인덱스 {idx}): {e}")
                continue
    
    def _load_image_from_url(self, image_url: str) -> Optional[Image.Image]:
        """URL에서 이미지 로드"""
        if not image_url:
            return None
            
        try:
            # HTTP 요청으로 이미지 다운로드
            response = requests.get(image_url, timeout=10, 
                                  headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            # PIL Image로 변환
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
            
        except Exception as e:
            # print(f"이미지 로드 실패 ({image_url}): {e}")
            return None
    
    def __len__(self):
        return len(self.products)
    
    def __getitem__(self, idx):
        """
        데이터 샘플 반환
        
        Returns:
            image: 전처리된 이미지 텐서
            product_id: 상품 ID
        """
        product_id, image_url = self.products[idx]
        
        # 캐시에서 이미지 가져오기
        if self.cache_images and idx in self.image_cache:
            image = self.image_cache[idx]
        else:
            image = self._load_image_from_url(image_url)
        
        # 이미지 로드 실패 시 더미 이미지 생성
        if image is None:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, product_id
    
    def get_product_info(self, idx: int) -> Dict[str, Any]:
        """특정 인덱스의 상품 정보 반환"""
        product_id, image_url = self.products[idx]
        return {
            'product_id': product_id,
            'image_url': image_url,
            'index': idx
        }
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, 'db_manager'):
            self.db_manager.dispose_all()

class DBMorigirlInferenceDataset(Dataset):
    """
    데이터베이스의 상품들에 대해 모리걸 추론을 수행하는 전용 데이터셋
    """
    
    def __init__(self, 
                 where_condition: str = "status = 'SALE' AND main_image IS NOT NULL",
                 transform=None,
                 batch_size: int = 1000):
        """
        Args:
            where_condition: 추론할 상품들의 조건
            transform: 이미지 전처리 함수  
            batch_size: 한 번에 로드할 상품 수
        """
        self.where_condition = where_condition
        self.transform = transform
        self.batch_size = batch_size
        
        # 데이터베이스 매니저
        self.db_manager = DatabaseManager()
        
        # 전체 상품 수 확인
        self.total_count = self.db_manager.mysql.get_product_count(where_condition)
        print(f"📊 추론 대상 상품 수: {self.total_count:,}개")
        
        # 현재 로드된 배치
        self.current_batch = []
        self.current_batch_no = 0
        self.current_index = 0
        
        # 첫 번째 배치 로드
        self._load_next_batch()
    
    def _load_next_batch(self):
        """다음 배치 로드"""
        self.current_batch = self.db_manager.mysql.get_product_images(
            where_condition=self.where_condition,
            limit=self.batch_size,
            batch_no=self.current_batch_no
        )
        self.current_batch_no += 1
        self.current_index = 0
        
        print(f"📦 배치 {self.current_batch_no} 로드: {len(self.current_batch)}개 상품")
    
    def __len__(self):
        return self.total_count
    
    def __getitem__(self, idx):
        # 현재 배치의 범위를 벗어나면 다음 배치 로드
        if self.current_index >= len(self.current_batch):
            if len(self.current_batch) < self.batch_size:
                # 더 이상 데이터가 없음
                raise StopIteration("No more data available")
            self._load_next_batch()
        
        product_id, image_url = self.current_batch[self.current_index]
        self.current_index += 1
        
        # 이미지 로드
        image = self._load_image_from_url(image_url)
        if image is None:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, product_id
    
    def _load_image_from_url(self, image_url: str) -> Optional[Image.Image]:
        """URL에서 이미지 로드"""
        if not image_url:
            return None
            
        try:
            response = requests.get(image_url, timeout=5,
                                  headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except:
            return None
    
    def get_progress(self) -> Dict[str, Any]:
        """현재 진행 상황 반환"""
        processed = (self.current_batch_no - 1) * self.batch_size + self.current_index
        return {
            'processed': processed,
            'total': self.total_count,
            'progress_pct': (processed / self.total_count) * 100 if self.total_count > 0 else 0,
            'current_batch': self.current_batch_no
        }

def save_morigirl_predictions_to_db(predictions: Dict[str, Dict[str, Any]]):
    """
    모리걸 예측 결과를 데이터베이스에 저장
    
    Args:
        predictions: {product_id: {'is_morigirl': bool, 'confidence': float}}
    """
    db_manager = DatabaseManager()
    
    try:
        # MySQL과 Vector DB 양쪽에 저장
        db_manager.mysql.update_morigirl_predictions(predictions)
        db_manager.vector_db.update_morigirl_predictions(predictions)
        
        print(f"✅ {len(predictions)}개 상품의 모리걸 예측 결과 저장 완료")
        
    except Exception as e:
        print(f"❌ 예측 결과 저장 실패: {e}")
        raise
    finally:
        db_manager.dispose_all()

# 사용 예시
if __name__ == "__main__":
    from torchvision import transforms
    
    # 이미지 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 생성 테스트
    try:
        dataset = DBProductDataset(
            where_condition="status = 'SALE'",
            transform=transform,
            limit=10  # 테스트용으로 10개만
        )
        
        print(f"데이터셋 크기: {len(dataset)}")
        
        # 첫 번째 샘플 테스트
        if len(dataset) > 0:
            image, product_id = dataset[0]
            print(f"첫 번째 샘플 - 상품 ID: {product_id}, 이미지 크기: {image.shape}")
            
    except Exception as e:
        print(f"데이터셋 테스트 실패: {e}")
        print("데이터베이스 연결을 확인하고 .env 파일을 설정하세요.") 