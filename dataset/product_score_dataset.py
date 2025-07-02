# dataset/product_score_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
import json
from typing import List, Tuple, Dict, Any, Optional
import os
from tqdm import tqdm

from database import DatabaseManager

class ProductScoreDataset(Dataset):
    """
    MySQL과 Vector DB에서 데이터를 가져와서 
    모리걸 확률과 판매 확률을 예측하는 데이터셋
    
    Input: [image_vector(1024dim), price] -> shape: (1025,)
    Output: [morigirl_prob, popularity_prob] -> shape: (2,)
    """
    
    def __init__(self, config_path: str = "./config.json", mode: str = "train"):
        """
        Args:
            config_path: 설정 파일 경로
            mode: 'train' 또는 'test'
        """
        self.mode = mode
        
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.style_id = self.config['style_id']
        self.status_filter = self.config['data']['status_filter']
        
        # 데이터베이스 매니저
        self.db_manager = DatabaseManager()
        
        # 데이터 로드
        print(f"📊 {mode} 데이터 로딩 중...")
        self.data = self._load_and_process_data()
        
        print(f"✅ {len(self.data)}개의 {mode} 샘플 로드 완료")
        self._print_statistics()
    
    def _load_and_process_data(self) -> List[Dict[str, Any]]:
        """데이터 로드 및 가공"""
        
        # 1. product_styles에서 style_id가 9인 product_id들 가져오기
        style_product_ids = self._get_style_product_ids()
        print(f"🎨 Style {self.style_id} 상품: {len(style_product_ids)}개")
        
        # 2. product 테이블에서 두 그룹 데이터 가져오기
        style_products = self._get_product_data(style_product_ids, is_style_group=True)
        non_style_products = self._get_product_data(style_product_ids, is_style_group=False)
        
        print(f"📦 Style 그룹: {len(style_products)}개")
        print(f"📦 Non-style 그룹: {len(non_style_products)}개")
        
        # 3. 두 그룹 합치기
        all_products = style_products + non_style_products
        
        # 4. Vector DB에서 임베딩 벡터 가져오기
        product_ids = [p['product_id'] for p in all_products]
        print(f"🔍 벡터 조회 대상: {len(product_ids)}개 상품")
        
        product_vectors = self._get_product_vectors(product_ids)
        print(f"📦 실제 벡터 발견: {len(product_vectors)}개")
        
        if len(product_vectors) == 0:
            print("⚠️ Vector DB에 벡터 데이터가 없습니다!")
            print("💡 해결 방법:")
            print("1. python setup_data.py 로 이미지 벡터 생성")
            print("2. 또는 더미 벡터로 테스트: self._create_dummy_vectors() 호출")
            
            # 더미 벡터 생성 (테스트용)
            product_vectors = self._create_dummy_vectors(product_ids[:100])  # 처음 100개만
            print(f"🎲 더미 벡터 생성: {len(product_vectors)}개")
        
        # 5. 데이터 결합 및 가공
        final_data = self._combine_and_process_data(all_products, product_vectors)
        
        return final_data
    
    def _get_style_product_ids(self) -> List[int]:
        """product_styles 테이블에서 특정 style_id인 product_id들 조회"""
        session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT product_id 
                FROM vingle.product_styles 
                WHERE styles_id LIKE :style_id
            """)
            
            result = session.execute(sql, {"style_id": str(self.style_id)})
            product_ids = [row[0] for row in result.fetchall()]
            
            return product_ids
            
        finally:
            session.close()
    
    def _get_product_data(self, style_product_ids: List[int], is_style_group: bool) -> List[Dict[str, Any]]:
        """
        product 테이블에서 데이터 조회
        
        Args:
            style_product_ids: style_id가 9인 product_id 리스트
            is_style_group: True면 style 그룹, False면 non-style 그룹
        """
        session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            # 조건 설정
            if is_style_group:
                # style_id가 9인 상품들
                id_condition = "id IN :product_ids"
                params = {"product_ids": style_product_ids}
            else:
                # style_id가 9가 아닌 상품들
                id_condition = "id NOT IN :product_ids"
                params = {"product_ids": style_product_ids}
            
            status_condition = " OR ".join([f"status = '{status}'" for status in self.status_filter])
            
            # MySQL IN 쿼리를 안전하게 처리
            if is_style_group and style_product_ids:
                # IN 쿼리용 플레이스홀더 생성
                placeholders = ','.join([':id' + str(j) for j in range(len(style_product_ids))])
                id_condition = f"id IN ({placeholders})"
                params = {f'id{j}': style_product_ids[j] for j in range(len(style_product_ids))}
            elif not is_style_group and style_product_ids:
                # NOT IN 쿼리용 플레이스홀더 생성
                placeholders = ','.join([':id' + str(j) for j in range(len(style_product_ids))])
                id_condition = f"id NOT IN ({placeholders})"
                params = {f'id{j}': style_product_ids[j] for j in range(len(style_product_ids))}
            else:
                # style_product_ids가 비어있는 경우
                id_condition = "1=1" if is_style_group else "1=0"
                params = {}
            
            sql = text(f"""
                SELECT 
                    id as product_id,
                    status,
                    views,
                    impressions,
                    amount
                FROM vingle.product 
                WHERE {id_condition}
                  AND ({status_condition})
                  AND views IS NOT NULL 
                  AND impressions IS NOT NULL
                  AND impressions > 0
                  AND amount IS NOT NULL
                ORDER BY RAND()
                LIMIT 10000
            """)
            
            result = session.execute(sql, params)
            
            products = []
            for row in result.fetchall():
                products.append({
                    'product_id': row[0],
                    'status': row[1],
                    'views': row[2],
                    'impressions': row[3],
                    'amount': row[4],
                    'is_style_group': is_style_group
                })
            
            return products
            
        finally:
            session.close()
    
    def _get_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """Vector DB에서 상품 임베딩 벡터 조회"""
        if not product_ids:
            return {}
            
        session = self.db_manager.vector_db.Session()
        try:
            from sqlalchemy import text
            
            # 배치 단위로 처리
            batch_size = self.config['database']['vector_batch_size']
            all_vectors = {}
            
            for i in tqdm(range(0, len(product_ids), batch_size), desc="벡터 로딩"):
                batch_ids = product_ids[i:i + batch_size]
                
                if not batch_ids:
                    continue
                
                # IN 쿼리용 플레이스홀더 생성
                placeholders = ','.join([':id' + str(j) for j in range(len(batch_ids))])
                sql = text(f"""
                    SELECT id, image_vector
                    FROM product_vectors
                    WHERE id IN ({placeholders})
                """)
                
                # 파라미터 딕셔너리 생성
                params = {f'id{j}': batch_ids[j] for j in range(len(batch_ids))}
                result = session.execute(sql, params)
                
                for product_id, vector_str in result.fetchall():
                    # PostgreSQL VECTOR 형식을 numpy 배열로 변환
                    if vector_str:
                        # "[1.0,2.0,3.0,...]" 형식을 파싱
                        vector_str = vector_str.strip('[]')
                        vector = np.array([float(x) for x in vector_str.split(',')])
                        all_vectors[product_id] = vector
            
            return all_vectors
            
        finally:
            session.close()
    
    def _create_dummy_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """테스트용 더미 벡터 생성"""
        dummy_vectors = {}
        np.random.seed(42)  # 재현 가능한 결과
        
        for product_id in product_ids:
            # 1024차원 더미 벡터 생성 (정규화됨)
            vector = np.random.randn(1024)
            vector = vector / np.linalg.norm(vector)  # L2 정규화
            dummy_vectors[product_id] = vector
            
        return dummy_vectors
    
    def _combine_and_process_data(self, products: List[Dict], vectors: Dict[int, np.ndarray]) -> List[Dict[str, Any]]:
        """데이터 결합 및 가공"""
        processed_data = []
        
        for product in tqdm(products, desc="데이터 가공"):
            product_id = product['product_id']
            
            # 벡터가 없는 상품은 제외
            if product_id not in vectors:
                continue
            
            # Popularity 계산
            if product['status'] == 'SOLD_OUT':
                popularity = 1.0
            else:  # SALE
                if product['impressions'] > 0:
                    popularity = min(product['views'] / product['impressions'], 1.0)
                else:
                    popularity = 0.0
            
            # Accuracy 계산 (모리걸 여부)
            accuracy = 1.0 if product['is_style_group'] else 0.0
            
            # 입력 벡터 구성: [image_vector(1024), price(1)]
            image_vector = vectors[product_id]
            price = float(product['amount'])
            
            # 가격 정규화 (log 스케일 적용)
            price_normalized = np.log(max(price, 1.0)) / 10.0  # 간단한 정규화
            
            input_vector = np.concatenate([image_vector, [price_normalized]])
            
            processed_data.append({
                'product_id': product_id,
                'input_vector': input_vector,  # shape: (1025,)
                'morigirl_prob': accuracy,     # 0 or 1
                'popularity_prob': popularity, # 0~1
                'raw_data': product
            })
        
        return processed_data
    
    def _print_statistics(self):
        """데이터 통계 출력"""
        if not self.data:
            return
            
        morigirl_count = sum(1 for d in self.data if d['morigirl_prob'] == 1.0)
        avg_popularity = np.mean([d['popularity_prob'] for d in self.data])
        
        print(f"📊 데이터 통계:")
        print(f"  - 총 샘플 수: {len(self.data)}")
        print(f"  - 모리걸 샘플: {morigirl_count} ({morigirl_count/len(self.data)*100:.1f}%)")
        print(f"  - 평균 인기도: {avg_popularity:.3f}")
        
        # 가격 분포
        prices = [d['raw_data']['amount'] for d in self.data]
        print(f"  - 가격 범위: {min(prices):,}원 ~ {max(prices):,}원")
        print(f"  - 평균 가격: {np.mean(prices):,.0f}원")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_vector: torch.Tensor, shape (1025,) - [image_vector(1024), price(1)]
            targets: torch.Tensor, shape (2,) - [morigirl_prob, popularity_prob]
        """
        sample = self.data[idx]
        
        input_vector = torch.FloatTensor(sample['input_vector'])
        targets = torch.FloatTensor([sample['morigirl_prob'], sample['popularity_prob']])
        
        return input_vector, targets
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """특정 샘플의 상세 정보 반환"""
        if idx >= len(self.data):
            return {}
        
        sample = self.data[idx]
        return {
            'product_id': sample['product_id'],
            'morigirl_prob': sample['morigirl_prob'],
            'popularity_prob': sample['popularity_prob'],
            'raw_data': sample['raw_data']
        }
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, 'db_manager'):
            self.db_manager.dispose_all()

def create_train_test_datasets(config_path: str = "./config.json") -> Tuple[ProductScoreDataset, ProductScoreDataset]:
    """훈련용과 테스트용 데이터셋 생성"""
    
    # 전체 데이터 로드
    full_dataset = ProductScoreDataset(config_path, mode="full")
    
    # train/test 분할
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    split_ratio = config['data']['train_test_split']
    total_size = len(full_dataset.data)
    train_size = int(total_size * split_ratio)
    
    # 랜덤 셔플
    np.random.shuffle(full_dataset.data)
    
    # 분할
    train_data = full_dataset.data[:train_size]
    test_data = full_dataset.data[train_size:]
    
    # 새로운 데이터셋 인스턴스 생성
    train_dataset = ProductScoreDataset.__new__(ProductScoreDataset)
    train_dataset.config = full_dataset.config
    train_dataset.mode = "train"
    train_dataset.data = train_data
    train_dataset.db_manager = full_dataset.db_manager
    
    test_dataset = ProductScoreDataset.__new__(ProductScoreDataset)
    test_dataset.config = full_dataset.config
    test_dataset.mode = "test"
    test_dataset.data = test_data
    test_dataset.db_manager = full_dataset.db_manager
    
    print(f"📊 데이터 분할 완료:")
    print(f"  - 훈련 데이터: {len(train_data)}개")
    print(f"  - 테스트 데이터: {len(test_data)}개")
    
    return train_dataset, test_dataset

# 사용 예시
if __name__ == "__main__":
    try:
        # 데이터셋 생성 테스트
        dataset = ProductScoreDataset()
        
        if len(dataset) > 0:
            # 첫 번째 샘플 확인
            input_vec, targets = dataset[0]
            sample_info = dataset.get_sample_info(0)
            
            print(f"\n📋 샘플 정보:")
            print(f"  - 상품 ID: {sample_info['product_id']}")
            print(f"  - 입력 벡터 크기: {input_vec.shape}")
            print(f"  - 모리걸 확률: {targets[0]:.3f}")
            print(f"  - 인기도 확률: {targets[1]:.3f}")
            print(f"  - 원본 데이터: {sample_info['raw_data']}")
        
    except Exception as e:
        print(f"❌ 데이터셋 테스트 실패: {e}")
        print("데이터베이스 연결을 확인하고 .env 파일을 설정하세요.") 