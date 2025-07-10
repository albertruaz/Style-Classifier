# dataset/morigirl_vector_dataset.py

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
import json

class MoriGirlVectorDataset(Dataset):
    """
    이미지 벡터 기반 모리걸 분류 데이터셋
    
    save_image_vectors.py에서 생성한 npy 파일들을 로드하여 사용
    """
    
    def __init__(self, 
                 data_path: str,
                 data_type: str = "all",  # "morigirl", "non_morigirl", "all"
                 normalize_vectors: bool = True,
                 normalize_prices: bool = True):
        """
        Args:
            data_path: data/training_data 폴더 경로
            data_type: 로드할 데이터 타입
            normalize_vectors: 벡터 정규화 여부
            normalize_prices: 가격 정규화 여부
        """
        self.data_path = data_path
        self.data_type = data_type
        self.normalize_vectors = normalize_vectors
        self.normalize_prices = normalize_prices
        
        # 데이터 로드
        self.data = self._load_data()
        
        print(f"✅ {data_type} 데이터 로드 완료: {len(self.data)}개")
        self._print_statistics()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """npy 파일들에서 데이터 로드"""
        all_data = []
        
        if self.data_type in ["morigirl", "all"]:
            morigirl_data = self._load_npy_files("morigirl")
            all_data.extend(morigirl_data)
            print(f"📦 모리걸 데이터: {len(morigirl_data)}개")
        
        if self.data_type in ["non_morigirl", "all"]:
            non_morigirl_data = self._load_npy_files("non_morigirl")
            all_data.extend(non_morigirl_data)
            print(f"📦 비모리걸 데이터: {len(non_morigirl_data)}개")
        
        if not all_data:
            raise ValueError(f"데이터를 찾을 수 없습니다: {self.data_path}")
        
        # 데이터 후처리
        all_data = self._postprocess_data(all_data)
        
        return all_data
    
    def _load_npy_files(self, file_prefix: str) -> List[Dict[str, Any]]:
        """특정 prefix의 npy 파일들 로드"""
        data = []
        
        # 폴더에서 해당 prefix로 시작하는 npy 파일들 찾기
        for filename in os.listdir(self.data_path):
            if filename.startswith(file_prefix) and filename.endswith('.npy'):
                file_path = os.path.join(self.data_path, filename)
                print(f"  📁 로딩: {filename}")
                
                try:
                    # npy 파일 로드
                    file_data = np.load(file_path, allow_pickle=True)
                    
                    # 리스트로 변환하여 추가
                    if isinstance(file_data, np.ndarray) and file_data.dtype == object:
                        # object 배열인 경우 (딕셔너리들의 배열)
                        data.extend(file_data.tolist())
                    else:
                        # 일반 배열인 경우
                        data.extend(file_data)
                    
                    print(f"    ✅ {len(file_data)}개 추가")
                    
                except Exception as e:
                    print(f"    ❌ 파일 로드 실패: {e}")
                    continue
        
        return data
    
    def _postprocess_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """데이터 후처리 (정규화 등)"""
        processed_data = []
        
        # 가격 정규화를 위한 통계
        if self.normalize_prices:
            prices = [item['price'] for item in data if 'price' in item]
            if prices:
                price_mean = np.mean(prices)
                price_std = np.std(prices)
                print(f"📊 가격 통계: 평균={price_mean:.2f}, 표준편차={price_std:.2f}")
        
        for item in data:
            try:
                # 필수 필드 확인
                if not all(key in item for key in ['vector', 'is_morigirl']):
                    continue
                
                # 벡터 처리
                vector = np.array(item['vector'], dtype=np.float32)
                if len(vector) != 1024:
                    print(f"⚠️ 벡터 차원 오류: {len(vector)}, 건너뜀")
                    continue
                
                # 벡터 정규화
                if self.normalize_vectors:
                    norm = np.linalg.norm(vector)
                    if norm > 0:
                        vector = vector / norm
                
                # 가격 정규화
                price = item.get('price', 0)
                if self.normalize_prices and 'price' in item and price_std > 0:
                    price = (price - price_mean) / price_std
                
                processed_item = {
                    'product_id': item.get('product_id', 0),
                    'vector': vector,
                    'price': price,
                    'is_morigirl': float(item['is_morigirl']),
                    'first_category': item.get('first_category', 0),
                    'second_category': item.get('second_category', 0),
                    'sales_score': item.get('sales_score', 0.0)
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"⚠️ 데이터 처리 오류: {e}")
                continue
        
        return processed_data
    
    def _print_statistics(self):
        """데이터 통계 출력"""
        if not self.data:
            return
        
        # 클래스 분포
        morigirl_count = sum(1 for item in self.data if item['is_morigirl'] == 1.0)
        non_morigirl_count = len(self.data) - morigirl_count
        
        print(f"📊 데이터 통계:")
        print(f"  - 모리걸: {morigirl_count}개 ({morigirl_count/len(self.data)*100:.1f}%)")
        print(f"  - 비모리걸: {non_morigirl_count}개 ({non_morigirl_count/len(self.data)*100:.1f}%)")
        
        # 가격 통계
        prices = [item['price'] for item in self.data]
        if prices:
            print(f"  - 가격 범위: {min(prices):.2f} ~ {max(prices):.2f}")
    
    def get_class_weights(self) -> float:
        """클래스 불균형 해결을 위한 가중치 계산"""
        morigirl_count = sum(1 for item in self.data if item['is_morigirl'] == 1.0)
        non_morigirl_count = len(self.data) - morigirl_count
        
        if morigirl_count > 0:
            pos_weight = non_morigirl_count / morigirl_count
            return pos_weight
        return 1.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        데이터 샘플 반환
        
        Returns:
            vector: (1024,) 이미지 벡터
            label: 모리걸 라벨 (0 또는 1)
        """
        item = self.data[idx]
        
        # 벡터와 라벨 반환
        vector = torch.tensor(item['vector'], dtype=torch.float32)
        label = torch.tensor(item['is_morigirl'], dtype=torch.float32)
        
        return vector, label
    
    def get_item_info(self, idx: int) -> Dict[str, Any]:
        """특정 인덱스의 상품 정보 반환"""
        return self.data[idx]

def create_train_test_datasets(
    data_path: str = "data/training_data",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    학습/검증/테스트 데이터셋 생성
    
    Args:
        data_path: 데이터 폴더 경로
        test_size: 테스트 셋 비율
        val_size: 검증 셋 비율 (훈련 셋 기준)
        random_state: 랜덤 시드
        **kwargs: 데이터셋 생성시 추가 파라미터
    
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    # 전체 데이터셋 로드
    full_dataset = MoriGirlVectorDataset(data_path, data_type="all", **kwargs)
    
    # 인덱스 분할
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.data[i]['is_morigirl'] for i in indices]
    
    # train+val / test 분할
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=random_state, 
        stratify=labels
    )
    
    # train / val 분할
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=val_size, random_state=random_state,
        stratify=train_val_labels
    )
    
    # 서브셋 생성
    train_data = [full_dataset.data[i] for i in train_indices]
    val_data = [full_dataset.data[i] for i in val_indices]
    test_data = [full_dataset.data[i] for i in test_indices]
    
    # 데이터셋 객체 생성
    train_dataset = MoriGirlVectorDataset.__new__(MoriGirlVectorDataset)
    train_dataset.__dict__.update(full_dataset.__dict__)
    train_dataset.data = train_data
    
    val_dataset = MoriGirlVectorDataset.__new__(MoriGirlVectorDataset)
    val_dataset.__dict__.update(full_dataset.__dict__)
    val_dataset.data = val_data
    
    test_dataset = MoriGirlVectorDataset.__new__(MoriGirlVectorDataset)
    test_dataset.__dict__.update(full_dataset.__dict__)
    test_dataset.data = test_data
    
    print(f"📊 데이터셋 분할 완료:")
    print(f"  - 훈련: {len(train_dataset)}개")
    print(f"  - 검증: {len(val_dataset)}개")
    print(f"  - 테스트: {len(test_dataset)}개")
    
    return train_dataset, val_dataset, test_dataset

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """DataLoader 생성"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        **kwargs
    )

# 테스트용 함수
if __name__ == "__main__":
    # 데이터셋 테스트
    try:
        dataset = MoriGirlVectorDataset("data/training_data")
        print(f"데이터셋 크기: {len(dataset)}")
        
        # 첫 번째 샘플 확인
        vector, label = dataset[0]
        print(f"벡터 형태: {vector.shape}")
        print(f"라벨: {label}")
        
        # 클래스 가중치
        pos_weight = dataset.get_class_weights()
        print(f"클래스 가중치: {pos_weight:.2f}")
        
    except Exception as e:
        print(f"테스트 실패: {e}") 