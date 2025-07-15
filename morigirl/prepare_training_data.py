#!/usr/bin/env python3
# prepare_training_data.py

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class MorigirlDataset(Dataset):
    """모리걸 분류를 위한 PyTorch Dataset"""
    
    def __init__(self, vectors: np.ndarray, labels: np.ndarray, product_ids: np.ndarray):
        """
        Args:
            vectors: 이미지 특징 벡터 (N, vector_dim)
            labels: 모리걸 여부 라벨 (N,) - 1: 모리걸, 0: 비모리걸
            product_ids: 상품 ID (N,)
        """
        self.vectors = torch.FloatTensor(vectors)
        self.labels = torch.FloatTensor(labels)  # BCELoss를 위해 Float 타입으로 변경
        self.product_ids = product_ids
        
    def __len__(self):
        return len(self.vectors)
    
    def __getitem__(self, idx):
        return {
            'vector': self.vectors[idx],
            'label': self.labels[idx],
            'product_id': self.product_ids[idx]
        }

class MorigirlDataProcessor:
    """모리걸 학습 데이터 처리기"""
    
    def __init__(self, data_dir: str = "../data/morigirl_50"):
        self.data_dir = data_dir
        self.vectors = []
        self.labels = []
        self.product_ids = []
        
    def load_npy_files(self, split_type: str = "all") -> bool:
        """npy 파일들을 로드하여 학습용 데이터로 변환
        
        Args:
            split_type: "all" (전체), "train" (train만), "test" (test만)
        """
        
        data_path = Path(self.data_dir)
        if not data_path.exists():
            print(f"❌ 데이터 폴더가 없습니다: {self.data_dir}")
            return False
        
        # split_type에 따라 파일 필터링
        if split_type == "train":
            pattern = "*_train.npy"
        elif split_type == "test":
            pattern = "*_test.npy"
        else:  # "all"
            pattern = "*.npy"
            
        npy_files = list(data_path.glob(pattern))
        if not npy_files:
            print(f"❌ {self.data_dir} 폴더에 {pattern} 파일이 없습니다.")
            return False
        
        print(f"📁 로딩할 npy 파일 ({split_type}): {len(npy_files)}개")
        
        total_loaded = 0
        total_filtered = 0  # 필터링된 데이터 개수
        morigirl_count = 0
        non_morigirl_count = 0
        
        for npy_file in sorted(npy_files):
            print(f"🔄 로딩 중: {npy_file.name}")
            
            try:
                data = np.load(npy_file, allow_pickle=True)
                
                for record in data:
                    if isinstance(record, dict):
                        vector = record.get('vector')
                        is_morigirl = record.get('is_morigirl')
                        product_id = record.get('product_id')
                        sales_score = record.get('sales_score', 0.0)  # 판매점수 추가
                        
                        # 필수 데이터 확인
                        if (vector is not None and 
                            is_morigirl is not None and 
                            product_id is not None and
                            sales_score is not None):
                            
                            # 판매점수 필터링: 0.01 이하면 제외
                            if sales_score <= 0.01:
                                total_filtered += 1
                                continue
                            
                            # 벡터를 numpy 배열로 변환
                            if isinstance(vector, list):
                                vector = np.array(vector)
                            elif not isinstance(vector, np.ndarray):
                                continue
                            
                            # 벡터 차원 확인 (일반적으로 1024차원)
                            if len(vector.shape) == 1 and len(vector) > 0:
                                # 가중치 적용: int(is_morigirl)*0.8 + 0.2*min(1, 판매점수*20)
                                weighted_label = int(is_morigirl) #* 0.8 + 0.2 * min(1.0, sales_score * 20)
                                
                                self.vectors.append(vector)
                                self.labels.append(weighted_label)  # 가중치 적용된 라벨
                                self.product_ids.append(product_id)
                                
                                if is_morigirl == 1:
                                    morigirl_count += 1
                                else:
                                    non_morigirl_count += 1
                                
                                total_loaded += 1
                
                print(f"  ✅ {npy_file.name}: {len(data)}개 중 {total_loaded}개 로딩")
                
            except Exception as e:
                print(f"❌ {npy_file.name} 로딩 실패: {e}")
        
        if total_loaded == 0:
            print("❌ 로딩된 데이터가 없습니다.")
            return False
        
        # numpy 배열로 변환
        self.vectors = np.vstack(self.vectors)
        self.labels = np.array(self.labels)
        self.product_ids = np.array(self.product_ids)
        
        print(f"\n📊 데이터 로딩 완료 ({split_type}):")
        print(f"  - 총 데이터: {total_loaded:,}개")
        print(f"  - 필터링된 데이터: {total_filtered:,}개 (판매점수 ≤ 0.01)")
        print(f"  - 모리걸: {morigirl_count:,}개 ({morigirl_count/total_loaded*100:.1f}%)")
        print(f"  - 비모리걸: {non_morigirl_count:,}개 ({non_morigirl_count/total_loaded*100:.1f}%)")
        print(f"  - 벡터 차원: {self.vectors.shape[1]}")
        print(f"  - 가중치 적용: int(is_morigirl)*0.8 + 0.2*min(1, 판매점수*20)")
        print(f"  - 라벨 범위: {self.labels.min():.3f} ~ {self.labels.max():.3f}")
        
        return True

 