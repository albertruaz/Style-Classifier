#!/usr/bin/env python3
# prepare_training_data.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import json
from typing import List, Dict, Tuple, Any

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
                        
                        # 필수 데이터 확인
                        if (vector is not None and 
                            is_morigirl is not None and 
                            product_id is not None):
                            
                            # 벡터를 numpy 배열로 변환
                            if isinstance(vector, list):
                                vector = np.array(vector)
                            elif not isinstance(vector, np.ndarray):
                                continue
                            
                            # 벡터 차원 확인 (일반적으로 1024차원)
                            if len(vector.shape) == 1 and len(vector) > 0:
                                self.vectors.append(vector)
                                self.labels.append(int(is_morigirl))
                                self.product_ids.append(product_id)
                                
                                if is_morigirl == 1:
                                    morigirl_count += 1
                                else:
                                    non_morigirl_count += 1
                                
                                total_loaded += 1
                
                print(f"  ✅ {npy_file.name}: {len(data)}개 로딩")
                
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
        print(f"  - 모리걸: {morigirl_count:,}개 ({morigirl_count/total_loaded*100:.1f}%)")
        print(f"  - 비모리걸: {non_morigirl_count:,}개 ({non_morigirl_count/total_loaded*100:.1f}%)")
        print(f"  - 벡터 차원: {self.vectors.shape[1]}")
        
        return True
    
    def create_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[MorigirlDataset, MorigirlDataset]:
        """train/test 데이터셋 분할"""
        
        if len(self.vectors) == 0:
            raise ValueError("데이터가 로딩되지 않았습니다. load_npy_files()를 먼저 실행하세요.")
        
        # 계층화 분할 (모리걸/비모리걸 비율 유지)
        X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
            self.vectors, self.labels, self.product_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels  # 클래스 비율 유지
        )
        
        # Dataset 객체 생성
        train_dataset = MorigirlDataset(X_train, y_train, ids_train)
        test_dataset = MorigirlDataset(X_test, y_test, ids_test)
        
        print(f"\n🔄 Train/Test 분할 완료:")
        print(f"  - Train: {len(train_dataset):,}개 (모리걸: {np.sum(y_train):,}개)")
        print(f"  - Test: {len(test_dataset):,}개 (모리걸: {np.sum(y_test):,}개)")
        print(f"  - Test 비율: {test_size*100:.1f}%")
        
        return train_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset: MorigirlDataset, test_dataset: MorigirlDataset,
                          batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
        """DataLoader 생성"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"📦 DataLoader 생성 완료:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        print(f"  - Batch size: {batch_size}")
        
        return train_loader, test_loader
    
    def save_processed_data(self, train_dataset: MorigirlDataset, test_dataset: MorigirlDataset,
                           output_dir: str = "../data/processed"):
        """처리된 데이터를 파일로 저장"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Train 데이터 저장
        train_data = {
            'vectors': train_dataset.vectors.numpy(),
            'labels': train_dataset.labels.numpy(),
            'product_ids': train_dataset.product_ids
        }
        np.save(f"{output_dir}/train_data.npy", train_data)
        
        # Test 데이터 저장
        test_data = {
            'vectors': test_dataset.vectors.numpy(),
            'labels': test_dataset.labels.numpy(),
            'product_ids': test_dataset.product_ids
        }
        np.save(f"{output_dir}/test_data.npy", test_data)
        
        # 메타데이터 저장
        metadata = {
            'vector_dim': train_dataset.vectors.shape[1],
            'num_classes': 2,
            'train_size': len(train_dataset),
            'test_size': len(test_dataset),
            'train_morigirl_count': int(torch.sum(train_dataset.labels)),
            'test_morigirl_count': int(torch.sum(test_dataset.labels)),
            'class_names': ['non_morigirl', 'morigirl']
        }
        
        with open(f"{output_dir}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 처리된 데이터 저장:")
        print(f"  - {output_dir}/train_data.npy")
        print(f"  - {output_dir}/test_data.npy")
        print(f"  - {output_dir}/metadata.json")
    
    def get_data_info(self) -> Dict[str, Any]:
        """데이터 정보 반환"""
        if len(self.vectors) == 0:
            return {}
        
        return {
            'total_samples': len(self.vectors),
            'vector_dim': self.vectors.shape[1],
            'morigirl_count': np.sum(self.labels),
            'non_morigirl_count': len(self.labels) - np.sum(self.labels),
            'morigirl_ratio': np.mean(self.labels)
        }

 