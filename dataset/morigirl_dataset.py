# dataset/morigirl_dataset.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import json

class MoriGirlDataset(Dataset):
    """
    모리걸 스타일 분류를 위한 데이터셋 클래스
    
    데이터 구조:
    data/
    ├── morigirl/          # 모리걸 스타일 이미지
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── non_morigirl/      # 일반/기타 스타일 이미지  
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── split_info.json    # 데이터 분할 정보 (자동 생성)
    """
    
    def __init__(self, root_dir, split='train', transform=None, test_size=0.2, val_size=0.1, random_state=42):
        """
        Args:
            root_dir: 데이터 루트 디렉토리
            split: 'train', 'val', 'test' 중 하나
            transform: 이미지 전처리 함수
            test_size: 테스트 세트 비율
            val_size: 검증 세트 비율 (훈련 세트 기준)
            random_state: 랜덤 시드
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 클래스 디렉토리
        self.morigirl_dir = os.path.join(root_dir, 'morigirl')
        self.non_morigirl_dir = os.path.join(root_dir, 'non_morigirl')
        
        # 데이터 로드 및 분할
        self.image_paths, self.labels = self._load_and_split_data(
            test_size, val_size, random_state
        )
        
        # 클래스 정보
        self.classes = ['non_morigirl', 'morigirl']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        print(f"{split} 세트: {len(self.image_paths)}개 이미지")
        self._print_class_distribution()
    
    def _load_and_split_data(self, test_size, val_size, random_state):
        """
        데이터 로드 및 train/val/test 분할
        """
        split_file = os.path.join(self.root_dir, 'split_info.json')
        
        # 기존 분할 정보가 있으면 로드
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_info = json.load(f)
            
            return split_info[self.split]['paths'], split_info[self.split]['labels']
        
        # 새로 분할 생성
        all_paths = []
        all_labels = []
        
        # 모리걸 이미지 로드
        if os.path.exists(self.morigirl_dir):
            morigirl_files = [f for f in os.listdir(self.morigirl_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for file in morigirl_files:
                all_paths.append(os.path.join(self.morigirl_dir, file))
                all_labels.append(1)  # 모리걸 = 1
        
        # 일반 이미지 로드
        if os.path.exists(self.non_morigirl_dir):
            non_morigirl_files = [f for f in os.listdir(self.non_morigirl_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            for file in non_morigirl_files:
                all_paths.append(os.path.join(self.non_morigirl_dir, file))
                all_labels.append(0)  # 일반 = 0
        
        if len(all_paths) == 0:
            raise ValueError(f"이미지를 찾을 수 없습니다. 경로를 확인하세요: {self.root_dir}")
        
        # 데이터 분할
        X_temp, X_test, y_temp, y_test = train_test_split(
            all_paths, all_labels, test_size=test_size, 
            random_state=random_state, stratify=all_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, 
            random_state=random_state, stratify=y_temp
        )
        
        # 분할 정보 저장
        split_info = {
            'train': {'paths': X_train, 'labels': y_train},
            'val': {'paths': X_val, 'labels': y_val},
            'test': {'paths': X_test, 'labels': y_test}
        }
        
        with open(split_file, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"데이터 분할 완료 및 저장: {split_file}")
        print(f"전체: {len(all_paths)}, 훈련: {len(X_train)}, 검증: {len(X_val)}, 테스트: {len(X_test)}")
        
        return split_info[self.split]['paths'], split_info[self.split]['labels']
    
    def _print_class_distribution(self):
        """클래스 분포 출력"""
        unique, counts = np.unique(self.labels, return_counts=True)
        for cls_idx, count in zip(unique, counts):
            cls_name = self.classes[cls_idx]
            percentage = count / len(self.labels) * 100
            print(f"  {cls_name}: {count}개 ({percentage:.1f}%)")
    
    def get_class_weights(self):
        """
        클래스 불균형 해결을 위한 가중치 계산
        
        Returns:
            class_weights: 클래스별 가중치 (BCEWithLogitsLoss의 pos_weight용)
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        if len(unique) == 2:
            # 이진 분류의 경우 양성 클래스 가중치만 반환
            neg_count = counts[0] if unique[0] == 0 else counts[1]
            pos_count = counts[1] if unique[1] == 1 else counts[0]
            pos_weight = neg_count / pos_count
            return pos_weight
        return None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        데이터 샘플 반환
        
        Returns:
            image: 전처리된 이미지 텐서
            label: 클래스 라벨 (0: 일반, 1: 모리걸)
        """
        # 이미지 로드
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 오류: {e}")
            # 더미 이미지 생성
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # 라벨
        label = float(self.labels[idx])
        
        # 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_sample_images(self, num_samples=5):
        """
        샘플 이미지들을 반환 (시각화용)
        
        Args:
            num_samples: 반환할 샘플 수
            
        Returns:
            samples: (이미지, 라벨, 경로) 튜플들의 리스트
        """
        indices = np.random.choice(len(self), min(num_samples, len(self)), replace=False)
        samples = []
        
        for idx in indices:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]
            samples.append((image, label, image_path))
        
        return samples

class MoriGirlInferenceDataset(Dataset):
    """
    추론 전용 데이터셋 (라벨 없음)
    """
    
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths: 이미지 경로들의 리스트
            transform: 이미지 전처리 함수
        """
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {image_path}, 오류: {e}")
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, image_path

# 데이터 준비 유틸리티 함수들
def create_data_directories(root_dir):
    """
    데이터 디렉토리 구조 생성
    """
    os.makedirs(os.path.join(root_dir, 'morigirl'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'non_morigirl'), exist_ok=True)
    print(f"데이터 디렉토리 생성 완료: {root_dir}")
    print("이제 다음 폴더에 이미지들을 넣어주세요:")
    print(f"  - 모리걸 이미지: {os.path.join(root_dir, 'morigirl')}")
    print(f"  - 일반 이미지: {os.path.join(root_dir, 'non_morigirl')}")

def check_dataset(root_dir):
    """
    데이터셋 상태 확인
    """
    morigirl_dir = os.path.join(root_dir, 'morigirl')
    non_morigirl_dir = os.path.join(root_dir, 'non_morigirl')
    
    morigirl_count = 0
    non_morigirl_count = 0
    
    if os.path.exists(morigirl_dir):
        morigirl_files = [f for f in os.listdir(morigirl_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        morigirl_count = len(morigirl_files)
    
    if os.path.exists(non_morigirl_dir):
        non_morigirl_files = [f for f in os.listdir(non_morigirl_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        non_morigirl_count = len(non_morigirl_files)
    
    total = morigirl_count + non_morigirl_count
    
    print(f"=== 데이터셋 상태 ===")
    print(f"모리걸 이미지: {morigirl_count}개")
    print(f"일반 이미지: {non_morigirl_count}개")
    print(f"총 이미지: {total}개")
    
    if total < 100:
        print("⚠️ 데이터가 부족합니다. 최소 수백 장 이상 권장")
    elif morigirl_count == 0 or non_morigirl_count == 0:
        print("⚠️ 한쪽 클래스의 데이터가 없습니다.")
    else:
        print("✅ 데이터셋 준비 완료")

# 사용 예시
if __name__ == "__main__":
    # 데이터 디렉토리 생성
    create_data_directories('./data')
    
    # 데이터셋 상태 확인
    check_dataset('./data') 