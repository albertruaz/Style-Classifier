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
import hashlib
from uuid import UUID
from typing import List, Dict, Any, Tuple
import sys
sys.path.append('..')
from database import DatabaseManager

class ImageVectorExtractor:
    """이미지를 다운로드하고 벡터로 변환하여 DB에 저장"""
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 데이터 저장 폴더는 나중에 max_products에 따라 설정
        self.output_dir = None
        
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
        
        print(f"✅ 이미지 벡터 추출기 초기화 완료")
        print(f"  - 디바이스: {self.device}")

    def _get_output_dir(self, max_products_per_type: int) -> str:
        """config에서 출력 경로 가져오기"""
        data_config = self.config["data"]
        data_paths = data_config.get("data_paths", {})
        
        # 1. base_data_dir 사용 (자동 경로 생성)
        if data_paths.get("auto_generate_path", True):
            base_path = data_paths.get("base_data_dir", "../data/morigirl_{max_products}")
            final_path = base_path.format(max_products=max_products_per_type)
            print(f"📁 자동 생성 출력 경로: {final_path}")
            return final_path
        
        # 2. 기본값
        default_path = f"../data/morigirl_{max_products_per_type}"
        print(f"📁 기본 출력 경로: {default_path}")
        return default_path

    def uuid_to_bigint(self, uuid_val) -> int:
        """UUID를 BIGINT로 변환"""
        if isinstance(uuid_val, str):
            uuid_val = UUID(uuid_val)
        elif isinstance(uuid_val, UUID):
            pass
        else:
            # 이미 int인 경우
            return int(uuid_val)
        
        # UUID를 bytes로 변환 후 해시하여 64bit 정수로 변환
        uuid_bytes = uuid_val.bytes
        hash_bytes = hashlib.sha256(uuid_bytes).digest()
        # 첫 8바이트를 사용하여 64bit 정수 생성
        return int.from_bytes(hash_bytes[:8], byteorder='big', signed=True)

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

    def get_morigirl_products(self, limit: int = 1000) -> List[Dict]:
        """모리걸 스타일 상품 조회 (styles_id = 9)"""
        mysql_session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT DISTINCT 
                    p.id,
                    p.status,
                    p.amount,
                    p.views,
                    p.impressions,
                    p.primary_category_id,
                    p.secondary_category_id
                FROM vingle.product p
                JOIN vingle.product_styles ps ON p.id = ps.product_id
                WHERE ps.styles_id = '9'
                  AND p.status IN ('SALE', 'SOLD_OUT')
                  AND p.amount IS NOT NULL
                  AND p.views IS NOT NULL
                  AND p.impressions IS NOT NULL
                  AND p.impressions > 0
                ORDER BY RAND()
                LIMIT :limit
            """)
            
            result = mysql_session.execute(sql, {"limit": limit})
            
            products = []
            for row in result.fetchall():
                products.append({
                    'product_id': row[0],
                    'status': row[1],
                    'amount': row[2],
                    'views': row[3],
                    'impressions': row[4],
                    'primary_category_id': row[5],
                    'secondary_category_id': row[6],
                    'is_morigirl': 1  # 모리걸
                })
            
            print(f"🎨 모리걸 스타일 상품 {len(products)}개 조회 완료")
            return products
            
        finally:
            mysql_session.close()

    def get_non_morigirl_products(self, limit: int = 1000) -> List[Dict]:
        """비모리걸 상품 조회 (styles_id != 9)"""
        mysql_session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT DISTINCT 
                    p.id,
                    p.status,
                    p.amount,
                    p.views,
                    p.impressions,
                    p.primary_category_id,
                    p.secondary_category_id
                FROM vingle.product p
                WHERE p.id NOT IN (
                    SELECT DISTINCT product_id 
                    FROM vingle.product_styles 
                    WHERE styles_id = '9'
                )
                  AND p.status IN ('SALE', 'SOLD_OUT')
                  AND p.amount IS NOT NULL
                  AND p.views IS NOT NULL
                  AND p.impressions IS NOT NULL
                  AND p.impressions > 0
                ORDER BY RAND()
                LIMIT :limit
            """)
            
            result = mysql_session.execute(sql, {"limit": limit})
            
            products = []
            for row in result.fetchall():
                products.append({
                    'product_id': row[0],
                    'status': row[1],
                    'amount': row[2],
                    'views': row[3],
                    'impressions': row[4],
                    'primary_category_id': row[5],
                    'secondary_category_id': row[6],
                    'is_morigirl': 0  # 비모리걸
                })
            
            print(f"📦 비모리걸 상품 {len(products)}개 조회 완료")
            return products
            
        finally:
            mysql_session.close()

    def get_product_vectors(self, product_ids: List[int]) -> Dict[int, List[float]]:
        """Vector DB에서 상품 벡터 조회"""
        if not product_ids:
            return {}
            
        vector_session = self.db_manager.vector_db.Session()
        try:
            from sqlalchemy import text
            
            # 배치 단위로 처리
            batch_size = 1000
            all_vectors = {}
            
            for i in range(0, len(product_ids), batch_size):
                batch_ids = product_ids[i:i + batch_size]
                
                # product_id로 직접 비교
                placeholders = ','.join([str(batch_id) for batch_id in batch_ids])
                sql = text(f"""
                    SELECT product_id, vector
                    FROM product_image_vector
                    WHERE product_id IN ({placeholders})
                      AND vector IS NOT NULL
                """)
                
                result = vector_session.execute(sql)
                
                for product_id, vector_str in result.fetchall():
                    if vector_str:
                        # "[1.0,2.0,3.0,...]" 형식을 파싱
                        if isinstance(vector_str, str):
                            vector_str = vector_str.strip('[]')
                            vector = [float(x) for x in vector_str.split(',')]
                        else:
                            # 이미 리스트나 배열인 경우
                            vector = list(vector_str)
                        all_vectors[int(product_id)] = vector
            
            return all_vectors
            
        finally:
            vector_session.close()

    def calculate_sales_score(self, status: str, views: int, impressions: int) -> float:
        """판매 점수 계산 (0~1 사이 값)"""
        if status == 'SOLD_OUT':
            return 1.0
        
        # views / impressions 비율로 계산
        if impressions > 0:
            ratio = min(views / impressions, 1.0)  # 최대 1.0으로 제한
            return ratio
        else:
            return 0.0

    def save_training_data_split(self, products_data: List[Dict[str, Any]], data_type: str):
        """학습용 데이터를 train/test로 분할하여 저장"""
        if not products_data:
            return 0, 0
        
        # 세션별 폴더 생성
        os.makedirs(self.output_dir, exist_ok=True)
        
        try:
            # 데이터 정리
            training_data = []
            for item in products_data:
                training_item = {
                    'product_id': item['product_id'],
                    'price': item['amount'],
                    'vector': item['vector'],
                    'first_category': item['primary_category_id'],
                    'second_category': item['secondary_category_id'],
                    'is_morigirl': item['is_morigirl'],
                    'sales_score': item['sales_score']
                }
                training_data.append(training_item)
            
            # config에서 train/test 비율 읽기
            train_ratio = self.config["data"]["train_test_split"]  # 0.8
            
            # train/test 분할
            from sklearn.model_selection import train_test_split
            train_data, test_data = train_test_split(
                training_data, 
                test_size=1-train_ratio, 
                random_state=42,
                stratify=[item['is_morigirl'] for item in training_data]  # 클래스 비율 유지
            )
            
            # train 파일 저장
            train_filename = f"{self.output_dir}/{data_type}_train.npy"
            np.save(train_filename, train_data)
            print(f"📁 {data_type} train 데이터 저장: {train_filename} ({len(train_data)}개)")
            
            # test 파일 저장
            test_filename = f"{self.output_dir}/{data_type}_test.npy"
            np.save(test_filename, test_data)
            print(f"📁 {data_type} test 데이터 저장: {test_filename} ({len(test_data)}개)")
            
            return len(train_data), len(test_data)
            
        except Exception as e:
            print(f"❌ {data_type} 데이터 저장 실패: {e}")
            return 0, 0

    def process_training_data(self, max_products_per_type: int = 5000):
        """학습용 데이터 처리 - 모리걸/비모리걸 분리"""
        
        # 데이터 폴더 설정 (config에서 읽기)
        self.output_dir = self._get_output_dir(max_products_per_type)
        
        print(f"🚀 학습용 데이터 생성 시작")
        print(f"  - 모리걸 최대: {max_products_per_type:,}개")
        print(f"  - 비모리걸 최대: {max_products_per_type:,}개")
        print(f"  - 저장 폴더: {self.output_dir}")
        
        # 1. 모리걸 데이터 처리
        print(f"\n=== 모리걸 데이터 처리 ===")
        morigirl_products = self.get_morigirl_products(max_products_per_type)
        morigirl_train, morigirl_test = self._process_product_batch(morigirl_products, "morigirl")
        
        # 2. 비모리걸 데이터 처리
        print(f"\n=== 비모리걸 데이터 처리 ===")
        non_morigirl_products = self.get_non_morigirl_products(max_products_per_type)
        non_morigirl_train, non_morigirl_test = self._process_product_batch(non_morigirl_products, "non_morigirl")
        
        # 결과 요약
        print(f"\n🎉 학습용 데이터 생성 완료!")
        print(f"  - 모리걸 train: {morigirl_train}개, test: {morigirl_test}개 (총 {morigirl_train + morigirl_test}개)")
        print(f"  - 비모리걸 train: {non_morigirl_train}개, test: {non_morigirl_test}개 (총 {non_morigirl_train + non_morigirl_test}개)")
        print(f"  - 총 train: {morigirl_train + non_morigirl_train}개")
        print(f"  - 총 test: {morigirl_test + non_morigirl_test}개")
        print(f"  - 전체 총합: {morigirl_train + morigirl_test + non_morigirl_train + non_morigirl_test}개")
        
        # 데이터 정보 JSON 파일 생성
        data_info = {
            "dataset_info": {
                "folder_name": self.output_dir,
                "creation_time": str(np.datetime64('now')),
                "max_products_per_type": max_products_per_type,
                "train_ratio": self.config["data"]["train_test_split"],
                "total_count": morigirl_train + morigirl_test + non_morigirl_train + non_morigirl_test
            },
            "file_counts": {
                "morigirl_train.npy": morigirl_train,
                "morigirl_test.npy": morigirl_test,
                "non_morigirl_train.npy": non_morigirl_train,
                "non_morigirl_test.npy": non_morigirl_test
            },
            "summary": {
                "total_train": morigirl_train + non_morigirl_train,
                "total_test": morigirl_test + non_morigirl_test,
                "morigirl_total": morigirl_train + morigirl_test,
                "non_morigirl_total": non_morigirl_train + non_morigirl_test,
                "train_ratio_actual": (morigirl_train + non_morigirl_train) / (morigirl_train + morigirl_test + non_morigirl_train + non_morigirl_test),
                "morigirl_ratio": (morigirl_train + morigirl_test) / (morigirl_train + morigirl_test + non_morigirl_train + non_morigirl_test)
            },
            "files_created": [
                f"{self.output_dir}/morigirl_train.npy",
                f"{self.output_dir}/morigirl_test.npy", 
                f"{self.output_dir}/non_morigirl_train.npy",
                f"{self.output_dir}/non_morigirl_test.npy"
            ]
        }
        
        # JSON 파일 저장
        data_info_file = f"{self.output_dir}/data_info.json"
        with open(data_info_file, 'w', encoding='utf-8') as f:
            json.dump(data_info, f, ensure_ascii=False, indent=2)
        print(f"📄 데이터 정보 저장: {data_info_file}")
        
        # 호환성을 위한 기존 변수명 유지
        result_data = data_info
        
        # try:
        #     result_file = f"{self.output_dir}/training_data_result.json"
        #     with open(result_file, 'w', encoding='utf-8') as f:
        #         json.dump(result_data, f, ensure_ascii=False, indent=2)
        #     print(f"📁 결과 파일 저장: {result_file}")
        # except Exception as e:
        #     print(f"⚠️ 결과 파일 저장 실패: {e}")
        
        # 폴더 경로 반환
        return self.output_dir

    def _process_product_batch(self, products: List[Dict], data_type: str) -> Tuple[int, int]:
        """상품 배치 처리 (train/test 분할)"""
        if not products:
            return 0, 0
        
        # 벡터 조회
        product_ids = [p['product_id'] for p in products]
        product_vectors = self.get_product_vectors(product_ids)
        
        # 벡터가 있는 상품만 필터링
        valid_products = []
        for product in tqdm(products, desc=f"{data_type} 데이터 처리"):
            product_id = product['product_id']
            
            if product_id in product_vectors:
                # 판매 점수 계산
                sales_score = self.calculate_sales_score(
                    product['status'], 
                    product['views'], 
                    product['impressions']
                )
                
                # 최종 데이터 구성
                product_data = {
                    'product_id': product_id,
                    'amount': product['amount'],
                    'vector': product_vectors[product_id],
                    'primary_category_id': product['primary_category_id'],
                    'secondary_category_id': product['secondary_category_id'],
                    'is_morigirl': product['is_morigirl'],
                    'sales_score': sales_score
                }
                
                valid_products.append(product_data)
        
        print(f"💡 {data_type}: 벡터 있는 상품 {len(valid_products)}개 / 전체 {len(products)}개")
        
        # 파일 저장 (train/test 분할)
        if valid_products:
            train_count, test_count = self.save_training_data_split(valid_products, data_type)
            return train_count, test_count
        
        return 0, 0

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'db_manager'):
            self.db_manager.dispose_all()

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='모리걸 학습용 데이터 생성')
    parser.add_argument('--config-path', default='config.json', help='설정 파일 경로')
    parser.add_argument('--max-products', type=int, default=None, 
                       help='각 타입별 최대 상품 수 (설정 파일 우선)')
    
    args = parser.parse_args()
    
    try:
        # config.json에서 max_products 읽기
        if args.max_products is None:
            try:
                with open(args.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                max_products = config["data"]["max_products_per_type"]
                print(f"✅ config.json에서 max_products 로드: {max_products}")
            except Exception as e:
                print(f"⚠️  config.json 로드 실패, 기본값 사용: {e}")
                max_products = 5000
        else:
            max_products = args.max_products
            print(f"✅ 커맨드라인에서 max_products 설정: {max_products}")
        
        extractor = ImageVectorExtractor()
        
        print(f"🚀 모리걸 학습용 데이터 생성 시작")
        print(f"  - 각 타입별 최대: {max_products:,}개")
        
        # 학습용 데이터 생성
        data_folder = extractor.process_training_data(
            max_products_per_type=max_products
        )
        
        print(f"\n🎉 데이터 생성 완료!")
        print(f"📁 데이터 폴더: {data_folder}")
        print(f"💡 학습 시 --data-path {data_folder} 옵션을 사용하세요.")
        
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