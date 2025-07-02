# inference_score_model.py

import torch
import numpy as np
import json
import os
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm

from dataset.product_score_dataset import ProductScoreDataset
from model.score_prediction_model import ProductScorePredictor
from database import DatabaseManager
from utils.train_utils import load_checkpoint

class ProductScoreInference:
    """상품 점수 추론기"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "./config.json",
                 device: str = None):
        """
        Args:
            model_path: 학습된 모델 체크포인트 경로
            config_path: 설정 파일 경로
            device: 사용할 디바이스 ('cuda', 'cpu', None=자동)
        """
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🚀 추론 디바이스: {self.device}")
        
        # 모델 로드
        self.model = self._load_model(model_path)
        
        # 데이터베이스 매니저
        self.db_manager = DatabaseManager()
        
        print("✅ 추론기 초기화 완료")
    
    def _load_model(self, model_path: str) -> ProductScorePredictor:
        """모델 로드"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 모델 생성
        model_config = self.config['model']
        model = ProductScorePredictor(
            input_dim=model_config['input_vector_dim'] + 1,
            hidden_dim=model_config['hidden_dim'],
            dropout=model_config['dropout']
        )
        
        # 체크포인트 로드
        checkpoint = load_checkpoint(model_path, self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        print(f"📦 모델 로드 완료: {model_path}")
        print(f"  - 에폭: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - 검증 손실: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        return model
    
    def predict_single(self, 
                      image_vector: np.ndarray, 
                      price: float) -> Dict[str, float]:
        """
        단일 상품 점수 예측
        
        Args:
            image_vector: 이미지 임베딩 벡터 (1024,)
            price: 상품 가격
            
        Returns:
            dict: {'morigirl_prob': float, 'popularity_prob': float}
        """
        # 입력 준비
        price_normalized = np.log(max(price, 1.0)) / 10.0
        input_vector = np.concatenate([image_vector, [price_normalized]])
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0).to(self.device)
        
        # 예측
        with torch.no_grad():
            morigirl_prob, popularity_prob = self.model(input_tensor)
            
            return {
                'morigirl_prob': morigirl_prob.item(),
                'popularity_prob': popularity_prob.item()
            }
    
    def predict_batch(self, 
                     product_ids: List[int],
                     batch_size: int = 32) -> pd.DataFrame:
        """
        배치 단위 상품 점수 예측
        
        Args:
            product_ids: 예측할 상품 ID 리스트
            batch_size: 배치 크기
            
        Returns:
            DataFrame: 상품별 예측 결과
        """
        results = []
        
        # 상품 데이터 조회
        print("📊 상품 데이터 조회 중...")
        product_data = self._get_product_data(product_ids)
        product_vectors = self._get_product_vectors(product_ids)
        
        # 배치별 예측
        print("🔮 예측 수행 중...")
        
        for i in tqdm(range(0, len(product_data), batch_size)):
            batch_data = product_data[i:i + batch_size]
            batch_inputs = []
            batch_results = []
            
            for product in batch_data:
                product_id = product['product_id']
                
                # 벡터가 없는 상품 건너뛰기
                if product_id not in product_vectors:
                    continue
                
                # 입력 벡터 준비
                image_vector = product_vectors[product_id]
                price = float(product['amount'])
                price_normalized = np.log(max(price, 1.0)) / 10.0
                
                input_vector = np.concatenate([image_vector, [price_normalized]])
                batch_inputs.append(input_vector)
                batch_results.append(product)
            
            if not batch_inputs:
                continue
            
            # 배치 예측
            batch_tensor = torch.FloatTensor(np.array(batch_inputs)).to(self.device)
            
            with torch.no_grad():
                morigirl_probs, popularity_probs = self.model(batch_tensor)
                
                for j, product in enumerate(batch_results):
                    results.append({
                        'product_id': product['product_id'],
                        'morigirl_prob': morigirl_probs[j].item(),
                        'popularity_prob': popularity_probs[j].item(),
                        'price': product['amount'],
                        'status': product['status'],
                        'view': product.get('view', 0),
                        'impression': product.get('impression', 0)
                    })
        
        return pd.DataFrame(results)
    
    def _get_product_data(self, product_ids: List[int]) -> List[Dict[str, Any]]:
        """상품 기본 데이터 조회"""
        session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            # 배치 단위로 처리
            batch_size = self.config['database']['mysql_batch_size']
            all_products = []
            
            for i in range(0, len(product_ids), batch_size):
                batch_ids = product_ids[i:i + batch_size]
                
                sql = text("""
                    SELECT 
                        id,
                        status,
                        views,
                        impressions,
                        amount
                    FROM vingle.product 
                    WHERE id IN :product_ids
                      AND amount IS NOT NULL
                """)
                
                result = session.execute(sql, {"product_ids": tuple(batch_ids)})
                
                for row in result.fetchall():
                    all_products.append({
                        'product_id': row[0],
                        'status': row[1],
                        'views': row[2] or 0,
                        'impressions': row[3] or 0,
                        'amount': row[4]
                    })
            
            return all_products
            
        finally:
            session.close()
    
    def _get_product_vectors(self, product_ids: List[int]) -> Dict[int, np.ndarray]:
        """상품 임베딩 벡터 조회"""
        session = self.db_manager.vector_db.Session()
        try:
            from sqlalchemy import text
            
            # 배치 단위로 처리
            batch_size = self.config['database']['vector_batch_size']
            all_vectors = {}
            
            for i in range(0, len(product_ids), batch_size):
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
                    if vector_str:
                        # PostgreSQL VECTOR 형식을 numpy 배열로 변환
                        vector_str = vector_str.strip('[]')
                        vector = np.array([float(x) for x in vector_str.split(',')])
                        all_vectors[product_id] = vector
            
            return all_vectors
            
        finally:
            session.close()
    
    def predict_style_products(self, 
                              style_id: int = None,
                              limit: int = 1000) -> pd.DataFrame:
        """
        특정 스타일의 모든 상품 예측
        
        Args:
            style_id: 스타일 ID (None이면 config에서 가져옴)
            limit: 예측할 최대 상품 수
            
        Returns:
            DataFrame: 예측 결과
        """
        if style_id is None:
            style_id = self.config['style_id']
        
        # 스타일 상품 조회
        print(f"🎨 Style {style_id} 상품 조회 중...")
        product_ids = self._get_style_product_ids(style_id, limit)
        
        if not product_ids:
            print(f"⚠️ Style {style_id}에 해당하는 상품이 없습니다.")
            return pd.DataFrame()
        
        print(f"📦 {len(product_ids)}개 상품 발견")
        
        # 예측 수행
        results_df = self.predict_batch(product_ids)
        
        # 결과 정렬 (모리걸 확률 높은 순)
        results_df = results_df.sort_values('morigirl_prob', ascending=False)
        
        return results_df
    
    def _get_style_product_ids(self, style_id: int, limit: int) -> List[int]:
        """특정 스타일의 상품 ID 조회"""
        session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT DISTINCT ps.product_id 
                FROM vingle.product_styles AS ps
                JOIN vingle.product AS p ON ps.product_id = p.id
                WHERE ps.styles_id LIKE :style_id
                  AND p.status IN ('SALE', 'SOLD_OUT')
                  AND p.amount IS NOT NULL
                ORDER BY p.id DESC
                LIMIT :limit
            """)
            
            result = session.execute(sql, {
                "style_id": str(style_id),
                "limit": limit
            })
            
            return [row[0] for row in result.fetchall()]
            
        finally:
            session.close()
    
    def save_predictions(self, 
                        predictions_df: pd.DataFrame,
                        output_path: str):
        """예측 결과 저장"""
        predictions_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"💾 예측 결과 저장: {output_path}")
    
    def update_database_predictions(self, 
                                   predictions_df: pd.DataFrame,
                                   table_name: str = 'product_morigirl_scores'):
        """데이터베이스에 예측 결과 업데이트"""
        if predictions_df.empty:
            print("⚠️ 업데이트할 예측 결과가 없습니다.")
            return
        
        session = self.db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            
            # 배치 업데이트
            print(f"💾 데이터베이스 업데이트 중... ({len(predictions_df)}개 상품)")
            
            for _, row in tqdm(predictions_df.iterrows(), total=len(predictions_df)):
                sql = text(f"""
                    INSERT INTO vingle.{table_name} 
                    (product_id, morigirl_prob, popularity_prob, updated_at)
                    VALUES (:product_id, :morigirl_prob, :popularity_prob, NOW())
                    ON DUPLICATE KEY UPDATE
                    morigirl_prob = :morigirl_prob,
                    popularity_prob = :popularity_prob,
                    updated_at = NOW()
                """)
                
                session.execute(sql, {
                    'product_id': int(row['product_id']),
                    'morigirl_prob': float(row['morigirl_prob']),
                    'popularity_prob': float(row['popularity_prob'])
                })
            
            session.commit()
            print("✅ 데이터베이스 업데이트 완료")
            
        except Exception as e:
            session.rollback()
            print(f"❌ 데이터베이스 업데이트 실패: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """리소스 정리"""
        if hasattr(self, 'db_manager'):
            self.db_manager.dispose_all()

def main():
    """메인 함수"""
    # 설정
    model_path = "./checkpoints/best_model.pth"  # 학습된 모델 경로
    config_path = "./config.json"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("먼저 train_score_model.py를 실행하여 모델을 학습하세요.")
        return
    
    try:
        # 추론기 생성
        inferencer = ProductScoreInference(model_path, config_path)
        
        # 특정 스타일 상품들에 대한 예측
        print("🔮 모리걸 스타일 상품 예측 중...")
        predictions_df = inferencer.predict_style_products(limit=500)
        
        if not predictions_df.empty:
            # 결과 출력
            print(f"\n📊 예측 결과 요약:")
            print(f"  - 총 예측 상품: {len(predictions_df)}개")
            print(f"  - 평균 모리걸 확률: {predictions_df['morigirl_prob'].mean():.3f}")
            print(f"  - 평균 인기도 확률: {predictions_df['popularity_prob'].mean():.3f}")
            print(f"  - 모리걸 확률 > 0.7: {(predictions_df['morigirl_prob'] > 0.7).sum()}개")
            print(f"  - 인기도 확률 > 0.5: {(predictions_df['popularity_prob'] > 0.5).sum()}개")
            
            # 상위 결과 출력
            print(f"\n🏆 상위 10개 모리걸 상품:")
            top_morigirl = predictions_df.head(10)
            for _, row in top_morigirl.iterrows():
                print(f"  상품 {row['product_id']}: 모리걸={row['morigirl_prob']:.3f}, "
                      f"인기도={row['popularity_prob']:.3f}, 가격={row['price']:,}원")
            
            # 결과 저장
            output_path = f"./predictions_morigirl_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
            inferencer.save_predictions(predictions_df, output_path)
            
            # 데이터베이스 업데이트 (선택사항)
            update_db = input("\n데이터베이스에 결과를 업데이트하시겠습니까? (y/N): ")
            if update_db.lower() == 'y':
                inferencer.update_database_predictions(predictions_df)
        
        inferencer.close()
        
    except Exception as e:
        print(f"❌ 추론 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 