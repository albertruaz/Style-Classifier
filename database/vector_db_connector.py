# database/vector_db_connector.py

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder
from typing import List, Dict, Optional, Tuple, Any
import os
import numpy as np

from .base_connector import BaseConnector

class VectorDBConnector(BaseConnector):
    """
    PostgreSQL + PGVector 데이터베이스 커넥터
    벡터 유사도 검색을 위한 전용 커넥터
    """
    
    def __init__(self):
        super().__init__()
        
        # SSH 터널 설정
        self.ssh_host = os.getenv('PG_SSH_HOST')
        self.ssh_username = os.getenv('PG_SSH_USERNAME')
        self.ssh_pkey_path = os.getenv('PG_SSH_PKEY_PATH')
        
        # PostgreSQL 설정
        self.pg_host = os.getenv('PG_HOST', 'localhost')
        self.pg_port = int(os.getenv('PG_PORT', 5432))
        self.pg_user = self.get_env_variable('PG_USER')
        self.pg_password = self.get_env_variable('PG_PASSWORD')
        self.pg_dbname = self.get_env_variable('PG_DB_NAME')
        
        # 커넥션 풀 설정
        self.pool_size = int(os.getenv('PG_POOL_SIZE', 5))
        self.max_overflow = int(os.getenv('PG_MAX_OVERFLOW', 10))
        self.pool_timeout = int(os.getenv('PG_POOL_TIMEOUT', 30))
        self.pool_recycle = int(os.getenv('PG_POOL_RECYCLE', 3600))
        
        # Private Session 변수
        self._Session = None
        
        # 자동 연결
        self.connect()
    
    def connect(self):
        """PostgreSQL 데이터베이스 연결 (SSH 터널 포함)"""
        if self._is_connected:
            return
            
        try:
            # SSH 터널 설정 (필요한 경우)
            if self.ssh_host and self.ssh_username and self.ssh_pkey_path:
                self.tunnel = SSHTunnelForwarder(
                    (self.ssh_host, 22),
                    ssh_username=self.ssh_username,
                    ssh_pkey=self.ssh_pkey_path,
                    remote_bind_address=(self.pg_host, self.pg_port)
                )
                self.tunnel.start()
                db_host = '127.0.0.1'
                db_port = self.tunnel.local_bind_port
            else:
                db_host = self.pg_host
                db_port = self.pg_port
            
            # SQLAlchemy 엔진 생성
            db_url = f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}@{db_host}:{db_port}/{self.pg_dbname}"
            
            self.engine = create_engine(
                db_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=False
            )
            
            self._Session = sessionmaker(bind=self.engine)
            self._is_connected = True
            
            print("✅ PostgreSQL Vector DB 연결 성공")
            
        except Exception as e:
            print(f"❌ PostgreSQL Vector DB 연결 실패: {e}")
            self.close()
            raise
    
    @property
    def Session(self):
        """Session에 접근할 때 자동으로 연결 상태 확인 및 재연결"""
        if self._Session is None or not self._is_connected:
            print("🔄 PostgreSQL 세션이 없습니다. 자동 재연결 중...")
            self.connect()
        return self._Session
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self._Session:
            self._Session.close_all()
            self._Session = None
            
        if self.engine:
            self.engine.dispose()
            self.engine = None
            
        if self.tunnel and self.tunnel.is_active:
            self.tunnel.close()
            self.tunnel = None
            
        self._is_connected = False
        print("📡 PostgreSQL Vector DB 연결 종료")
    
    def create_vector_extension(self):
        """PGVector 확장 설치"""
        session = self.Session()
        try:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
            print("✅ PGVector 확장 설치 완료")
        except Exception as e:
            session.rollback()
            print(f"❌ PGVector 확장 설치 실패: {e}")
            raise
        finally:
            session.close()
    
    def create_product_table(self, dimension: int = 1024):
        """
        상품 벡터 테이블 생성
        
        Args:
            dimension: 벡터 차원 수
        """
        session = self.Session()
        try:
            # PGVector 확장 먼저 설치
            self.create_vector_extension()
            
            # 상품 벡터 테이블 생성
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS product_vectors (
                id                    BIGINT         PRIMARY KEY,
                status                VARCHAR(255),
                primary_category_id   BIGINT,
                secondary_category_id BIGINT,
                image_vector          VECTOR({dimension}),
                is_morigirl          BOOLEAN DEFAULT FALSE,
                morigirl_confidence  FLOAT DEFAULT 0.0,
                created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            session.execute(text(create_table_sql))
            
            # 인덱스 생성 (벡터 유사도 검색 최적화)
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_product_vectors_similarity 
            ON product_vectors USING ivfflat (image_vector vector_cosine_ops)
            WITH (lists = 100);
            """
            
            session.execute(text(index_sql))
            session.commit()
            
            print(f"✅ 상품 벡터 테이블 생성 완료 (차원: {dimension})")
            
        except Exception as e:
            session.rollback()
            print(f"❌ 상품 벡터 테이블 생성 실패: {e}")
            raise
        finally:
            session.close()
    
    def upsert_product_vectors(self, products_data: List[Dict[str, Any]]):
        """
        상품 벡터 데이터 삽입/업데이트
        
        Args:
            products_data: [
                {
                    'product_id': int,
                    'image_vector': List[float],
                    'status': str,
                    'primary_category_id': int,
                    'secondary_category_id': int,
                    'is_morigirl': bool (optional),
                    'morigirl_confidence': float (optional)
                }
            ]
        """
        if not products_data:
            return
            
        session = self.Session()
        try:
            for item in products_data:
                product_id = item['product_id']
                image_vector = item['image_vector']
                status = item['status']
                primary_category_id = item['primary_category_id']
                secondary_category_id = item['secondary_category_id']
                is_morigirl = item.get('is_morigirl', False)
                morigirl_confidence = item.get('morigirl_confidence', 0.0)
                
                # 벡터 데이터 검증
                if not all(isinstance(val, (int, float)) for val in image_vector):
                    raise ValueError(f"Vector contains non-numeric values: {image_vector}")
                
                vector_str = "[" + ",".join(map(str, image_vector)) + "]"
                
                # UPSERT 쿼리
                sql = text("""
                    INSERT INTO product_vectors 
                    (id, status, primary_category_id, secondary_category_id, 
                     image_vector, is_morigirl, morigirl_confidence, updated_at)
                    VALUES (:pid, :status, :primary_cat, :secondary_cat, 
                            :vec, :is_morigirl, :confidence, CURRENT_TIMESTAMP)
                    ON CONFLICT (id)
                    DO UPDATE SET 
                        status = EXCLUDED.status,
                        primary_category_id = EXCLUDED.primary_category_id,
                        secondary_category_id = EXCLUDED.secondary_category_id,
                        image_vector = EXCLUDED.image_vector,
                        is_morigirl = EXCLUDED.is_morigirl,
                        morigirl_confidence = EXCLUDED.morigirl_confidence,
                        updated_at = EXCLUDED.updated_at;
                """)
                
                session.execute(sql, {
                    "pid": product_id,
                    "status": status,
                    "primary_cat": primary_category_id,
                    "secondary_cat": secondary_category_id,
                    "vec": vector_str,
                    "is_morigirl": is_morigirl,
                    "confidence": morigirl_confidence
                })
            
            session.commit()
            print(f"✅ {len(products_data)}개 상품 벡터 저장 완료")
            
        except Exception as e:
            session.rollback()
            print(f"❌ 상품 벡터 저장 실패: {e}")
            raise
        finally:
            session.close()
    
    def update_morigirl_predictions(self, predictions: Dict[str, Dict[str, Any]]):
        """
        모리걸 예측 결과 업데이트
        
        Args:
            predictions: {product_id: {'is_morigirl': bool, 'confidence': float}}
        """
        if not predictions:
            return
            
        session = self.Session()
        try:
            for product_id, pred_data in predictions.items():
                sql = text("""
                    UPDATE product_vectors 
                    SET is_morigirl = :is_morigirl,
                        morigirl_confidence = :confidence,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = :product_id
                """)
                
                session.execute(sql, {
                    'product_id': int(product_id),
                    'is_morigirl': pred_data['is_morigirl'],
                    'confidence': pred_data['confidence']
                })
            
            session.commit()
            print(f"✅ {len(predictions)}개 상품의 모리걸 예측 결과 업데이트 완료")
            
        except Exception as e:
            session.rollback()
            print(f"❌ 모리걸 예측 결과 업데이트 실패: {e}")
            raise
        finally:
            session.close()
    


    def get_product_vectors_by_ids(self, product_ids: List[int]) -> List[Tuple]:
        """
        특정 상품들의 벡터 조회
        
        Returns:
            List[Tuple]: (product_id, image_vector_array)
        """
        if not product_ids:
            return []
            
        session = self.Session()
        try:
            # IN 쿼리용 플레이스홀더 생성
            placeholders = ','.join([':id' + str(j) for j in range(len(product_ids))])
            params = {f'id{j}': product_ids[j] for j in range(len(product_ids))}
            
            sql = text(f"""
                SELECT id, image_vector
                FROM product_image_vector
                WHERE id IN ({placeholders})
            """)
            
            result = session.execute(sql, params)
            return result.fetchall()
            
        finally:
            session.close()

    def get_existing_product_ids(self, batch_size: int = 10000) -> List[int]:
        """
        이미 벡터가 생성된 상품 ID들을 조회
        
        Args:
            batch_size: 한 번에 조회할 배치 크기
            
        Returns:
            List[int]: 이미 벡터가 있는 상품 ID 리스트
        """
        session = self.Session()
        try:
            # 기존 상품 ID들 조회
            sql = text("""
                SELECT id 
                FROM product_image_vector
                WHERE image_vector IS NOT NULL
            """)
            
            result = session.execute(sql)
            existing_ids = [row[0] for row in result.fetchall()]
            
            print(f"✅ 기존 벡터 데이터: {len(existing_ids):,}개")
            return existing_ids
            
        except Exception as e:
            print(f"❌ 기존 상품 ID 조회 실패: {e}")
            return []
        finally:
            session.close()

    def get_similar_products(self, product_ids: List[int], top_k: int = 50, 
                           same_category_only: bool = True) -> Dict[int, List[Tuple]]:
        """
        유사한 상품들 조회
        
        Args:
            product_ids: 기준 상품 ID 리스트
            top_k: 각 상품별 유사 상품 개수
            same_category_only: 같은 카테고리 내에서만 검색할지 여부
            
        Returns:
            Dict[int, List[Tuple]]: {product_id: [(similar_id, distance), ...]}
        """
        if not product_ids:
            return {}
        
        session = self.Session()
        try:
            category_condition = """
                AND p1.primary_category_id = p2.primary_category_id
                AND p1.secondary_category_id = p2.secondary_category_id
            """ if same_category_only else ""
            
            # IN 쿼리용 플레이스홀더 생성
            placeholders = ','.join([':id' + str(j) for j in range(len(product_ids))])
            params = {f'id{j}': product_ids[j] for j in range(len(product_ids))}
            params['top_k'] = top_k
            
            sql = text(f"""
                WITH ranked AS (
                    SELECT 
                        p1.id AS product_id,
                        p2.id AS similar_id,
                        (p1.image_vector <#> p2.image_vector) AS distance,
                        ROW_NUMBER() OVER (
                            PARTITION BY p1.id 
                            ORDER BY (p1.image_vector <#> p2.image_vector)
                        ) AS rn
                    FROM product_image_vector p1
                    JOIN product_image_vector p2 ON p1.id != p2.id
                    WHERE p1.id IN ({placeholders})
                      AND p2.status = 'SALE'
                      {category_condition}
                )
                SELECT product_id, similar_id, distance
                FROM ranked
                WHERE rn <= :top_k
                ORDER BY product_id, distance
            """)
            
            rows = session.execute(sql, params).fetchall()
            
            # 결과를 Dict 형태로 변환
            result = {}
            for product_id, similar_id, distance in rows:
                if product_id not in result:
                    result[product_id] = []
                result[product_id].append((similar_id, float(distance)))
            
            return result
            
        finally:
            session.close()

    def get_morigirl_products(self, limit: Optional[int] = None, 
                             min_confidence: float = 0.5) -> List[Tuple]:
        """
        모리걸로 분류된 상품들 조회
        
        Args:
            limit: 조회할 최대 개수
            min_confidence: 최소 신뢰도
            
        Returns:
            List[Tuple]: (product_id, confidence, primary_category_id, secondary_category_id)
        """
        session = self.Session()
        try:
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            sql = text(f"""
                SELECT id, morigirl_confidence, primary_category_id, secondary_category_id
                FROM product_image_vector
                WHERE is_morigirl = TRUE 
                  AND morigirl_confidence >= :min_confidence
                ORDER BY morigirl_confidence DESC
                {limit_clause}
            """)
            
            result = session.execute(sql, {"min_confidence": min_confidence})
            return result.fetchall()
            
        finally:
            session.close()
    
    def check_table_schema(self):
        """Vector DB 테이블 스키마 확인"""
        session = self.Session()
        try:
            # 테이블 목록 조회
            tables_sql = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE '%product%'
            """)
            
            result = session.execute(tables_sql)
            tables = [row[0] for row in result.fetchall()]
            
            print("📋 발견된 상품 관련 테이블들:")
            for table in tables:
                print(f"  - {table}")
                
            return tables
            
        except Exception as e:
            print(f"❌ 테이블 스키마 확인 실패: {e}")
            return []
        finally:
            session.close() 