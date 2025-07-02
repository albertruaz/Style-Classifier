# database/mysql_connector.py

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sshtunnel import SSHTunnelForwarder
from typing import List, Dict, Optional, Tuple, Any
import os
from abc import ABCMeta

from .base_connector import BaseConnector

class SingletonMeta(ABCMeta):
    """
    Singleton을 위한 메타클래스 (ABCMeta를 상속받아 충돌 해결)
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MySQLConnector(BaseConnector, metaclass=SingletonMeta):
    """
    MySQL 데이터베이스 커넥터 (SSH 터널 지원)
    """
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        super().__init__()
        
        # SSH 터널 설정
        self.ssh_host = os.getenv('SSH_HOST')
        self.ssh_username = os.getenv('SSH_USERNAME')
        self.ssh_pkey_path = os.getenv('SSH_PKEY_PATH')
        
        # MySQL 설정
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = int(os.getenv('DB_PORT', 3306))
        self.db_user = self.get_env_variable('DB_USER')
        self.db_password = self.get_env_variable('DB_PASSWORD')
        self.db_name = self.get_env_variable('DB_NAME')
        
        # 커넥션 풀 설정
        self.pool_size = int(os.getenv('DB_POOL_SIZE', 10))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', 20))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', 30))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', 3600))
        
        # S3/CloudFront 설정
        self.cloudfront_domain = os.getenv('S3_CLOUDFRONT_DOMAIN')
        
        # Private Session 변수
        self._Session = None
        
        self._initialized = True
        
        # 자동 연결
        self.connect()
    
    def connect(self):
        """MySQL 데이터베이스 연결 (SSH 터널 포함)"""
        if self._is_connected:
            return
            
        try:
            # SSH 터널 설정 (필요한 경우)
            if self.ssh_host and self.ssh_username and self.ssh_pkey_path:
                self.tunnel = SSHTunnelForwarder(
                    (self.ssh_host, 22),
                    ssh_username=self.ssh_username,
                    ssh_pkey=self.ssh_pkey_path,
                    remote_bind_address=(self.db_host, self.db_port)
                )
                self.tunnel.start()
                db_host = '127.0.0.1'
                db_port = self.tunnel.local_bind_port
            else:
                db_host = self.db_host
                db_port = self.db_port
            
            # SQLAlchemy 엔진 생성
            db_url = f"mysql+pymysql://{self.db_user}:{self.db_password}@{db_host}:{db_port}/{self.db_name}"
            
            self.engine = create_engine(
                db_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=False  # SQL 로깅 끄기
            )
            
            self._Session = sessionmaker(bind=self.engine)
            self._is_connected = True
            
            print("✅ MySQL 데이터베이스 연결 성공")
            
        except Exception as e:
            print(f"❌ MySQL 연결 실패: {e}")
            self.close()
            raise
    
    @property
    def Session(self):
        """Session에 접근할 때 자동으로 연결 상태 확인 및 재연결"""
        if self._Session is None or not self._is_connected:
            print("🔄 MySQL 세션이 없습니다. 자동 재연결 중...")
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
        print("📡 MySQL 데이터베이스 연결 종료")
    
    def get_s3_url(self, file_name: str) -> Optional[str]:
        """S3/CloudFront URL 생성"""
        if not file_name or not self.cloudfront_domain:
            return None
        return f"https://{self.cloudfront_domain}/{file_name}"
    
    def get_product_data(self, where_condition: str = "1=1", limit: int = 500, batch_no: int = 0) -> List[Tuple]:
        """
        상품 데이터 조회
        
        Args:
            where_condition: WHERE 조건문
            limit: 조회할 개수
            batch_no: 배치 번호 (페이징용)
            
        Returns:
            List[Tuple]: (id, main_image_url, status, primary_category_id, secondary_category_id)
        """
        offset = batch_no * limit
        
        session = self.Session()
        try:
            sql = text(f"""
                SELECT 
                    id,
                    main_image,
                    status,
                    primary_category_id,
                    secondary_category_id
                FROM product
                WHERE {where_condition}
                LIMIT {limit} OFFSET {offset}
            """)
            
            result = session.execute(sql)
            products = []
            
            for row in result.fetchall():
                products.append((
                    row[0],  # id
                    self.get_s3_url(row[1]) if row[1] else None,  # main_image -> S3 URL
                    row[2],  # status
                    row[3],  # primary_category_id
                    row[4],  # secondary_category_id
                ))
            
            return products
            
        finally:
            session.close()
    
    def get_product_images(self, where_condition: str = "1=1", limit: int = 1000, batch_no: int = 0) -> List[Tuple]:
        """
        상품 이미지 URL 조회 (모리걸 분류용)
        
        Returns:
            List[Tuple]: (id, main_image_url)
        """
        offset = batch_no * limit
        
        session = self.Session()
        try:
            sql = text(f"""
                SELECT 
                    id,
                    main_image
                FROM product
                WHERE {where_condition}
                LIMIT {limit} OFFSET {offset}
            """)
            
            result = session.execute(sql)
            products = []
            
            for row in result.fetchall():
                image_url = self.get_s3_url(row[1]) if row[1] else None
                if image_url:  # 이미지가 있는 경우만
                    products.append((row[0], image_url))
            
            return products
            
        finally:
            session.close()
    
    def get_product_count(self, where_condition: str = "1=1") -> int:
        """상품 총 개수 조회"""
        session = self.Session()
        try:
            sql = text(f"SELECT COUNT(*) FROM product WHERE {where_condition}")
            result = session.execute(sql)
            return result.scalar()
        finally:
            session.close()
    
    def update_morigirl_predictions(self, predictions: Dict[str, Dict[str, Any]]):
        """
        모리걸 예측 결과를 데이터베이스에 저장
        
        Args:
            predictions: {product_id: {'is_morigirl': bool, 'confidence': float}}
        """
        if not predictions:
            return
            
        session = self.Session()
        try:
            # 모리걸 예측 결과 저장 테이블이 있다고 가정
            for product_id, pred_data in predictions.items():
                sql = text("""
                    INSERT INTO product_morigirl_prediction 
                    (product_id, is_morigirl, confidence, updated_at)
                    VALUES (:product_id, :is_morigirl, :confidence, NOW())
                    ON DUPLICATE KEY UPDATE
                        is_morigirl = VALUES(is_morigirl),
                        confidence = VALUES(confidence),
                        updated_at = VALUES(updated_at)
                """)
                
                session.execute(sql, {
                    'product_id': product_id,
                    'is_morigirl': pred_data['is_morigirl'],
                    'confidence': pred_data['confidence']
                })
            
            session.commit()
            print(f"✅ {len(predictions)}개 상품의 모리걸 예측 결과 저장 완료")
            
        except Exception as e:
            session.rollback()
            print(f"❌ 모리걸 예측 결과 저장 실패: {e}")
            raise
        finally:
            session.close()
    
    def get_morigirl_products(self, limit: int = None) -> List[Tuple]:
        """
        모리걸로 예측된 상품들 조회
        
        Returns:
            List[Tuple]: (product_id, confidence, main_image_url)
        """
        session = self.Session()
        try:
            limit_clause = f"LIMIT {limit}" if limit else ""
            
            sql = text(f"""
                SELECT 
                    p.id,
                    m.confidence,
                    p.main_image
                FROM product p
                JOIN product_morigirl_prediction m ON p.id = m.product_id
                WHERE m.is_morigirl = 1
                ORDER BY m.confidence DESC
                {limit_clause}
            """)
            
            result = session.execute(sql)
            products = []
            
            for row in result.fetchall():
                products.append((
                    row[0],  # product_id
                    row[1],  # confidence
                    self.get_s3_url(row[2]) if row[2] else None  # image_url
                ))
            
            return products
            
        finally:
            session.close() 