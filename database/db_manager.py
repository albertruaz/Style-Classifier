# database/db_manager.py

from typing import Optional
from .mysql_connector import MySQLConnector
from .vector_db_connector import VectorDBConnector

class DatabaseManager:
    """
    여러 데이터베이스 커넥터를 중앙에서 관리하는 매니저 클래스
    모리걸 분류 프로젝트용으로 MySQL과 PostgreSQL Vector DB를 관리
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """데이터베이스 매니저 초기화"""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._mysql_connector = None
        self._vector_db_connector = None
        self._initialized = True
    
    @property
    def mysql(self) -> MySQLConnector:
        """
        MySQL 커넥터 인스턴스 반환
        - 상품 기본 정보 저장
        - 모리걸 예측 결과 저장
        
        Returns:
            MySQLConnector: MySQL 커넥터 인스턴스
        """
        if self._mysql_connector is None:
            self._mysql_connector = MySQLConnector()
        return self._mysql_connector
    
    @property
    def vector_db(self) -> VectorDBConnector:
        """
        PostgreSQL Vector DB 커넥터 인스턴스 반환
        - 이미지 임베딩 벡터 저장
        - 벡터 유사도 검색
        
        Returns:
            VectorDBConnector: Vector DB 커넥터 인스턴스
        """
        if self._vector_db_connector is None:
            self._vector_db_connector = VectorDBConnector()
        return self._vector_db_connector
    
    def setup_databases(self):
        """
        데이터베이스 초기 설정
        - 필요한 테이블 생성
        - 인덱스 생성
        """
        print("🔧 데이터베이스 초기 설정 시작...")
        
        try:
            # Vector DB 테이블 생성
            self.vector_db.create_product_table(dimension=1024)
            print("✅ Vector DB 설정 완료")
            
        except Exception as e:
            print(f"❌ 데이터베이스 설정 실패: {e}")
            raise
    
    def test_connections(self):
        """
        모든 데이터베이스 연결 테스트
        """
        print("🧪 데이터베이스 연결 테스트 시작...")
        
        # MySQL 연결 테스트
        try:
            mysql_count = self.mysql.get_product_count()
            print(f"✅ MySQL 연결 성공 - 총 상품 수: {mysql_count:,}개")
        except Exception as e:
            print(f"❌ MySQL 연결 실패: {e}")
        
        # Vector DB 연결 테스트
        try:
            # 간단한 쿼리로 연결 테스트
            from sqlalchemy import text
            session = self.vector_db.Session()
            result = session.execute(text("SELECT 1 as test")).scalar()
            session.close()
            if result == 1:
                print("✅ Vector DB 연결 성공")
        except Exception as e:
            print(f"❌ Vector DB 연결 실패: {e}")
    
    def dispose_all(self):
        """모든 데이터베이스 연결 정리"""
        if self._mysql_connector:
            self._mysql_connector.close()
            self._mysql_connector = None
            
        if self._vector_db_connector:
            self._vector_db_connector.close()
            self._vector_db_connector = None
        
        print("🧹 모든 데이터베이스 연결 정리 완료")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose_all()
    
    def __del__(self):
        """소멸자 - 모든 연결 정리"""
        self.dispose_all()

# 전역 데이터베이스 매니저 인스턴스
db_manager = DatabaseManager() 