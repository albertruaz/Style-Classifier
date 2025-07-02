# database/base_connector.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class BaseConnector(ABC):
    """
    모든 데이터베이스 커넥터의 기본 클래스
    """
    
    def __init__(self):
        self.engine = None
        self.tunnel = None
        self._is_connected = False
    
    @abstractmethod
    def connect(self):
        """데이터베이스 연결"""
        pass
    
    @abstractmethod
    def close(self):
        """데이터베이스 연결 종료"""
        pass
    
    def __enter__(self):
        if not self._is_connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @property
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._is_connected and self.engine is not None
    
    def get_env_variable(self, key: str, default: Any = None) -> str:
        """환경 변수 가져오기"""
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Environment variable '{key}' is not set")
        return value 