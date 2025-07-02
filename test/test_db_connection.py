# test_db_connection.py

import os
from database import DatabaseManager

def test_database_connections():
    """
    데이터베이스 연결 테스트
    .env 파일을 먼저 설정해야 합니다.
    """
    print("🧪 데이터베이스 연결 테스트 시작")
    print("="*50)
    
    # .env 파일 존재 확인
    if not os.path.exists('.env'):
        print("❌ .env 파일이 없습니다!")
        print("\n📋 다음 단계:")
        print("1. .env.example을 참고해서 .env 파일을 생성하세요")
        print("2. 데이터베이스 연결 정보를 입력하세요")
        print("3. 다시 이 스크립트를 실행하세요")
        return
    
    try:
        # 데이터베이스 매니저 초기화
        db_manager = DatabaseManager()
        
        # 연결 테스트
        db_manager.test_connections()
        
        # 데이터베이스 설정 (테이블 생성 등)
        print("\n🔧 데이터베이스 초기 설정...")
        try:
            db_manager.setup_databases()
        except Exception as e:
            print(f"⚠️ 데이터베이스 설정 중 오류 (무시 가능): {e}")
        
        # MySQL 기본 기능 테스트
        print("\n📊 MySQL 기능 테스트...")
        try:
            # 상품 데이터 샘플 조회
            products = db_manager.mysql.get_product_images(
                where_condition="status = 'SALE'", 
                limit=5
            )
            print(f"✅ 상품 이미지 샘플 조회 성공: {len(products)}개")
            
            for product_id, image_url in products[:3]:
                print(f"  - 상품 {product_id}: {image_url[:50]}...")
                
        except Exception as e:
            print(f"❌ MySQL 기능 테스트 실패: {e}")
        
        # Vector DB 기본 기능 테스트 (연결만 확인)
        print("\n🔍 Vector DB 기능 테스트...")
        try:
            # 간단한 쿼리 테스트
            from sqlalchemy import text
            session = db_manager.vector_db.Session()
            result = session.execute(text("SELECT version()")).scalar()
            session.close()
            print(f"✅ PostgreSQL 버전: {result}")
            
        except Exception as e:
            print(f"❌ Vector DB 기능 테스트 실패: {e}")
        
        print("\n✨ 데이터베이스 연결 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 데이터베이스 연결 실패: {e}")
        print("\n💡 해결 방법:")
        print("1. .env 파일의 데이터베이스 연결 정보를 확인하세요")
        print("2. SSH 터널이 필요한 경우 키 파일 경로를 확인하세요")
        print("3. 네트워크 연결을 확인하세요")
        
    finally:
        # 연결 정리
        try:
            db_manager.dispose_all()
        except:
            pass

if __name__ == "__main__":
    test_database_connections() 