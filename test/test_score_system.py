# test_score_system.py

import os
import sys
import json
import numpy as np
from datetime import datetime

def test_config():
    """설정 파일 테스트"""
    print("=== 설정 파일 테스트 ===")
    
    config_path = "./config.json"
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일이 없습니다: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        required_keys = ['style_id', 'model', 'data', 'database']
        for key in required_keys:
            if key not in config:
                print(f"❌ 설정에서 '{key}' 키가 누락되었습니다.")
                return False
        
        print(f"✅ 설정 파일 로드 성공")
        print(f"  - 스타일 ID: {config['style_id']}")
        print(f"  - 모델 히든 차원: {config['model']['hidden_dim']}")
        print(f"  - 배치 크기: {config['model']['batch_size']}")
        return True
        
    except Exception as e:
        print(f"❌ 설정 파일 오류: {e}")
        return False

def test_database_connection():
    """데이터베이스 연결 테스트"""
    print("\n=== 데이터베이스 연결 테스트 ===")
    
    try:
        from database import DatabaseManager
        
        db_manager = DatabaseManager()
        
        # MySQL 연결 테스트
        print("📊 MySQL 연결 테스트...")
        mysql_session = db_manager.mysql.Session()
        try:
            from sqlalchemy import text
            result = mysql_session.execute(text("SELECT 1 as test")).fetchone()
            if result[0] == 1:
                print("✅ MySQL 연결 성공")
            else:
                print("❌ MySQL 연결 실패")
                return False
        finally:
            mysql_session.close()
        
        # Vector DB 연결 테스트
        print("🔍 Vector DB 연결 테스트...")
        vector_session = db_manager.vector_db.Session()
        try:
            from sqlalchemy import text
            result = vector_session.execute(text("SELECT 1 as test")).fetchone()
            if result[0] == 1:
                print("✅ Vector DB 연결 성공")
            else:
                print("❌ Vector DB 연결 실패")
                return False
        finally:
            vector_session.close()
        
        db_manager.dispose_all()
        return True
        
    except Exception as e:
        print(f"❌ 데이터베이스 연결 오류: {e}")
        return False

def test_model_creation():
    """모델 생성 테스트"""
    print("\n=== 모델 생성 테스트 ===")
    
    try:
        from model.score_prediction_model import ProductScorePredictor, ProductScoreLoss, get_model_info
        import torch
        
        # 모델 생성
        model = ProductScorePredictor(
            input_dim=1025,  # 1024 + 1 (price)
            hidden_dim=512,
            dropout=0.3
        )
        
        # 모델 정보
        info = get_model_info(model)
        print(f"✅ 모델 생성 성공")
        print(f"  - 총 파라미터: {info['total_params']:,}")
        print(f"  - 모델 크기: {info['model_size_mb']:.2f} MB")
        
        # 순전파 테스트
        test_input = torch.randn(4, 1025)
        morigirl_prob, popularity_prob = model(test_input)
        
        print(f"  - 입력 크기: {test_input.shape}")
        print(f"  - 모리걸 출력 크기: {morigirl_prob.shape}")
        print(f"  - 인기도 출력 크기: {popularity_prob.shape}")
        
        # 손실함수 테스트
        criterion = ProductScoreLoss()
        morigirl_target = torch.randint(0, 2, (4, 1)).float()
        popularity_target = torch.rand(4, 1)
        
        losses = criterion(morigirl_prob, popularity_prob, morigirl_target, popularity_target)
        print(f"  - 총 손실: {losses['total_loss']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 생성 오류: {e}")
        return False

def test_data_query():
    """데이터 쿼리 테스트"""
    print("\n=== 데이터 쿼리 테스트 ===")
    
    try:
        from database import DatabaseManager
        from sqlalchemy import text
        
        db_manager = DatabaseManager()
        
        # product_styles 테이블에서 style_id 9인 상품 확인
        print("🎨 Style 9 상품 조회 테스트...")
        mysql_session = db_manager.mysql.Session()
        try:
            sql = text("""
                SELECT COUNT(*) as count
                FROM vingle.product_styles 
                WHERE styles_id LIKE '9'
            """)
            result = mysql_session.execute(sql).fetchone()
            style_count = result[0]
            print(f"  - Style 9 상품 수: {style_count:,}개")
            
            if style_count == 0:
                print("⚠️ Style 9 상품이 없습니다. 다른 스타일 ID를 확인하세요.")
            
        finally:
            mysql_session.close()
        
        # product 테이블에서 SALE/SOLD_OUT 상품 확인
        print("📦 판매 상품 조회 테스트...")
        mysql_session = db_manager.mysql.Session()
        try:
            sql = text("""
                SELECT 
                    status,
                    COUNT(*) as count,
                    AVG(amount) as avg_price
                FROM vingle.product 
                WHERE status IN ('SALE', 'SOLD_OUT')
                  AND amount IS NOT NULL
                  AND views IS NOT NULL
                  AND impressions IS NOT NULL
                GROUP BY status
            """)
            results = mysql_session.execute(sql).fetchall()
            
            for row in results:
                status, count, avg_price = row
                print(f"  - {status}: {count:,}개, 평균 가격: {avg_price:,.0f}원")
            
        finally:
            mysql_session.close()
        
        # Vector DB에서 벡터 데이터 확인
        print("🔍 벡터 데이터 조회 테스트...")
        vector_session = db_manager.vector_db.Session()
        try:
            sql = text("SELECT COUNT(*) as count FROM product_vectors")
            result = vector_session.execute(sql).fetchone()
            vector_count = result[0]
            print(f"  - 벡터 데이터 수: {vector_count:,}개")
            
            if vector_count == 0:
                print("⚠️ 벡터 데이터가 없습니다. 임베딩을 먼저 생성하세요.")
            
        finally:
            vector_session.close()
        
        db_manager.dispose_all()
        return True
        
    except Exception as e:
        print(f"❌ 데이터 쿼리 오류: {e}")
        return False

def test_dataset_creation():
    """데이터셋 생성 테스트"""
    print("\n=== 데이터셋 생성 테스트 ===")
    
    try:
        from dataset.product_score_dataset import ProductScoreDataset
        
        print("📊 데이터셋 로딩 중... (시간이 걸릴 수 있습니다)")
        
        # 소량의 데이터로 테스트
        dataset = ProductScoreDataset(mode="test")
        
        if len(dataset) == 0:
            print("⚠️ 데이터셋이 비어있습니다. 데이터베이스의 데이터를 확인하세요.")
            return False
        
        print(f"✅ 데이터셋 생성 성공: {len(dataset)}개 샘플")
        
        # 첫 번째 샘플 확인
        if len(dataset) > 0:
            input_vec, targets = dataset[0]
            sample_info = dataset.get_sample_info(0)
            
            print(f"  - 입력 벡터 크기: {input_vec.shape}")
            print(f"  - 타겟 크기: {targets.shape}")
            print(f"  - 모리걸 확률: {targets[0]:.3f}")
            print(f"  - 인기도 확률: {targets[1]:.3f}")
            print(f"  - 상품 ID: {sample_info['product_id']}")
        
        dataset.close()
        return True
        
    except Exception as e:
        print(f"❌ 데이터셋 생성 오류: {e}")
        return False

def test_system_integration():
    """시스템 통합 테스트"""
    print("\n=== 시스템 통합 테스트 ===")
    
    try:
        # 모든 주요 컴포넌트가 잘 작동하는지 확인
        from model.score_prediction_model import ProductScorePredictor
        from database import DatabaseManager
        import torch
        import numpy as np
        
        # 더미 데이터로 전체 파이프라인 테스트
        model = ProductScorePredictor()
        model.eval()
        
        # 가상의 이미지 벡터와 가격으로 예측 테스트
        dummy_image_vector = np.random.randn(1024)
        dummy_price = 50000.0
        
        # 입력 준비
        price_normalized = np.log(max(dummy_price, 1.0)) / 10.0
        input_vector = np.concatenate([dummy_image_vector, [price_normalized]])
        input_tensor = torch.FloatTensor(input_vector).unsqueeze(0)
        
        # 예측
        with torch.no_grad():
            morigirl_prob, popularity_prob = model(input_tensor)
            
            print(f"✅ 예측 파이프라인 테스트 성공")
            print(f"  - 모리걸 확률: {morigirl_prob.item():.3f}")
            print(f"  - 인기도 확률: {popularity_prob.item():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 시스템 통합 테스트 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print(f"🧪 모리걸 점수 예측 시스템 테스트")
    print(f"테스트 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    tests = [
        ("설정 파일", test_config),
        ("데이터베이스 연결", test_database_connection),
        ("모델 생성", test_model_creation),
        ("데이터 쿼리", test_data_query),
        ("데이터셋 생성", test_dataset_creation),
        ("시스템 통합", test_system_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
    
    print("\n" + "=" * 60)
    print(f"🏁 테스트 완료: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 통과했습니다! 시스템이 정상적으로 작동합니다.")
        print("\n다음 단계:")
        print("1. python train_score_model.py - 모델 학습")
        print("2. python inference_score_model.py - 예측 수행")
    else:
        print("⚠️ 일부 테스트가 실패했습니다. 문제를 해결한 후 다시 시도하세요.")
        
        print("\n문제 해결 방법:")
        print("- .env 파일에 데이터베이스 연결 정보가 올바른지 확인")
        print("- 데이터베이스에 필요한 테이블과 데이터가 있는지 확인")
        print("- requirements.txt의 패키지들이 모두 설치되어 있는지 확인")

if __name__ == "__main__":
    main() 