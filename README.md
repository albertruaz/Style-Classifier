# 🌿 모리걸 스타일 분류기 (Mori Girl Style Classifier)

EfficientNet-B0 기반의 모리걸 스타일 이진 분류 모델입니다.
데이터베이스 연동을 통해 대용량 상품 데이터에 대한 실시간 분류가 가능합니다.

## 📁 프로젝트 구조

```
mori-look/
├── main.py                     # 메인 학습 스크립트
├── inference.py                # 추론 스크립트
├── run_db_inference.py         # DB 상품 대량 추론
├── test_db_connection.py       # DB 연결 테스트
├── visualize.py                # 시각화 스크립트
├── requirements.txt            # 의존성 패키지
├── model/
│   └── morigirl_model.py      # 모델 정의
├── dataset/
│   ├── morigirl_dataset.py    # 로컬 이미지 데이터셋
│   └── db_dataset.py          # DB 연동 데이터셋
├── database/                   # 데이터베이스 연동
│   ├── __init__.py
│   ├── base_connector.py      # 기본 커넥터 클래스
│   ├── mysql_connector.py     # MySQL 커넥터
│   ├── vector_db_connector.py # PostgreSQL Vector DB 커넥터
│   └── db_manager.py          # DB 매니저
├── utils/
│   └── train_utils.py         # 학습 유틸리티
├── data/                      # 로컬 데이터 폴더
│   ├── morigirl/             # 모리걸 이미지
│   └── non_morigirl/         # 일반 이미지
└── checkpoints/              # 모델 체크포인트
```

## 🚀 시작하기

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 데이터베이스 연결 정보를 설정하세요:

```bash
# MySQL 설정
DB_HOST=localhost
DB_PORT=3306
DB_USER=your-username
DB_PASSWORD=your-password
DB_NAME=your-database

# PostgreSQL Vector DB 설정 (선택사항)
PG_HOST=localhost
PG_PORT=5432
PG_USER=your-pg-username
PG_PASSWORD=your-pg-password
PG_DB_NAME=your-vector-db

# S3/CloudFront 설정
S3_CLOUDFRONT_DOMAIN=your-domain.cloudfront.net

# SSH 터널 설정 (필요한 경우)
SSH_HOST=your-ssh-server.com
SSH_USERNAME=your-ssh-username
SSH_PKEY_PATH=/path/to/private-key
```

### 3. 데이터베이스 연결 테스트

```bash
python test_db_connection.py
```

## 📊 사용 방법

### 로컬 이미지로 학습

```bash
# 1. 데이터 준비
python setup_data.py

# 2. 학습 실행
python main.py
```

### 데이터베이스 상품 추론

```bash
# 단일 배치 테스트
python run_db_inference.py --checkpoint ./checkpoints/best_model.pth --max_products 100

# 전체 상품 추론 및 DB 저장
python run_db_inference.py --checkpoint ./checkpoints/best_model.pth --save_to_db

# 조건부 추론
python run_db_inference.py \
    --checkpoint ./checkpoints/best_model.pth \
    --where_condition "status = 'SALE' AND primary_category_id = 1" \
    --save_to_db
```

### 로컬 이미지 추론

```bash
# 단일 이미지
python inference.py --checkpoint ./checkpoints/best_model.pth --image ./test_image.jpg

# 폴더 일괄 처리
python inference.py --checkpoint ./checkpoints/best_model.pth --image_dir ./test_images/
```

## 🎯 모델 특징

### 기본 모델

- **백본**: EfficientNet-B0 (ImageNet pretrained)
- **클래스**: 이진 분류 (모리걸 vs 일반)
- **입력 크기**: 224x224 RGB
- **파라미터 수**: ~5.3M
- **모델 크기**: ~21MB

### 경량 모델 (모바일용)

- **파라미터 수**: ~200K
- **모델 크기**: ~1MB
- **추론 속도**: 2-3배 빠름

## 🗄️ 데이터베이스 구조

### MySQL (상품 기본 정보)

```sql
-- 상품 테이블
CREATE TABLE product (
    id BIGINT PRIMARY KEY,
    main_image VARCHAR(255),
    status VARCHAR(50),
    primary_category_id BIGINT,
    secondary_category_id BIGINT
);

-- 모리걸 예측 결과 테이블
CREATE TABLE product_morigirl_prediction (
    product_id BIGINT PRIMARY KEY,
    is_morigirl BOOLEAN,
    confidence FLOAT,
    updated_at TIMESTAMP
);
```

### PostgreSQL + PGVector (벡터 검색)

```sql
-- 상품 벡터 테이블
CREATE TABLE product_vectors (
    id BIGINT PRIMARY KEY,
    status VARCHAR(255),
    primary_category_id BIGINT,
    secondary_category_id BIGINT,
    image_vector VECTOR(1024),
    is_morigirl BOOLEAN DEFAULT FALSE,
    morigirl_confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 📈 모니터링 및 시각화

```bash
# 학습 과정 시각화
python visualize.py

# 모델 성능 테스트
python test_model.py
```

학습 과정에서 자동 생성되는 파일들:

- `training_history.png`: 손실/정확도 그래프
- `./checkpoints/best_model.pth`: 최고 성능 모델
- `morigirl_model_traced.pt`: TorchScript 모델

## 🔧 설정 변경

### 학습 설정

`main.py`의 config 딕셔너리에서 하이퍼파라미터 조정:

```python
config = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-4,
    'patience': 7,  # early stopping
    'data_root': './data',
}
```

### 데이터베이스 설정

`database/` 폴더의 커넥터 클래스들을 통해 연결 설정 관리

## 🚀 확장 기능

### 1. 벡터 유사도 검색

```python
from database import DatabaseManager

db_manager = DatabaseManager()
similar_products = db_manager.vector_db.get_similar_products([product_id], top_k=10)
```

### 2. 배치 처리

```python
from dataset.db_dataset import DBProductDataset

dataset = DBProductDataset(
    where_condition="status = 'SALE'",
    limit=1000,
    cache_images=True
)
```

### 3. 실시간 추론 API

데이터베이스와 연동된 FastAPI 서버 구축 가능

## 🔍 문제 해결

### 데이터베이스 연결 실패

1. `.env` 파일의 연결 정보 확인
2. SSH 터널 설정 확인 (필요한 경우)
3. 방화벽 및 네트워크 설정 확인

### 이미지 로드 실패

1. S3/CloudFront 도메인 설정 확인
2. 인터넷 연결 상태 확인
3. 이미지 URL 형식 확인

### 메모리 부족

1. 배치 크기 축소
2. 이미지 캐싱 비활성화
3. num_workers 조정

## 📚 추가 자료

- [EfficientNet 논문](https://arxiv.org/abs/1905.11946)
- [PGVector 문서](https://github.com/pgvector/pgvector)
- [모리걸 패션 가이드](https://en.wikipedia.org/wiki/Mori_girl)
# Style-Classifier
