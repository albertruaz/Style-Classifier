# 🎯 Mori-Look: 모리걸 스타일 분류 시스템

> 딥러닝을 활용하여 패션 아이템의 모리걸 스타일을 자동으로 분류하는 시스템

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
conda create -n mori-look python=3.9 -y
conda activate mori-look

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env 파일 생성)
cp .env.example .env

# wandb 로그인 (선택사항)
wandb login
```

### 2. 설정 파일 확인

`config.json` 파일에서 모든 설정을 관리합니다:

```json
{
  "style_id": 9,
  "data": {
    "max_products_per_type": 5000,
    "train_test_split": 0.8,
    "val_split": 0.1
  },
  "model": {
    "input_vector_dim": 1024,
    "hidden_dim": 128,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "epochs": 50
  },
  "wandb": {
    "enabled": true,
    "project": "morigirl-classification"
  }
}
```

## 📊 주요 기능

### 🎨 **모리걸 스타일 분류**

- EfficientNet 기반 이미지 임베딩 벡터 사용
- 1024차원 → 128차원 → 1 구조의 경량 분류기
- 높은 정확도의 이진 분류 (모리걸 vs 비모리걸)

### 📈 **실험 추적 (Wandb)**

- 학습 과정 실시간 모니터링
- 모델 성능 메트릭 자동 로깅
- 하이퍼파라미터 실험 관리

### ⚡ **통합 설정 관리**

- config.json을 통한 중앙 집중식 설정
- 모델 구조, 학습 파라미터, 데이터 설정 일원화

## 🛠️ 사용법

### 📥 **1. 학습 데이터 생성**

```bash
# config.json 설정 사용 (기본 5,000개)
python save_image_vectors.py

# 또는 개수 직접 지정
python save_image_vectors.py --max-products 1000
```

**결과**: `data/morigirl_5000/` 폴더에 저장

- `morigirl_5000.npy`: 모리걸 상품 데이터
- `non_morigirl_5000.npy`: 비모리걸 상품 데이터

### 🏋️ **2. 모델 학습**

```bash
# config.json 기반 학습 (wandb 자동 연동)
python train_model.py

# 커스텀 설정으로 학습
python train_model.py --config-path custom_config.json

# 특정 데이터로 학습
python train_model.py --data-path data/morigirl_1000
```

**결과**: `result/{MMDDHHMM_RR}/` 폴더에 저장

- `checkpoints/best_model.pth`: 최고 성능 모델
- `training_history.json`: 학습 기록

### 🔮 **3. 모델 테스트**

```bash
# 종합 성능 평가 (시각화 포함)
python test_model.py --checkpoint result/12151430_42/checkpoints/best_model.pth

# 빠른 테스트 (10개 샘플)
python test_model.py --checkpoint result/12151430_42/checkpoints/best_model.pth --quick-test

# 커스텀 설정으로 테스트
python test_model.py --checkpoint model.pth --config-path custom_config.json
```

**결과**: 새로운 `result/{timestamp}/` 폴더에 저장

- `test_results_visualization.png`: 성능 시각화
- `metrics.json`: 상세 성능 메트릭
- `predictions.csv`: 예측 결과
- `classification_report.txt`: 분류 보고서

## 📁 프로젝트 구조

```
mori-look/
├── 🔧 설정 파일
│   ├── config.json              # 통합 설정 파일
│   └── requirements.txt         # 패키지 목록
├── 🚀 메인 스크립트
│   ├── save_image_vectors.py    # 데이터 생성
│   ├── train_model.py           # 모델 학습
│   └── test_model.py            # 모델 테스트
├── 🧠 모델 & 데이터
│   ├── model/
│   │   └── morigirl_model.py    # 모델 정의
│   ├── dataset/
│   │   └── morigirl_vector_dataset.py
│   └── prepare_training_data.py # 데이터 처리
├── 💾 데이터 & 결과
│   ├── data/                    # 학습 데이터
│   │   └── morigirl_5000/
│   └── result/                  # 실험 결과
│       └── {timestamp}/
├── 🗄️ 데이터베이스
│   ├── database/                # DB 연결 관리
│   └── utils/                   # 유틸리티
└── 📋 기타
    └── README.md
```

## 🎯 주요 스크립트

| 스크립트                | 기능                   | 설정 파일   | 사용 예시                                     |
| ----------------------- | ---------------------- | ----------- | --------------------------------------------- |
| `save_image_vectors.py` | 학습 데이터 생성       | config.json | `python save_image_vectors.py`                |
| `train_model.py`        | 모델 학습 (wandb 연동) | config.json | `python train_model.py`                       |
| `test_model.py`         | 모델 평가 및 시각화    | config.json | `python test_model.py --checkpoint model.pth` |

## ⚙️ 설정 옵션

### config.json 상세 설정

```json
{
  "data": {
    "max_products_per_type": 5000, // 각 클래스별 최대 상품 수
    "train_test_split": 0.8, // 학습/테스트 분할 비율
    "val_split": 0.1 // 검증 데이터 비율
  },
  "model": {
    "input_vector_dim": 1024, // 입력 벡터 차원
    "hidden_dim": 128, // 히든 레이어 차원
    "dropout_rate": 0.1, // 드롭아웃 비율
    "learning_rate": 1e-4, // 학습률
    "weight_decay": 0.01, // 가중치 감쇠
    "batch_size": 64, // 배치 크기
    "epochs": 50, // 학습 에포크
    "patience": 10 // Early stopping 인내
  },
  "wandb": {
    "enabled": true, // wandb 사용 여부
    "project": "morigirl-classification", // 프로젝트 명
    "entity": null, // 팀/사용자 명
    "log_frequency": 10, // 로그 주기
    "save_model": true, // 모델 아티팩트 저장
    "watch_model": true // 모델 gradients 감시
  }
}
```

### 환경 변수 (.env)

```bash
# 데이터베이스 연결
MYSQL_HOST=your_host
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password

# 벡터 DB 연결
MILVUS_HOST=your_host
MILVUS_PORT=19530

# 이미지 저장소
S3_CLOUDFRONT_DOMAIN=your_domain
```

## 📈 모델 성능

### 아키텍처

- **입력**: 1024차원 EfficientNet 임베딩 벡터
- **구조**: 1024 → 128 → 1 (약 130K 파라미터)
- **출력**: 모리걸 확률 (0~1)

### 성능 지표

- **정확도**: ~92%
- **F1-Score**: ~91%
- **AUC**: ~95%

## 🔄 전체 워크플로우

```bash
# 1️⃣ 설정 확인
vim config.json

# 2️⃣ 데이터 생성 (5,000개 x 2클래스)
python save_image_vectors.py
# 결과: data/morigirl_5000/

# 3️⃣ 모델 학습 (wandb 자동 로깅)
python train_model.py
# 결과: result/12151430_42/checkpoints/

# 4️⃣ 모델 평가 (시각화 생성)
python test_model.py --checkpoint result/12151430_42/checkpoints/best_model.pth
# 결과: result/{new_timestamp}/test_results_visualization.png

# 5️⃣ Wandb에서 실험 결과 확인
# https://wandb.ai/your-username/morigirl-classification
```

## 🎛️ 명령어 옵션

### save_image_vectors.py

```bash
--config-path    # 설정 파일 경로 (기본: config.json)
--max-products   # 상품 수 override (설정 파일 우선)
```

### train_model.py

```bash
--config-path      # 설정 파일 경로
--data-path        # 데이터 경로 override
--experiment-name  # 실험 이름 지정
```

### test_model.py

```bash
--checkpoint       # 모델 체크포인트 경로 (필수)
--config-path      # 설정 파일 경로
--data-path        # 데이터 경로 override
--quick-test       # 빠른 테스트 모드
--num-samples      # 빠른 테스트 샘플 수
```

## 🔍 트러블슈팅

### 1. wandb 오류

```bash
# wandb 로그인 확인
wandb login

# wandb 비활성화
# config.json에서 "wandb.enabled": false
```

### 2. 메모리 부족

```bash
# 배치 크기 줄이기
# config.json에서 "model.batch_size": 32
```

### 3. 데이터 로딩 실패

```bash
# 데이터 경로 확인
ls data/morigirl_5000/
# morigirl_5000.npy, non_morigirl_5000.npy 존재 확인
```

## 🤝 기여

이슈 제기 및 PR을 환영합니다!

## �� 라이선스

MIT License
