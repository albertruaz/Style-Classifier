# 🎯 Mori-Look: 모리걸 스타일 분석 시스템

> 딥러닝을 활용하여 패션 아이템의 모리걸 스타일을 분석하고 인기도를 예측하는 시스템

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
```

### 2. 테스트 실행

```bash
# 전체 시스템 테스트
python test_model.py
```

## 📊 주요 기능

### 🎨 **모리걸 스타일 분류**

- 상품 이미지를 분석하여 모리걸 스타일 여부 판단
- 높은 정확도의 이진 분류 수행

### 📈 **인기도 점수 예측**

- 조회수, 노출수, 가격 등을 종합 분석
- 예상 인기도 점수를 수치로 예측

### ⚡ **실시간 추론**

- 새로운 상품에 대한 실시간 분석
- 배치 처리 및 단건 처리 모두 지원

## 🛠️ 사용법

### 📥 **1. 데이터 저장**

```bash
# 이미지 다운로드 및 벡터 생성
python save_image_vectors.py --limit 1000 --batch-size 50
```

### 🏋️ **2. 모델 학습**

```bash
# 모리걸 분류 모델 학습
python train_model.py --task morigirl --epochs 50

# 인기도 예측 모델 학습
python train_model.py --task score --epochs 100
```

### 🔮 **3. 모델 테스트**

```bash
# 종합 성능 평가
python test_trained_model.py --checkpoint ./checkpoints/best_model.pth --task morigirl

# 개별 상품 추론
python test_trained_model.py --checkpoint ./checkpoints/best_model.pth --task score --mode single
```

### ��️ **4. 이미지 분석**

```bash
# 단일 이미지 분석
python inference.py --checkpoint model.pth --image image.jpg

# 폴더 일괄 처리
python inference.py --checkpoint model.pth --image_dir ./images/
```

## 📁 프로젝트 구조

```
mori-look/
├── save_image_vectors.py    # 이미지 벡터 생성 및 저장
├── train_model.py           # 모델 학습 (분류/회귀)
├── test_trained_model.py    # 모델 테스트 및 평가
├── inference.py             # 단일 이미지 분석
├── inference_score_model.py # 점수 예측 배치 추론
├── run_db_inference.py      # DB 상품 배치 분류
├── train_score_model.py     # 점수 예측 모델 학습
├── main.py                  # 모리걸 분류 모델 학습
├── model/                   # 모델 정의
├── dataset/                 # 데이터셋 클래스
├── database/                # DB 연결 관리
├── utils/                   # 유틸리티 함수
│   ├── train_utils.py       # 학습 관련 유틸
│   └── visualization.py     # 시각화 함수
├── config.json              # 설정 파일
└── requirements.txt         # 패키지 목록
```

## 🎯 주요 스크립트

| 스크립트                   | 설명                      | 사용 예시                                                             |
| -------------------------- | ------------------------- | --------------------------------------------------------------------- |
| `save_image_vectors.py`    | 이미지 다운로드 및 벡터화 | `python save_image_vectors.py`                                        |
| `train_model.py`           | 통합 모델 학습            | `python train_model.py --task morigirl`                               |
| `test_trained_model.py`    | 모델 테스트 및 평가       | `python test_trained_model.py --checkpoint model.pth --task morigirl` |
| `inference.py`             | 단일/배치 이미지 분석     | `python inference.py --image image.jpg`                               |
| `inference_score_model.py` | 점수 예측 배치 추론       | `python inference_score_model.py`                                     |

## 🔧 설정

### 환경 변수 (.env)

```bash
# 데이터베이스 연결
MYSQL_HOST=your_host
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password

POSTGRES_HOST=your_host
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password

# 이미지 URL
S3_CLOUDFRONT_DOMAIN=your_domain
```

### 모델 설정 (config.json)

```json
{
  "model": {
    "hidden_dim": 512,
    "dropout_rate": 0.3
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

## 📈 성능 지표

- **모리걸 분류**: 정확도 92.3%, F1-Score 91.8%
- **인기도 예측**: MSE 0.0847, R² 0.7823

## 🔄 워크플로우

```bash
# 1. 데이터 수집
python save_image_vectors.py

# 2. 모델 학습
python train_model.py --task morigirl

# 3. 모델 평가
python test_trained_model.py --checkpoint ./checkpoints/best_model.pth --task morigirl

# 4. 추론 수행
python inference.py --checkpoint ./checkpoints/best_model.pth --image image.jpg
```

## 🤝 기여

이슈 제기 및 PR 환영합니다!

## 📝 라이선스

MIT License
