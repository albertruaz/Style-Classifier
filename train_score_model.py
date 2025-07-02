# train_score_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import time
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import os
from dotenv import load_dotenv

from dataset.product_score_dataset import ProductScoreDataset, create_train_test_datasets
from model.score_prediction_model import ProductScorePredictor, ProductScoreLoss, get_model_info
from utils.train_utils import EarlyStopping, save_checkpoint, load_checkpoint

class ScoreModelTrainer:
    """상품 점수 예측 모델 학습기"""
    
    def __init__(self, config_path: str = "./config.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        # 설정 로드
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 사용 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = self._create_model()
        
        # 손실함수 및 옵티마이저
        self.criterion = ProductScoreLoss(
            morigirl_weight=1.0,
            popularity_weight=0.5  # 인기도는 조금 더 낮은 가중치
        )
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['model']['learning_rate'],
            weight_decay=1e-4
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=10, min_delta=0.001, restore_best_weights=True
        )
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = "./checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # WandB 설정
        self.use_wandb = False
        self.wandb_run = None
        
        # 학습 기록
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_morigirl_acc': [], 'val_morigirl_acc': [],
            'train_popularity_mse': [], 'val_popularity_mse': []
        }
    
    def _create_model(self) -> ProductScorePredictor:
        """모델 생성"""
        model_config = self.config['model']
        
        model = ProductScorePredictor(
            input_dim=model_config['input_vector_dim'] + 1,  # +1 for price
            hidden_dim=model_config['hidden_dim'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        # 모델 정보 출력
        info = get_model_info(model)
        print(f"📊 모델 정보:")
        print(f"  - 총 파라미터: {info['total_params']:,}")
        print(f"  - 모델 크기: {info['model_size_mb']:.2f} MB")
        
        return model
    
    def _prepare_data(self):
        """데이터셋 준비"""
        print("📊 데이터셋 준비 중...")
        
        # 훈련/테스트 데이터셋 생성
        train_dataset, val_dataset = create_train_test_datasets(self.config_path)
        
        # 데이터로더 생성
        batch_size = self.config['model']['batch_size']
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"✅ 데이터 준비 완료:")
        print(f"  - 훈련 배치 수: {len(self.train_loader)}")
        print(f"  - 검증 배치 수: {len(self.val_loader)}")
    
    def _train_epoch(self) -> dict:
        """한 에폭 훈련"""
        self.model.train()
        
        total_loss = 0
        morigirl_preds, morigirl_targets = [], []
        popularity_losses = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            morigirl_target = targets[:, 0:1].to(self.device)
            popularity_target = targets[:, 1:2].to(self.device)
            
            # 순전파
            morigirl_pred, popularity_pred = self.model(inputs)
            
            # 손실 계산
            losses = self.criterion(
                morigirl_pred, popularity_pred,
                morigirl_target, popularity_target
            )
            
            # 역전파
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # 그래디언트 클리핑
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 메트릭 수집
            total_loss += losses['total_loss'].item()
            
            # 모리걸 예측 정확도를 위한 데이터 수집
            morigirl_preds.extend((morigirl_pred > 0.5).cpu().numpy().flatten())
            morigirl_targets.extend(morigirl_target.cpu().numpy().flatten())
            
            # 인기도 MSE
            popularity_losses.append(losses['popularity_loss'].item())
            
            # 진행바 업데이트
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'mori_loss': f"{losses['morigirl_loss'].item():.4f}",
                'pop_loss': f"{losses['popularity_loss'].item():.4f}"
            })
        
        # 에폭 메트릭 계산
        avg_loss = total_loss / len(self.train_loader)
        morigirl_acc = accuracy_score(morigirl_targets, morigirl_preds)
        avg_popularity_mse = np.mean(popularity_losses)
        
        return {
            'loss': avg_loss,
            'morigirl_acc': morigirl_acc,
            'popularity_mse': avg_popularity_mse
        }
    
    def _validate_epoch(self) -> dict:
        """한 에폭 검증"""
        self.model.eval()
        
        total_loss = 0
        morigirl_preds, morigirl_targets = [], []
        popularity_losses = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                morigirl_target = targets[:, 0:1].to(self.device)
                popularity_target = targets[:, 1:2].to(self.device)
                
                # 순전파
                morigirl_pred, popularity_pred = self.model(inputs)
                
                # 손실 계산
                losses = self.criterion(
                    morigirl_pred, popularity_pred,
                    morigirl_target, popularity_target
                )
                
                total_loss += losses['total_loss'].item()
                
                # 메트릭 수집
                morigirl_preds.extend((morigirl_pred > 0.5).cpu().numpy().flatten())
                morigirl_targets.extend(morigirl_target.cpu().numpy().flatten())
                popularity_losses.append(losses['popularity_loss'].item())
        
        # 검증 메트릭 계산
        avg_loss = total_loss / len(self.val_loader)
        morigirl_acc = accuracy_score(morigirl_targets, morigirl_preds)
        avg_popularity_mse = np.mean(popularity_losses)
        
        return {
            'loss': avg_loss,
            'morigirl_acc': morigirl_acc,
            'popularity_mse': avg_popularity_mse
        }
    
    def train(self):
        """모델 학습"""
        print(f"🚀 학습 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 데이터 준비
        self._prepare_data()
        
        # WandB 초기화
        self.use_wandb = self._initialize_wandb()
        
        num_epochs = self.config['model']['epochs']
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
            
            # 훈련
            train_metrics = self._train_epoch()
            
            # 검증
            val_metrics = self._validate_epoch()
            
            # 학습률 스케줄링
            self.scheduler.step(val_metrics['loss'])
            
            epoch_time = time.time() - start_time
            
            # 결과 출력
            print(f"훈련 - Loss: {train_metrics['loss']:.4f}, "
                  f"Mori Acc: {train_metrics['morigirl_acc']:.4f}, "
                  f"Pop MSE: {train_metrics['popularity_mse']:.4f}")
            print(f"검증 - Loss: {val_metrics['loss']:.4f}, "
                  f"Mori Acc: {val_metrics['morigirl_acc']:.4f}, "
                  f"Pop MSE: {val_metrics['popularity_mse']:.4f}")
            print(f"시간: {epoch_time:.1f}초, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 기록 저장
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['train_morigirl_acc'].append(train_metrics['morigirl_acc'])
            self.train_history['val_morigirl_acc'].append(val_metrics['morigirl_acc'])
            self.train_history['train_popularity_mse'].append(train_metrics['popularity_mse'])
            self.train_history['val_popularity_mse'].append(val_metrics['popularity_mse'])
            
            # WandB 로깅
            self._log_to_wandb(epoch, train_metrics, val_metrics)
            
            # 체크포인트 저장
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                save_checkpoint(
                    self.model, self.optimizer, epoch + 1, val_metrics['loss'],
                    os.path.join(self.checkpoint_dir, 'best_model.pth')
                )
                print("💾 최고 성능 모델 저장")
            
            # Early stopping 체크
            if self.early_stopping(val_metrics['loss'], self.model):
                print(f"⏹️ Early stopping at epoch {epoch + 1}")
                break
        
        print(f"✅ 학습 완료!")
        
        # 최종 모델 저장
        final_path = os.path.join(self.checkpoint_dir, f'final_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
        save_checkpoint(self.model, self.optimizer, epoch + 1, val_metrics['loss'], final_path)
        
        # 학습 곡선 그리기
        self._plot_training_curves()
        
        # WandB 아티팩트 저장 및 종료
        if self.use_wandb:
            self._save_wandb_artifacts()
            wandb.finish()
            print("🎯 WandB 실험 종료")
    
    def _initialize_wandb(self) -> bool:
        """WandB 초기화"""
        try:
            import os
            from dotenv import load_dotenv
            
            # .env 파일 로드
            load_dotenv()
            
            wandb_config = {
                **self.config,
                'model_architecture': 'ProductScorePredictor',
                'device': str(self.device),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
            
            # WandB 프로젝트 설정
            project_name = os.getenv('WANDB_PROJECT', 'mori-look-score-prediction')
            entity_name = os.getenv('WANDB_ENTITY', 'albertruaz')
            
            self.wandb_run = wandb.init(
                project=project_name,
                entity=entity_name,
                config=wandb_config,
                name=f"score_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['morigirl', 'popularity', 'multitask', 'score-prediction'],
                notes="모리걸 스타일 분류와 상품 인기도 예측을 위한 멀티태스크 모델"
            )
            
            # 모델 아키텍처 로깅
            wandb.watch(self.model, log='all', log_freq=10)
            
            print("✅ WandB 초기화 성공")
            print(f"📊 프로젝트: {project_name}")
            print(f"👤 엔티티: {entity_name}")
            print(f"🔗 실험 URL: {self.wandb_run.url}")
            
            return True
            
        except ImportError:
            print("⚠️ WandB 라이브러리가 설치되지 않았습니다.")
            print("pip install wandb 명령어로 설치하세요.")
            return False
        except Exception as e:
            print(f"⚠️ WandB 초기화 실패: {e}")
            print("로컬 학습으로 진행합니다.")
            return False
    
    def _log_to_wandb(self, epoch: int, train_metrics: dict, val_metrics: dict):
        """WandB에 메트릭 로깅"""
        if not self.use_wandb or not self.wandb_run:
            return
        
        try:
            # 메트릭 로깅
            wandb.log({
                'epoch': epoch + 1,
                
                # 훈련 메트릭
                'train/total_loss': train_metrics['loss'],
                'train/morigirl_accuracy': train_metrics['morigirl_acc'],
                'train/popularity_mse': train_metrics['popularity_mse'],
                
                # 검증 메트릭
                'val/total_loss': val_metrics['loss'],
                'val/morigirl_accuracy': val_metrics['morigirl_acc'],
                'val/popularity_mse': val_metrics['popularity_mse'],
                
                # 시스템 메트릭
                'system/learning_rate': self.optimizer.param_groups[0]['lr'],
                'system/epoch': epoch + 1
            })
            
            # 매 10 에폭마다 추가 분석
            if (epoch + 1) % 10 == 0:
                self._log_detailed_analysis(epoch, val_metrics)
                
        except Exception as e:
            print(f"⚠️ WandB 로깅 실패: {e}")
    
    def _log_detailed_analysis(self, epoch: int, val_metrics: dict):
        """상세 분석 로깅"""
        if not self.use_wandb or not self.wandb_run:
            return
        
        try:
            # 모델 가중치 히스토그램
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    wandb.log({
                        f"weights/{name}": wandb.Histogram(param.data.cpu().numpy()),
                        f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy())
                    })
            
            # 검증 메트릭 요약
            wandb.log({
                f"summary/val_loss_epoch_{epoch+1}": val_metrics['loss'],
                f"summary/val_morigirl_acc_epoch_{epoch+1}": val_metrics['morigirl_acc'],
                f"summary/val_popularity_mse_epoch_{epoch+1}": val_metrics['popularity_mse']
            })
            
        except Exception as e:
            print(f"⚠️ 상세 분석 로깅 실패: {e}")
    
    def _save_wandb_artifacts(self):
        """WandB에 아티팩트 저장"""
        if not self.use_wandb or not self.wandb_run:
            return
        
        try:
            # 모델 체크포인트 아티팩트
            model_artifact = wandb.Artifact(
                name=f"mori_score_model",
                type="model",
                description="모리걸 점수 예측 모델 체크포인트"
            )
            
            # 최고 성능 모델 파일 추가
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_model_path):
                model_artifact.add_file(best_model_path, name="best_model.pth")
            
            # 설정 파일 추가
            if os.path.exists(self.config_path):
                model_artifact.add_file(self.config_path, name="config.json")
            
            # 학습 곡선 이미지 추가
            training_curves_path = os.path.join(self.checkpoint_dir, 'training_curves.png')
            if os.path.exists(training_curves_path):
                model_artifact.add_file(training_curves_path, name="training_curves.png")
                # 이미지를 WandB에도 로깅
                wandb.log({"training_curves": wandb.Image(training_curves_path)})
            
            # 아티팩트 로깅
            self.wandb_run.log_artifact(model_artifact)
            
            print("💾 WandB 아티팩트 저장 완료")
            
        except Exception as e:
            print(f"⚠️ WandB 아티팩트 저장 실패: {e}")
    
    def _plot_training_curves(self):
        """학습 곡선 시각화"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 손실 곡선
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 모리걸 정확도
        ax2.plot(epochs, self.train_history['train_morigirl_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.train_history['val_morigirl_acc'], 'r-', label='Val Acc')
        ax2.set_title('Morigirl Classification Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # 인기도 MSE
        ax3.plot(epochs, self.train_history['train_popularity_mse'], 'b-', label='Train MSE')
        ax3.plot(epochs, self.train_history['val_popularity_mse'], 'r-', label='Val MSE')
        ax3.set_title('Popularity Prediction MSE')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('MSE')
        ax3.legend()
        ax3.grid(True)
        
        # 학습률
        ax4.plot(epochs, [self.optimizer.param_groups[0]['lr']] * len(epochs), 'g-')
        ax4.set_title('Learning Rate')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Learning Rate')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 학습 곡선 저장: {os.path.join(self.checkpoint_dir, 'training_curves.png')}")

def main():
    """메인 함수"""
    config_path = "./config.json"
    
    # 설정 파일 확인
    if not os.path.exists(config_path):
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
        return
    
    try:
        # 트레이너 생성 및 학습
        trainer = ScoreModelTrainer(config_path)
        trainer.train()
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 학습이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 