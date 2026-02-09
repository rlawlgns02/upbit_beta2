"""
LSTM 기반 가격 예측 모델
시계열 데이터 + 소셜 미디어 감정 분석을 활용한 암호화폐 가격 예측
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from sklearn.preprocessing import MinMaxScaler
import os
import sys
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 소셜 감정 분석 모듈 임포트
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from news.social_sentiment import get_social_sentiment_aggregator
    SOCIAL_SENTIMENT_AVAILABLE = True
except ImportError:
    SOCIAL_SENTIMENT_AVAILABLE = False
    logger.debug("[LSTM] 소셜 감정 분석 모듈 사용 불가")


class LSTMModel(nn.Module):
    """LSTM 가격 예측 신경망"""

    def __init__(
        self,
        input_size: int = 5,      # OHLCV 기본 5개 특징
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,     # 다음 가격 예측
        dropout: float = 0.2
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention 메커니즘 (선택적)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파

        Args:
            x: (batch_size, seq_len, input_size)

        Returns:
            예측값: (batch_size, output_size)
        """
        # LSTM 처리
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)

        # Attention 가중치 계산
        attention_weights = self.attention(lstm_out)  # (batch, seq, 1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden)

        # 출력
        output = self.fc(context)  # (batch, output_size)
        return output


class LSTMPredictor:
    """LSTM 가격 예측기"""

    def __init__(
        self,
        model_path: str = 'models/lstm_predictor',
        seq_length: int = 60,      # 60개 시퀀스 사용
        feature_columns: List[str] = None,
        device: str = None
    ):
        self.model_path = model_path
        self.seq_length = seq_length
        self.feature_columns = feature_columns or ['open', 'high', 'low', 'close', 'volume']
        self.scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()  # 가격 전용 스케일러

        # 디바이스 설정
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: Optional[LSTMModel] = None
        self.is_trained = False
        # 학습 중단 플래그
        self._stop_training = False

    def request_stop(self):
        """학습 중단 요청"""
        self._stop_training = True

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 추가"""
        df = df.copy()

        # 이동평균선
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # 볼린저 밴드
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # 거래량 변화율
        df['volume_change'] = df['volume'].pct_change()

        # 가격 변화율
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)

        # NaN 제거
        df = df.dropna()

        return df

    def _prepare_data(
        self,
        df: pd.DataFrame,
        add_indicators: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """데이터 전처리

        Args:
            df: OHLCV 데이터프레임
            add_indicators: 기술적 지표 추가 여부

        Returns:
            (X, y) 튜플
        """
        if add_indicators:
            df = self._add_technical_indicators(df)

        # 사용할 컬럼 선택
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                        'sma_5', 'sma_20', 'rsi', 'macd', 'macd_signal',
                        'bb_position', 'volume_change', 'price_change', 'price_change_5']

        # 존재하는 컬럼만 사용
        available_cols = [col for col in feature_cols if col in df.columns]
        data = df[available_cols].values

        # 가격 데이터 별도 저장 (역정규화용)
        prices = df['close'].values.reshape(-1, 1)
        self.price_scaler.fit(prices)

        # 정규화
        scaled_data = self.scaler.fit_transform(data)

        # 시퀀스 생성
        X, y = [], []
        for i in range(self.seq_length, len(scaled_data)):
            X.append(scaled_data[i - self.seq_length:i])
            # 다음 종가 예측 (정규화된 close 인덱스 = 3)
            y.append(scaled_data[i, 3])  # close 컬럼

        return np.array(X), np.array(y)

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        verbose: bool = True,
        progress_callback: Optional[callable] = None,
        resume_from_checkpoint: bool = False,
        checkpoint_interval: int = 10  # 체크포인트 저장 간격 (디스크 공간 절약을 위해 10으로 증가)
    ) -> dict:
        """모델 학습 (진행 콜백 및 중단 지원)

        Args:
            df: OHLCV 데이터프레임
            epochs: 에폭 수
            batch_size: 배치 크기
            learning_rate: 학습률
            validation_split: 검증 데이터 비율
            early_stopping_patience: 조기 종료 인내심
            verbose: 출력 여부
            progress_callback: 학습 진행 정보를 전달받는 콜백 (dict를 인자로 받음)

        Returns:
            학습 결과 딕셔너리
        """
        if verbose:
            print(f"[LSTM] 데이터 전처리 중... (총 {len(df)}개 데이터)")

        X, y = self._prepare_data(df)

        if len(X) < 100:
            raise ValueError(f"데이터가 부족합니다. 최소 100개 필요 (현재: {len(X)})")

        # 학습/검증 분할
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        if verbose:
            print(f"[LSTM] 학습 데이터: {len(X_train)}, 검증 데이터: {len(X_val)}")

        # 텐서 변환
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # 모델 생성
        input_size = X_train.shape[2]
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout=0.2
        ).to(self.device)

        # 손실 함수 및 옵티마이저
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 학습
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []

        if verbose:
            print(f"[LSTM] 학습 시작 (디바이스: {self.device})")

        start_epoch = 0
        # resume support: if requested, load latest epoch checkpoint and continue
        if resume_from_checkpoint:
            latest = self._find_latest_epoch_checkpoint()
            if latest:
                epoch_num, path = latest
                meta = self._load_checkpoint(path)
                start_epoch = epoch_num
                if verbose:
                    print(f"[LSTM] 체크포인트에서 재개: epoch {start_epoch}")

        # Determine end epoch (support resume meaning: 'epochs' additional epochs when resume_from_checkpoint=True)
        if resume_from_checkpoint and start_epoch > 0:
            end_epoch = start_epoch + epochs
        else:
            end_epoch = epochs

        for epoch in range(start_epoch, end_epoch):
            current_epoch_index = epoch + 1

            if self._stop_training:
                if verbose:
                    print(f"[LSTM] 학습 중단 요청 수신: Epoch {current_epoch_index}")
                # save partial checkpoint
                try:
                    self._save_checkpoint(epoch=current_epoch_index, is_best=False, extra={'cancelled': True})
                except Exception:
                    pass
                return {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'epochs_trained': len(train_losses),
                    'cancelled': True
                }

            self.model.train()

            # 미니배치 학습
            total_train_loss = 0
            num_batches = 0

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

            avg_train_loss = total_train_loss / num_batches
            train_losses.append(avg_train_loss)

            # 검증
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_losses.append(val_loss)

            scheduler.step(val_loss)

            # 진행 콜백
            if progress_callback:
                try:
                    progress_callback({
                        'epoch': current_epoch_index,
                        'train_loss': avg_train_loss,
                        'val_loss': val_loss,
                        'best_val_loss': best_val_loss
                    })
                except Exception:
                    pass

            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 최적 모델 저장
                try:
                    self._save_checkpoint(epoch=current_epoch_index, is_best=True)
                except Exception:
                    pass
            else:
                patience_counter += 1

            # 주기적 체크포인트 저장 (예: 매 5 epoch)
            try:
                if checkpoint_interval and current_epoch_index % checkpoint_interval == 0:
                    self._save_checkpoint(epoch=current_epoch_index)
            except Exception:
                pass

            if verbose and (current_epoch_index) % 10 == 0:
                print(f"[LSTM] Epoch {current_epoch_index}/{epochs} - "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"[LSTM] 조기 종료 (Epoch {current_epoch_index})")
                break

        # 최적 모델 로드
        self._load_checkpoint()
        self.is_trained = True

        if verbose:
            print(f"[LSTM] 학습 완료! 최적 검증 손실: {best_val_loss:.6f}")

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs_trained': len(train_losses)
        }

    def predict(self, df: pd.DataFrame) -> Tuple[float, float, str]:
        """가격 예측

        Args:
            df: 최근 OHLCV 데이터 (최소 seq_length + 30개)

        Returns:
            (예측 가격, 변화율 예측, 방향)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("모델이 학습되지 않았습니다.")

        # 데이터 준비
        df_processed = self._add_technical_indicators(df)

        # 지표 추가 후 길이 재확인 (dropna로 인한 감소 고려)
        if len(df_processed) < self.seq_length:
            raise ValueError(
                f"데이터가 부족합니다. 최소 {self.seq_length}개 필요 (현재: {len(df_processed)}개)"
            )

        # 특징 추출
        feature_cols = ['open', 'high', 'low', 'close', 'volume',
                        'sma_5', 'sma_20', 'rsi', 'macd', 'macd_signal',
                        'bb_position', 'volume_change', 'price_change', 'price_change_5']
        available_cols = [col for col in feature_cols if col in df_processed.columns]
        data = df_processed[available_cols].values

        # Scaler 검증
        if not hasattr(self.scaler, 'data_min_') or not hasattr(self.scaler, 'data_max_'):
            raise ValueError("Scaler가 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        # 정규화
        scaled_data = self.scaler.transform(data)

        # 마지막 시퀀스 추출
        if len(scaled_data) < self.seq_length:
            raise ValueError(
                f"정규화 후 데이터가 부족합니다. 필요: {self.seq_length}, 현재: {len(scaled_data)}"
            )

        last_sequence = scaled_data[-self.seq_length:]
        X = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        # 예측
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(X).cpu().numpy()[0, 0]

        # 역정규화 (close 컬럼의 스케일 사용)
        current_price = df_processed['close'].iloc[-1]

        # close 컬럼의 인덱스를 동적으로 찾기
        try:
            close_idx = available_cols.index('close')
            close_min = self.scaler.data_min_[close_idx]
            close_max = self.scaler.data_max_[close_idx]
        except (ValueError, IndexError) as e:
            raise ValueError(f"역정규화 오류: close 컬럼을 찾을 수 없습니다 - {e}")

        # 역정규화 수행
        predicted_price = pred_scaled * (close_max - close_min) + close_min

        # 변화율 계산
        change_rate = ((predicted_price - current_price) / current_price) * 100

        # 방향 결정
        if change_rate > 1.0:
            direction = "STRONG_UP"
        elif change_rate > 0.3:
            direction = "UP"
        elif change_rate < -1.0:
            direction = "STRONG_DOWN"
        elif change_rate < -0.3:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"

        return predicted_price, change_rate, direction

    def predict_next_n(self, df: pd.DataFrame, n_steps: int = 5) -> List[dict]:
        """다음 n개 스텝 예측

        Args:
            df: OHLCV 데이터
            n_steps: 예측할 스텝 수

        Returns:
            예측 결과 리스트
        """
        predictions = []
        temp_df = df.copy()

        for i in range(n_steps):
            try:
                pred_price, change_rate, direction = self.predict(temp_df)
                predictions.append({
                    'step': i + 1,
                    'predicted_price': pred_price,
                    'change_rate': change_rate,
                    'direction': direction
                })

                # 예측값을 다음 입력으로 사용 (간단한 방식)
                last_row = temp_df.iloc[-1].copy()
                last_row['close'] = pred_price
                last_row['open'] = temp_df.iloc[-1]['close']
                last_row['high'] = max(pred_price, last_row['open'])
                last_row['low'] = min(pred_price, last_row['open'])
                temp_df = pd.concat([temp_df, pd.DataFrame([last_row])], ignore_index=True)

            except Exception as e:
                print(f"[LSTM] 예측 오류 (step {i + 1}): {e}")
                break

        return predictions

    def _save_checkpoint(self, epoch: Optional[int] = None, is_best: bool = False, extra: Optional[dict] = None):
        """모델 체크포인트 저장

        Args:
            epoch: 현재 epoch (있다면 파일명에 포함)
            is_best: 현재 모델이 베스트 모델인지 여부
            extra: 추가 메타데이터 저장
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        data = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'price_scaler': self.price_scaler,
            'seq_length': self.seq_length,
            'feature_columns': self.feature_columns,
            'meta': {
                'saved_at': datetime.now().isoformat(),
                'epoch': epoch,
            }
        }

        if extra:
            data['meta'].update(extra)

        # 기본 이름 (최신/베스트용)
        base_path = f"{self.model_path}.pt"
        # 에폭 별 파일도 저장
        if epoch is not None:
            epoch_path = f"{self.model_path}_epoch{epoch}.pt"
            torch.save(data, epoch_path)

        # 항상 최신으로 기본 파일 덮어쓰기
        torch.save(data, base_path)

        # 별도 베스트 파일
        if is_best:
            best_path = f"{self.model_path}_best.pt"
            torch.save(data, best_path)

    def _find_latest_epoch_checkpoint(self) -> Optional[tuple]:
        """가장 최신의 epoch 체크포인트 파일을 찾아 (epoch, path) 반환"""
        directory = os.path.dirname(self.model_path)
        basename = os.path.basename(self.model_path)
        files = []
        try:
            for fname in os.listdir(directory or '.'):
                if fname.startswith(basename + '_epoch') and fname.endswith('.pt'):
                    # 파일명에서 epoch 추출
                    try:
                        epoch_str = fname.split('_epoch')[-1].split('.pt')[0]
                        epoch = int(epoch_str)
                        files.append((epoch, os.path.join(directory, fname)))
                    except Exception:
                        continue
            if not files:
                return None
            files.sort(key=lambda x: x[0])
            return files[-1]
        except FileNotFoundError:
            return None

    def _load_checkpoint(self, path: Optional[str] = None) -> Optional[dict]:
        """모델 체크포인트 로드

        Args:
            path: 체크포인트 파일 경로 (None이면 기본 경로 사용)

        Returns:
            체크포인트 메타 정보 (있으면)
        """
        cp_path = path or f"{self.model_path}.pt"
        checkpoint = torch.load(cp_path, map_location=self.device)

        # 모델 생성 (입력 크기는 스케일러에서 추론)
        input_size = len(checkpoint['scaler'].data_min_)
        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=1,
            dropout=0.2
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.price_scaler = checkpoint['price_scaler']
        self.seq_length = checkpoint['seq_length']
        self.feature_columns = checkpoint['feature_columns']
        self.is_trained = True

        meta = checkpoint.get('meta', {})
        return meta

    def save(self):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        self._save_checkpoint()
        logger.info(f"[LSTM] 모델 저장 완료: {self.model_path}.pt")

    def load(self) -> bool:
        """모델 로드 (기본 경로의 최신 체크포인트 로드)

        Returns:
            로드 성공 여부
        """
        try:
            if not os.path.exists(f"{self.model_path}.pt"):
                logger.debug(f"[LSTM] 모델 파일이 없습니다: {self.model_path}.pt")
                return False

            meta = self._load_checkpoint()

            logger.info(f"[LSTM] 모델 로드 완료: {self.model_path}.pt")
            return True

        except Exception as e:
            logger.exception(f"[LSTM] 모델 로드 실패: {e}")
            return False

    def load_latest_checkpoint(self) -> Optional[dict]:
        """가장 최신의 epoch 체크포인트를 찾아 로드하고 메타 정보를 반환"""
        latest = self._find_latest_epoch_checkpoint()
        if not latest:
            return None
        epoch, path = latest
        meta = self._load_checkpoint(path)
        return {'epoch': epoch, 'meta': meta}


def get_device() -> str:
    """최적 디바이스 반환"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'
