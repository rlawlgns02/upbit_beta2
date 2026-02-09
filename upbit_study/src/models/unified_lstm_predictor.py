"""
통합 LSTM 예측 모델
- 여러 코인의 차트 데이터를 통합하여 하나의 모델 학습
- 가격 변화율(%) 기반으로 정규화하여 코인 간 스케일 차이 해결
- 범용적인 차트 패턴 학습
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class UnifiedModelConfig:
    """통합 모델 설정"""
    # 모델 구조
    input_size: int = 20          # 입력 특성 수
    hidden_size: int = 128        # LSTM 히든 크기
    num_layers: int = 3           # LSTM 레이어 수
    dropout: float = 0.3          # 드롭아웃

    # 학습 설정
    seq_length: int = 60          # 시퀀스 길이 (60분 = 1시간)
    batch_size: int = 64          # 배치 크기
    epochs: int = 100             # 에폭 수
    learning_rate: float = 0.001  # 학습률

    # 데이터 설정
    min_coins: int = 5            # 최소 학습 코인 수
    samples_per_coin: int = 1000  # 코인당 샘플 수

    # 출력
    output_size: int = 3          # 0: 하락, 1: 유지, 2: 상승


class AttentionLayer(nn.Module):
    """Attention 메커니즘"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        # lstm_output: (batch, seq_len, hidden_size)
        weights = self.attention(lstm_output)  # (batch, seq_len, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, weights


class UnifiedLSTMModel(nn.Module):
    """통합 LSTM 모델"""

    def __init__(self, config: UnifiedModelConfig):
        super().__init__()
        self.config = config

        # 입력 정규화 제거 (배치 크기 1에서 문제 발생)
        # 대신 forward에서 수동 정규화

        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention
        self.attention = AttentionLayer(config.hidden_size * 2)

        # 분류 레이어
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )

        # 회귀 레이어 (가격 변화율 예측)
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            classification: (batch, 3) - 하락/유지/상승 확률
            regression: (batch, 1) - 예측 변화율 (%)
        """
        # 입력 정규화 (수동 - 배치 크기 1에서도 작동)
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True) + 1e-8
        x = (x - mean) / std

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention
        context, _ = self.attention(lstm_out)  # (batch, hidden*2)

        # 분류 (상승/하락/유지)
        classification = self.classifier(context)

        # 회귀 (변화율 예측)
        regression = self.regressor(context)

        return classification, regression


class ChartFeatureExtractor:
    """차트 특성 추출기"""

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        OHLCV 데이터에서 기술적 지표 계산

        Returns:
            20개 특성 포함 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. 가격 변화율 (%)
        features['price_change'] = df['close'].pct_change() * 100

        # 2. 거래량 변화율 (%)
        features['volume_change'] = df['volume'].pct_change() * 100

        # 3. 고가-저가 변동폭 (%)
        features['high_low_range'] = (df['high'] - df['low']) / df['close'] * 100

        # 4. 시가-종가 차이 (%)
        features['open_close_diff'] = (df['close'] - df['open']) / df['open'] * 100

        # 5-7. 이동평균 대비 위치
        for period in [5, 10, 20]:
            ma = df['close'].rolling(period).mean()
            features[f'ma{period}_position'] = (df['close'] - ma) / ma * 100

        # 8. RSI (14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))

        # 9-10. MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = (ema12 - ema26) / df['close'] * 100
        features['macd_signal'] = features['macd'].ewm(span=9).mean()

        # 11-12. 볼린저밴드 위치
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        features['bb_upper_dist'] = (df['close'] - (bb_ma + 2*bb_std)) / df['close'] * 100
        features['bb_lower_dist'] = (df['close'] - (bb_ma - 2*bb_std)) / df['close'] * 100

        # 13. 거래량 MA 대비
        vol_ma = df['volume'].rolling(20).mean()
        features['volume_ma_ratio'] = df['volume'] / (vol_ma + 1e-10)

        # 14. ATR (변동성)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr'] = tr.rolling(14).mean() / df['close'] * 100

        # 15. Stochastic RSI
        rsi = features['rsi']
        rsi_min = rsi.rolling(14).min()
        rsi_max = rsi.rolling(14).max()
        features['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

        # 16. OBV 변화율
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv_change'] = obv.pct_change() * 100

        # 17. 가격 모멘텀 (10분)
        features['momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100

        # 18. Williams %R
        highest = df['high'].rolling(14).max()
        lowest = df['low'].rolling(14).min()
        features['williams_r'] = (highest - df['close']) / (highest - lowest + 1e-10) * -100

        # 19. CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(20).mean()
        tp_std = tp.rolling(20).std()
        features['cci'] = (tp - tp_ma) / (0.015 * tp_std + 1e-10)

        # 20. 연속 상승/하락 수
        changes = df['close'].diff()
        streak = changes.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        features['streak'] = streak.rolling(5).sum()

        # NaN 처리
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        # 이상치 클리핑
        for col in features.columns:
            q1, q99 = features[col].quantile([0.01, 0.99])
            features[col] = features[col].clip(q1, q99)

        return features


class UnifiedLSTMPredictor:
    """통합 LSTM 예측기"""

    def __init__(self, config: Optional[UnifiedModelConfig] = None):
        self.config = config or UnifiedModelConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[UnifiedLSTMModel] = None
        self.feature_extractor = ChartFeatureExtractor()
        self.is_training = False
        self.training_progress = {"current": 0, "total": 0, "loss": 0, "accuracy": 0}

        # 콜백
        self.on_progress_callback = None
        self.on_complete_callback = None

        logger.info(f"UnifiedLSTMPredictor initialized on {self.device}")

    def _create_model(self) -> UnifiedLSTMModel:
        """모델 생성"""
        model = UnifiedLSTMModel(self.config)
        return model.to(self.device)

    def _prepare_dataset(self, all_data: Dict[str, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        여러 코인 데이터로 학습 데이터셋 생성

        Args:
            all_data: {market: DataFrame} 형태의 OHLCV 데이터

        Returns:
            X: (samples, seq_length, features)
            y_class: (samples,) - 0: 하락, 1: 유지, 2: 상승
            y_reg: (samples,) - 변화율 (%)
        """
        X_list = []
        y_class_list = []
        y_reg_list = []

        for market, df in all_data.items():
            if len(df) < self.config.seq_length + 10:
                continue

            # 특성 추출
            features = self.feature_extractor.calculate_features(df)

            # 시퀀스 생성
            for i in range(self.config.seq_length, len(features) - 1):
                seq = features.iloc[i - self.config.seq_length:i].values

                # 레이블: 다음 캔들 변화율
                current_close = df['close'].iloc[i]
                next_close = df['close'].iloc[i + 1]
                change_rate = (next_close - current_close) / current_close * 100

                # 분류 레이블
                if change_rate > 0.3:
                    label = 2  # 상승
                elif change_rate < -0.3:
                    label = 0  # 하락
                else:
                    label = 1  # 유지

                X_list.append(seq)
                y_class_list.append(label)
                y_reg_list.append(change_rate)

        X = np.array(X_list, dtype=np.float32)
        y_class = np.array(y_class_list, dtype=np.int64)
        y_reg = np.array(y_reg_list, dtype=np.float32)

        logger.info(f"Dataset prepared: {len(X)} samples from {len(all_data)} coins")

        return X, y_class, y_reg

    async def train(self, all_data: Dict[str, pd.DataFrame],
                    epochs: Optional[int] = None,
                    callback: Optional[callable] = None) -> Dict:
        """
        여러 코인 데이터로 통합 모델 학습

        Args:
            all_data: {market: DataFrame} - 각 코인의 OHLCV 데이터
            epochs: 에폭 수 (None이면 config 사용)
            callback: 진행 상황 콜백

        Returns:
            학습 결과 딕셔너리
        """
        if self.is_training:
            return {"success": False, "error": "이미 학습 중입니다"}

        if len(all_data) < self.config.min_coins:
            return {"success": False, "error": f"최소 {self.config.min_coins}개 코인 데이터가 필요합니다"}

        self.is_training = True
        epochs = epochs or self.config.epochs

        try:
            # 데이터 준비
            logger.info(f"Preparing dataset from {len(all_data)} coins...")
            X, y_class, y_reg = self._prepare_dataset(all_data)

            if len(X) < 100:
                return {"success": False, "error": "충분한 학습 데이터가 없습니다"}

            # 데이터 셔플 및 분할
            indices = np.random.permutation(len(X))
            split_idx = int(len(X) * 0.8)

            train_idx = indices[:split_idx]
            val_idx = indices[split_idx:]

            X_train = torch.tensor(X[train_idx]).to(self.device)
            y_class_train = torch.tensor(y_class[train_idx]).to(self.device)
            y_reg_train = torch.tensor(y_reg[train_idx]).to(self.device)

            X_val = torch.tensor(X[val_idx]).to(self.device)
            y_class_val = torch.tensor(y_class[val_idx]).to(self.device)
            y_reg_val = torch.tensor(y_reg[val_idx]).to(self.device)

            # 모델 생성
            self.model = self._create_model()

            # 옵티마이저 및 손실 함수
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

            class_criterion = nn.CrossEntropyLoss()
            reg_criterion = nn.MSELoss()

            # 학습
            best_val_acc = 0
            best_model_state = None
            history = {"train_loss": [], "val_loss": [], "val_acc": []}

            self.training_progress["total"] = epochs

            for epoch in range(epochs):
                self.training_progress["current"] = epoch + 1
                self.model.train()

                # 미니배치 학습
                total_loss = 0
                num_batches = 0

                for i in range(0, len(X_train), self.config.batch_size):
                    batch_X = X_train[i:i+self.config.batch_size]
                    batch_y_class = y_class_train[i:i+self.config.batch_size]
                    batch_y_reg = y_reg_train[i:i+self.config.batch_size]

                    optimizer.zero_grad()

                    class_out, reg_out = self.model(batch_X)

                    # 복합 손실: 분류 + 회귀
                    loss_class = class_criterion(class_out, batch_y_class)
                    loss_reg = reg_criterion(reg_out.squeeze(), batch_y_reg)
                    loss = loss_class + 0.5 * loss_reg

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                scheduler.step()
                avg_train_loss = total_loss / num_batches

                # 검증
                self.model.eval()
                with torch.no_grad():
                    val_class_out, val_reg_out = self.model(X_val)
                    val_loss_class = class_criterion(val_class_out, y_class_val)
                    val_loss_reg = reg_criterion(val_reg_out.squeeze(), y_reg_val)
                    val_loss = val_loss_class + 0.5 * val_loss_reg

                    # 정확도 계산
                    predictions = torch.argmax(val_class_out, dim=1)
                    val_acc = (predictions == y_class_val).float().mean().item()

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(val_loss.item())
                history["val_acc"].append(val_acc)

                self.training_progress["loss"] = avg_train_loss
                self.training_progress["accuracy"] = val_acc

                # 최고 모델 저장
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = self.model.state_dict().copy()

                # 콜백
                if callback:
                    await callback({
                        "epoch": epoch + 1,
                        "total_epochs": epochs,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss.item(),
                        "val_accuracy": val_acc
                    })

                # 비동기 양보
                await asyncio.sleep(0)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 최고 모델 복원
            if best_model_state:
                self.model.load_state_dict(best_model_state)

            # 모델 저장
            model_path = self._save_model()

            return {
                "success": True,
                "model_path": model_path,
                "final_accuracy": best_val_acc,
                "total_samples": len(X),
                "num_coins": len(all_data),
                "history": history
            }

        except Exception as e:
            logger.error(f"Training error: {e}")
            return {"success": False, "error": str(e)}

        finally:
            self.is_training = False

    def _save_model(self) -> str:
        """모델 저장"""
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/unified_lstm_{timestamp}.pt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "timestamp": timestamp
        }, model_path)

        # 최신 모델 심볼릭 링크 (또는 복사)
        latest_path = "models/unified_lstm_latest.pt"
        if os.path.exists(latest_path):
            os.remove(latest_path)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": self.config.__dict__,
            "timestamp": timestamp
        }, latest_path)

        logger.info(f"Model saved: {model_path}")
        return model_path

    def load_model(self, model_path: Optional[str] = None) -> bool:
        """모델 로드"""
        try:
            if model_path is None:
                model_path = "models/unified_lstm_latest.pt"

            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                return False

            checkpoint = torch.load(model_path, map_location=self.device)

            # Config 복원
            if "config" in checkpoint:
                for key, value in checkpoint["config"].items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

            # 모델 생성 및 로드 (strict=False로 호환성 유지)
            self.model = self._create_model()
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.model.eval()

            logger.info(f"Model loaded: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Model load error: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        가격 예측

        Args:
            df: OHLCV DataFrame (최소 seq_length 이상)

        Returns:
            예측 결과 딕셔너리
        """
        if self.model is None:
            if not self.load_model():
                return {"success": False, "error": "모델이 로드되지 않았습니다"}

        # DataFrame 유효성 검증
        if df is None or df.empty:
            return {"success": False, "error": "데이터가 없습니다"}

        if len(df) < self.config.seq_length:
            return {"success": False, "error": f"최소 {self.config.seq_length}개 데이터가 필요합니다"}

        try:
            # 특성 추출
            features = self.feature_extractor.calculate_features(df)

            # 마지막 시퀀스
            seq = features.iloc[-self.config.seq_length:].values
            X = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            # 예측
            self.model.eval()
            with torch.no_grad():
                class_out, reg_out = self.model(X)

                probs = torch.softmax(class_out, dim=1)[0].cpu().numpy()
                predicted_change = reg_out[0].item()
                predicted_class = torch.argmax(class_out, dim=1)[0].item()

            # 결과 해석
            class_names = ["하락", "유지", "상승"]
            direction = class_names[predicted_class]

            current_price = df['close'].iloc[-1]
            predicted_price = current_price * (1 + predicted_change / 100)

            # 신뢰도 계산
            confidence = probs[predicted_class]

            return {
                "success": True,
                "direction": direction,
                "direction_en": ["DOWN", "NEUTRAL", "UP"][predicted_class],
                "predicted_change_rate": predicted_change,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "confidence": confidence,
                "probabilities": {
                    "down": float(probs[0]),
                    "neutral": float(probs[1]),
                    "up": float(probs[2])
                }
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"success": False, "error": str(e)}

    def get_training_progress(self) -> Dict:
        """학습 진행 상황 반환"""
        return {
            "is_training": self.is_training,
            "current_epoch": self.training_progress["current"],
            "total_epochs": self.training_progress["total"],
            "loss": self.training_progress["loss"],
            "accuracy": self.training_progress["accuracy"]
        }


# 전역 인스턴스
unified_predictor = UnifiedLSTMPredictor()
