"""
LSTM 기반 단타 트레이딩 봇
학습된 LSTM 모델의 예측을 활용한 자동매매 시스템

주요 기능:
- LSTM 예측 + 기술적 지표 복합 신호 생성
- 급락 감지 시 즉시 매도
- 동적 손절/익절 (변동성 기반)
- 추세 반전 감지
- VWAP 기반 매매
"""
import asyncio
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.upbit_client import UpbitClient

logger = logging.getLogger(__name__)


@dataclass
class LSTMScalpingConfig:
    """LSTM 단타 봇 설정"""
    # LSTM vs 기술적 지표 가중치
    lstm_weight: float = 0.6
    technical_weight: float = 0.4

    # LSTM 방향 임계값 (변화율 %)
    strong_buy_threshold: float = 1.0
    buy_threshold: float = 0.3
    sell_threshold: float = -0.3
    strong_sell_threshold: float = -1.0

    # 동적 손절/익절 설정
    use_dynamic_stops: bool = True
    base_profit_percent: float = 0.5
    base_loss_percent: float = 0.3
    volatility_multiplier: float = 1.5
    min_stop_percent: float = 0.2
    max_stop_percent: float = 2.0

    # 급락 감지 설정
    crash_detection_enabled: bool = True
    crash_price_threshold: float = -2.0
    crash_volume_spike: float = 3.0
    consecutive_red_candles: int = 3

    # 거래 설정
    trade_amount: float = 50000
    min_trade_amount: float = 10000  # 업비트 최소 거래금액 (5천원 미만 불가)
    max_positions: int = 5
    cooldown_seconds: int = 60
    max_holding_minutes: int = 60

    # 거래 수수료
    commission: float = 0.0005


@dataclass
class Position:
    """포지션 정보"""
    market: str
    entry_price: float
    entry_time: datetime
    amount: float
    volume: float
    uuid: Optional[str] = None
    highest_price: float = 0
    lowest_price: float = float('inf')


class CrashDetector:
    """급락 감지 알고리즘"""

    def __init__(self, config: LSTMScalpingConfig):
        self.config = config

    def detect(self, df: pd.DataFrame) -> Tuple[bool, str, float]:
        """
        급락 신호 감지

        Returns:
            (is_crash, reason, severity)
            - is_crash: 급락 여부
            - reason: 급락 사유
            - severity: 심각도 (높을수록 심각)
        """
        if len(df) < 20:
            return False, "", 0

        reasons = []
        severity = 0

        # 1. 순간 가격 급락 (최근 1분)
        price_change_1m = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        if price_change_1m <= self.config.crash_price_threshold:
            reasons.append(f"1분 급락 {price_change_1m:.2f}%")
            severity += abs(price_change_1m)

        # 2. 거래량 스파이크 + 하락
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        if volume_ma > 0:
            volume_ratio = df['volume'].iloc[-1] / volume_ma
            if volume_ratio >= self.config.crash_volume_spike and price_change_1m < -0.5:
                reasons.append(f"거래량 {volume_ratio:.1f}배 + 하락")
                severity += 1

        # 3. 연속 하락 캔들
        red_count = 0
        for i in range(-self.config.consecutive_red_candles, 0):
            if df['close'].iloc[i] < df['open'].iloc[i]:
                red_count += 1

        if red_count >= self.config.consecutive_red_candles:
            reasons.append(f"연속 {red_count}개 하락캔들")
            severity += 0.5

        # 4. 볼린저 밴드 하단 급락
        bb_mid = df['close'].rolling(20).mean().iloc[-1]
        bb_std = df['close'].rolling(20).std().iloc[-1]
        bb_lower = bb_mid - 2 * bb_std

        if df['close'].iloc[-1] < bb_lower:
            reasons.append("볼린저밴드 하단 돌파")
            severity += 0.5

        # 5. 급격한 하락 모멘텀 (3분 내 -3%)
        if len(df) >= 4:
            price_change_3m = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100
            if price_change_3m <= -3.0:
                reasons.append(f"3분 내 {price_change_3m:.2f}% 급락")
                severity += 1.5

        is_crash = len(reasons) >= 2 or severity >= 2
        return is_crash, ", ".join(reasons), severity


class TrendReversalDetector:
    """추세 반전 감지"""

    def detect(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        추세 반전 감지

        Returns:
            (reversal_type, confidence)
            - reversal_type: 'bullish', 'bearish', 'none'
            - confidence: 신뢰도 (0~1)
        """
        if len(df) < 30:
            return 'none', 0

        reversal_type = 'none'
        confidence = 0

        # RSI 계산
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # RSI 다이버전스
        price_trend = df['close'].iloc[-5:].mean() - df['close'].iloc[-10:-5].mean()
        rsi_trend = rsi.iloc[-5:].mean() - rsi.iloc[-10:-5].mean()

        if price_trend < 0 and rsi_trend > 0:
            reversal_type = 'bullish'
            confidence = 0.6
        elif price_trend > 0 and rsi_trend < 0:
            reversal_type = 'bearish'
            confidence = 0.6

        # MACD 크로스
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()

        if len(macd) >= 2 and len(signal) >= 2:
            if macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
                reversal_type = 'bullish'
                confidence = max(confidence, 0.7)
            elif macd.iloc[-2] > signal.iloc[-2] and macd.iloc[-1] < signal.iloc[-1]:
                reversal_type = 'bearish'
                confidence = max(confidence, 0.7)

        # RSI 극단값 반전
        current_rsi = rsi.iloc[-1]
        if current_rsi < 25 and rsi.iloc[-2] < rsi.iloc[-1]:
            reversal_type = 'bullish'
            confidence = max(confidence, 0.65)
        elif current_rsi > 75 and rsi.iloc[-2] > rsi.iloc[-1]:
            reversal_type = 'bearish'
            confidence = max(confidence, 0.65)

        return reversal_type, confidence


class VWAPStrategy:
    """VWAP 기반 매매 전략"""

    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """VWAP 계산"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.iloc[-1]

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        """
        VWAP 기반 신호 생성

        Returns:
            (signal, confidence)
            - signal: 'buy', 'sell', 'hold'
            - confidence: 신뢰도
        """
        if len(df) < 10:
            return 'hold', 0

        vwap = self.calculate_vwap(df)
        current_price = df['close'].iloc[-1]
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-3]) / df['close'].iloc[-3] * 100

        # VWAP 대비 위치
        vwap_diff_percent = (current_price - vwap) / vwap * 100

        signal = 'hold'
        confidence = 0.5

        # 가격 < VWAP이고 상승 모멘텀 → 매수
        if vwap_diff_percent < -0.5 and price_change > 0.1:
            signal = 'buy'
            confidence = min(0.7, 0.5 + abs(vwap_diff_percent) / 10)

        # 가격 > VWAP이고 하락 모멘텀 → 매도
        elif vwap_diff_percent > 0.5 and price_change < -0.1:
            signal = 'sell'
            confidence = min(0.7, 0.5 + abs(vwap_diff_percent) / 10)

        return signal, confidence


class VolumeAnomalyDetector:
    """거래량 이상 감지"""

    def detect(self, df: pd.DataFrame) -> Tuple[bool, str, float]:
        """
        비정상 거래량 패턴 감지

        Returns:
            (is_anomaly, anomaly_type, magnitude)
            - is_anomaly: 이상 여부
            - anomaly_type: 'accumulation', 'distribution', 'normal'
            - magnitude: 크기
        """
        if len(df) < 20:
            return False, 'normal', 0

        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / (volume_ma + 1e-10)

        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100

        if volume_ratio > 2.5:
            if price_change > 0.3:
                return True, 'accumulation', volume_ratio
            elif price_change < -0.3:
                return True, 'distribution', volume_ratio

        return False, 'normal', volume_ratio


class TechnicalSignalGenerator:
    """기술적 지표 기반 신호 생성기 (ScalpingBot 스타일)"""

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        기술적 지표 기반 신호 생성

        Returns:
            (signal, confidence, details)
        """
        if len(df) < 30:
            return 'hold', 0, {}

        buy_score = 0
        sell_score = 0
        details = {}

        # 1. RSI (7일)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        details['rsi'] = rsi

        if rsi < 25:
            buy_score += 3
        elif rsi < 40:
            buy_score += 1
        elif rsi > 75:
            sell_score += 3
        elif rsi > 60:
            sell_score += 1

        # 2. 스토캐스틱 RSI
        rsi_series = 100 - (100 / (1 + rs))
        rsi_min = rsi_series.rolling(14).min()
        rsi_max = rsi_series.rolling(14).max()
        stoch_rsi = ((rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100).iloc[-1]
        details['stoch_rsi'] = stoch_rsi

        if stoch_rsi < 20:
            buy_score += 2
        elif stoch_rsi > 80:
            sell_score += 2

        # 3. 볼린저 밴드 위치
        bb_mid = df['close'].rolling(10).mean()
        bb_std = df['close'].rolling(10).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        current_price = df['close'].iloc[-1]
        bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10)
        details['bb_position'] = bb_position

        if bb_position < 0.1:
            buy_score += 2
        elif bb_position > 0.9:
            sell_score += 2

        # 4. 거래량 분석
        volume_ma = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / (volume_ma + 1e-10)
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        details['volume_ratio'] = volume_ratio

        if volume_ratio > 2:
            if price_change > 0:
                buy_score += 2
            else:
                sell_score += 2

        # 5. 가격 모멘텀 (1분, 3분)
        mom_1m = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
        details['momentum_1m'] = mom_1m

        if len(df) >= 4:
            mom_3m = (df['close'].iloc[-1] - df['close'].iloc[-4]) / df['close'].iloc[-4] * 100
            details['momentum_3m'] = mom_3m

            if mom_3m > 0.5:
                buy_score += 2
            elif mom_3m < -0.5:
                sell_score += 2

        if mom_1m > 0.2:
            buy_score += 1
        elif mom_1m < -0.2:
            sell_score += 1

        # 최종 신호 결정
        details['buy_score'] = buy_score
        details['sell_score'] = sell_score

        if buy_score >= 4 and buy_score > sell_score:
            confidence = min(buy_score / 10, 0.9)
            return 'buy', confidence, details
        elif sell_score >= 4 and sell_score > buy_score:
            confidence = min(sell_score / 10, 0.9)
            return 'sell', confidence, details
        else:
            return 'hold', 0.5, details


class LSTMScalpingSignal:
    """LSTM + 기술적 지표 복합 신호 생성기"""

    def __init__(self, lstm_predictor, config: LSTMScalpingConfig):
        self.lstm = lstm_predictor
        self.config = config
        self.crash_detector = CrashDetector(config)
        self.reversal_detector = TrendReversalDetector()
        self.vwap_strategy = VWAPStrategy()
        self.volume_detector = VolumeAnomalyDetector()
        self.technical_generator = TechnicalSignalGenerator()

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """
        복합 신호 생성

        우선순위:
        1. 급락 감지 (최우선) → 즉시 매도
        2. LSTM 예측 신호
        3. 기술적 지표 신호
        4. 추세 반전 감지
        5. VWAP 위치

        Returns:
            (signal, confidence, details)
        """
        details = {
            'timestamp': datetime.now().isoformat(),
            'current_price': df['close'].iloc[-1] if len(df) > 0 else 0
        }

        # 1. 급락 감지 (최우선)
        if self.config.crash_detection_enabled:
            is_crash, crash_reason, severity = self.crash_detector.detect(df)
            details['crash_detection'] = {
                'is_crash': is_crash,
                'reason': crash_reason,
                'severity': severity
            }

            if is_crash:
                return 'emergency_sell', 1.0, details

        # 2. LSTM 예측 신호
        lstm_signal = 'hold'
        lstm_confidence = 0.5

        if self.lstm is not None:
            try:
                result = self.lstm.predict(df)

                # Dict 반환 (UnifiedLSTMPredictor) 또는 튜플 반환 처리
                if isinstance(result, dict):
                    if result.get('success'):
                        predicted_price = result.get('predicted_price', df['close'].iloc[-1])
                        change_rate = result.get('predicted_change_rate', 0)
                        direction = (result.get('direction_en') or 'NEUTRAL').lower()
                    else:
                        # 예측 실패 시 기본값
                        predicted_price = df['close'].iloc[-1]
                        change_rate = 0
                        direction = 'neutral'
                elif isinstance(result, tuple) and len(result) == 3:
                    # 튜플 반환 (기존 LSTM)
                    predicted_price, change_rate, direction = result
                else:
                    # 알 수 없는 형식
                    predicted_price = df['close'].iloc[-1]
                    change_rate = 0
                    direction = 'neutral'

                details['lstm'] = {
                    'predicted_price': predicted_price,
                    'change_rate': change_rate,
                    'direction': direction
                }

                if change_rate >= self.config.strong_buy_threshold:
                    lstm_signal = 'strong_buy'
                    lstm_confidence = 0.9
                elif change_rate >= self.config.buy_threshold:
                    lstm_signal = 'buy'
                    lstm_confidence = 0.7
                elif change_rate <= self.config.strong_sell_threshold:
                    lstm_signal = 'strong_sell'
                    lstm_confidence = 0.9
                elif change_rate <= self.config.sell_threshold:
                    lstm_signal = 'sell'
                    lstm_confidence = 0.7
            except Exception as e:
                logger.warning(f"LSTM 예측 실패: {e}")
                details['lstm'] = {'error': str(e)}

        # 3. 기술적 지표 신호
        tech_signal, tech_confidence, tech_details = self.technical_generator.generate_signal(df)
        details['technical'] = tech_details

        # 4. 추세 반전 감지
        reversal_type, reversal_confidence = self.reversal_detector.detect(df)
        details['reversal'] = {
            'type': reversal_type,
            'confidence': reversal_confidence
        }

        # 5. VWAP 신호
        vwap_signal, vwap_confidence = self.vwap_strategy.generate_signal(df)
        details['vwap'] = {
            'signal': vwap_signal,
            'confidence': vwap_confidence
        }

        # 6. 거래량 이상 감지
        is_volume_anomaly, anomaly_type, magnitude = self.volume_detector.detect(df)
        details['volume_anomaly'] = {
            'is_anomaly': is_volume_anomaly,
            'type': anomaly_type,
            'magnitude': magnitude
        }

        # 복합 신호 계산
        final_signal, final_confidence = self._combine_signals(
            lstm_signal, lstm_confidence,
            tech_signal, tech_confidence,
            reversal_type, reversal_confidence,
            vwap_signal, vwap_confidence,
            is_volume_anomaly, anomaly_type
        )

        details['final'] = {
            'signal': final_signal,
            'confidence': final_confidence
        }

        return final_signal, final_confidence, details

    def _combine_signals(
        self,
        lstm_signal: str, lstm_conf: float,
        tech_signal: str, tech_conf: float,
        reversal_type: str, reversal_conf: float,
        vwap_signal: str, vwap_conf: float,
        volume_anomaly: bool, anomaly_type: str
    ) -> Tuple[str, float]:
        """신호 결합"""

        # 가중치 기반 점수 계산
        buy_score = 0
        sell_score = 0

        # LSTM 신호 (가중치 0.6)
        lstm_weight = self.config.lstm_weight
        if lstm_signal in ['strong_buy', 'buy']:
            buy_score += lstm_conf * lstm_weight * (1.2 if lstm_signal == 'strong_buy' else 1.0)
        elif lstm_signal in ['strong_sell', 'sell']:
            sell_score += lstm_conf * lstm_weight * (1.2 if lstm_signal == 'strong_sell' else 1.0)

        # 기술적 지표 신호 (가중치 0.4)
        tech_weight = self.config.technical_weight
        if tech_signal == 'buy':
            buy_score += tech_conf * tech_weight
        elif tech_signal == 'sell':
            sell_score += tech_conf * tech_weight

        # 추세 반전 보조
        if reversal_type == 'bullish':
            buy_score += reversal_conf * 0.15
        elif reversal_type == 'bearish':
            sell_score += reversal_conf * 0.15

        # VWAP 보조
        if vwap_signal == 'buy':
            buy_score += vwap_conf * 0.1
        elif vwap_signal == 'sell':
            sell_score += vwap_conf * 0.1

        # 거래량 이상 시 신호 강화
        if volume_anomaly:
            if anomaly_type == 'accumulation':
                buy_score *= 1.2
            elif anomaly_type == 'distribution':
                sell_score *= 1.2

        # 최종 결정
        threshold = 0.4

        if buy_score > sell_score and buy_score >= threshold:
            confidence = min(buy_score, 1.0)
            return 'buy', confidence
        elif sell_score > buy_score and sell_score >= threshold:
            confidence = min(sell_score, 1.0)
            return 'sell', confidence
        else:
            return 'hold', max(0.5, 1 - abs(buy_score - sell_score))


class LSTMScalpingBot:
    """LSTM 기반 스캘핑 봇"""

    def __init__(
        self,
        client: UpbitClient,
        lstm_predictors: Dict,
        config: LSTMScalpingConfig = None
    ):
        self.client = client
        self.lstm_predictors = lstm_predictors
        self.config = config or LSTMScalpingConfig()

        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.last_trade_time: Dict[str, datetime] = {}
        self.last_signals: Dict[str, str] = {}  # 마켓별 마지막 LSTM 신호

        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0,
            'total_fees': 0,
            'win_rate': 0
        }

        self.is_running = False
        self._stop_requested = False

        # 자동 탐색 모드 설정
        self.auto_discover = False
        self.discovered_markets: List[str] = []  # 자동 발견된 유망 코인
        self.last_scan_time: Optional[datetime] = None
        self.scan_interval = 300  # 5분마다 전체 스캔
        self.top_coin_count = 10  # 상위 N개 코인만 모니터링

        # 콜백 함수들
        self.on_trade_callback = None
        self.on_status_update_callback = None
        self.on_signal_callback = None  # 신호 발생 시 호출

    def get_signal_generator(self, market: str) -> LSTMScalpingSignal:
        """마켓별 신호 생성기 반환"""
        predictor = self.lstm_predictors.get(market)
        return LSTMScalpingSignal(predictor, self.config)

    async def scan_all_markets(self) -> List[Dict]:
        """
        전체 KRW 마켓 스캔하여 유망한 코인 탐색

        Returns:
            유망 코인 리스트 (점수 순 정렬)
        """
        logger.info("전체 마켓 스캔 시작...")

        # 1. 모든 KRW 마켓 조회
        all_markets = self.client.get_markets()
        krw_markets = [m['market'] for m in all_markets if m['market'].startswith('KRW-')]

        # 2. 거래량 상위 코인 필터링 (API 호출 최소화)
        tickers = self.client.get_ticker(krw_markets)
        if not tickers:
            return []

        # 거래대금 기준 상위 50개만 분석 (효율성)
        sorted_tickers = sorted(tickers, key=lambda x: x.get('acc_trade_price_24h', 0), reverse=True)
        top_markets = [t['market'] for t in sorted_tickers[:50]]

        # 3. 각 코인 분석
        coin_scores = []

        for market in top_markets:
            try:
                # 분봉 데이터 조회
                candles = self.client.get_candles_minute(market, unit=1, count=100)
                if not candles or len(candles) < 50:
                    continue

                df = pd.DataFrame(candles)
                df = df.rename(columns={
                    'opening_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'trade_price': 'close',
                    'candle_acc_trade_volume': 'volume'
                })
                df = df.sort_values('candle_date_time_kst').reset_index(drop=True)

                # 점수 계산
                score = await self._calculate_opportunity_score(market, df)

                if score > 0.5:  # 임계값 이상만 포함
                    ticker_data = next((t for t in tickers if t['market'] == market), {})
                    coin_scores.append({
                        'market': market,
                        'score': score,
                        'current_price': df['close'].iloc[-1],
                        'change_rate': ticker_data.get('signed_change_rate', 0) * 100,
                        'volume_24h': ticker_data.get('acc_trade_price_24h', 0)
                    })

                await asyncio.sleep(0.1)  # API 제한 방지

            except Exception as e:
                logger.debug(f"[{market}] 분석 실패: {e}")
                continue

        # 점수 순 정렬
        coin_scores.sort(key=lambda x: x['score'], reverse=True)

        logger.info(f"스캔 완료: {len(coin_scores)}개 유망 코인 발견")
        return coin_scores

    async def _calculate_opportunity_score(self, market: str, df: pd.DataFrame) -> float:
        """
        코인의 단타 기회 점수 계산 (0~1)

        높은 점수 기준:
        - 적절한 변동성 (너무 낮거나 높지 않은)
        - 상승 모멘텀
        - 거래량 증가
        - LSTM 매수 신호
        """
        score = 0.0

        try:
            # 1. 변동성 점수 (최근 20봉 기준, 적정 변동성이 좋음)
            returns = df['close'].pct_change().dropna()
            volatility = returns.tail(20).std() * 100

            # 변동성 0.5~2% 가 이상적
            if 0.3 <= volatility <= 3.0:
                vol_score = 1.0 - abs(volatility - 1.0) / 2.0
                score += max(0, vol_score) * 0.15

            # 2. 모멘텀 점수 (최근 상승 추세)
            ma5 = df['close'].rolling(5).mean().iloc[-1]
            ma20 = df['close'].rolling(20).mean().iloc[-1]
            current = df['close'].iloc[-1]

            if current > ma5 > ma20:
                score += 0.2  # 상승 추세
            elif current > ma5:
                score += 0.1  # 단기 상승

            # 3. 거래량 증가 점수
            vol_ma = df['volume'].rolling(20).mean().iloc[-1]
            recent_vol = df['volume'].tail(5).mean()

            if recent_vol > vol_ma * 1.5:
                score += 0.2  # 거래량 급증
            elif recent_vol > vol_ma:
                score += 0.1  # 거래량 증가

            # 4. RSI 점수 (30~50 매수 구간이 좋음)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, 0.0001)
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

            if 30 <= rsi <= 50:
                score += 0.2  # 매수 적정 구간
            elif 25 <= rsi < 30:
                score += 0.15  # 과매도 근접
            elif 50 < rsi <= 60:
                score += 0.1  # 중립

            # 5. LSTM 신호 (모델이 있는 경우)
            predictor = self.lstm_predictors.get(market)
            if predictor:
                try:
                    result = predictor.predict(df)
                    if isinstance(result, dict) and result.get('success'):
                        change_rate = result.get('predicted_change_rate', 0)
                        if change_rate > 1.0:
                            score += 0.25  # 강한 상승 예측
                        elif change_rate > 0.3:
                            score += 0.15  # 상승 예측
                    elif isinstance(result, tuple) and len(result) == 3:
                        _, change_rate, direction = result
                        if change_rate > 1.0:
                            score += 0.25
                        elif change_rate > 0.3:
                            score += 0.15
                except:
                    pass
            else:
                # LSTM 없어도 기술적 지표만으로 분석
                score += 0.1

        except Exception as e:
            logger.debug(f"점수 계산 오류: {e}")
            return 0.0

        return min(score, 1.0)

    def calculate_dynamic_stops(self, df: pd.DataFrame) -> Tuple[float, float]:
        """ATR 기반 동적 손절/익절 계산"""
        if not self.config.use_dynamic_stops or len(df) < 15:
            return self.config.base_profit_percent, self.config.base_loss_percent

        # ATR 계산
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]

        current_price = df['close'].iloc[-1]
        atr_percent = (atr / current_price) * 100

        # 변동성에 따른 조절 (0.5 ~ 2.0 범위)
        multiplier = min(max(atr_percent / 1.0, 0.5), 2.0)

        dynamic_profit = self.config.base_profit_percent * multiplier
        dynamic_loss = self.config.base_loss_percent * multiplier

        # 범위 제한
        dynamic_profit = max(self.config.min_stop_percent, min(dynamic_profit, self.config.max_stop_percent))
        dynamic_loss = max(self.config.min_stop_percent, min(dynamic_loss, self.config.max_stop_percent))

        return dynamic_profit, dynamic_loss

    def check_position_profit(self, market: str, current_price: float, df: pd.DataFrame) -> Tuple[bool, str, float]:
        """포지션 수익률 체크 및 청산 조건 확인"""
        if market not in self.positions:
            return False, '', 0

        position = self.positions[market]
        profit_rate = ((current_price - position.entry_price) / position.entry_price) * 100

        # 동적 손절/익절 계산
        take_profit, stop_loss = self.calculate_dynamic_stops(df)

        # 최고가/최저가 업데이트
        position.highest_price = max(position.highest_price, current_price)
        position.lowest_price = min(position.lowest_price, current_price)

        # 익절 조건
        if profit_rate >= take_profit:
            return True, f'익절 ({profit_rate:.2f}% >= {take_profit:.2f}%)', profit_rate

        # 손절 조건
        if profit_rate <= -stop_loss:
            return True, f'손절 ({profit_rate:.2f}% <= -{stop_loss:.2f}%)', profit_rate

        # 추적 손절 (Trailing Stop) - 최고점 대비 하락
        if position.highest_price > position.entry_price:
            from_high = ((current_price - position.highest_price) / position.highest_price) * 100
            trailing_stop = -stop_loss * 0.7
            if from_high <= trailing_stop:
                return True, f'추적손절 (최고점 대비 {from_high:.2f}%)', profit_rate

        # 최대 보유 시간 초과
        holding_time = (datetime.now() - position.entry_time).total_seconds() / 60
        if holding_time >= self.config.max_holding_minutes:
            return True, f'보유시간 초과 ({holding_time:.0f}분)', profit_rate

        return False, '', profit_rate

    def is_cooldown(self, market: str) -> bool:
        """쿨다운 체크"""
        if market not in self.last_trade_time:
            return False

        elapsed = (datetime.now() - self.last_trade_time[market]).total_seconds()
        return elapsed < self.config.cooldown_seconds

    async def execute_buy(self, market: str, signal_details: Dict) -> Optional[Dict]:
        """매수 실행"""
        # 쿨다운 체크
        if self.is_cooldown(market):
            return None

        # 기존 포지션 체크
        if market in self.positions:
            return None

        # 최대 포지션 수 체크
        if len(self.positions) >= self.config.max_positions:
            return None

        # 잔고 확인
        balance = self.client.get_balance('KRW')
        trade_amount = min(self.config.trade_amount, balance * 0.95)

        if trade_amount < self.config.min_trade_amount:
            return None

        # 시장가 매수
        result = self.client.buy_market_order(market, trade_amount)

        if isinstance(result, dict) and 'uuid' in result and 'error' not in result:
            # 현재가 조회
            ticker = self.client.get_ticker([market])
            current_price = ticker[0]['trade_price'] if ticker else 0

            # 포지션 기록
            self.positions[market] = Position(
                market=market,
                entry_price=current_price,
                entry_time=datetime.now(),
                amount=trade_amount,
                volume=trade_amount / current_price if current_price > 0 else 0,
                uuid=result.get('uuid'),
                highest_price=current_price,
                lowest_price=current_price
            )

            self.last_trade_time[market] = datetime.now()
            self.stats['total_trades'] += 1

            trade_record = {
                'type': 'buy',
                'market': market,
                'price': current_price,
                'amount': trade_amount,
                'timestamp': datetime.now().isoformat(),
                'signal_details': signal_details,
                'uuid': result.get('uuid')
            }
            self.trade_history.append(trade_record)

            if self.on_trade_callback:
                await self._safe_callback(self.on_trade_callback, trade_record)

            return trade_record

        return None

    async def execute_sell(self, market: str, reason: str, profit_rate: float = 0) -> Optional[Dict]:
        """매도 실행"""
        if market not in self.positions:
            return None

        position = self.positions[market]

        # 보유량 확인
        crypto_symbol = market.split('-')[1]
        crypto_balance = self.client.get_balance(crypto_symbol)

        if crypto_balance <= 0:
            del self.positions[market]
            return None

        # 시장가 매도
        result = self.client.sell_market_order(market, crypto_balance)

        if isinstance(result, dict) and 'uuid' in result and 'error' not in result:
            # 현재가
            ticker = self.client.get_ticker([market])
            current_price = ticker[0]['trade_price'] if ticker else position.entry_price

            # 수익 계산
            sell_amount = crypto_balance * current_price
            profit = sell_amount - position.amount
            fee = sell_amount * self.config.commission
            net_profit = profit - fee

            # 통계 업데이트
            self.stats['total_trades'] += 1
            self.stats['total_profit'] += net_profit
            self.stats['total_fees'] += fee

            if net_profit > 0:
                self.stats['winning_trades'] += 1
            else:
                self.stats['losing_trades'] += 1

            total = self.stats['winning_trades'] + self.stats['losing_trades']
            if total > 0:
                self.stats['win_rate'] = round(self.stats['winning_trades'] / total * 100, 1)

            self.last_trade_time[market] = datetime.now()

            trade_record = {
                'type': 'sell',
                'market': market,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'profit': net_profit,
                'profit_rate': profit_rate,
                'reason': reason,
                'holding_time': (datetime.now() - position.entry_time).total_seconds() / 60,
                'timestamp': datetime.now().isoformat(),
                'uuid': result.get('uuid')
            }
            self.trade_history.append(trade_record)

            # 포지션 제거
            del self.positions[market]

            if self.on_trade_callback:
                await self._safe_callback(self.on_trade_callback, trade_record)

            return trade_record

        return None

    async def run_single_check(self, market: str) -> Optional[Dict]:
        """단일 마켓 체크 및 거래"""
        try:
            # 분봉 데이터 조회
            candles = self.client.get_candles_minute(market, unit=1, count=100)
            if not candles:
                return None

            df = pd.DataFrame(candles)
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            })
            df = df.sort_values('candle_date_time_kst').reset_index(drop=True)

            current_price = df['close'].iloc[-1]

            # 신호 생성
            signal_generator = self.get_signal_generator(market)
            signal, confidence, details = signal_generator.generate_signal(df)

            # 마지막 신호 저장 (UI 표시용)
            self.last_signals[market] = signal.upper()

            # 신호 콜백 호출 (실시간 로그용)
            if self.on_signal_callback:
                crash_reason = details.get('crash_detection', {}).get('reason', '')
                await self._safe_callback(self.on_signal_callback, {
                    "market": market,
                    "signal": signal.upper(),
                    "confidence": confidence,
                    "reason": crash_reason if signal == 'emergency_sell' else '',
                    "current_price": current_price
                })

            # 1. 긴급 매도 (급락 감지)
            if signal == 'emergency_sell' and market in self.positions:
                return await self.execute_sell(market, f"급락 감지: {details.get('crash_detection', {}).get('reason', '')}")

            # 2. 기존 포지션 수익률 체크
            if market in self.positions:
                should_sell, reason, profit_rate = self.check_position_profit(market, current_price, df)
                if should_sell:
                    return await self.execute_sell(market, reason, profit_rate)

                # 매도 신호
                if signal == 'sell' and confidence >= 0.6:
                    return await self.execute_sell(market, f"매도 신호 (신뢰도: {confidence:.1%})", profit_rate)

            # 3. 매수 신호
            elif signal == 'buy' and confidence >= 0.6:
                return await self.execute_buy(market, details)

            return None

        except Exception as e:
            logger.error(f"[{market}] 체크 오류: {e}")
            return None

    async def emergency_sell_all(self) -> List[Dict]:
        """긴급 전체 매도"""
        results = []
        markets = list(self.positions.keys())

        for market in markets:
            result = await self.execute_sell(market, "긴급 전체 매도")
            if result:
                results.append(result)

        return results

    async def run_loop(self, markets: List[str], interval: int = 30):
        """메인 트레이딩 루프"""
        self.is_running = True
        self._stop_requested = False

        # 자동 탐색 모드면 초기 스캔 수행
        if self.auto_discover:
            logger.info(f"LSTM 단타 봇 시작 (자동 탐색 모드), 간격: {interval}초")
            await self._do_market_scan()
            active_markets = self.discovered_markets
        else:
            logger.info(f"LSTM 단타 봇 시작: {markets}, 간격: {interval}초")
            active_markets = markets

        while not self._stop_requested:
            try:
                # 자동 탐색 모드: 주기적으로 마켓 재스캔
                if self.auto_discover:
                    if self.last_scan_time is None or \
                       (datetime.now() - self.last_scan_time).total_seconds() > self.scan_interval:
                        await self._do_market_scan()
                        active_markets = self.discovered_markets

                # 마켓별 체크
                for market in active_markets:
                    if self._stop_requested:
                        break

                    await self.run_single_check(market)
                    await asyncio.sleep(0.5)  # API 제한 방지

                # 보유 중인 포지션도 체크 (자동 탐색 모드에서 발견되지 않은 코인)
                for market in list(self.positions.keys()):
                    if market not in active_markets:
                        await self.run_single_check(market)
                        await asyncio.sleep(0.5)

                # 상태 업데이트 콜백
                if self.on_status_update_callback:
                    status = self.get_status()
                    status['discovered_markets'] = self.discovered_markets if self.auto_discover else []
                    await self._safe_callback(self.on_status_update_callback, status)

                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"루프 오류: {e}")
                await asyncio.sleep(5)

        self.is_running = False
        logger.info("LSTM 단타 봇 종료")

    async def _do_market_scan(self):
        """마켓 스캔 수행 및 결과 저장"""
        try:
            scores = await self.scan_all_markets()
            self.discovered_markets = [s['market'] for s in scores[:self.top_coin_count]]
            self.last_scan_time = datetime.now()

            if self.on_signal_callback:
                await self._safe_callback(self.on_signal_callback, {
                    "market": "SYSTEM",
                    "signal": "SCAN_COMPLETE",
                    "confidence": 1.0,
                    "reason": f"유망 코인 {len(self.discovered_markets)}개 발견: {', '.join([m.replace('KRW-', '') for m in self.discovered_markets[:5]])}..."
                })

            logger.info(f"자동 탐색 완료: {self.discovered_markets}")
        except Exception as e:
            logger.error(f"마켓 스캔 오류: {e}")

    def stop(self):
        """봇 중지 요청"""
        self._stop_requested = True

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        positions_data = {}
        for market, pos in self.positions.items():
            ticker = self.client.get_ticker([market])
            current_price = ticker[0]['trade_price'] if ticker else pos.entry_price
            profit_rate = ((current_price - pos.entry_price) / pos.entry_price) * 100

            positions_data[market] = {
                'entry_price': pos.entry_price,
                'current_price': current_price,
                'profit_rate': round(profit_rate, 2),
                'amount': pos.amount,
                'holding_minutes': round((datetime.now() - pos.entry_time).total_seconds() / 60, 1),
                'lstm_signal': self.last_signals.get(market, 'N/A')
            }

        return {
            'is_running': self.is_running,
            'positions': positions_data,
            'stats': self.stats,
            'config': {
                'trade_amount': self.config.trade_amount,
                'max_positions': self.config.max_positions,
                'lstm_weight': self.config.lstm_weight,
                'crash_detection': self.config.crash_detection_enabled
            }
        }

    async def _safe_callback(self, callback, data):
        """안전한 콜백 호출"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
        except Exception as e:
            logger.error(f"콜백 오류: {e}")
