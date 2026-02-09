"""
스캘핑(단타) 전용 트레이딩 봇
빠른 매수/매도로 작은 수익을 자주 실현
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time


@dataclass
class ScalpingConfig:
    """스캘핑 설정"""
    # 수익/손실 설정
    min_profit_percent: float = 0.15   # 최소 수익률 (%) - 수수료 0.1% 고려
    max_loss_percent: float = 0.5      # 최대 손실률 (%)
    target_profit_percent: float = 0.3  # 목표 수익률 (%)

    # 거래 설정
    trade_amount: float = 50000        # 1회 거래 금액 (KRW)
    max_positions: int = 3             # 동시 보유 가능 포지션 수
    cooldown_seconds: int = 30         # 거래 후 쿨다운 (초)

    # 시그널 설정
    rsi_oversold: float = 25           # RSI 과매도 기준
    rsi_overbought: float = 75         # RSI 과매수 기준
    volume_spike_ratio: float = 2.0    # 거래량 급증 배율
    price_momentum_threshold: float = 0.1  # 가격 모멘텀 임계값 (%)


class ScalpingSignal:
    """스캘핑 시그널 생성기"""

    def __init__(self, config: ScalpingConfig = None):
        self.config = config or ScalpingConfig()

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """초단기 기술적 지표 계산

        Args:
            df: 분봉 OHLCV 데이터

        Returns:
            지표 딕셔너리
        """
        if len(df) < 20:
            return {}

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # RSI (짧은 기간)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=7).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # 스토캐스틱 RSI
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

        # 볼린저 밴드 (짧은 기간)
        sma = close.rolling(window=10).mean()
        std = close.rolling(window=10).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_position = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # 거래량 분석
        volume_sma = volume.rolling(window=10).mean()
        volume_ratio = volume / (volume_sma + 1e-10)

        # 가격 모멘텀
        momentum_1 = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
        momentum_3 = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100 if len(close) > 3 else 0
        momentum_5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100 if len(close) > 5 else 0

        # VWAP (Volume Weighted Average Price)
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        # 현재 가격과 VWAP 비교
        current_price = close.iloc[-1]
        vwap_position = (current_price - vwap.iloc[-1]) / vwap.iloc[-1] * 100

        return {
            'current_price': current_price,
            'rsi': rsi.iloc[-1],
            'stoch_rsi': stoch_rsi.iloc[-1],
            'bb_position': bb_position.iloc[-1],
            'bb_upper': bb_upper.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'volume_ratio': volume_ratio.iloc[-1],
            'momentum_1': momentum_1,
            'momentum_3': momentum_3,
            'momentum_5': momentum_5,
            'vwap': vwap.iloc[-1],
            'vwap_position': vwap_position,
            'high_low_range': (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] * 100
        }

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """스캘핑 시그널 생성

        Args:
            df: 분봉 데이터

        Returns:
            (signal, confidence, details)
            signal: 'BUY', 'SELL', 'HOLD'
        """
        indicators = self.calculate_indicators(df)

        if not indicators:
            return 'HOLD', 0.0, {}

        rsi = indicators['rsi']
        stoch_rsi = indicators['stoch_rsi']
        bb_position = indicators['bb_position']
        volume_ratio = indicators['volume_ratio']
        momentum_1 = indicators['momentum_1']
        momentum_3 = indicators['momentum_3']
        vwap_position = indicators['vwap_position']

        buy_score = 0
        sell_score = 0
        reasons = []

        # 1. RSI 과매도/과매수
        if rsi < self.config.rsi_oversold:
            buy_score += 3
            reasons.append(f"RSI 과매도 ({rsi:.1f})")
        elif rsi > self.config.rsi_overbought:
            sell_score += 3
            reasons.append(f"RSI 과매수 ({rsi:.1f})")
        elif rsi < 40:
            buy_score += 1
        elif rsi > 60:
            sell_score += 1

        # 2. 스토캐스틱 RSI
        if stoch_rsi < 20:
            buy_score += 2
            reasons.append("스토캐스틱 RSI 과매도")
        elif stoch_rsi > 80:
            sell_score += 2
            reasons.append("스토캐스틱 RSI 과매수")

        # 3. 볼린저 밴드 위치
        if bb_position < 0.1:
            buy_score += 2
            reasons.append("볼린저 하단 접근")
        elif bb_position > 0.9:
            sell_score += 2
            reasons.append("볼린저 상단 접근")

        # 4. 거래량 급증
        if volume_ratio > self.config.volume_spike_ratio:
            if momentum_1 > 0:
                buy_score += 2
                reasons.append(f"거래량 급증 (상승, {volume_ratio:.1f}x)")
            else:
                sell_score += 2
                reasons.append(f"거래량 급증 (하락, {volume_ratio:.1f}x)")

        # 5. 가격 모멘텀
        if momentum_1 > self.config.price_momentum_threshold:
            buy_score += 1
            if momentum_3 > 0:
                buy_score += 1
                reasons.append("상승 모멘텀")
        elif momentum_1 < -self.config.price_momentum_threshold:
            sell_score += 1
            if momentum_3 < 0:
                sell_score += 1
                reasons.append("하락 모멘텀")

        # 6. VWAP 위치
        if vwap_position < -0.5:
            buy_score += 1
            reasons.append("VWAP 아래 (저평가)")
        elif vwap_position > 0.5:
            sell_score += 1
            reasons.append("VWAP 위 (고평가)")

        # 시그널 결정
        total_score = buy_score + sell_score
        if total_score == 0:
            return 'HOLD', 0.0, indicators

        if buy_score > sell_score and buy_score >= 4:
            confidence = min(buy_score / 10, 0.9)
            return 'BUY', confidence, {**indicators, 'reasons': reasons}
        elif sell_score > buy_score and sell_score >= 4:
            confidence = min(sell_score / 10, 0.9)
            return 'SELL', confidence, {**indicators, 'reasons': reasons}
        else:
            return 'HOLD', 0.3, indicators


class ScalpingBot:
    """스캘핑 전용 트레이딩 봇"""

    def __init__(self, client, config: ScalpingConfig = None):
        """
        Args:
            client: UpbitClient 인스턴스
            config: 스캘핑 설정
        """
        self.client = client
        self.config = config or ScalpingConfig()
        self.signal_generator = ScalpingSignal(self.config)

        # 포지션 관리
        self.positions: Dict[str, Dict] = {}  # market -> position info
        self.trade_history: List[Dict] = []
        self.last_trade_time: Dict[str, datetime] = {}

        # 통계
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_fees': 0.0
        }

    def get_minute_candles(self, market: str, count: int = 60, unit: int = 1) -> pd.DataFrame:
        """분봉 데이터 조회

        Args:
            market: 마켓 코드
            count: 캔들 개수
            unit: 분 단위 (1, 3, 5, 15, 30, 60)

        Returns:
            OHLCV 데이터프레임
        """
        candles = self.client.get_candles_minute(market, unit=unit, count=count)

        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles)
        df = df.rename(columns={
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        })
        df = df.sort_values('candle_date_time_kst').reset_index(drop=True)

        return df[['open', 'high', 'low', 'close', 'volume']]

    def check_cooldown(self, market: str) -> bool:
        """쿨다운 체크

        Args:
            market: 마켓 코드

        Returns:
            거래 가능 여부
        """
        if market not in self.last_trade_time:
            return True

        elapsed = (datetime.now() - self.last_trade_time[market]).total_seconds()
        return elapsed >= self.config.cooldown_seconds

    def check_position_profit(self, market: str, current_price: float) -> Tuple[bool, float]:
        """포지션 수익률 체크

        Args:
            market: 마켓 코드
            current_price: 현재 가격

        Returns:
            (목표 도달 여부, 수익률)
        """
        if market not in self.positions:
            return False, 0.0

        position = self.positions[market]
        entry_price = position['entry_price']
        profit_rate = ((current_price - entry_price) / entry_price) * 100

        # 목표 수익 도달
        if profit_rate >= self.config.target_profit_percent:
            return True, profit_rate

        # 손절 도달
        if profit_rate <= -self.config.max_loss_percent:
            return True, profit_rate

        return False, profit_rate

    def execute_scalp_trade(self, market: str, signal: str, current_price: float, confidence: float) -> Optional[Dict]:
        """스캘핑 거래 실행

        Args:
            market: 마켓 코드
            signal: 시그널 ('BUY', 'SELL')
            current_price: 현재 가격
            confidence: 신뢰도

        Returns:
            거래 결과
        """
        # 쿨다운 체크
        if not self.check_cooldown(market):
            return None

        if signal == 'BUY':
            # 이미 포지션이 있으면 스킵
            if market in self.positions:
                return None

            # 최대 포지션 수 체크
            if len(self.positions) >= self.config.max_positions:
                return None

            # 잔고 체크
            krw_balance = self.client.get_balance('KRW')
            trade_amount = min(self.config.trade_amount, krw_balance * 0.95)

            if trade_amount < 5000:
                return None

            # 매수 실행
            try:
                result = self.client.buy_market_order(market, trade_amount)

                if 'error' not in result:
                    # 포지션 기록
                    self.positions[market] = {
                        'entry_price': current_price,
                        'entry_time': datetime.now(),
                        'amount': trade_amount,
                        'volume': trade_amount / current_price,
                        'uuid': result.get('uuid')
                    }
                    self.last_trade_time[market] = datetime.now()
                    self.stats['total_trades'] += 1

                    return {
                        'action': 'BUY',
                        'market': market,
                        'price': current_price,
                        'amount': trade_amount,
                        'confidence': confidence,
                        'uuid': result.get('uuid')
                    }
            except Exception as e:
                print(f"[SCALPING] 매수 오류: {e}")

        elif signal == 'SELL':
            # 포지션이 없으면 스킵
            if market not in self.positions:
                return None

            position = self.positions[market]
            coin_symbol = market.replace('KRW-', '')
            coin_balance = self.client.get_balance(coin_symbol)

            if coin_balance <= 0:
                del self.positions[market]
                return None

            # 매도 실행
            try:
                result = self.client.sell_market_order(market, coin_balance)

                if 'error' not in result:
                    # 수익 계산
                    entry_price = position['entry_price']
                    profit_rate = ((current_price - entry_price) / entry_price) * 100
                    profit_amount = position['amount'] * (profit_rate / 100)

                    # 통계 업데이트
                    if profit_rate > 0:
                        self.stats['winning_trades'] += 1
                    else:
                        self.stats['losing_trades'] += 1
                    self.stats['total_profit'] += profit_amount
                    self.stats['total_trades'] += 1

                    # 거래 기록
                    trade_record = {
                        'action': 'SELL',
                        'market': market,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_rate': profit_rate,
                        'profit_amount': profit_amount,
                        'hold_time': (datetime.now() - position['entry_time']).total_seconds(),
                        'uuid': result.get('uuid'),
                        'time': datetime.now().isoformat()
                    }
                    self.trade_history.append(trade_record)

                    # 포지션 정리
                    del self.positions[market]
                    self.last_trade_time[market] = datetime.now()

                    return trade_record
            except Exception as e:
                print(f"[SCALPING] 매도 오류: {e}")

        return None

    def check_and_close_positions(self, markets: List[str] = None):
        """포지션 체크 및 청산

        Args:
            markets: 체크할 마켓 리스트 (None이면 전체)
        """
        markets_to_check = markets or list(self.positions.keys())

        for market in markets_to_check:
            if market not in self.positions:
                continue

            # 현재가 조회
            ticker = self.client.get_ticker([market])
            if not ticker:
                continue

            current_price = ticker[0]['trade_price']

            # 수익률 체크
            should_close, profit_rate = self.check_position_profit(market, current_price)

            if should_close:
                reason = "목표 수익 도달" if profit_rate > 0 else "손절"
                print(f"[SCALPING] {market} 청산 ({reason}: {profit_rate:+.2f}%)")
                self.execute_scalp_trade(market, 'SELL', current_price, 1.0)

    def get_stats(self) -> Dict:
        """통계 조회"""
        win_rate = 0
        if self.stats['winning_trades'] + self.stats['losing_trades'] > 0:
            win_rate = self.stats['winning_trades'] / (self.stats['winning_trades'] + self.stats['losing_trades']) * 100

        return {
            **self.stats,
            'win_rate': win_rate,
            'active_positions': len(self.positions),
            'positions': list(self.positions.keys())
        }

    def run_single_check(self, market: str) -> Optional[Dict]:
        """단일 마켓 체크 및 거래

        Args:
            market: 마켓 코드

        Returns:
            거래 결과
        """
        # 분봉 데이터 조회
        df = self.get_minute_candles(market, count=60, unit=1)

        if len(df) < 20:
            return None

        current_price = df['close'].iloc[-1]

        # 기존 포지션 수익률 체크
        if market in self.positions:
            should_close, profit_rate = self.check_position_profit(market, current_price)
            if should_close:
                return self.execute_scalp_trade(market, 'SELL', current_price, 1.0)

        # 시그널 생성
        signal, confidence, details = self.signal_generator.generate_signal(df)

        if signal == 'HOLD':
            return None

        # 거래 실행
        return self.execute_scalp_trade(market, signal, current_price, confidence)
