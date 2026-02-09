"""
강화학습을 위한 가상화폐 트레이딩 환경
Gymnasium 기반 커스텀 환경
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Tuple, Optional, Dict
import ta  # Technical Analysis library


class CryptoTradingEnv(gym.Env):
    """가상화폐 트레이딩 강화학습 환경"""

    metadata = {'render_modes': ['human']}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 1000000,
                 commission: float = 0.0005,
                 render_mode: Optional[str] = None):
        """
        Args:
            df: OHLCV 데이터프레임 (columns: open, high, low, close, volume)
            initial_balance: 초기 자금 (KRW)
            commission: 거래 수수료 (0.05%)
            render_mode: 렌더 모드
        """
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.commission = commission
        self.render_mode = render_mode

        # 기술적 지표 추가
        self._add_technical_indicators()

        # 액션 스페이스: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # 관측 스페이스: 가격 데이터 + 기술적 지표 + 보유 현황
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._get_observation_size(),),
            dtype=np.float32
        )

        # 상태 변수
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = 0.0
        self.total_trades = 0
        self.total_profit = 0.0

    def _add_technical_indicators(self):
        """기술적 지표 추가"""
        # 이동평균선
        self.df['sma_5'] = ta.trend.sma_indicator(self.df['close'], window=5)
        self.df['sma_20'] = ta.trend.sma_indicator(self.df['close'], window=20)
        self.df['sma_60'] = ta.trend.sma_indicator(self.df['close'], window=60)

        # 지수이동평균
        self.df['ema_12'] = ta.trend.ema_indicator(self.df['close'], window=12)
        self.df['ema_26'] = ta.trend.ema_indicator(self.df['close'], window=26)

        # MACD
        macd = ta.trend.MACD(self.df['close'])
        self.df['macd'] = macd.macd()
        self.df['macd_signal'] = macd.macd_signal()
        self.df['macd_diff'] = macd.macd_diff()

        # RSI
        self.df['rsi'] = ta.momentum.rsi(self.df['close'], window=14)

        # 볼린저 밴드
        bollinger = ta.volatility.BollingerBands(self.df['close'])
        self.df['bb_high'] = bollinger.bollinger_hband()
        self.df['bb_mid'] = bollinger.bollinger_mavg()
        self.df['bb_low'] = bollinger.bollinger_lband()

        # 스토캐스틱
        stoch = ta.momentum.StochasticOscillator(
            self.df['high'], self.df['low'], self.df['close']
        )
        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()

        # ATR (변동성)
        self.df['atr'] = ta.volatility.average_true_range(
            self.df['high'], self.df['low'], self.df['close']
        )

        # 거래량 지표
        self.df['volume_sma'] = ta.trend.sma_indicator(self.df['volume'], window=20)

        # NaN 처리: 전방/후방 채움 사용 (0으로 대체하면 지표 왜곡)
        self.df = self.df.ffill().bfill()
        # 그래도 남은 NaN은 0으로 처리 (첫 행 등)
        self.df = self.df.fillna(0)

    def _get_observation_size(self) -> int:
        """관측 벡터 크기 계산"""
        # OHLCV(5) + 이동평균(5) + MACD(3) + RSI(1) + 볼린저(3) + 스토캐스틱(2) + ATR(1) + 계좌상태(3) + 수익률(1) = 24
        return 24

    def _get_observation(self) -> np.ndarray:
        """현재 관측값 반환"""
        row = self.df.iloc[self.current_step]

        # 가격 정규화 (현재가 기준)
        current_price = row['close']

        obs = np.array([
            # OHLCV (정규화)
            row['open'] / current_price,
            row['high'] / current_price,
            row['low'] / current_price,
            1.0,  # close는 항상 1
            row['volume'] / (row['volume_sma'] + 1e-8),

            # 이동평균선
            row['sma_5'] / current_price,
            row['sma_20'] / current_price,
            row['sma_60'] / current_price,
            row['ema_12'] / current_price,
            row['ema_26'] / current_price,

            # MACD
            row['macd'] / current_price,
            row['macd_signal'] / current_price,
            row['macd_diff'] / current_price,

            # RSI (0~100 -> 0~1)
            row['rsi'] / 100.0,

            # 볼린저 밴드
            row['bb_high'] / current_price,
            row['bb_mid'] / current_price,
            row['bb_low'] / current_price,

            # 스토캐스틱
            row['stoch_k'] / 100.0,
            row['stoch_d'] / 100.0,

            # ATR
            row['atr'] / current_price,

            # 계좌 상태
            self.balance / self.initial_balance,
            self.crypto_held * current_price / self.initial_balance,
            (self.balance + self.crypto_held * current_price) / self.initial_balance,

            # 수익률
            self.total_profit / self.initial_balance,
        ], dtype=np.float32)

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """환경 초기화"""
        super().reset(seed=seed)

        self.current_step = 60  # 기술적 지표 계산을 위한 최소 스텝
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.total_trades = 0
        self.total_profit = 0.0

        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """한 스텝 실행

        Args:
            action: 0=Hold, 1=Buy, 2=Sell

        Returns:
            observation, reward, terminated, truncated, info
        """
        # 현재 스텝이 유효한지 확인
        if self.current_step >= len(self.df):
            raise IndexError(f"current_step ({self.current_step}) exceeds dataframe length ({len(self.df)})")

        current_price = self.df.iloc[self.current_step]['close']
        reward = 0.0

        # 이전 총 자산
        prev_net_worth = self.balance + self.crypto_held * current_price

        # 액션 실행
        if action == 1:  # Buy
            if self.balance > 0:
                # 수수료 포함하여 최대 매수 가능 금액 계산 (잔액의 95% 범위 내)
                max_buy_amount = (self.balance * 0.95) / (1 + self.commission)
                cost = max_buy_amount * (1 + self.commission)

                if cost <= self.balance:
                    self.crypto_held += max_buy_amount / current_price
                    self.balance -= cost
                    self.total_trades += 1

        elif action == 2:  # Sell
            if self.crypto_held > 0:
                # 전량 매도
                sell_amount = self.crypto_held * current_price
                proceeds = sell_amount * (1 - self.commission)

                self.balance += proceeds
                self.crypto_held = 0.0
                self.total_trades += 1

        # 다음 스텝으로 이동하기 전에 범위 체크
        self.current_step += 1

        # 에피소드 종료 조건 먼저 체크
        terminated = False
        truncated = self.current_step >= len(self.df) - 1

        # 마지막 스텝이거나 초과한 경우 현재 가격으로 평가
        if truncated or self.current_step >= len(self.df):
            next_price = current_price  # 현재 가격 사용
        else:
            # 다음 스텝의 가격으로 자산 계산 (정확한 보상 계산을 위해)
            next_price = self.df.iloc[self.current_step]['close']

        current_net_worth = self.balance + self.crypto_held * next_price

        # 보상 계산: 자산 변화율 (0으로 나누기 방지)
        if prev_net_worth > 0:
            reward = (current_net_worth - prev_net_worth) / prev_net_worth
        else:
            reward = 0.0

        # 총 수익 업데이트
        self.total_profit = current_net_worth - self.initial_balance

        # 파산 체크
        if current_net_worth < self.initial_balance * 0.1:
            terminated = True
            reward = -10.0  # 큰 패널티

        info = {
            'net_worth': current_net_worth,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'total_profit': self.total_profit,
            'total_trades': self.total_trades,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
        """렌더링 (현재 상태 출력)"""
        if self.render_mode == 'human':
            current_price = self.df.iloc[self.current_step]['close']
            net_worth = self.balance + self.crypto_held * current_price

            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:,.0f} KRW")
            print(f"Balance: {self.balance:,.0f} KRW")
            print(f"Crypto: {self.crypto_held:.8f}")
            print(f"Net Worth: {net_worth:,.0f} KRW")
            print(f"Profit: {self.total_profit:,.0f} KRW ({self.total_profit/self.initial_balance*100:.2f}%)")
            print("-" * 50)
