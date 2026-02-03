"""
백테스팅 시스템
과거 데이터로 전략 성능 평가
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from datetime import datetime


class Backtester:
    """트레이딩 전략 백테스팅"""

    def __init__(self,
                 df: pd.DataFrame,
                 initial_balance: float = 1000000,
                 commission: float = 0.0005):
        """
        Args:
            df: OHLCV 데이터
            initial_balance: 초기 자금
            commission: 거래 수수료
        """
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.commission = commission

        # 백테스트 결과
        self.trades: List[Dict] = []
        self.balance_history: List[float] = []
        self.portfolio_value_history: List[float] = []

    def run_backtest(self, actions: np.ndarray) -> Dict:
        """백테스트 실행

        Args:
            actions: 각 스텝의 액션 배열 (0=Hold, 1=Buy, 2=Sell)

        Returns:
            백테스트 결과
        """
        # 빈 데이터 처리
        if len(actions) == 0 or len(self.df) == 0:
            return self._get_empty_results()

        balance = self.initial_balance
        crypto_held = 0.0
        trades = []

        self.balance_history = []
        self.portfolio_value_history = []

        for i, action in enumerate(actions):
            if i >= len(self.df):
                break

            price = self.df.iloc[i]['close']

            # Buy
            if action == 1 and balance > 0:
                # 수수료 포함하여 최대 매수 가능 금액 계산 (잔액의 95% 범위 내)
                max_buy_amount = (balance * 0.95) / (1 + self.commission)
                cost = max_buy_amount * (1 + self.commission)

                if cost <= balance:
                    quantity = max_buy_amount / price
                    crypto_held += quantity
                    balance -= cost

                    trades.append({
                        'type': 'BUY',
                        'index': i,
                        'price': price,
                        'quantity': quantity,
                        'balance': balance
                    })

            # Sell
            elif action == 2 and crypto_held > 0:
                sell_amount = crypto_held * price
                proceeds = sell_amount * (1 - self.commission)

                balance += proceeds

                trades.append({
                    'type': 'SELL',
                    'index': i,
                    'price': price,
                    'quantity': crypto_held,
                    'balance': balance
                })

                crypto_held = 0.0

            # 포트폴리오 가치 기록
            portfolio_value = balance + crypto_held * price
            self.balance_history.append(balance)
            self.portfolio_value_history.append(portfolio_value)

        # 최종 정산
        final_price = self.df.iloc[-1]['close']
        final_value = balance + crypto_held * final_price

        # 결과 계산
        total_profit = final_value - self.initial_balance
        total_return = (final_value / self.initial_balance - 1) * 100

        # 거래 통계
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']

        # 승률 계산 (수수료 포함)
        profitable_trades = 0
        if len(buy_trades) > 0 and len(sell_trades) > 0:
            for i in range(min(len(buy_trades), len(sell_trades))):
                # 수수료 포함하여 실제 수익 계산
                buy_cost = buy_trades[i]['price'] * (1 + self.commission)
                sell_revenue = sell_trades[i]['price'] * (1 - self.commission)
                if sell_revenue > buy_cost:
                    profitable_trades += 1

        win_rate = (profitable_trades / len(sell_trades) * 100) if sell_trades else 0

        # MDD (Maximum Drawdown) 계산
        mdd = self._calculate_mdd(self.portfolio_value_history)

        # Sharpe Ratio 계산
        sharpe = self._calculate_sharpe_ratio(self.portfolio_value_history)

        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_profit': total_profit,
            'total_return': total_return,
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_rate,
            'mdd': mdd,
            'sharpe_ratio': sharpe,
            'trades': trades
        }

        self.trades = trades
        return results

    def _get_empty_results(self) -> Dict:
        """빈 결과 반환 (데이터가 없을 때)"""
        return {
            'initial_balance': self.initial_balance,
            'final_value': self.initial_balance,
            'total_profit': 0,
            'total_return': 0,
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'win_rate': 0,
            'mdd': 0,
            'sharpe_ratio': 0,
            'trades': []
        }

    def _calculate_mdd(self, values: List[float]) -> float:
        """MDD (Maximum Drawdown) 계산"""
        if not values:
            return 0.0

        peak = values[0]
        max_dd = 0.0

        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def _calculate_sharpe_ratio(self, values: List[float], risk_free_rate: float = 0.02) -> float:
        """Sharpe Ratio 계산

        Args:
            values: 포트폴리오 가치 히스토리
            risk_free_rate: 무위험 수익률 (연 2%)
        """
        if len(values) < 2:
            return 0.0

        returns = np.diff(values) / values[:-1]
        excess_returns = returns - risk_free_rate / 252  # 일간 수익률

        if np.std(excess_returns) == 0:
            return 0.0

        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return sharpe

    def print_results(self, results: Dict):
        """백테스트 결과 출력"""
        print("\n" + "="*60)
        print("📊 백테스트 결과")
        print("="*60)
        print(f"초기 자금:     {results['initial_balance']:>15,.0f} KRW")
        print(f"최종 자산:     {results['final_value']:>15,.0f} KRW")
        print(f"총 수익:       {results['total_profit']:>15,.0f} KRW")
        print(f"수익률:        {results['total_return']:>15.2f} %")
        print("-"*60)
        print(f"총 거래 횟수:  {results['total_trades']:>15}")
        print(f"매수 횟수:     {results['buy_trades']:>15}")
        print(f"매도 횟수:     {results['sell_trades']:>15}")
        print(f"승률:          {results['win_rate']:>15.2f} %")
        print("-"*60)
        print(f"MDD:           {results['mdd']:>15.2f} %")
        print(f"Sharpe Ratio:  {results['sharpe_ratio']:>15.2f}")
        print("="*60)

    def plot_results(self, save_path: str = None):
        """백테스트 결과 시각화"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))

        # 1. 가격 차트 + 매수/매도 시그널
        ax1 = axes[0]
        ax1.plot(self.df.index, self.df['close'], label='Price', alpha=0.7)

        # 매수 시그널
        buy_signals = [t for t in self.trades if t['type'] == 'BUY']
        if buy_signals:
            buy_indices = [t['index'] for t in buy_signals]
            buy_prices = [t['price'] for t in buy_signals]
            ax1.scatter(buy_indices, buy_prices, color='green', marker='^',
                       s=100, label='Buy', zorder=5)

        # 매도 시그널
        sell_signals = [t for t in self.trades if t['type'] == 'SELL']
        if sell_signals:
            sell_indices = [t['index'] for t in sell_signals]
            sell_prices = [t['price'] for t in sell_signals]
            ax1.scatter(sell_indices, sell_prices, color='red', marker='v',
                       s=100, label='Sell', zorder=5)

        ax1.set_title('Price Chart with Trading Signals', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price (KRW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 포트폴리오 가치 변화
        ax2 = axes[1]
        ax2.plot(self.portfolio_value_history, label='Portfolio Value', color='blue')
        ax2.axhline(y=self.initial_balance, color='gray', linestyle='--',
                   label='Initial Balance', alpha=0.5)
        ax2.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Value (KRW)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 수익률 변화
        ax3 = axes[2]
        returns = [(v / self.initial_balance - 1) * 100
                   for v in self.portfolio_value_history]
        ax3.plot(returns, label='Return (%)', color='purple')
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_title('Return Over Time', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 차트 저장: {save_path}")

        plt.show()
