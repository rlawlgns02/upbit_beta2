"""
실시간 자동 매매 봇
학습된 AI 모델을 사용하여 실시간 트레이딩 수행
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import sys
import os

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.upbit_client import UpbitClient
from models.rl_agent import TradingAgent
from environment.trading_env import CryptoTradingEnv


class TradingBot:
    """실시간 자동 매매 봇"""

    def __init__(self,
                 access_key: str,
                 secret_key: str,
                 market: str = 'KRW-BTC',
                 model_path: str = 'models/crypto_trader',
                 interval: int = 60,
                 max_trade_amount: float = 100000):
        """
        Args:
            access_key: 업비트 Access Key
            secret_key: 업비트 Secret Key
            market: 거래 마켓 (예: 'KRW-BTC')
            model_path: 학습된 모델 경로
            interval: 트레이딩 주기 (초)
            max_trade_amount: 최대 거래 금액 (KRW)
        """
        self.client = UpbitClient(access_key, secret_key)
        self.market = market
        self.interval = interval
        self.max_trade_amount = max_trade_amount

        # 모델 로드
        print("🤖 AI 모델 로딩 중...")
        self.agent = None
        self.model_path = model_path

        # 상태 변수
        self.is_running = False
        self.trade_count = 0
        self.start_balance = 0
        self.current_position = None  # None, 'long'

        # 환경 객체 캐싱 (성능 개선)
        self._cached_env = None

    def _prepare_observation(self, candles: pd.DataFrame) -> np.ndarray:
        """현재 시장 데이터를 관측값으로 변환

        Args:
            candles: OHLCV 캔들 데이터

        Returns:
            관측 벡터
        """
        # 환경 객체 캐싱으로 불필요한 재생성 방지 (성능 개선)
        if self._cached_env is None:
            self._cached_env = CryptoTradingEnv(candles, initial_balance=1000000)
        else:
            # 기존 환경의 데이터만 업데이트
            self._cached_env.df = candles.reset_index(drop=True)
            self._cached_env._add_technical_indicators()

        self._cached_env.reset()

        # 최신 관측값 반환
        return self._cached_env._get_observation()

    def _get_market_data(self, count: int = 200) -> pd.DataFrame:
        """시장 데이터 가져오기

        Args:
            count: 가져올 캔들 개수

        Returns:
            OHLCV 데이터프레임
        """
        candles = self.client.get_candles_minute(self.market, unit=1, count=count)

        # 데이터프레임 변환
        df = pd.DataFrame(candles)
        df = df.rename(columns={
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        })

        # 최신 순으로 정렬
        df = df.sort_values('candle_date_time_kst').reset_index(drop=True)

        return df[['open', 'high', 'low', 'close', 'volume']]

    def _execute_action(self, action: int, current_price: float):
        """액션 실행

        Args:
            action: 0=Hold, 1=Buy, 2=Sell
            current_price: 현재 가격
        """
        # 잔고 조회 - API 오류 검증
        accounts = self.client.get_accounts()
        if not accounts:
            print("❌ 계좌 정보 조회 실패 - 거래 중단")
            return

        balance = self.client.get_balance('KRW')
        # 마켓에서 암호화폐 심볼 추출 (예: 'KRW-BTC' -> 'BTC')
        try:
            crypto_symbol = self.market.split('-')[1]
        except IndexError:
            print(f"❌ 잘못된 마켓 형식: {self.market}")
            return

        crypto_balance = self.client.get_balance(crypto_symbol)

        # Hold
        if action == 0:
            print("⏸️  HOLD - 대기")
            return

        # Buy
        elif action == 1:
            if self.current_position is None and balance > 5000:  # 최소 거래 금액
                # 거래 가능 금액 계산
                trade_amount = min(balance * 0.5, self.max_trade_amount)

                if trade_amount >= 5000:
                    try:
                        result = self.client.buy_market_order(self.market, trade_amount)

                        # 주문 결과 상세 검증
                        if isinstance(result, dict) and 'error' not in result and 'uuid' in result:
                            # 주문 상태 확인 (wait/watch/done)
                            order_uuid = result.get('uuid')
                            print(f"✅ 매수 주문 접수: {trade_amount:,.0f} KRW @ {current_price:,.0f}")
                            print(f"   주문 UUID: {order_uuid}")

                            # 실제 체결 여부는 별도로 확인해야 하지만,
                            # 주문이 성공적으로 접수되면 포지션 업데이트
                            self.current_position = 'long'
                            self.trade_count += 1
                        elif isinstance(result, dict) and 'error' in result:
                            error_msg = result.get('error', {}).get('message', '알 수 없는 오류')
                            print(f"❌ 매수 실패: {error_msg}")
                        else:
                            print(f"❌ 매수 실패: 예상치 못한 응답 형식")

                    except Exception as e:
                        print(f"❌ 매수 오류: {str(e)}")
            else:
                print(f"⚠️  매수 불가 (잔고: {balance:,.0f} KRW, 포지션: {self.current_position})")

        # Sell
        elif action == 2:
            if self.current_position == 'long' and crypto_balance > 0:
                try:
                    result = self.client.sell_market_order(self.market, crypto_balance)

                    # 주문 결과 상세 검증
                    if isinstance(result, dict) and 'error' not in result and 'uuid' in result:
                        order_uuid = result.get('uuid')
                        print(f"✅ 매도 주문 접수: {crypto_balance:.8f} @ {current_price:,.0f}")
                        print(f"   주문 UUID: {order_uuid}")

                        # 포지션 정리
                        self.current_position = None
                        self.trade_count += 1
                    elif isinstance(result, dict) and 'error' in result:
                        error_msg = result.get('error', {}).get('message', '알 수 없는 오류')
                        print(f"❌ 매도 실패: {error_msg}")
                    else:
                        print(f"❌ 매도 실패: 예상치 못한 응답 형식")

                except Exception as e:
                    print(f"❌ 매도 오류: {str(e)}")
            else:
                print(f"⚠️  매도 불가 (보유량: {crypto_balance:.8f}, 포지션: {self.current_position})")

    def _print_status(self, current_price: float):
        """현재 상태 출력"""
        balance = self.client.get_balance('KRW')
        crypto = self.client.get_balance(self.market.split('-')[1])

        total_value = balance + crypto * current_price
        profit = total_value - self.start_balance
        profit_rate = (profit / self.start_balance * 100) if self.start_balance > 0 else 0

        print("\n" + "="*60)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 현재가: {current_price:,.0f} KRW")
        print(f"💵 원화 잔고: {balance:,.0f} KRW")
        print(f"🪙 코인 보유: {crypto:.8f} ({crypto * current_price:,.0f} KRW)")
        print(f"📊 총 자산: {total_value:,.0f} KRW")
        print(f"📈 수익: {profit:,.0f} KRW ({profit_rate:+.2f}%)")
        print(f"🔄 거래 횟수: {self.trade_count}")
        print(f"📍 포지션: {self.current_position or 'None'}")
        print("="*60)

    def start(self, use_model: bool = True):
        """봇 시작

        Args:
            use_model: AI 모델 사용 여부 (False면 랜덤 트레이딩)
        """
        print("\n🚀 업비트 자동 매매 봇 시작")
        print(f"📊 마켓: {self.market}")
        print(f"⏱️  간격: {self.interval}초")
        print(f"💰 최대 거래 금액: {self.max_trade_amount:,.0f} KRW")

        # 모델 로드
        if use_model:
            try:
                # 더미 환경 생성 (모델 로드용)
                dummy_data = self._get_market_data(200)
                dummy_env = CryptoTradingEnv(dummy_data)
                self.agent = TradingAgent(dummy_env)
                self.agent.load(self.model_path)
                print("✅ AI 모델 로드 완료!")
            except Exception as e:
                print(f"⚠️  모델 로드 실패: {str(e)}")
                print("⚠️  랜덤 트레이딩 모드로 전환")
                use_model = False

        # 초기 잔고 기록
        self.start_balance = self.client.get_balance('KRW')
        print(f"💵 시작 잔고: {self.start_balance:,.0f} KRW")

        self.is_running = True

        try:
            while self.is_running:
                # 시장 데이터 가져오기
                df = self._get_market_data(200)
                current_price = df.iloc[-1]['close']

                # 현재 상태 출력
                self._print_status(current_price)

                # 액션 결정
                if use_model and self.agent:
                    # AI 모델 사용
                    obs = self._prepare_observation(df)
                    action, _ = self.agent.predict(obs, deterministic=True)
                    action = int(action)
                    print(f"🤖 AI 판단: {['HOLD', 'BUY', 'SELL'][action]}")
                else:
                    # 랜덤 액션 (테스트용)
                    action = np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15])
                    print(f"🎲 랜덤 액션: {['HOLD', 'BUY', 'SELL'][action]}")

                # 액션 실행
                self._execute_action(action, current_price)

                # 대기
                print(f"\n⏳ {self.interval}초 대기 중... (Ctrl+C로 종료)\n")
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print("\n\n⏹️  사용자에 의해 중단됨")
            self.stop()

        except Exception as e:
            print(f"\n❌ 오류 발생: {str(e)}")
            self.stop()

    def stop(self):
        """봇 중지"""
        self.is_running = False

        # 최종 결과
        final_balance = self.client.get_balance('KRW')
        crypto = self.client.get_balance(self.market.split('-')[1])

        if crypto > 0:
            ticker_list = self.client.get_ticker([self.market])
            if ticker_list and len(ticker_list) > 0:
                current_price = ticker_list[0]['trade_price']
                final_value = final_balance + crypto * current_price
            else:
                # 티커 조회 실패 시 코인 가치를 0으로 처리
                print("⚠️  티커 조회 실패, 코인 가치를 계산할 수 없습니다.")
                final_value = final_balance
        else:
            final_value = final_balance

        profit = final_value - self.start_balance
        profit_rate = (profit / self.start_balance * 100) if self.start_balance > 0 else 0

        print("\n" + "="*60)
        print("📊 최종 결과")
        print("="*60)
        print(f"시작 자산: {self.start_balance:,.0f} KRW")
        print(f"최종 자산: {final_value:,.0f} KRW")
        print(f"총 수익: {profit:,.0f} KRW ({profit_rate:+.2f}%)")
        print(f"거래 횟수: {self.trade_count}")
        print("="*60)
        print("👋 봇 종료")
