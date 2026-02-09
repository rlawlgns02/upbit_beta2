"""
AI 기반 종목 예측 시스템
학습된 모델 또는 앙상블 방식으로 상승/하락 예측
LSTM 가격 예측 + PPO 강화학습 + 뉴스 감정 분석 통합
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sys
import os
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.trading_env import CryptoTradingEnv
from models.rl_agent import TradingAgent

# LSTM 예측기 (lazy import) - 스레드 안전성 보장
_lstm_predictor = None
_lstm_predictor_lock = threading.Lock()

def get_lstm_predictor(market: str = 'KRW-BTC'):
    """LSTM 예측기 지연 로딩 (스레드 안전)"""
    global _lstm_predictor

    # 이미 로드되었으면 락 없이 반환 (성능 최적화)
    if _lstm_predictor is not None:
        return _lstm_predictor

    # 락을 획득하여 초기화
    with _lstm_predictor_lock:
        # Double-checked locking: 락 획득 후 다시 확인
        if _lstm_predictor is None:
            try:
                from models.lstm_predictor import LSTMPredictor
                coin = market.replace('KRW-', '').lower()
                model_path = f'models/lstm_{coin}'
                predictor = LSTMPredictor(model_path=model_path)
                if not predictor.load():
                    # BTC 모델로 폴백
                    predictor = LSTMPredictor(model_path='models/lstm_btc')
                    if not predictor.load():
                        try:
                            import logging
                            logging.getLogger(__name__).debug("[LSTM] LSTM 모델을 찾을 수 없습니다.")
                        except Exception:
                            pass
                        return None
                _lstm_predictor = predictor
            except Exception as e:
                print(f"[WARNING] LSTM 예측기 로드 실패: {e}")
                return None
        return _lstm_predictor

# 뉴스 신호 생성기 (lazy import) - 스레드 안전성 보장
_news_signal_generator = None
_news_signal_generator_lock = threading.Lock()

def get_news_signal_generator():
    """뉴스 신호 생성기 지연 로딩 (스레드 안전)"""
    global _news_signal_generator

    # 이미 로드되었으면 락 없이 반환
    if _news_signal_generator is not None:
        return _news_signal_generator

    # 락을 획득하여 초기화
    with _news_signal_generator_lock:
        # Double-checked locking
        if _news_signal_generator is None:
            try:
                from news.signal_generator import NewsSignalGenerator
                _news_signal_generator = NewsSignalGenerator()
            except Exception as e:
                print(f"[WARNING] 뉴스 신호 생성기 로드 실패: {e}")
                return None
        return _news_signal_generator


class AIPredictor:
    """AI 기반 예측 시스템"""

    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 학습된 모델 경로 (None이면 규칙 기반)
        """
        self.model_path = model_path
        self.agent = None
        self.use_ai_model = False

        if model_path and os.path.exists(model_path + '.zip'):
            try:
                print("[AI] 모델 로딩 중...")
                # 더미 환경 생성
                dummy_data = pd.DataFrame({
                    'open': [50000] * 100,
                    'high': [51000] * 100,
                    'low': [49000] * 100,
                    'close': [50000] * 100,
                    'volume': [1000] * 100
                })
                dummy_env = CryptoTradingEnv(dummy_data)
                self.agent = TradingAgent(dummy_env)
                self.agent.load(model_path)
                self.use_ai_model = True
                print("[AI] 모델 로드 완료!")
            except Exception as e:
                print(f"[WARNING] AI 모델 로드 실패: {str(e)}")
                print("[WARNING] 규칙 기반 예측으로 전환")
                self.use_ai_model = False
        else:
            print("[INFO] 규칙 기반 예측 모드")

    def predict_with_ai(self, df: pd.DataFrame) -> Tuple[int, float]:
        """AI 모델을 사용한 예측

        Args:
            df: OHLCV 데이터

        Returns:
            (action, confidence) - action: 0=Hold, 1=Buy, 2=Sell
        """
        if not self.use_ai_model or self.agent is None:
            return self.predict_with_rules(df)

        try:
            # 환경 생성
            env = CryptoTradingEnv(df, initial_balance=1000000)
            obs, _ = env.reset()

            # 확정적 예측 (deterministic=True)으로 일관성 있는 결과
            action, _ = self.agent.predict(obs, deterministic=True)
            predicted_action = int(action)

            # 신뢰도: 추가로 몇 번 샘플링하여 일관성 측정 (5회로 축소)
            consistent_count = 1  # 첫 번째 예측 포함
            for _ in range(4):
                sample_action, _ = self.agent.predict(obs, deterministic=False)
                if int(sample_action) == predicted_action:
                    consistent_count += 1

            confidence = consistent_count / 5

            return predicted_action, confidence

        except Exception as e:
            print(f"⚠️  AI 예측 오류: {str(e)}")
            return self.predict_with_rules(df)

    def predict_with_rules(self, df: pd.DataFrame) -> Tuple[int, float]:
        """규칙 기반 예측

        Args:
            df: OHLCV 데이터

        Returns:
            (action, confidence)
        """
        import ta

        # 기술적 지표 계산
        sma_5 = ta.trend.sma_indicator(df['close'], window=5)
        sma_20 = ta.trend.sma_indicator(df['close'], window=20)
        rsi = ta.momentum.rsi(df['close'], window=14)
        macd = ta.trend.MACD(df['close'])

        current_price = df.iloc[-1]['close']
        current_rsi = rsi.iloc[-1]
        current_sma5 = sma_5.iloc[-1]
        current_sma20 = sma_20.iloc[-1]
        macd_line = macd.macd().iloc[-1]
        macd_signal = macd.macd_signal().iloc[-1]

        # 점수 시스템
        buy_score = 0
        sell_score = 0

        # RSI 기반
        if current_rsi < 30:
            buy_score += 3
        elif current_rsi < 40:
            buy_score += 1
        elif current_rsi > 70:
            sell_score += 3
        elif current_rsi > 60:
            sell_score += 1

        # 이동평균선 기반
        if current_sma5 > current_sma20:
            if current_price > current_sma5:
                buy_score += 2
        else:
            if current_price < current_sma5:
                sell_score += 2

        # MACD 기반
        if macd_line > macd_signal:
            buy_score += 1
        else:
            sell_score += 1

        # 단기 추세
        price_change = (df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close']
        if price_change > 0.02:  # 2% 이상 상승
            buy_score += 1
        elif price_change < -0.02:  # 2% 이상 하락
            sell_score += 1

        # 결정 (매수/매도 임계값 균형 조정)
        total_score = buy_score + sell_score
        THRESHOLD = 3  # 동일한 임계값 사용
        if buy_score > sell_score and buy_score >= THRESHOLD:
            return 1, buy_score / (total_score + 1)  # Buy
        elif sell_score > buy_score and sell_score >= THRESHOLD:
            return 2, sell_score / (total_score + 1)  # Sell
        else:
            return 0, 0.5  # Hold

    def predict_with_lstm(self, df: pd.DataFrame, market: str) -> Optional[Dict]:
        """LSTM 기반 가격 예측

        Args:
            df: OHLCV 데이터
            market: 마켓 코드

        Returns:
            LSTM 예측 결과 또는 None
        """
        lstm = get_lstm_predictor(market)
        if lstm is None:
            return None

        try:
            pred_price, change_rate, direction = lstm.predict(df)
            return {
                'predicted_price': pred_price,
                'change_rate': change_rate,
                'direction': direction,
                'available': True
            }
        except Exception as e:
            print(f"[LSTM] 예측 실패: {e}")
            return None

    def predict_market(self, df: pd.DataFrame, market: str) -> Dict:
        """종목 예측 (LSTM + PPO/규칙 앙상블)

        Args:
            df: OHLCV 데이터
            market: 마켓 코드

        Returns:
            예측 결과
        """
        # PPO/규칙 기반 예측
        action, confidence = self.predict_with_ai(df) if self.use_ai_model else self.predict_with_rules(df)

        # LSTM 가격 예측 (보조 지표)
        lstm_result = self.predict_with_lstm(df, market)

        # LSTM 신호를 PPO/규칙 예측과 결합
        if lstm_result and lstm_result.get('available'):
            lstm_direction = lstm_result['direction']
            lstm_change = lstm_result['change_rate']

            # LSTM이 강한 신호를 보내면 액션 조정
            if lstm_direction == 'STRONG_UP' and action != 1:
                # LSTM이 강한 상승 예측이면 매수 신호 강화
                if action == 0:  # Hold -> Buy 고려
                    if confidence < 0.6:
                        action = 1
                        confidence = min(confidence + 0.2, 0.9)
            elif lstm_direction == 'STRONG_DOWN' and action != 2:
                # LSTM이 강한 하락 예측이면 매도 신호 강화
                if action == 0:  # Hold -> Sell 고려
                    if confidence < 0.6:
                        action = 2
                        confidence = min(confidence + 0.2, 0.9)

            # LSTM과 PPO가 일치하면 신뢰도 상승
            if (action == 1 and lstm_direction in ['UP', 'STRONG_UP']) or \
               (action == 2 and lstm_direction in ['DOWN', 'STRONG_DOWN']):
                confidence = min(confidence * 1.15, 0.95)

        # 액션 해석
        action_map = {
            0: "보유 (HOLD)",
            1: "매수 (BUY)",
            2: "매도 (SELL)"
        }

        # 예측 방향
        direction_map = {
            0: "중립",
            1: "상승 예상",
            2: "하락 예상"
        }

        # 신뢰도 해석
        if confidence >= 0.8:
            confidence_level = "매우 높음"
        elif confidence >= 0.6:
            confidence_level = "높음"
        elif confidence >= 0.4:
            confidence_level = "보통"
        else:
            confidence_level = "낮음"

        # 현재가 정보
        current_price = df.iloc[-1]['close']
        price_change_1d = (df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100

        result = {
            'market': market,
            'current_price': current_price,
            'price_change_1d': price_change_1d,
            'action': action,
            'action_text': action_map[action],
            'direction': direction_map[action],
            'confidence': confidence,
            'confidence_level': confidence_level,
            'method': 'LSTM + AI 앙상블' if lstm_result else ('AI 모델' if self.use_ai_model else '규칙 기반')
        }

        # LSTM 예측 정보 추가
        if lstm_result:
            result['lstm_prediction'] = {
                'predicted_price': lstm_result['predicted_price'],
                'change_rate': lstm_result['change_rate'],
                'direction': lstm_result['direction']
            }

        return result

    def predict_with_news(self, df: pd.DataFrame, market: str) -> Dict:
        """뉴스 감정을 통합한 예측

        기술적 분석 + 뉴스 감정 분석을 결합하여 최종 신호 생성

        Args:
            df: OHLCV 데이터
            market: 마켓 코드

        Returns:
            통합 예측 결과
        """
        # 기본 예측
        base_result = self.predict_market(df, market)

        # 뉴스 신호 가져오기
        news_generator = get_news_signal_generator()
        news_signal = None

        if news_generator:
            try:
                news_signal = news_generator.generate_signal(page_size=100)
            except Exception as e:
                print(f"[WARNING] 뉴스 신호 생성 실패: {e}")

        if not news_signal:
            # 뉴스 신호 없으면 기본 예측 반환
            base_result['news_integrated'] = False
            return base_result

        # 신호 통합 (기술적 분석 70%, 뉴스 30%)
        tech_weight = 0.7
        news_weight = 0.3

        # 뉴스 신호를 숫자로 변환
        news_action_map = {"BUY": 1, "SELL": 2, "HOLD": 0}
        news_action = news_action_map.get(news_signal['signal'], 0)

        # 가중 평균 계산
        combined_confidence = (
            base_result['confidence'] * tech_weight +
            news_signal['confidence'] * news_weight
        )

        # 최종 액션 결정
        if base_result['action'] == news_action:
            # 기술/뉴스 일치 -> 신뢰도 상승
            final_action = base_result['action']
            combined_confidence = min(1.0, combined_confidence * 1.2)
        elif news_signal['confidence'] > 0.7 and base_result['confidence'] < 0.5:
            # 뉴스 신호가 강하고 기술 신호가 약함 -> 뉴스 우선
            final_action = news_action
        else:
            # 기술적 분석 우선
            final_action = base_result['action']

        # 액션 해석
        action_map = {
            0: "보유 (HOLD)",
            1: "매수 (BUY)",
            2: "매도 (SELL)"
        }
        direction_map = {
            0: "중립",
            1: "상승 예상",
            2: "하락 예상"
        }

        # 결과 통합
        result = {
            'market': market,
            'current_price': base_result['current_price'],
            'price_change_1d': base_result['price_change_1d'],
            'action': final_action,
            'action_text': action_map[final_action],
            'direction': direction_map[final_action],
            'confidence': round(combined_confidence, 4),
            'confidence_level': self._get_confidence_level(combined_confidence),
            'method': 'AI + 뉴스 통합',
            'news_integrated': True,
            'technical_signal': {
                'action': base_result['action'],
                'confidence': base_result['confidence']
            },
            'news_signal': {
                'signal': news_signal['signal'],
                'positive_ratio': news_signal['positive_ratio'],
                'confidence': news_signal['confidence'],
                'news_count': news_signal['news_count']
            }
        }

        return result

    def _get_confidence_level(self, confidence: float) -> str:
        """신뢰도 레벨 반환"""
        if confidence >= 0.8:
            return "매우 높음"
        elif confidence >= 0.6:
            return "높음"
        elif confidence >= 0.4:
            return "보통"
        else:
            return "낮음"

    def batch_predict(self, market_data: Dict[str, pd.DataFrame], top_n: int = 10) -> List[Dict]:
        """여러 종목 일괄 예측

        Args:
            market_data: {market: df} 딕셔너리
            top_n: 상위 N개

        Returns:
            예측 결과 리스트 (신뢰도 순)
        """
        results = []

        print("\n🔮 AI 예측 시작...")
        total = len(market_data)

        for i, (market, df) in enumerate(market_data.items(), 1):
            print(f"[{i}/{total}] {market} 예측 중...", end='\r')

            try:
                result = self.predict_market(df, market)
                # 매수 신호만 수집 (상승 예상)
                if result['action'] == 1:
                    results.append(result)
            except Exception as e:
                print(f"\n⚠️  {market} 예측 실패: {str(e)}")
                continue

        print(f"\n✅ 예측 완료: {len(results)} 종목")

        # 신뢰도 순으로 정렬
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results[:top_n]

    def print_prediction(self, result: Dict):
        """예측 결과 출력"""
        print("\n" + "="*60)
        print(f"🔮 {result['market']} AI 예측 결과")
        print("="*60)
        print(f"💰 현재가: {result['current_price']:,.0f} KRW")
        print(f"📊 전일 대비: {result['price_change_1d']:+.2f}%")
        print()
        print(f"🎯 예측: {result['direction']}")
        print(f"📌 추천 액션: {result['action_text']}")
        print(f"💯 신뢰도: {result['confidence']*100:.1f}% ({result['confidence_level']})")
        print(f"🤖 예측 방법: {result['method']}")
        print("="*60)

        # 신뢰도에 따른 조언
        if result['action'] == 1:  # Buy
            if result['confidence'] >= 0.7:
                print("💡 강력한 매수 신호입니다!")
            elif result['confidence'] >= 0.5:
                print("💡 매수를 고려해볼 만합니다.")
            else:
                print("💡 신중한 판단이 필요합니다.")
        elif result['action'] == 2:  # Sell
            if result['confidence'] >= 0.7:
                print("💡 매도 타이밍일 수 있습니다.")
            else:
                print("💡 추가 분석이 필요합니다.")
        else:
            print("💡 관망하는 것이 좋겠습니다.")

        print()

    def print_predictions_ranking(self, results: List[Dict]):
        """예측 결과 순위 출력"""
        print("\n" + "="*80)
        print("🔥 AI 추천 상승 예상 종목")
        print("="*80)
        print(f"{'순위':<6} {'종목':<15} {'현재가':<15} {'1일 변화':<12} {'신뢰도':<12} {'방향'}")
        print("-"*80)

        for i, result in enumerate(results, 1):
            market = result['market']
            price = result['current_price']
            change = result['price_change_1d']
            confidence = result['confidence'] * 100
            direction = result['direction']

            # 신뢰도에 따른 이모지
            if confidence >= 80:
                emoji = '🔥'
            elif confidence >= 60:
                emoji = '⭐'
            else:
                emoji = '💡'

            print(f"{i:<6} {market:<15} {price:>12,.0f} {change:>+10.2f}% {confidence:>10.1f}% {emoji} {direction}")

        print("="*80)
