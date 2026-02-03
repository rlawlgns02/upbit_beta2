"""
시장 분석기
전체 종목 분석 및 기술적 지표 계산
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import ta
from datetime import datetime
import time


class MarketAnalyzer:
    """시장 종목 분석기"""

    def __init__(self, client):
        """
        Args:
            client: UpbitClient 인스턴스
        """
        self.client = client

    def get_all_krw_markets(self) -> List[Dict]:
        """KRW 마켓 전체 종목 조회

        Returns:
            KRW 마켓 종목 리스트
        """
        all_markets = self.client.get_market_all()
        krw_markets = [m for m in all_markets if m['market'].startswith('KRW-')]

        print(f"📊 전체 KRW 마켓 종목 수: {len(krw_markets)}")
        return krw_markets

    def get_market_data(self, market: str, days: int = 30) -> Optional[pd.DataFrame]:
        """특정 종목의 시장 데이터 가져오기

        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            days: 일수

        Returns:
            OHLCV 데이터프레임
        """
        try:
            candles = self.client.get_candles_day(market, count=days)

            if not candles:
                return None

            df = pd.DataFrame(candles)
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume',
                'candle_acc_trade_price': 'value'
            })

            df = df.sort_values('candle_date_time_kst').reset_index(drop=True)
            df['market'] = market

            return df

        except Exception as e:
            print(f"⚠️  {market} 데이터 수집 실패: {str(e)}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """기술적 지표 계산

        Args:
            df: OHLCV 데이터프레임

        Returns:
            기술적 지표 딕셔너리
        """
        if len(df) < 20:
            return None

        try:
            # 현재가
            current_price = df.iloc[-1]['close']

            # 이동평균선
            sma_5 = ta.trend.sma_indicator(df['close'], window=5)
            sma_20 = ta.trend.sma_indicator(df['close'], window=20)
            sma_60 = ta.trend.sma_indicator(df['close'], window=60) if len(df) >= 60 else sma_20

            # RSI
            rsi = ta.momentum.rsi(df['close'], window=14)

            # MACD
            macd = ta.trend.MACD(df['close'])
            macd_line = macd.macd()
            macd_signal = macd.macd_signal()

            # 볼린저 밴드
            bollinger = ta.volatility.BollingerBands(df['close'])
            bb_high = bollinger.bollinger_hband()
            bb_low = bollinger.bollinger_lband()

            # 거래량 분석
            volume_sma = ta.trend.sma_indicator(df['volume'], window=20)

            # 변동률 계산 (Division by zero 및 NaN 보호)
            prev_close_1d = df.iloc[-2]['close'] if len(df) >= 2 else 0
            prev_close_7d = df.iloc[-8]['close'] if len(df) >= 8 else 0
            prev_close_30d = df.iloc[0]['close'] if len(df) >= 1 else 0
            current_close = df.iloc[-1]['close']

            price_change_1d = ((current_close - prev_close_1d) / prev_close_1d * 100) if prev_close_1d > 0 else 0.0
            price_change_7d = ((current_close - prev_close_7d) / prev_close_7d * 100) if prev_close_7d > 0 else 0.0
            price_change_30d = ((current_close - prev_close_30d) / prev_close_30d * 100) if prev_close_30d > 0 else 0.0

            # 거래량 변화 (Division by zero 보호)
            volume_sma_val = volume_sma.iloc[-1]
            if volume_sma_val > 0:
                volume_change = (df.iloc[-1]['volume'] - volume_sma_val) / volume_sma_val * 100
            else:
                volume_change = 0.0

            # 기술적 신호 계산
            indicators = {
                'current_price': current_price,
                'sma_5': sma_5.iloc[-1],
                'sma_20': sma_20.iloc[-1],
                'sma_60': sma_60.iloc[-1],
                'rsi': rsi.iloc[-1],
                'macd': macd_line.iloc[-1],
                'macd_signal': macd_signal.iloc[-1],
                'bb_high': bb_high.iloc[-1],
                'bb_low': bb_low.iloc[-1],
                'price_change_1d': price_change_1d,
                'price_change_7d': price_change_7d,
                'price_change_30d': price_change_30d,
                'volume': df.iloc[-1]['volume'],
                'volume_change': volume_change,
                'volume_avg': volume_sma.iloc[-1],
            }

            return indicators

        except Exception as e:
            print(f"⚠️  지표 계산 실패: {str(e)}")
            return None

    def generate_signals(self, indicators: Dict) -> Dict:
        """매매 신호 생성 (세밀한 점수 계산)

        Args:
            indicators: 기술적 지표

        Returns:
            신호 점수 및 이유
        """
        signals = []
        score = 0.0  # 소수점 단위 점수

        # ==================== RSI 분석 (최대 ±3점) ====================
        rsi = indicators['rsi']
        if rsi < 20:
            signals.append("RSI 극과매도 (강력 매수)")
            score += 3.0
        elif rsi < 30:
            signals.append("RSI 과매도 (매수)")
            score += 2.5
        elif rsi < 40:
            signals.append("RSI 저평가")
            score += 1.5
        elif rsi < 45:
            signals.append("RSI 약간 저평가")
            score += 0.8
        elif rsi > 80:
            signals.append("RSI 극과매수 (강력 매도)")
            score -= 3.0
        elif rsi > 70:
            signals.append("RSI 과매수 (매도)")
            score -= 2.5
        elif rsi > 60:
            signals.append("RSI 고평가")
            score -= 1.5
        elif rsi > 55:
            signals.append("RSI 약간 고평가")
            score -= 0.8

        # ==================== 이동평균선 분석 (최대 ±4점) ====================
        current = indicators['current_price']
        sma5 = indicators['sma_5']
        sma20 = indicators['sma_20']
        sma60 = indicators['sma_60']

        # 골든크로스/데드크로스
        if sma5 > sma20 > sma60:
            signals.append("완벽한 정배열 (강세)")
            score += 3.0
        elif sma5 > sma20:
            signals.append("단기 골든크로스")
            score += 2.0
        elif sma5 < sma20 < sma60:
            signals.append("완벽한 역배열 (약세)")
            score -= 3.0
        elif sma5 < sma20:
            signals.append("단기 데드크로스")
            score -= 2.0

        # 현재가와 이동평균선 위치
        if current > sma5 > sma20:
            signals.append("이평선 위 강세 유지")
            score += 1.5
        elif current < sma5 < sma20:
            signals.append("이평선 아래 약세 유지")
            score -= 1.5

        # 이평선 간격 (모멘텀)
        sma_gap = abs(sma5 - sma20) / sma20 * 100
        if sma_gap > 5 and sma5 > sma20:
            signals.append("강한 상승 모멘텀")
            score += 1.0
        elif sma_gap > 5 and sma5 < sma20:
            signals.append("강한 하락 모멘텀")
            score -= 1.0

        # ==================== MACD 분석 (최대 ±2.5점) ====================
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_diff = macd - macd_signal

        if macd > macd_signal and macd_diff > 0:
            if macd_diff > abs(macd) * 0.1:
                signals.append("MACD 강한 매수 신호")
                score += 2.0
            else:
                signals.append("MACD 매수 신호")
                score += 1.2
        elif macd < macd_signal and macd_diff < 0:
            if abs(macd_diff) > abs(macd) * 0.1:
                signals.append("MACD 강한 매도 신호")
                score -= 2.0
            else:
                signals.append("MACD 매도 신호")
                score -= 1.2

        # MACD 제로라인 교차
        if macd > 0 and macd_signal > 0:
            signals.append("MACD 상승국면")
            score += 0.5
        elif macd < 0 and macd_signal < 0:
            signals.append("MACD 하락국면")
            score -= 0.5

        # ==================== 볼린저 밴드 분석 (최대 ±3점) ====================
        bb_low = indicators['bb_low']
        bb_high = indicators['bb_high']
        bb_mid = (bb_high + bb_low) / 2
        bb_width = (bb_high - bb_low) / bb_mid * 100

        # 볼린저 밴드 위치
        if current < bb_low:
            bb_position = (bb_low - current) / bb_low * 100
            if bb_position > 2:
                signals.append("볼린저 하단 돌파 (강한 반등 기대)")
                score += 2.5
            else:
                signals.append("볼린저 하단 접근 (반등 가능)")
                score += 1.8
        elif current > bb_high:
            bb_position = (current - bb_high) / bb_high * 100
            if bb_position > 2:
                signals.append("볼린저 상단 돌파 (조정 위험)")
                score -= 1.5
            else:
                signals.append("볼린저 상단 접근 (저항)")
                score -= 1.0
        elif current < bb_mid:
            signals.append("볼린저 하단 부근")
            score += 0.5
        elif current > bb_mid:
            signals.append("볼린저 상단 부근")
            score -= 0.3

        # 볼린저 밴드 폭 (변동성)
        if bb_width < 5:
            signals.append("변동성 수축 (큰 움직임 예상)")
            score += 0.8
        elif bb_width > 15:
            signals.append("변동성 확대 (주의)")
            score -= 0.5

        # ==================== 거래량 분석 (최대 ±2.5점) ====================
        volume_change = indicators['volume_change']

        if volume_change > 200:
            signals.append("거래량 폭발! (200%↑)")
            score += 2.5
        elif volume_change > 150:
            signals.append("거래량 급증 (150%↑)")
            score += 2.0
        elif volume_change > 100:
            signals.append("거래량 급증 (100%↑)")
            score += 1.5
        elif volume_change > 50:
            signals.append("거래량 증가 (50%↑)")
            score += 1.0
        elif volume_change > 20:
            signals.append("거래량 소폭 증가")
            score += 0.5
        elif volume_change < -50:
            signals.append("거래량 급감 (관심 저하)")
            score -= 1.0
        elif volume_change < -30:
            signals.append("거래량 감소")
            score -= 0.5

        # ==================== 가격 변동 분석 (최대 ±3점) ====================
        price_1d = indicators['price_change_1d']
        price_7d = indicators['price_change_7d']
        price_30d = indicators['price_change_30d']

        # 1일 변동
        if price_1d > 15:
            signals.append("전일 대비 급등 (+15%↑)")
            score += 2.0
        elif price_1d > 10:
            signals.append("전일 대비 강한 상승 (+10%↑)")
            score += 1.5
        elif price_1d > 5:
            signals.append("전일 대비 상승 (+5%↑)")
            score += 1.0
        elif price_1d > 2:
            signals.append("전일 대비 소폭 상승")
            score += 0.5
        elif price_1d < -15:
            signals.append("전일 대비 급락 (-15%↓)")
            score -= 2.0
        elif price_1d < -10:
            signals.append("전일 대비 강한 하락 (-10%↓)")
            score -= 1.5
        elif price_1d < -5:
            signals.append("전일 대비 하락 (-5%↓)")
            score -= 1.0
        elif price_1d < -2:
            signals.append("전일 대비 소폭 하락")
            score -= 0.5

        # 7일 추세
        if price_7d > 30:
            signals.append("주간 강한 상승세 (+30%↑)")
            score += 1.5
        elif price_7d > 15:
            signals.append("주간 상승세 (+15%↑)")
            score += 1.0
        elif price_7d < -30:
            signals.append("주간 강한 하락세 (-30%↓)")
            score -= 1.5
        elif price_7d < -15:
            signals.append("주간 하락세 (-15%↓)")
            score -= 1.0

        # 30일 추세
        if price_30d > 50:
            signals.append("월간 상승 추세 (+50%↑)")
            score += 1.0
        elif price_30d < -50:
            signals.append("월간 하락 추세 (-50%↓)")
            score -= 1.0

        # 추세 일관성 보너스
        if price_1d > 0 and price_7d > 0 and price_30d > 0:
            signals.append("일관된 상승 추세")
            score += 1.5
        elif price_1d < 0 and price_7d < 0 and price_30d < 0:
            signals.append("일관된 하락 추세")
            score -= 1.5

        # ==================== 최종 판단 ====================
        if score >= 10:
            recommendation = "강력 매수"
        elif score >= 6:
            recommendation = "매수"
        elif score >= 3:
            recommendation = "약한 매수"
        elif score >= 0:
            recommendation = "중립 (관망)"
        elif score >= -3:
            recommendation = "약한 매도"
        elif score >= -6:
            recommendation = "매도"
        else:
            recommendation = "강력 매도"

        # 소수점 1자리로 반올림
        score = round(score, 1)

        return {
            'score': score,
            'recommendation': recommendation,
            'signals': signals
        }

    def analyze_market(self, market: str, days: int = 30) -> Optional[Dict]:
        """종목 분석

        Args:
            market: 마켓 코드
            days: 분석 기간

        Returns:
            분석 결과
        """
        print(f"🔍 {market} 분석 중...")

        # 데이터 수집
        df = self.get_market_data(market, days)
        if df is None or len(df) < 20:
            return None

        # 기술적 지표 계산
        indicators = self.calculate_technical_indicators(df)
        if indicators is None:
            return None

        # 신호 생성
        signals = self.generate_signals(indicators)

        # 결과 조합
        result = {
            'market': market,
            'name': market.split('-')[1],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **indicators,
            **signals
        }

        return result

    def scan_all_markets(self, top_n: int = 10, delay: float = 0.1) -> List[Dict]:
        """전체 시장 스캔

        Args:
            top_n: 상위 N개 종목
            delay: API 호출 간 대기 시간 (초)

        Returns:
            분석 결과 리스트
        """
        print("\n" + "="*60)
        print("🔍 전체 시장 스캔 시작")
        print("="*60)

        # 전체 종목 조회
        markets = self.get_all_krw_markets()

        results = []
        total = len(markets)

        for i, market_info in enumerate(markets, 1):
            market = market_info['market']

            print(f"[{i}/{total}] {market} 분석 중...", end='\r')

            result = self.analyze_market(market, days=30)

            if result:
                results.append(result)

            # API 제한 방지
            time.sleep(delay)

        print(f"\n✅ 분석 완료: {len(results)}/{total} 종목")

        # 점수 기준 정렬
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_n]

    def print_analysis(self, result: Dict):
        """분석 결과 출력"""
        print("\n" + "="*60)
        print(f"📊 {result['market']} ({result['name']}) 분석 결과")
        print("="*60)
        print(f"⏰ 분석 시간: {result['timestamp']}")
        print(f"💰 현재가: {result['current_price']:,.2f} KRW")
        print()
        print("📈 가격 변동:")
        print(f"   1일:  {result['price_change_1d']:+.2f}%")
        print(f"   7일:  {result['price_change_7d']:+.2f}%")
        print(f"   30일: {result['price_change_30d']:+.2f}%")
        print()
        print("📊 기술적 지표:")
        print(f"   RSI:         {result['rsi']:.2f}")
        print(f"   SMA(5):      {result['sma_5']:,.2f}")
        print(f"   SMA(20):     {result['sma_20']:,.2f}")
        print(f"   MACD:        {result['macd']:.4f}")
        print(f"   거래량 변화: {result['volume_change']:+.2f}%")
        print()
        print("🎯 매매 신호:")
        for signal in result['signals']:
            print(f"   • {signal}")
        print()
        print(f"💡 추천: {result['recommendation']} (점수: {result['score']})")
        print("="*60)

    def print_ranking(self, results: List[Dict]):
        """종목 순위 출력"""
        print("\n" + "="*80)
        print("🏆 추천 종목 순위")
        print("="*80)
        print(f"{'순위':<6} {'종목':<15} {'현재가':<15} {'1일':<10} {'7일':<10} {'RSI':<8} {'점수':<8} {'추천'}")
        print("-"*80)

        for i, result in enumerate(results, 1):
            market = result['market']
            price = result['current_price']
            change_1d = result['price_change_1d']
            change_7d = result['price_change_7d']
            rsi = result['rsi']
            score = result['score']
            rec = result['recommendation']

            # 추천에 따른 이모지
            if '강력 매수' in rec:
                emoji = '🔥'
            elif '매수' in rec:
                emoji = '📈'
            elif '매도' in rec:
                emoji = '📉'
            else:
                emoji = '⏸️'

            print(f"{i:<6} {market:<15} {price:>12,.0f} {change_1d:>+8.2f}% {change_7d:>+8.2f}% {rsi:>6.1f} {score:>6} {emoji} {rec}")

        print("="*80)
