"""
뉴스 감정 분석 기반 트레이딩 신호 생성기
"""
from typing import Dict, List, Optional
from datetime import datetime

from .news_collector import NewsCollector
from .sentiment_analyzer import SentimentAnalyzer


class NewsSignalGenerator:
    """뉴스 감정 분석을 기반으로 매수/매도/보유 신호를 생성하는 클래스"""

    # 신호 임계값
    BUY_THRESHOLD = 0.6   # 긍정 비율 > 0.6 -> BUY
    SELL_THRESHOLD = 0.4  # 긍정 비율 < 0.4 -> SELL

    # 신호 상수
    SIGNAL_BUY = "BUY"
    SIGNAL_SELL = "SELL"
    SIGNAL_HOLD = "HOLD"

    # 코인별 검색 쿼리 매핑
    COIN_QUERY_MAP = {
        # 주요 코인
        'BTC': 'bitcoin OR BTC',
        'ETH': 'ethereum OR ETH',
        'XRP': 'ripple OR XRP',
        'SOL': 'solana OR SOL',
        'DOGE': 'dogecoin OR DOGE',
        'ADA': 'cardano OR ADA',
        'AVAX': 'avalanche OR AVAX',
        'DOT': 'polkadot OR DOT',
        'MATIC': 'polygon OR MATIC',
        'LINK': 'chainlink OR LINK',
        'ATOM': 'cosmos OR ATOM',
        'UNI': 'uniswap OR UNI',
        'LTC': 'litecoin OR LTC',
        'BCH': 'bitcoin cash OR BCH',
        'ETC': 'ethereum classic OR ETC',
        'XLM': 'stellar OR XLM',
        'ALGO': 'algorand OR ALGO',
        'VET': 'vechain OR VET',
        'FIL': 'filecoin OR FIL',
        'AAVE': 'aave crypto',
        'EOS': 'EOS crypto OR EOS blockchain',
        'XTZ': 'tezos OR XTZ',
        'THETA': 'theta network OR THETA',
        'AXS': 'axie infinity OR AXS',
        'SAND': 'sandbox crypto OR SAND metaverse',
        'MANA': 'decentraland OR MANA',
        'SHIB': 'shiba inu OR SHIB',
        'APT': 'aptos OR APT crypto',
        'ARB': 'arbitrum OR ARB',
        'OP': 'optimism crypto OR OP layer2',
        'SUI': 'sui blockchain OR SUI crypto',
        'SEI': 'sei network OR SEI crypto',
        'TRX': 'tron OR TRX',
        'NEAR': 'near protocol OR NEAR',
        'ICP': 'internet computer OR ICP',
        'HBAR': 'hedera OR HBAR',
        'CRO': 'cronos OR CRO crypto',
        'IMX': 'immutable x OR IMX',
        'PEPE': 'pepe coin OR PEPE crypto',
    }

    # 캐시 설정
    MAX_CACHE_SIZE = 50  # 최대 캐시 항목 수

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: NewsAPI 키. None이면 환경변수에서 로드
        """
        self.collector = NewsCollector(api_key=api_key)
        self.analyzer = SentimentAnalyzer()
        self._cache = {}
        self._cache_ttl = 600  # 캐시 유효시간 (10분) - API 호출 제한 고려

    def generate_signal(self, query: str = "cryptocurrency", page_size: int = 100) -> Dict:
        """뉴스 기반 트레이딩 신호 생성

        Args:
            query: 검색 쿼리
            page_size: 분석할 뉴스 개수

        Returns:
            신호 정보 딕셔너리
            - signal: BUY, SELL, HOLD
            - positive_ratio: 긍정 비율 (0.0 ~ 1.0)
            - confidence: 신호 신뢰도 (0.0 ~ 1.0)
            - news_count: 분석된 뉴스 개수
            - top_news: 상위 뉴스 리스트
            - timestamp: 생성 시간
        """
        # 캐시 확인
        cache_key = f"{query}_{page_size}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        # 뉴스 수집
        articles = self.collector.get_crypto_news(
            query=query,
            page_size=page_size
        )

        if not articles:
            return self._get_default_signal()

        # 감정 분석
        analysis_result = self.analyzer.analyze_news_batch(articles)
        summary = analysis_result['summary']
        analyzed_articles = analysis_result['articles']

        # 긍정 비율
        positive_ratio = summary['positive_ratio']

        # 신호 결정 (실제 뉴스 수 전달)
        signal, confidence = self._determine_signal(positive_ratio, news_count=summary['total_articles'])

        # 상위 뉴스 (긍정/부정 순)
        sorted_articles = sorted(
            analyzed_articles,
            key=lambda x: abs(x['polarity']),
            reverse=True
        )[:5]

        result = {
            'signal': signal,
            'signal_kr': self._get_signal_korean(signal),
            'positive_ratio': round(positive_ratio, 4),
            'average_polarity': summary['average_polarity'],
            'confidence': round(confidence, 4),
            'news_count': summary['total_articles'],
            'positive_count': summary['positive_count'],
            'negative_count': summary['negative_count'],
            'neutral_count': summary['neutral_count'],
            'overall_sentiment': summary['overall_sentiment'],
            'top_news': sorted_articles,
            'timestamp': datetime.now().isoformat(),
            'query': query
        }

        # 캐시 저장
        self._set_cache(cache_key, result)

        return result

    def generate_coin_signal(self, market: str, page_size: int = 100) -> Dict:
        """특정 코인의 뉴스 신호 생성

        Args:
            market: 마켓 코드 (예: "KRW-BTC", "BTC")
            page_size: 분석할 뉴스 개수

        Returns:
            해당 코인의 뉴스 신호
        """
        # 마켓 코드에서 코인 심볼 추출 (KRW-BTC -> BTC)
        coin_symbol = market.replace('KRW-', '').replace('USDT-', '').upper()

        # 쿼리 매핑에서 검색어 가져오기
        query = self.COIN_QUERY_MAP.get(coin_symbol)

        if not query:
            # 매핑에 없으면 심볼 자체로 검색
            query = f"{coin_symbol} crypto OR {coin_symbol} coin"

        signal = self.generate_signal(query=query, page_size=page_size)

        # 뉴스가 없으면 해당 코인 뉴스 없음으로 표시
        if signal.get('news_count', 0) == 0:
            signal['no_news'] = True
            signal['message'] = f'{coin_symbol} 관련 뉴스를 찾을 수 없습니다'

        signal['market'] = market
        signal['coin_symbol'] = coin_symbol
        signal['query_used'] = query

        return signal

    def generate_bitcoin_signal(self) -> Dict:
        """비트코인 전용 신호 생성"""
        return self.generate_coin_signal("KRW-BTC", page_size=100)

    def generate_ethereum_signal(self) -> Dict:
        """이더리움 전용 신호 생성"""
        return self.generate_coin_signal("KRW-ETH", page_size=100)

    def generate_market_signal(self) -> Dict:
        """암호화폐 시장 전체 신호 생성"""
        return self.generate_signal(
            query="cryptocurrency market OR crypto trading",
            page_size=100
        )

    def get_coin_query(self, market: str) -> str:
        """코인의 검색 쿼리 반환"""
        coin_symbol = market.replace('KRW-', '').replace('USDT-', '').upper()
        return self.COIN_QUERY_MAP.get(coin_symbol, f"{coin_symbol} crypto")

    def _determine_signal(self, positive_ratio: float, news_count: int = 10) -> tuple:
        """긍정 비율을 기반으로 신호 결정

        Args:
            positive_ratio: 긍정 비율 (0.0 ~ 1.0)
            news_count: 분석된 뉴스 수

        Returns:
            (signal, confidence) 튜플
        """
        # 뉴스 수에 따른 신뢰도 가중치 (뉴스가 많을수록 신뢰도 증가)
        news_weight = min(1.0, news_count / 10)

        if positive_ratio > self.BUY_THRESHOLD:
            # 매수 신호 (0.6 초과)
            # 0.6 초과 시 기본 60% + 추가 (최대 100%)
            extra = (positive_ratio - self.BUY_THRESHOLD) / (1.0 - self.BUY_THRESHOLD)
            base_confidence = 0.6 + (extra * 0.4)
            confidence = base_confidence * news_weight
            return self.SIGNAL_BUY, min(1.0, max(0.5, confidence))

        elif positive_ratio < self.SELL_THRESHOLD:
            # 매도 신호 (0.4 미만)
            # 0.4 미만 시 기본 60% + 추가 (최대 100%)
            extra = (self.SELL_THRESHOLD - positive_ratio) / self.SELL_THRESHOLD
            base_confidence = 0.6 + (extra * 0.4)
            confidence = base_confidence * news_weight
            return self.SIGNAL_SELL, min(1.0, max(0.5, confidence))

        else:
            # 보유 신호 (0.4 ~ 0.6)
            # 0.5에 가까울수록 신뢰도 높음
            # distance_from_center가 0.1일 때 (0.4 또는 0.6) confidence = 50%
            # distance_from_center가 0일 때 (0.5) confidence = 100%
            distance_from_center = abs(positive_ratio - 0.5)
            # 정규화: 0 ~ 0.1 범위를 1.0 ~ 0.5로 변환
            base_confidence = 1.0 - (distance_from_center / 0.1) * 0.5
            confidence = base_confidence * news_weight
            return self.SIGNAL_HOLD, min(1.0, max(0.5, confidence))

    def _get_signal_korean(self, signal: str) -> str:
        """신호의 한국어 레이블 반환"""
        signal_map = {
            self.SIGNAL_BUY: "매수",
            self.SIGNAL_SELL: "매도",
            self.SIGNAL_HOLD: "보유"
        }
        return signal_map.get(signal, "보유")

    def _get_default_signal(self) -> Dict:
        """기본 신호 반환 (API 실패 시)"""
        return {
            'signal': self.SIGNAL_HOLD,
            'signal_kr': "보유",
            'positive_ratio': 0.5,
            'average_polarity': 0.0,
            'confidence': 0.0,
            'news_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'overall_sentiment': 'neutral',
            'top_news': [],
            'timestamp': datetime.now().isoformat(),
            'query': '',
            'error': 'News API unavailable'
        }

    def _get_cached(self, key: str) -> Optional[Dict]:
        """캐시에서 결과 조회"""
        if key not in self._cache:
            return None

        cached_data = self._cache[key]
        cached_time = datetime.fromisoformat(cached_data['timestamp'])
        now = datetime.now()

        # TTL 확인
        if (now - cached_time).total_seconds() > self._cache_ttl:
            del self._cache[key]
            return None

        return cached_data

    def _set_cache(self, key: str, data: Dict):
        """결과를 캐시에 저장 (크기 제한 적용)"""
        # 캐시 크기 제한 체크
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            # 가장 오래된 항목 삭제
            self._cleanup_old_cache()

        self._cache[key] = data

    def _cleanup_old_cache(self):
        """만료된 캐시 정리 및 가장 오래된 항목 삭제"""
        now = datetime.now()
        expired_keys = []

        # 만료된 항목 찾기
        for key, cached_data in self._cache.items():
            try:
                cached_time = datetime.fromisoformat(cached_data['timestamp'])
                if (now - cached_time).total_seconds() > self._cache_ttl:
                    expired_keys.append(key)
            except (KeyError, ValueError):
                expired_keys.append(key)

        # 만료된 항목 삭제
        for key in expired_keys:
            del self._cache[key]

        # 아직 꽉 찼으면 가장 오래된 항목 삭제
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            try:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: datetime.fromisoformat(self._cache[k].get('timestamp', '2000-01-01'))
                )
                del self._cache[oldest_key]
            except (ValueError, KeyError):
                # 에러 시 첫 번째 항목 삭제
                if self._cache:
                    first_key = next(iter(self._cache))
                    del self._cache[first_key]

    def clear_cache(self):
        """캐시 초기화"""
        self._cache = {}

    def get_combined_signal(self) -> Dict:
        """여러 소스의 신호를 결합한 종합 신호 생성

        Returns:
            종합 신호 정보
        """
        # 각 카테고리별 신호 수집
        btc_signal = self.generate_bitcoin_signal()
        market_signal = self.generate_market_signal()

        # 가중치 적용 (비트코인 60%, 시장 40%)
        weighted_ratio = (
            btc_signal['positive_ratio'] * 0.6 +
            market_signal['positive_ratio'] * 0.4
        )

        # 뉴스 카운트 합산
        total_news = btc_signal['news_count'] + market_signal['news_count']

        # 신호 결정 (실제 뉴스 수 전달)
        signal, confidence = self._determine_signal(weighted_ratio, news_count=total_news)

        # 상위 뉴스 결합
        all_top_news = btc_signal['top_news'] + market_signal['top_news']
        sorted_news = sorted(
            all_top_news,
            key=lambda x: abs(x['polarity']),
            reverse=True
        )[:5]

        return {
            'signal': signal,
            'signal_kr': self._get_signal_korean(signal),
            'positive_ratio': round(weighted_ratio, 4),
            'confidence': round(confidence, 4),
            'news_count': total_news,
            'top_news': sorted_news,
            'timestamp': datetime.now().isoformat(),
            'sources': {
                'bitcoin': {
                    'signal': btc_signal['signal'],
                    'positive_ratio': btc_signal['positive_ratio'],
                    'news_count': btc_signal['news_count']
                },
                'market': {
                    'signal': market_signal['signal'],
                    'positive_ratio': market_signal['positive_ratio'],
                    'news_count': market_signal['news_count']
                }
            }
        }

    def print_signal(self, signal_data: Dict):
        """신호 정보 출력"""
        print("\n" + "=" * 60)
        print("NEWS SENTIMENT SIGNAL")
        print("=" * 60)
        print(f"Signal: {signal_data['signal']} ({signal_data['signal_kr']})")
        print(f"Positive Ratio: {signal_data['positive_ratio'] * 100:.1f}%")
        print(f"Confidence: {signal_data['confidence'] * 100:.1f}%")
        print(f"Analyzed News: {signal_data['news_count']} articles")
        print(f"Timestamp: {signal_data['timestamp']}")
        print("-" * 60)

        if signal_data.get('top_news'):
            print("\nTop News:")
            for i, news in enumerate(signal_data['top_news'][:3], 1):
                sentiment_emoji = ""
                if news['sentiment'] == 'positive':
                    sentiment_emoji = "+"
                elif news['sentiment'] == 'negative':
                    sentiment_emoji = "-"
                else:
                    sentiment_emoji = "="

                print(f"  {i}. [{sentiment_emoji}] {news['title'][:50]}...")
                print(f"      Source: {news['source']}")

        print("=" * 60)
