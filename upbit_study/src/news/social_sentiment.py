"""
소셜 미디어 감정 분석 모듈
트위터/X, Reddit, CryptoPanic 등 소셜 데이터 수집 및 분석
"""
import os
import re
import requests
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

# 감정 분석용
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class SentimentResult:
    """감정 분석 결과"""
    score: float          # -1.0 ~ 1.0
    label: str            # 'positive', 'negative', 'neutral'
    confidence: float     # 0.0 ~ 1.0
    source: str           # 데이터 소스
    text: str             # 원본 텍스트
    timestamp: str        # 시간


class CryptoPanicCollector:
    """CryptoPanic API를 통한 암호화폐 뉴스/소셜 수집

    무료 API로 암호화폐 관련 뉴스와 소셜 미디어 데이터 수집
    https://cryptopanic.com/developers/api/
    """

    BASE_URL = "https://cryptopanic.com/api/v1/posts/"

    # 코인별 필터 매핑
    COIN_FILTERS = {
        'BTC': 'BTC',
        'ETH': 'ETH',
        'XRP': 'XRP',
        'SOL': 'SOL',
        'DOGE': 'DOGE',
        'ADA': 'ADA',
        'AVAX': 'AVAX',
        'DOT': 'DOT',
        'MATIC': 'MATIC',
        'LINK': 'LINK',
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: CryptoPanic API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv('CRYPTOPANIC_API_KEY', '')
        self._cache = {}
        self._cache_ttl = 300  # 5분 캐시

    def get_posts(
        self,
        currencies: str = 'BTC',
        filter_type: str = 'hot',  # 'rising', 'hot', 'bullish', 'bearish', 'important'
        kind: str = 'all',  # 'news', 'media', 'all'
        limit: int = 50
    ) -> List[Dict]:
        """암호화폐 관련 포스트 수집

        Args:
            currencies: 코인 심볼 (쉼표로 구분)
            filter_type: 필터 유형
            kind: 콘텐츠 유형
            limit: 최대 개수

        Returns:
            포스트 리스트
        """
        cache_key = f"{currencies}_{filter_type}_{kind}"

        # 캐시 확인
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                return cached_data

        params = {
            'auth_token': self.api_key,
            'currencies': currencies,
            'filter': filter_type,
            'kind': kind,
            'public': 'true'
        }

        # API 키가 없으면 public API 사용
        if not self.api_key:
            params.pop('auth_token', None)

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])[:limit]

                # 캐시 저장
                self._cache[cache_key] = (datetime.now(), results)

                return results
            else:
                print(f"[SOCIAL] CryptoPanic API 오류: {response.status_code}")
                return []

        except Exception as e:
            print(f"[SOCIAL] CryptoPanic 요청 실패: {e}")
            return []

    def get_sentiment_summary(self, currencies: str = 'BTC') -> Dict:
        """감정 요약 조회

        Args:
            currencies: 코인 심볼

        Returns:
            감정 요약 딕셔너리
        """
        # 각 필터별로 데이터 수집
        bullish_posts = self.get_posts(currencies, filter_type='bullish', limit=20)
        bearish_posts = self.get_posts(currencies, filter_type='bearish', limit=20)
        hot_posts = self.get_posts(currencies, filter_type='hot', limit=30)

        bullish_count = len(bullish_posts)
        bearish_count = len(bearish_posts)
        total_count = bullish_count + bearish_count

        if total_count > 0:
            bullish_ratio = bullish_count / total_count
        else:
            bullish_ratio = 0.5

        # 감정 점수 (-1 ~ 1)
        sentiment_score = (bullish_ratio - 0.5) * 2

        return {
            'sentiment_score': sentiment_score,
            'bullish_ratio': bullish_ratio,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'total_posts': len(hot_posts),
            'timestamp': datetime.now().isoformat()
        }


class TwitterSentimentAnalyzer:
    """트위터/X 감정 분석기

    Twitter API v2 또는 대안 소스 사용
    """

    # 암호화폐 관련 키워드
    CRYPTO_KEYWORDS = {
        'BTC': ['bitcoin', 'btc', '#bitcoin', '#btc', 'satoshi'],
        'ETH': ['ethereum', 'eth', '#ethereum', '#eth', 'vitalik'],
        'XRP': ['ripple', 'xrp', '#xrp', '#ripple'],
        'SOL': ['solana', 'sol', '#solana', '#sol'],
        'DOGE': ['dogecoin', 'doge', '#dogecoin', '#doge', 'elon'],
    }

    # 긍정/부정 키워드
    BULLISH_KEYWORDS = [
        'moon', 'bullish', 'pump', 'buy', 'long', 'breakout', 'rally',
        'ath', 'all time high', 'gains', 'profit', 'hodl', 'accumulate',
        'undervalued', 'bullrun', 'green', 'up', 'rise', 'soar'
    ]

    BEARISH_KEYWORDS = [
        'dump', 'bearish', 'sell', 'short', 'crash', 'drop', 'fall',
        'correction', 'fear', 'panic', 'loss', 'red', 'down', 'sink',
        'overvalued', 'bubble', 'scam', 'rug', 'rekt'
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Twitter API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv('TWITTER_BEARER_TOKEN', '')
        self.sentiment_pipeline = None

        # Transformers 감정 분석 모델 로드 (선택적)
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=-1  # CPU
                )
                print("[SOCIAL] Twitter 감정 분석 모델 로드 완료")
            except Exception as e:
                print(f"[SOCIAL] 감정 분석 모델 로드 실패: {e}")

    def analyze_text(self, text: str) -> SentimentResult:
        """텍스트 감정 분석

        Args:
            text: 분석할 텍스트

        Returns:
            감정 분석 결과
        """
        # 텍스트 정제
        clean_text = self._clean_text(text)

        # 1. Transformers 모델 사용 (가장 정확)
        if self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(clean_text[:512])[0]
                label = result['label'].lower()
                score = result['score']

                if 'positive' in label:
                    sentiment_score = score
                elif 'negative' in label:
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0

                return SentimentResult(
                    score=sentiment_score,
                    label=label,
                    confidence=score,
                    source='transformer',
                    text=text[:100],
                    timestamp=datetime.now().isoformat()
                )
            except Exception:
                pass

        # 2. TextBlob 사용 (폴백)
        if TextBlob:
            try:
                blob = TextBlob(clean_text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'

                return SentimentResult(
                    score=polarity,
                    label=label,
                    confidence=abs(polarity),
                    source='textblob',
                    text=text[:100],
                    timestamp=datetime.now().isoformat()
                )
            except Exception:
                pass

        # 3. 키워드 기반 분석 (최후 수단)
        return self._keyword_analysis(clean_text, text)

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        # URL 제거
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # 멘션 제거
        text = re.sub(r'@\w+', '', text)
        # 특수문자 정제 (해시태그는 유지)
        text = re.sub(r'[^\w\s#]', ' ', text)
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def _keyword_analysis(self, clean_text: str, original_text: str) -> SentimentResult:
        """키워드 기반 감정 분석"""
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in clean_text)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in clean_text)

        total = bullish_count + bearish_count
        if total == 0:
            return SentimentResult(
                score=0.0,
                label='neutral',
                confidence=0.3,
                source='keyword',
                text=original_text[:100],
                timestamp=datetime.now().isoformat()
            )

        score = (bullish_count - bearish_count) / total

        if score > 0.2:
            label = 'positive'
        elif score < -0.2:
            label = 'negative'
        else:
            label = 'neutral'

        return SentimentResult(
            score=score,
            label=label,
            confidence=min(total / 5, 1.0),
            source='keyword',
            text=original_text[:100],
            timestamp=datetime.now().isoformat()
        )

    def analyze_batch(self, texts: List[str]) -> Dict:
        """배치 감정 분석

        Args:
            texts: 텍스트 리스트

        Returns:
            분석 요약
        """
        if not texts:
            return {
                'average_score': 0.0,
                'positive_ratio': 0.5,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'count': 0
            }

        results = [self.analyze_text(text) for text in texts]

        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores)

        positive_count = sum(1 for r in results if r.label == 'positive')
        negative_count = sum(1 for r in results if r.label == 'negative')

        positive_ratio = positive_count / len(results)

        if avg_score > 0.15:
            label = 'bullish'
        elif avg_score < -0.15:
            label = 'bearish'
        else:
            label = 'neutral'

        return {
            'average_score': avg_score,
            'positive_ratio': positive_ratio,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': len(results) - positive_count - negative_count,
            'sentiment_label': label,
            'confidence': sum(r.confidence for r in results) / len(results),
            'count': len(results)
        }


class SocialSentimentAggregator:
    """소셜 미디어 감정 통합 분석기

    여러 소스의 감정 데이터를 통합하여 최종 점수 산출
    """

    def __init__(self):
        self.crypto_panic = CryptoPanicCollector()
        self.twitter_analyzer = TwitterSentimentAnalyzer()
        self._cache = {}
        self._cache_ttl = 300  # 5분

    def get_social_sentiment(self, market: str = 'KRW-BTC') -> Dict:
        """통합 소셜 감정 분석

        Args:
            market: 마켓 코드

        Returns:
            통합 감정 분석 결과
        """
        coin = market.replace('KRW-', '').replace('USDT-', '').upper()

        # 캐시 확인
        cache_key = f"social_{coin}"
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                return cached_data

        # 1. CryptoPanic 데이터
        panic_summary = self.crypto_panic.get_sentiment_summary(coin)
        panic_posts = self.crypto_panic.get_posts(coin, filter_type='hot', limit=30)

        # 2. 포스트 텍스트 감정 분석
        texts = [post.get('title', '') for post in panic_posts if post.get('title')]
        text_sentiment = self.twitter_analyzer.analyze_batch(texts)

        # 3. 통합 점수 계산 (가중 평균)
        # CryptoPanic 불리쉬/베어리쉬: 40%
        # 텍스트 감정 분석: 60%

        panic_score = panic_summary.get('sentiment_score', 0)
        text_score = text_sentiment.get('average_score', 0)

        combined_score = (panic_score * 0.4) + (text_score * 0.6)

        # 신뢰도 계산
        total_data_points = panic_summary.get('total_posts', 0) + text_sentiment.get('count', 0)
        confidence = min(total_data_points / 50, 1.0)  # 50개 이상이면 최대 신뢰도

        # 감정 라벨
        if combined_score > 0.2:
            sentiment_label = 'BULLISH'
        elif combined_score < -0.2:
            sentiment_label = 'BEARISH'
        else:
            sentiment_label = 'NEUTRAL'

        result = {
            'coin': coin,
            'combined_score': round(combined_score, 4),
            'sentiment_label': sentiment_label,
            'confidence': round(confidence, 4),
            'panic_data': {
                'score': panic_score,
                'bullish_ratio': panic_summary.get('bullish_ratio', 0.5),
                'bullish_count': panic_summary.get('bullish_count', 0),
                'bearish_count': panic_summary.get('bearish_count', 0)
            },
            'text_analysis': {
                'score': text_score,
                'positive_ratio': text_sentiment.get('positive_ratio', 0.5),
                'analyzed_count': text_sentiment.get('count', 0)
            },
            'timestamp': datetime.now().isoformat()
        }

        # 캐시 저장
        self._cache[cache_key] = (datetime.now(), result)

        return result

    def get_sentiment_for_lstm(self, market: str = 'KRW-BTC') -> Tuple[float, float]:
        """LSTM 입력용 감정 점수 반환

        Args:
            market: 마켓 코드

        Returns:
            (감정 점수, 신뢰도) - 둘 다 0~1 범위로 정규화
        """
        sentiment = self.get_social_sentiment(market)

        # -1~1 범위를 0~1로 정규화
        normalized_score = (sentiment['combined_score'] + 1) / 2
        confidence = sentiment['confidence']

        return normalized_score, confidence


# 싱글톤 인스턴스
_social_aggregator = None

def get_social_sentiment_aggregator() -> SocialSentimentAggregator:
    """소셜 감정 분석기 싱글톤 반환"""
    global _social_aggregator
    if _social_aggregator is None:
        _social_aggregator = SocialSentimentAggregator()
    return _social_aggregator
