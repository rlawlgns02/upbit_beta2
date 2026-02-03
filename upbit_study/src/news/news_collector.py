"""
NewsAPI를 사용한 암호화폐 뉴스 수집기
"""
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()


class NewsCollector:
    """NewsAPI를 사용하여 암호화폐 관련 뉴스를 수집하는 클래스"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: NewsAPI 키. None이면 환경변수에서 로드
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.newsapi = None
        self._init_client()

    def _init_client(self):
        """NewsAPI 클라이언트 초기화"""
        if not self.api_key:
            print("[WARNING] NEWS_API_KEY가 설정되지 않았습니다.")
            return

        try:
            from newsapi import NewsApiClient
            self.newsapi = NewsApiClient(api_key=self.api_key)
            print("[INFO] NewsAPI 클라이언트 초기화 완료")
        except ImportError:
            print("[WARNING] newsapi-python 패키지를 설치하세요: pip install newsapi-python")
        except Exception as e:
            print(f"[ERROR] NewsAPI 초기화 실패: {str(e)}")

    def get_crypto_news(
        self,
        query: str = "cryptocurrency",
        language: str = "en",
        page_size: int = 20,
        days_back: int = 7
    ) -> List[Dict]:
        """암호화폐 관련 뉴스 수집

        Args:
            query: 검색 쿼리 (기본: "cryptocurrency")
            language: 언어 코드 (기본: "en")
            page_size: 가져올 뉴스 개수 (기본: 20)
            days_back: 며칠 전까지의 뉴스를 가져올지 (기본: 7)

        Returns:
            뉴스 기사 리스트
        """
        if not self.newsapi:
            return self._get_fallback_news()

        try:
            # 날짜 범위 설정
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)

            # NewsAPI 호출
            response = self.newsapi.get_everything(
                q=query,
                language=language,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                sort_by='publishedAt',
                page_size=page_size
            )

            if response.get('status') != 'ok':
                print(f"[WARNING] NewsAPI 응답 오류: {response.get('message', 'Unknown error')}")
                return self._get_fallback_news()

            articles = response.get('articles', [])

            # 결과 정리
            news_list = []
            for article in articles:
                news_item = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', 'Unknown')
                }
                news_list.append(news_item)

            print(f"[INFO] {len(news_list)}개의 뉴스 기사 수집 완료")
            return news_list

        except Exception as e:
            print(f"[ERROR] 뉴스 수집 실패: {str(e)}")
            return self._get_fallback_news()

    def get_bitcoin_news(self, page_size: int = 20) -> List[Dict]:
        """비트코인 관련 뉴스 수집"""
        return self.get_crypto_news(query="bitcoin OR BTC", page_size=page_size)

    def get_ethereum_news(self, page_size: int = 20) -> List[Dict]:
        """이더리움 관련 뉴스 수집"""
        return self.get_crypto_news(query="ethereum OR ETH", page_size=page_size)

    def get_altcoin_news(self, page_size: int = 20) -> List[Dict]:
        """알트코인 관련 뉴스 수집"""
        return self.get_crypto_news(
            query="altcoin OR solana OR ripple OR XRP OR dogecoin",
            page_size=page_size
        )

    def get_market_news(self, page_size: int = 20) -> List[Dict]:
        """암호화폐 시장 전반 뉴스 수집"""
        return self.get_crypto_news(
            query="crypto market OR cryptocurrency trading OR crypto regulation",
            page_size=page_size
        )

    def _get_fallback_news(self) -> List[Dict]:
        """API 실패 시 반환할 기본 뉴스 (빈 리스트)"""
        print("[INFO] 뉴스 API 사용 불가 - 기본값 반환")
        return []

    def get_headlines(self, category: str = "business", country: str = "us") -> List[Dict]:
        """최신 헤드라인 뉴스 수집 (암호화폐 관련 필터링)

        Args:
            category: 카테고리 (business, technology 등)
            country: 국가 코드

        Returns:
            뉴스 헤드라인 리스트
        """
        if not self.newsapi:
            return self._get_fallback_news()

        try:
            response = self.newsapi.get_top_headlines(
                category=category,
                country=country,
                page_size=100
            )

            if response.get('status') != 'ok':
                return self._get_fallback_news()

            articles = response.get('articles', [])

            # 암호화폐 관련 키워드 필터링
            crypto_keywords = [
                'bitcoin', 'btc', 'ethereum', 'eth', 'crypto',
                'cryptocurrency', 'blockchain', 'defi', 'nft',
                'altcoin', 'ripple', 'xrp', 'solana', 'dogecoin'
            ]

            crypto_news = []
            for article in articles:
                title = (article.get('title', '') or '').lower()
                description = (article.get('description', '') or '').lower()

                # 키워드 매칭
                if any(keyword in title or keyword in description for keyword in crypto_keywords):
                    news_item = {
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', '')
                    }
                    crypto_news.append(news_item)

            return crypto_news

        except Exception as e:
            print(f"[ERROR] 헤드라인 수집 실패: {str(e)}")
            return self._get_fallback_news()
