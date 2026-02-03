"""
TextBlob을 사용한 뉴스 감정 분석기
"""
from typing import List, Dict, Tuple, Optional


class SentimentAnalyzer:
    """TextBlob을 사용하여 뉴스 텍스트의 감정을 분석하는 클래스"""

    def __init__(self):
        """감정 분석기 초기화"""
        self.textblob = None
        self._init_textblob()

    def _init_textblob(self):
        """TextBlob 라이브러리 초기화"""
        try:
            from textblob import TextBlob
            self.textblob = TextBlob
            print("[INFO] TextBlob 초기화 완료")
        except ImportError:
            print("[WARNING] textblob 패키지를 설치하세요: pip install textblob")
            print("[WARNING] 설치 후 'python -m textblob.download_corpora' 실행 필요")

    def analyze_text(self, text: str) -> Dict:
        """단일 텍스트 감정 분석

        Args:
            text: 분석할 텍스트

        Returns:
            감정 분석 결과 딕셔너리
            - polarity: 감정 극성 (-1.0 ~ 1.0, 음수=부정, 양수=긍정)
            - subjectivity: 주관성 (0.0 ~ 1.0, 0=객관적, 1=주관적)
            - sentiment: 감정 레이블 (positive, negative, neutral)
        """
        if not self.textblob or not text:
            return self._get_neutral_result()

        try:
            blob = self.textblob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # 감정 레이블 결정
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'text_length': len(text)
            }

        except Exception as e:
            print(f"[ERROR] 텍스트 분석 실패: {str(e)}")
            return self._get_neutral_result()

    def analyze_news_article(self, article: Dict) -> Dict:
        """뉴스 기사 감정 분석

        Args:
            article: 뉴스 기사 딕셔너리 (title, description, content 포함)

        Returns:
            감정 분석 결과
        """
        # 제목과 설명을 합쳐서 분석 (제목에 더 높은 가중치)
        title = article.get('title', '') or ''
        description = article.get('description', '') or ''

        # 제목 분석 (가중치 2)
        title_result = self.analyze_text(title)

        # 설명 분석 (가중치 1)
        desc_result = self.analyze_text(description)

        # 가중 평균 계산
        if title_result['text_length'] > 0 or desc_result['text_length'] > 0:
            total_weight = 2 + 1  # 제목 2, 설명 1
            weighted_polarity = (
                title_result['polarity'] * 2 +
                desc_result['polarity'] * 1
            ) / total_weight
            weighted_subjectivity = (
                title_result['subjectivity'] * 2 +
                desc_result['subjectivity'] * 1
            ) / total_weight
        else:
            weighted_polarity = 0.0
            weighted_subjectivity = 0.5

        # 감정 레이블 결정
        if weighted_polarity > 0.1:
            sentiment = "positive"
        elif weighted_polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            'title': title,
            'source': article.get('source', 'Unknown'),
            'url': article.get('url', ''),
            'polarity': weighted_polarity,
            'subjectivity': weighted_subjectivity,
            'sentiment': sentiment,
            'published_at': article.get('published_at', '')
        }

    def analyze_news_batch(self, articles: List[Dict]) -> Dict:
        """여러 뉴스 기사 일괄 분석

        Args:
            articles: 뉴스 기사 리스트

        Returns:
            종합 분석 결과
            - articles: 개별 기사 분석 결과
            - summary: 종합 요약 (평균 극성, 긍정 비율 등)
        """
        if not articles:
            return {
                'articles': [],
                'summary': self._get_empty_summary()
            }

        analyzed_articles = []
        total_polarity = 0.0
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            result = self.analyze_news_article(article)
            analyzed_articles.append(result)

            total_polarity += result['polarity']

            if result['sentiment'] == 'positive':
                positive_count += 1
            elif result['sentiment'] == 'negative':
                negative_count += 1
            else:
                neutral_count += 1

        total_articles = len(analyzed_articles)
        avg_polarity = total_polarity / total_articles if total_articles > 0 else 0.0

        # 긍정 비율 계산 (0.0 ~ 1.0)
        positive_ratio = positive_count / total_articles if total_articles > 0 else 0.5

        # 전체 감정 레이블
        if avg_polarity > 0.1:
            overall_sentiment = "positive"
        elif avg_polarity < -0.1:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"

        summary = {
            'total_articles': total_articles,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_ratio': round(positive_ratio, 4),
            'average_polarity': round(avg_polarity, 4),
            'overall_sentiment': overall_sentiment
        }

        return {
            'articles': analyzed_articles,
            'summary': summary
        }

    def get_sentiment_score(self, articles: List[Dict]) -> Tuple[float, str]:
        """뉴스 기사들의 감정 점수 반환

        Args:
            articles: 뉴스 기사 리스트

        Returns:
            (positive_ratio, sentiment_label) 튜플
            - positive_ratio: 긍정 비율 (0.0 ~ 1.0)
            - sentiment_label: 전체 감정 레이블
        """
        result = self.analyze_news_batch(articles)
        summary = result['summary']

        return (
            summary['positive_ratio'],
            summary['overall_sentiment']
        )

    def _get_neutral_result(self) -> Dict:
        """중립 결과 반환 (분석 실패 시)"""
        return {
            'polarity': 0.0,
            'subjectivity': 0.5,
            'sentiment': 'neutral',
            'text_length': 0
        }

    def _get_empty_summary(self) -> Dict:
        """빈 요약 결과 반환"""
        return {
            'total_articles': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_ratio': 0.5,
            'average_polarity': 0.0,
            'overall_sentiment': 'neutral'
        }
