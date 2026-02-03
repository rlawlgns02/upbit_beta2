"""
업비트 종목 추천 시스템
기술적 분석 + AI 예측으로 상승 예상 종목 추천
"""
import os
import sys
import pandas as pd
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api.upbit_client import UpbitClient
from analyzer.market_analyzer import MarketAnalyzer
from analyzer.ai_predictor import AIPredictor


def recommend_by_technical_analysis(top_n: int = 10):
    """기술적 분석 기반 종목 추천

    Args:
        top_n: 추천 종목 수

    Returns:
        분석 결과 리스트 또는 None (실패 시)
    """
    print("\n" + "="*60)
    print("📊 기술적 분석 기반 종목 추천")
    print("="*60)

    # API 클라이언트 생성
    client = UpbitClient("", "")
    analyzer = MarketAnalyzer(client)

    try:
        # 전체 시장 스캔
        results = analyzer.scan_all_markets(top_n=top_n, delay=0.1)

        # 결과 없음 체크
        if not results:
            print("❌ 분석 결과가 없습니다. API 연결을 확인해주세요.")
            return None

        # 결과 출력
        analyzer.print_ranking(results)

        # 상위 3개 상세 분석
        print("\n" + "="*60)
        print("🔍 상위 3개 종목 상세 분석")
        print("="*60)

        for i, result in enumerate(results[:3], 1):
            print(f"\n[{i}위]")
            analyzer.print_analysis(result)

        return results

    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {str(e)}")
        return None


def recommend_by_ai_prediction(top_n: int = 10, model_path: Optional[str] = None):
    """AI 예측 기반 종목 추천

    Args:
        top_n: 추천 종목 수
        model_path: 학습된 모델 경로

    Returns:
        예측 결과 리스트 또는 None (실패 시)
    """
    print("\n" + "="*60)
    print("🤖 AI 예측 기반 종목 추천")
    print("="*60)

    # API 클라이언트 생성
    client = UpbitClient("", "")
    analyzer = MarketAnalyzer(client)
    predictor = AIPredictor(model_path)

    try:
        # 전체 종목 조회
        markets = analyzer.get_all_krw_markets()

        # API 실패 체크
        if not markets:
            print("❌ 마켓 조회 실패: API 연결을 확인해주세요.")
            return None

        # 데이터 수집
        print("\n📊 시장 데이터 수집 중...")
        market_data = {}

        import time
        for i, market_info in enumerate(markets, 1):
            market = market_info['market']
            print(f"[{i}/{len(markets)}] {market} 데이터 수집...", end='\r')

            try:
                df = analyzer.get_market_data(market, days=30)
                if df is not None and len(df) >= 20:
                    market_data[market] = df
            except Exception as e:
                # 개별 종목 에러는 스킵
                continue

            # API 제한 방지
            time.sleep(0.1)

        print(f"\n✅ {len(market_data)} 종목 데이터 수집 완료")

        # 데이터 수집 실패 체크
        if not market_data:
            print("❌ 데이터 수집 실패: 분석할 수 있는 종목이 없습니다.")
            return None

        # AI 예측
        results = predictor.batch_predict(market_data, top_n=top_n)

        # 결과 출력
        if results:
            predictor.print_predictions_ranking(results)

            # 상위 3개 상세 예측
            print("\n" + "="*60)
            print("🔍 상위 3개 종목 AI 상세 예측")
            print("="*60)

            for i, result in enumerate(results[:3], 1):
                print(f"\n[{i}위]")
                predictor.print_prediction(result)
        else:
            print("❌ 매수 신호가 있는 종목이 없습니다.")

        return results

    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {str(e)}")
        return None


def search_and_predict(market: str, model_path: Optional[str] = None):
    """특정 종목 검색 및 예측

    Args:
        market: 마켓 코드 (예: 'KRW-BTC' 또는 'BTC')
        model_path: 학습된 모델 경로

    Returns:
        분석 결과 딕셔너리 또는 None (실패 시)
    """
    # 마켓 코드 정규화
    if not market.startswith('KRW-'):
        market = f'KRW-{market.upper()}'

    print("\n" + "="*60)
    print(f"🔍 {market} 종목 분석 및 예측")
    print("="*60)

    # API 클라이언트 생성
    client = UpbitClient("", "")
    analyzer = MarketAnalyzer(client)
    predictor = AIPredictor(model_path)

    try:
        # 1. 기술적 분석
        print("\n📊 1단계: 기술적 분석")
        print("-"*60)

        tech_result = analyzer.analyze_market(market, days=30)

        if tech_result is None:
            print(f"❌ {market} 종목을 찾을 수 없거나 데이터가 부족합니다.")
            return None

        analyzer.print_analysis(tech_result)

        # 2. AI 예측
        print("\n🤖 2단계: AI 예측")
        print("-"*60)

        ai_result = None
        df = analyzer.get_market_data(market, days=30)
        if df is not None and len(df) >= 20:
            try:
                ai_result = predictor.predict_market(df, market)
                predictor.print_prediction(ai_result)
            except Exception as e:
                print(f"⚠️  AI 예측 오류: {str(e)}")
                ai_result = None
        else:
            print("❌ AI 예측 실패: 데이터 부족")
            ai_result = None

        # 3. 종합 판단
        print("\n💡 3단계: 종합 판단")
        print("="*60)

        if tech_result and ai_result:
            tech_score = tech_result['score']
            tech_rec = tech_result['recommendation']
            ai_direction = ai_result['direction']
            ai_confidence = ai_result['confidence']

            print(f"기술적 분석: {tech_rec} (점수: {tech_score})")
            print(f"AI 예측:     {ai_direction} (신뢰도: {ai_confidence*100:.1f}%)")
            print()

            # 종합 판단
            if '매수' in tech_rec and ai_result['action'] == 1:
                if tech_score >= 5 and ai_confidence >= 0.7:
                    final = "🔥 강력 추천: 두 분석 모두 강한 매수 신호!"
                else:
                    final = "📈 추천: 두 분석 모두 매수 신호입니다."
            elif '매도' in tech_rec and ai_result['action'] == 2:
                final = "📉 주의: 두 분석 모두 매도/하락 신호입니다."
            elif tech_score > 0 and ai_result['action'] == 1:
                final = "💡 긍정적: 전반적으로 긍정적인 신호가 많습니다."
            elif tech_score < 0 and ai_result['action'] == 2:
                final = "⚠️  부정적: 전반적으로 부정적인 신호가 많습니다."
            else:
                final = "⏸️  중립: 분석 결과가 혼재되어 있습니다. 신중한 판단이 필요합니다."

            print(f"종합 판단: {final}")
        else:
            print("❌ 종합 판단 불가")

        print("="*60)

        return {
            'technical': tech_result,
            'ai': ai_result
        }

    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {str(e)}")
        return None


def main():
    """메인 함수"""
    print("="*60)
    print("🎯 업비트 종목 추천 시스템")
    print("="*60)
    print()
    print("1. 기술적 분석 기반 추천 (빠름)")
    print("2. AI 예측 기반 추천 (느림, 정확)")
    print("3. 특정 종목 검색 및 분석")
    print("4. 종료")
    print()

    choice = input("선택하세요 (1-4): ").strip()

    if choice == '1':
        top_n = input("추천 종목 수 (기본: 10): ").strip()
        top_n = int(top_n) if top_n else 10

        recommend_by_technical_analysis(top_n)

    elif choice == '2':
        top_n = input("추천 종목 수 (기본: 10): ").strip()
        top_n = int(top_n) if top_n else 10

        model_path = input("모델 경로 (기본: models/crypto_trader, 엔터=규칙기반): ").strip()
        if not model_path:
            model_path = None
        elif not os.path.exists(model_path + '.zip'):
            print(f"⚠️  모델 파일이 없습니다: {model_path}")
            print("⚠️  규칙 기반 예측으로 진행합니다.")
            model_path = None

        recommend_by_ai_prediction(top_n, model_path)

    elif choice == '3':
        market = input("종목 코드 (예: BTC, ETH, KRW-BTC): ").strip()

        if not market:
            print("❌ 종목 코드를 입력해주세요.")
            return

        model_path = input("모델 경로 (기본: models/crypto_trader, 엔터=규칙기반): ").strip()
        if not model_path:
            model_path = None
        elif not os.path.exists(model_path + '.zip'):
            print(f"⚠️  모델 파일이 없습니다: {model_path}")
            print("⚠️  규칙 기반 예측으로 진행합니다.")
            model_path = None

        search_and_predict(market, model_path)

    elif choice == '4':
        print("👋 프로그램을 종료합니다.")

    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
