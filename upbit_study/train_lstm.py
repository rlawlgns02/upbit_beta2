"""
LSTM 가격 예측 모델 학습 스크립트
업비트 API에서 실시간 데이터 수집 후 학습
"""
import os
import sys
import pandas as pd
from datetime import datetime

# 프로젝트 경로 추가 (우선순위로 삽입)
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    # insert at front so local 'api'/'models' packages are found before system packages
    sys.path.insert(0, src_path)

# 임포트 시 더 친절한 오류 메시지를 제공하고, 폴백으로 'src.' 접두사도 시도
try:
    from api.upbit_client import UpbitClient
except Exception:
    try:
        from src.api.upbit_client import UpbitClient
    except Exception as e:
        print(f"[IMPORT ERROR] 'api.upbit_client' 모듈을 찾을 수 없습니다: {e}")
        print("실행 방법: 프로젝트 루트에서 'python train_lstm.py' 명령으로 실행하거나, pip로 패키지를 설치하세요.")
        raise

try:
    from models.lstm_predictor import LSTMPredictor, get_device
except Exception:
    try:
        from src.models.lstm_predictor import LSTMPredictor, get_device
    except Exception as e:
        print(f"[IMPORT ERROR] 'models.lstm_predictor' 모듈을 찾을 수 없습니다: {e}")
        print("실행 방법: 프로젝트 루트에서 'python train_lstm.py' 명령으로 실행하거나, pip로 패키지를 설치하세요.")
        raise


def collect_training_data(
    market: str = 'KRW-BTC',
    days: int = 200,
    use_minute: bool = False
) -> pd.DataFrame:
    """학습용 데이터 수집

    Args:
        market: 마켓 코드
        days: 수집할 일수 (일봉 기준)
        use_minute: 분봉 사용 여부

    Returns:
        OHLCV 데이터프레임
    """
    print(f"\n[DATA] {market} 데이터 수집 중...")
    client = UpbitClient("", "")

    if use_minute:
        # 분봉 데이터 (최대 200개씩)
        candles = client.get_candles_minute(market, unit=60, count=200)
    else:
        # 일봉 데이터
        candles = client.get_candles_day(market, count=days)

    if not candles:
        raise ValueError("데이터 수집 실패")

    df = pd.DataFrame(candles)
    df = df.rename(columns={
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    })

    # 시간순 정렬
    df = df.sort_values('candle_date_time_kst').reset_index(drop=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"[DATA] 수집 완료: {len(df)}개 데이터")
    print(f"[DATA] 기간: {df.index[0]} ~ {df.index[-1]}")
    print(f"[DATA] 가격 범위: {df['close'].min():,.0f} ~ {df['close'].max():,.0f}")

    return df


def train_lstm_model(
    market: str = 'KRW-BTC',
    days: int = 200,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    model_name: str = None
):
    """LSTM 모델 학습

    Args:
        market: 마켓 코드
        days: 데이터 일수
        epochs: 에폭 수
        batch_size: 배치 크기
        learning_rate: 학습률
        model_name: 모델 저장 이름
    """
    print("\n" + "=" * 60)
    print("LSTM 가격 예측 모델 학습")
    print("=" * 60)

    # 모델 이름 설정
    if model_name is None:
        coin = market.replace('KRW-', '').lower()
        model_name = f"models/lstm_{coin}"

    # 데이터 수집
    df = collect_training_data(market, days)

    if len(df) < 100:
        print(f"[ERROR] 데이터가 부족합니다. 최소 100개 필요 (현재: {len(df)})")
        return None

    # 디바이스 확인
    device = get_device()
    print(f"[TRAIN] 디바이스: {device.upper()}")

    # 예측기 생성
    predictor = LSTMPredictor(
        model_path=model_name,
        seq_length=60,
        device=device
    )

    # 학습
    print(f"\n[TRAIN] 학습 시작...")
    print(f"  - 에폭: {epochs}")
    print(f"  - 배치 크기: {batch_size}")
    print(f"  - 학습률: {learning_rate}")

    try:
        results = predictor.train(
            df=df,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=0.2,
            early_stopping_patience=15,
            verbose=True
        )

        # 모델 저장
        predictor.save()

        # 결과 출력
        print("\n" + "=" * 60)
        print("학습 결과")
        print("=" * 60)
        print(f"  - 학습 에폭: {results['epochs_trained']}")
        print(f"  - 최적 검증 손실: {results['best_val_loss']:.6f}")
        print(f"  - 모델 저장: {model_name}.pt")

        # 테스트 예측
        print("\n[TEST] 테스트 예측...")
        try:
            pred_price, change_rate, direction = predictor.predict(df)
            current_price = df['close'].iloc[-1]

            print(f"  - 현재가: {current_price:,.0f} KRW")
            print(f"  - 예측가: {pred_price:,.0f} KRW")
            print(f"  - 변화율: {change_rate:+.2f}%")
            print(f"  - 방향: {direction}")
        except Exception as e:
            print(f"  - 테스트 예측 실패: {e}")

        print("=" * 60)
        return predictor

    except Exception as e:
        print(f"[ERROR] 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_multiple_coins(
    coins: list = None,
    days: int = 200,
    epochs: int = 100
):
    """여러 코인 모델 학습

    Args:
        coins: 코인 리스트 (None이면 주요 코인)
        days: 데이터 일수
        epochs: 에폭 수
    """
    if coins is None:
        coins = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE']

    print("\n" + "=" * 60)
    print(f"다중 코인 LSTM 모델 학습 ({len(coins)}개)")
    print("=" * 60)

    results = {}
    for i, market in enumerate(coins, 1):
        print(f"\n[{i}/{len(coins)}] {market} 학습 중...")
        try:
            predictor = train_lstm_model(
                market=market,
                days=days,
                epochs=epochs
            )
            results[market] = 'SUCCESS' if predictor else 'FAILED'
        except Exception as e:
            print(f"[ERROR] {market} 학습 실패: {e}")
            results[market] = f'ERROR: {str(e)}'

    # 결과 요약
    print("\n" + "=" * 60)
    print("학습 결과 요약")
    print("=" * 60)
    for market, status in results.items():
        emoji = "" if status == 'SUCCESS' else ""
        print(f"  {emoji} {market}: {status}")

    return results


def main():
    """메인 함수"""
    print("=" * 60)
    print("LSTM 가격 예측 모델 학습")
    print("=" * 60)
    print()
    print("1. 단일 코인 학습 (BTC)")
    print("2. 특정 코인 학습")
    print("3. 주요 코인 전체 학습 (BTC, ETH, XRP, SOL, DOGE)")
    print("4. 종료")
    print()

    choice = input("선택하세요 (1-4): ").strip()

    if choice == '1':
        train_lstm_model(market='KRW-BTC', days=200, epochs=100)

    elif choice == '2':
        market = input("마켓 코드 (예: KRW-ETH): ").strip().upper()
        if not market.startswith('KRW-'):
            market = f'KRW-{market}'

        days = input("데이터 일수 (기본: 200): ").strip()
        days = int(days) if days else 200

        epochs = input("에폭 수 (기본: 100): ").strip()
        epochs = int(epochs) if epochs else 100

        train_lstm_model(market=market, days=days, epochs=epochs)

    elif choice == '3':
        epochs = input("에폭 수 (기본: 100): ").strip()
        epochs = int(epochs) if epochs else 100

        train_multiple_coins(epochs=epochs)

    elif choice == '4':
        print("프로그램을 종료합니다.")

    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
