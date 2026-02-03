"""
업비트 AI 자동 매매 시스템
메인 실행 파일
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from api.upbit_client import UpbitClient
from environment.trading_env import CryptoTradingEnv
from models.rl_agent import TradingAgent, TrainingCallback, evaluate_agent
from backtesting.backtest import Backtester
from bot.trading_bot import TradingBot


def train_model(market: str = 'KRW-BTC', timesteps: int = 100000):
    """AI 모델 학습

    Args:
        market: 거래 마켓
        timesteps: 학습 스텝 수
    """
    print(f"\n🎓 AI 모델 학습 시작 ({market})")

    # API 클라이언트 생성 (공개 API만 사용)
    client = UpbitClient("", "")

    # 과거 데이터 가져오기 (일봉 200일치)
    print("📊 데이터 수집 중...")
    candles = client.get_candles_day(market, count=200)

    # API 실패 체크
    if not candles:
        print("❌ 데이터 수집 실패: API 응답이 비어있습니다.")
        return None, None

    # 데이터프레임 변환
    df = pd.DataFrame(candles)

    # 필수 컬럼 확인
    required_columns = ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']
    if not all(col in df.columns for col in required_columns):
        print("❌ 데이터 수집 실패: 필수 컬럼이 없습니다.")
        return None, None

    df = df.rename(columns={
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    })
    df = df.sort_values('candle_date_time_kst').reset_index(drop=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # 최소 데이터 확인
    if len(df) < 100:
        print(f"❌ 데이터 부족: {len(df)}일치 (최소 100일 필요)")
        return None, None

    print(f"✅ 데이터 수집 완료: {len(df)}일치")

    # 학습/검증 데이터 분할
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    print(f"📚 학습 데이터: {len(train_df)}일")
    print(f"📝 검증 데이터: {len(test_df)}일")

    # 환경 생성
    train_env = CryptoTradingEnv(train_df, initial_balance=1000000)
    test_env = CryptoTradingEnv(test_df, initial_balance=1000000)

    # 에이전트 생성
    device = TradingAgent.get_device()
    print(f"🖥️  디바이스: {device.upper()}")

    agent = TradingAgent(
        env=train_env,
        model_name="crypto_trader",
        learning_rate=0.0003,
        device=device
    )

    # 학습
    callback = TrainingCallback(check_freq=5000)
    agent.train(total_timesteps=timesteps, callback=callback)

    # 모델 저장
    agent.save()

    # 평가
    print("\n📊 모델 평가 (검증 데이터)")
    results = evaluate_agent(agent, test_env, n_episodes=5)

    return agent, results


def run_backtest(market: str = 'KRW-BTC', model_path: str = 'models/crypto_trader'):
    """백테스팅 실행

    Args:
        market: 거래 마켓
        model_path: 모델 경로
    """
    print(f"\n🔬 백테스팅 시작 ({market})")

    # API 클라이언트 생성
    client = UpbitClient("", "")

    # 데이터 수집
    print("📊 데이터 수집 중...")
    candles = client.get_candles_day(market, count=200)

    # API 실패 체크
    if not candles:
        print("❌ 데이터 수집 실패: API 응답이 비어있습니다.")
        return None

    df = pd.DataFrame(candles)

    # 필수 컬럼 확인
    required_columns = ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']
    if not all(col in df.columns for col in required_columns):
        print("❌ 데이터 수집 실패: 필수 컬럼이 없습니다.")
        return None

    df = df.rename(columns={
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    })
    df = df.sort_values('candle_date_time_kst').reset_index(drop=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    # 최소 데이터 확인
    if len(df) < 100:
        print(f"❌ 데이터 부족: {len(df)}일치 (최소 100일 필요)")
        return None

    print(f"✅ 데이터 수집 완료: {len(df)}일치")

    # 환경 생성
    env = CryptoTradingEnv(df, initial_balance=1000000)

    # 모델 로드
    print("🤖 AI 모델 로딩...")
    try:
        agent = TradingAgent(env)
        agent.load(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {str(e)}")
        return None

    # 액션 예측
    print("🎯 트레이딩 시뮬레이션...")
    actions = []
    obs, _ = env.reset()

    # env.reset()은 current_step을 60으로 설정
    start_index = 60

    for _ in range(len(df) - start_index - 1):
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

    # 백테스트 - env와 동일한 인덱스 범위 사용
    # df[start_index:]를 사용하여 인덱스 일치
    backtest_df = df.iloc[start_index:].reset_index(drop=True)
    backtester = Backtester(backtest_df, initial_balance=1000000)
    results = backtester.run_backtest(actions)

    # 결과 출력
    backtester.print_results(results)

    # 차트 저장
    os.makedirs('logs', exist_ok=True)
    backtester.plot_results(save_path='logs/backtest_result.png')

    return results


def run_live_trading(market: str = 'KRW-BTC'):
    """실시간 자동 매매 실행

    Args:
        market: 거래 마켓
    """
    # API 키 확인
    access_key = os.getenv('UPBIT_ACCESS_KEY')
    secret_key = os.getenv('UPBIT_SECRET_KEY')

    if not access_key or not secret_key:
        print("❌ API 키가 설정되지 않았습니다!")
        print("   .env 파일에 UPBIT_ACCESS_KEY와 UPBIT_SECRET_KEY를 설정하세요.")
        return

    # 봇 생성
    bot = TradingBot(
        access_key=access_key,
        secret_key=secret_key,
        market=market,
        model_path='models/crypto_trader',
        interval=60,  # 1분마다 체크
        max_trade_amount=100000  # 최대 10만원
    )

    # 봇 시작
    bot.start(use_model=True)


def main():
    """메인 함수"""
    print("="*60)
    print("💰 업비트 AI 자동 매매 시스템")
    print("="*60)
    print()
    print("1. AI 모델 학습")
    print("2. 백테스팅")
    print("3. 실시간 자동 매매 (주의!)")
    print("4. 종료")
    print()

    choice = input("선택하세요 (1-4): ").strip()

    if choice == '1':
        market = input("마켓 코드 (기본: KRW-BTC): ").strip() or 'KRW-BTC'
        timesteps = input("학습 스텝 수 (기본: 100000): ").strip()
        timesteps = int(timesteps) if timesteps else 100000

        train_model(market, timesteps)

    elif choice == '2':
        market = input("마켓 코드 (기본: KRW-BTC): ").strip() or 'KRW-BTC'
        run_backtest(market)

    elif choice == '3':
        print("\n⚠️  경고: 실제 자금을 사용한 실시간 트레이딩입니다!")
        print("   .env 파일에 API 키가 설정되어 있는지 확인하세요.")
        confirm = input("\n계속하시겠습니까? (yes/no): ").strip().lower()

        if confirm == 'yes':
            market = input("마켓 코드 (기본: KRW-BTC): ").strip() or 'KRW-BTC'
            run_live_trading(market)
        else:
            print("취소되었습니다.")

    elif choice == '4':
        print("👋 프로그램을 종료합니다.")

    else:
        print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()
