# 업비트 AI 자동 매매 시스템 - 코드 검수 보고서

> **검수일자**: 2026-01-31
> **검수자**: Claude Code
> **대상 프로젝트**: upbit_study

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [심각한 버그](#2-심각한-버그-즉시-수정-필요)
3. [논리적 오류](#3-논리적-오류)
4. [보안 취약점](#4-보안-취약점)
5. [성능 문제](#5-성능-문제)
6. [코드 품질 개선 권장](#6-코드-품질-개선-권장)
7. [잠재적 버그](#7-잠재적-버그)
8. [검수 요약](#8-검수-요약)
9. [권장 조치사항](#9-권장-조치사항)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 구조

```
upbit_study/
├── src/
│   ├── api/
│   │   └── upbit_client.py        # 업비트 API 클라이언트
│   ├── analyzer/
│   │   ├── market_analyzer.py     # 기술적 분석
│   │   └── ai_predictor.py        # AI 예측
│   ├── models/
│   │   └── rl_agent.py            # PPO 강화학습 에이전트
│   ├── environment/
│   │   └── trading_env.py         # 강화학습 환경
│   ├── backtesting/
│   │   └── backtest.py            # 백테스팅
│   ├── bot/
│   │   └── trading_bot.py         # 자동 매매 봇
│   └── news/
│       ├── news_collector.py      # 뉴스 수집
│       ├── sentiment_analyzer.py  # 감정 분석
│       └── signal_generator.py    # 신호 생성
├── main.py                        # 메인 실행 파일
├── recommend.py                   # 종목 추천 시스템
└── webapp.py                      # 웹 UI 애플리케이션
```

### 1.2 기술 스택

| 구분 | 기술 |
|------|------|
| 강화학습 | PPO (stable-baselines3) |
| 딥러닝 | PyTorch |
| 기술적 분석 | ta (Technical Analysis) |
| 웹 프레임워크 | FastAPI |
| 감정 분석 | TextBlob |

---

## 2. 심각한 버그 (즉시 수정 필요)

### 2.1 매수 로직 이중 감액

**파일**: `src/environment/trading_env.py:188-197`

```python
# 현재 코드 (문제)
buy_amount = self.balance * 0.95  # 5% 제외
cost = buy_amount * (1 + self.commission)  # 수수료 추가

if cost <= self.balance:
    self.crypto_held += buy_amount / current_price
    self.balance -= cost
```

**문제점**:
- 잔액의 95%를 매수 금액으로 설정한 후, 수수료를 추가로 적용
- 실제로는 약 95.05%가 빠져나가야 하지만 로직상 잔액이 남게 됨
- 학습과 실제 거래 결과가 왜곡됨

**수정 권장**:
```python
# 수수료 포함하여 최대 매수 가능 금액 계산
max_buy = self.balance / (1 + self.commission) * 0.95
cost = max_buy * (1 + self.commission)

if cost <= self.balance:
    self.crypto_held += max_buy / current_price
    self.balance -= cost
```

---

### 2.2 API 응답 타입 불일치

**파일**: `src/api/upbit_client.py:137-146`

```python
def get_accounts(self) -> List[Dict]:
    ...
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return response.json()  # List[Dict] 반환
    except Exception as e:
        return {'error': {'message': str(e)}}  # Dict 반환 (불일치!)
```

**문제점**:
- 정상 응답: `List[Dict]`
- 에러 응답: `Dict`
- 호출부에서 타입 확인 없이 리스트로 처리하면 런타임 에러 발생

**영향받는 코드**:
- `get_balance()` 메서드에서 `for account in accounts:` 순회 시 에러
- 다행히 `get_balance()`에는 타입 체크가 있음

**수정 권장**:
```python
except Exception as e:
    print(f"[UPBIT] 계좌 조회 예외: {e}")
    return []  # 빈 리스트 반환으로 통일
```

---

### 2.3 보상 계산 시점 오류

**파일**: `src/environment/trading_env.py:209-216`

```python
# 다음 스텝으로 이동
self.current_step += 1

# 현재 총 자산 (문제: 이전 스텝의 가격 사용)
current_net_worth = self.balance + self.crypto_held * current_price

# 보상 계산
reward = (current_net_worth - prev_net_worth) / prev_net_worth
```

**문제점**:
- 스텝을 이동한 후에도 이전 가격(`current_price`)으로 자산 계산
- 실제로는 다음 스텝의 가격으로 계산해야 정확한 보상

**수정 권장**:
```python
# 다음 스텝으로 이동
self.current_step += 1

# 다음 스텝 가격으로 자산 계산
next_price = self.df.iloc[self.current_step]['close']
current_net_worth = self.balance + self.crypto_held * next_price
```

---

## 3. 논리적 오류

### 3.1 매수/매도 임계값 불균형

**파일**: `src/analyzer/ai_predictor.py:164-169`

```python
if buy_score > sell_score and buy_score >= 2:  # 매수: 2점 이상
    return 1, buy_score / (total_score + 1)
elif sell_score > buy_score and sell_score >= 4:  # 매도: 4점 이상
    return 2, sell_score / (total_score + 1)
```

**문제점**:
- 매수 임계값(2)이 매도 임계값(4)의 절반
- 매수 신호가 매도 신호보다 2배 자주 발생
- 시장 상황과 관계없이 매수 편향 발생

**권장 수정**:
```python
THRESHOLD = 3  # 동일한 임계값 사용
if buy_score > sell_score and buy_score >= THRESHOLD:
    return 1, buy_score / (total_score + 1)
elif sell_score > buy_score and sell_score >= THRESHOLD:
    return 2, sell_score / (total_score + 1)
```

---

### 3.2 백테스트 승률 계산 오류

**파일**: `src/backtesting/backtest.py:109-115`

```python
# 승률 계산 (문제)
profitable_trades = 0
if len(buy_trades) > 0 and len(sell_trades) > 0:
    for i in range(min(len(buy_trades), len(sell_trades))):
        if sell_trades[i]['price'] > buy_trades[i]['price']:
            profitable_trades += 1
```

**문제점**:
1. 수수료를 고려하지 않음 (수수료 포함 시 손실인 거래도 수익으로 계산)
2. 매수/매도 순서가 맞지 않을 수 있음 (매도가 먼저 발생한 경우)
3. 인덱스로 단순 매칭하여 실제 쌍이 맞지 않을 수 있음

**권장 수정**:
```python
# 실제 거래 쌍 매칭 후 수수료 포함 계산
profitable_trades = 0
buy_idx = 0
for sell in sell_trades:
    if buy_idx < len(buy_trades):
        buy = buy_trades[buy_idx]
        # 수수료 포함 수익 계산
        buy_cost = buy['price'] * (1 + self.commission)
        sell_revenue = sell['price'] * (1 - self.commission)
        if sell_revenue > buy_cost:
            profitable_trades += 1
        buy_idx += 1
```

---

### 3.3 미사용 파라미터

**파일**: `src/news/signal_generator.py:200`

```python
def _determine_signal(self, positive_ratio: float, news_count: int = 10) -> tuple:
    # news_count 파라미터가 함수 내에서 사용되지 않음
    base_confidence = min(0.5, news_count / 10)  # 유일한 사용처
    ...
```

**문제점**:
- `news_count`가 기본값 10으로만 사용됨
- `generate_signal()` 호출 시 실제 뉴스 수가 전달되지 않음

**수정 권장**:
```python
# signal_generator.py:117
signal, confidence = self._determine_signal(
    positive_ratio,
    news_count=summary['total_articles']  # 실제 뉴스 수 전달
)
```

---

### 3.4 7일 변동률 인덱스 오류 가능성

**파일**: `src/analyzer/market_analyzer.py:109`

```python
price_change_7d = (df.iloc[-1]['close'] - df.iloc[-8]['close']) / df.iloc[-8]['close'] * 100 if len(df) >= 8 else 0
```

**문제점**:
- `df.iloc[-8]`은 현재(-1)로부터 7개 이전 = 8번째 전 데이터
- 실제로 7일 전 데이터를 원한다면 `df.iloc[-8]`이 맞음 (0-indexed)
- 그러나 일봉 데이터에서 주말 제외 등으로 정확히 7일이 아닐 수 있음

---

## 4. 보안 취약점

### 4.1 CORS 전체 허용

**파일**: `webapp.py:56-62`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**위험도**: 🔴 높음

**위협**:
- 악성 웹사이트에서 사용자의 API를 호출할 수 있음
- CSRF(Cross-Site Request Forgery) 공격에 취약
- 자동매매 명령이 외부에서 실행될 수 있음

**수정 권장**:
```python
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # 실제 서비스 도메인 추가
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

### 4.2 WebSocket 인증 부재

**파일**: `webapp.py:239-256`

```python
class ConnectionManager:
    async def connect(self, websocket: WebSocket):
        await websocket.accept()  # 인증 없이 바로 수락
        self.active_connections.append(websocket)
```

**위험도**: 🔴 높음

**위협**:
- 인증 없이 누구나 WebSocket 연결 가능
- 자동매매 시작/중지 명령을 외부에서 보낼 수 있음
- 실시간 거래 데이터가 노출됨

**수정 권장**:
```python
async def connect(self, websocket: WebSocket, token: str = None):
    # 토큰 검증
    if not self._verify_token(token):
        await websocket.close(code=4001)
        return False

    await websocket.accept()
    self.active_connections.append(websocket)
    return True
```

---

### 4.3 API 키 검증 미흡

**파일**: `webapp.py:78-79`

```python
def is_valid_api_key(key: str) -> bool:
    return key and key not in ['', 'your_access_key_here', 'your_secret_key_here']
```

**위험도**: 🟠 중간

**문제점**:
- 플레이스홀더만 체크하고 실제 키 형식 검증 없음
- 잘못된 형식의 키가 저장될 수 있음

**수정 권장**:
```python
import re

def is_valid_api_key(key: str) -> bool:
    if not key:
        return False
    # 플레이스홀더 체크
    placeholders = ['', 'your_access_key_here', 'your_secret_key_here']
    if key in placeholders:
        return False
    # 업비트 API 키 형식 검증 (예: UUID 형식)
    uuid_pattern = r'^[a-zA-Z0-9-]{36,}$'
    return bool(re.match(uuid_pattern, key))
```

---

### 4.4 .gitignore 확인 필요

**확인 사항**:
- `.env` 파일이 `.gitignore`에 포함되어 있는지 확인 필요
- API 키가 Git 히스토리에 노출되지 않도록 주의

---

## 5. 성능 문제

### 5.1 매번 환경 재생성

**파일**: `src/bot/trading_bot.py:56-70`

```python
def _prepare_observation(self, candles: pd.DataFrame) -> np.ndarray:
    # 매 호출마다 새 환경 생성 (비효율적)
    env = CryptoTradingEnv(candles, initial_balance=1000000)
    env.reset()
    return env._get_observation()
```

**영향**:
- API 호출(60초)마다 환경 객체 생성
- 기술적 지표 재계산 (약 20개 지표)
- 불필요한 CPU/메모리 사용

**수정 권장**:
```python
class TradingBot:
    def __init__(self, ...):
        ...
        self._cached_env = None

    def _prepare_observation(self, candles: pd.DataFrame) -> np.ndarray:
        # 환경을 캐시하고 데이터만 업데이트
        if self._cached_env is None:
            self._cached_env = CryptoTradingEnv(candles, initial_balance=1000000)
        else:
            self._cached_env.df = candles
            self._cached_env._add_technical_indicators()

        self._cached_env.reset()
        return self._cached_env._get_observation()
```

---

### 5.2 동기 API 호출

**파일**: `src/analyzer/market_analyzer.py:449-461`

```python
for i, market_info in enumerate(markets, 1):
    market = market_info['market']
    result = self.analyze_market(market, days=30)
    if result:
        results.append(result)
    time.sleep(delay)  # 순차 처리
```

**영향**:
- 전체 KRW 마켓 약 150개 종목
- 종목당 0.1초 대기 = 최소 15초 소요
- 실제로는 API 응답 시간 포함 수십 초 소요

**수정 권장** (비동기 처리):
```python
import asyncio
import aiohttp

async def scan_all_markets_async(self, top_n: int = 10):
    markets = self.get_all_krw_markets()

    async def analyze_one(market_info):
        market = market_info['market']
        return await self.analyze_market_async(market, days=30)

    # 동시에 10개씩 처리
    semaphore = asyncio.Semaphore(10)
    async with semaphore:
        tasks = [analyze_one(m) for m in markets]
        results = await asyncio.gather(*tasks)

    return sorted([r for r in results if r], key=lambda x: x['score'], reverse=True)[:top_n]
```

---

### 5.3 비효율적 신뢰도 계산

**파일**: `src/analyzer/ai_predictor.py:84-97`

```python
# 10번 반복 예측으로 신뢰도 계산
predictions = []
for _ in range(10):
    action, _ = self.agent.predict(obs, deterministic=False)
    predictions.append(int(action))

action_counts = np.bincount(predictions, minlength=3)
```

**영향**:
- 같은 관측값으로 10번 예측 수행
- 모델 추론 시간 10배 증가
- 결과의 일관성 부족 (non-deterministic)

**대안**:
```python
# 방법 1: 확정적 예측 + 모델 출력 확률 사용
action, _ = self.agent.predict(obs, deterministic=True)
# stable-baselines3는 action_probability 메서드 제공

# 방법 2: 단순화
action, _ = self.agent.predict(obs, deterministic=True)
confidence = 0.7  # 기본 신뢰도 또는 별도 신뢰도 모델 사용
```

---

## 6. 코드 품질 개선 권장

### 6.1 로깅 시스템 도입

**현재 상태**:
```python
print(f"[UPBIT] 마켓 조회 실패: {result}")
print(f"[WARNING] AI 모델 로드 실패: {str(e)}")
```

**권장 수정**:
```python
import logging

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 사용
logger.error(f"마켓 조회 실패: {result}")
logger.warning(f"AI 모델 로드 실패: {e}")
```

---

### 6.2 설정 파일 분리

**현재 상태** (하드코딩):
```python
# trading_env.py
commission = 0.0005
initial_balance = 1000000

# trading_bot.py
interval = 60
max_trade_amount = 100000
```

**권장 구조**:
```python
# config.py
from dataclasses import dataclass

@dataclass
class TradingConfig:
    commission: float = 0.0005
    initial_balance: float = 1000000
    interval: int = 60
    max_trade_amount: float = 100000

    # 강화학습 설정
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64

config = TradingConfig()

# 사용
from config import config
env = CryptoTradingEnv(df, initial_balance=config.initial_balance)
```

---

### 6.3 테스트 코드 추가

**권장 테스트 구조**:
```
tests/
├── test_upbit_client.py
├── test_market_analyzer.py
├── test_trading_env.py
├── test_backtest.py
└── conftest.py
```

**예시 테스트**:
```python
# tests/test_trading_env.py
import pytest
import pandas as pd
from src.environment.trading_env import CryptoTradingEnv

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'open': [100] * 100,
        'high': [110] * 100,
        'low': [90] * 100,
        'close': [105] * 100,
        'volume': [1000] * 100
    })

def test_env_reset(sample_df):
    env = CryptoTradingEnv(sample_df)
    obs, info = env.reset()
    assert obs.shape == (24,)
    assert env.balance == env.initial_balance

def test_buy_action(sample_df):
    env = CryptoTradingEnv(sample_df)
    env.reset()
    obs, reward, done, truncated, info = env.step(1)  # Buy
    assert env.crypto_held > 0
    assert env.balance < env.initial_balance
```

---

### 6.4 타입 힌트 완성

**현재 상태** (부분적):
```python
def get_market_data(self, market: str, days: int = 30) -> Optional[pd.DataFrame]:
```

**권장** (전체 적용):
```python
from typing import Dict, List, Optional, Tuple, Union
from pandas import DataFrame

def generate_signals(self, indicators: Dict[str, float]) -> Dict[str, Union[float, str, List[str]]]:
    """매매 신호 생성

    Args:
        indicators: 기술적 지표 딕셔너리

    Returns:
        신호 정보 (score, recommendation, signals)
    """
    ...
```

---

## 7. 잠재적 버그

### 7.1 NaN 값 처리

**파일**: `src/environment/trading_env.py:100`

```python
self.df = self.df.fillna(0)
```

**문제점**:
- 모든 NaN을 0으로 대체하면 지표가 왜곡됨
- 예: RSI가 NaN인 경우 0으로 대체되면 극단적 과매도 신호

**권장 수정**:
```python
# 전방/후방 채움 사용
self.df = self.df.ffill().bfill()

# 또는 시작 부분 NaN 행 제거
self.df = self.df.dropna()
```

---

### 7.2 Division by Zero 보호 미흡

**파일**: `src/analyzer/market_analyzer.py:113`

```python
volume_change = (df.iloc[-1]['volume'] - volume_sma.iloc[-1]) / volume_sma.iloc[-1] * 100
```

**문제점**:
- `volume_sma.iloc[-1]`이 0이면 ZeroDivisionError 발생
- 거래량이 없는 코인에서 발생 가능

**수정 권장**:
```python
volume_sma_val = volume_sma.iloc[-1]
if volume_sma_val > 0:
    volume_change = (df.iloc[-1]['volume'] - volume_sma_val) / volume_sma_val * 100
else:
    volume_change = 0.0
```

---

### 7.3 빈 데이터 처리

**파일**: `src/backtesting/backtest.py:97`

```python
final_price = self.df.iloc[-1]['close']
```

**문제점**:
- `self.df`가 비어있으면 IndexError 발생
- `actions` 배열이 비어있어도 에러 없이 진행됨

**수정 권장**:
```python
def run_backtest(self, actions: np.ndarray) -> Dict:
    if len(actions) == 0 or len(self.df) == 0:
        return self._get_empty_results()
    ...
```

---

## 8. 검수 요약

### 8.1 문제 분류표

| 구분 | 개수 | 심각도 | 조치 우선순위 |
|------|------|--------|---------------|
| 심각한 버그 | 3 | 🔴 높음 | 즉시 수정 |
| 논리적 오류 | 4 | 🟠 중간 | 1주 내 수정 |
| 보안 취약점 | 4 | 🔴 높음 | 즉시 수정 |
| 성능 문제 | 3 | 🟡 낮음 | 개선 권장 |
| 코드 품질 | 4 | 🟡 낮음 | 장기 개선 |
| 잠재적 버그 | 3 | 🟠 중간 | 1주 내 수정 |

### 8.2 파일별 문제 현황

| 파일 | 문제 수 | 주요 이슈 |
|------|---------|-----------|
| trading_env.py | 4 | 매수 로직, 보상 계산, NaN 처리 |
| upbit_client.py | 1 | API 응답 타입 불일치 |
| ai_predictor.py | 2 | 임계값 불균형, 비효율적 예측 |
| backtest.py | 2 | 승률 계산, 빈 데이터 처리 |
| webapp.py | 3 | CORS, WebSocket 인증, API 키 검증 |
| market_analyzer.py | 2 | Division by zero, 동기 처리 |
| signal_generator.py | 1 | 미사용 파라미터 |
| trading_bot.py | 1 | 환경 재생성 |

### 8.3 전체 평가

```
┌─────────────────────────────────────────────────────────────┐
│                      전체 코드 품질 평가                      │
├─────────────────────────────────────────────────────────────┤
│  구조 설계      ████████░░  80%  - 모듈화 양호               │
│  코드 가독성    ███████░░░  70%  - 주석 양호, 일관성 필요     │
│  에러 처리      █████░░░░░  50%  - print 기반, 로깅 필요     │
│  보안성         ████░░░░░░  40%  - 인증/인가 부족            │
│  테스트 커버리지 ░░░░░░░░░░   0%  - 테스트 없음              │
│  성능 최적화    █████░░░░░  50%  - 동기 처리, 캐싱 부족      │
├─────────────────────────────────────────────────────────────┤
│  종합 점수:  58/100                                         │
│  등급: C+ (교육/연구 목적 적합, 프로덕션 사용 전 개선 필요)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. 권장 조치사항

### 9.1 즉시 조치 (1-2일)

1. **보안 취약점 수정**
   - [ ] CORS 설정 제한
   - [ ] WebSocket 인증 추가
   - [ ] .gitignore에 .env 확인

2. **심각한 버그 수정**
   - [ ] 매수 로직 이중 감액 수정
   - [ ] API 응답 타입 통일
   - [ ] 보상 계산 시점 수정

### 9.2 단기 조치 (1주)

1. **논리적 오류 수정**
   - [ ] 매수/매도 임계값 균형 조정
   - [ ] 백테스트 승률 계산 수정
   - [ ] 미사용 파라미터 활용

2. **잠재적 버그 수정**
   - [ ] NaN 처리 방식 개선
   - [ ] Division by zero 보호
   - [ ] 빈 데이터 처리

### 9.3 장기 개선 (1개월)

1. **코드 품질 개선**
   - [ ] logging 모듈 도입
   - [ ] config.py 설정 분리
   - [ ] 타입 힌트 완성
   - [ ] 단위 테스트 작성

2. **성능 최적화**
   - [ ] 비동기 API 호출
   - [ ] 환경 객체 캐싱
   - [ ] 신뢰도 계산 최적화

---

## 부록: 참고 자료

- [업비트 API 문서](https://docs.upbit.com/reference)
- [stable-baselines3 문서](https://stable-baselines3.readthedocs.io/)
- [FastAPI 보안 가이드](https://fastapi.tiangolo.com/tutorial/security/)
- [Python 로깅 모범 사례](https://docs.python.org/3/howto/logging.html)

---

*이 보고서는 자동화된 코드 검수 도구와 수동 검토를 통해 작성되었습니다.*
