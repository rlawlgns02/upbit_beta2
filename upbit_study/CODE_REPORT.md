# 업비트 AI 자동매매 시스템 코드 분석 보고서

> **분석 일시**: 2026-02-01
> **분석 범위**: 전체 소스 코드 (13개 파일)
> **분석자**: Claude AI

---

## 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [심각한 버그 (Critical)](#2-심각한-버그-critical)
3. [논리적 오류 (Logic Errors)](#3-논리적-오류-logic-errors)
4. [보안 취약점 (Security)](#4-보안-취약점-security)
5. [성능 이슈 (Performance)](#5-성능-이슈-performance)
6. [잠재적 버그 (Potential Bugs)](#6-잠재적-버그-potential-bugs)
7. [코드 품질 개선 제안](#7-코드-품질-개선-제안)
8. [수정 완료 항목](#8-수정-완료-항목)
9. [권장 조치 사항](#9-권장-조치-사항)

---

## 1. 프로젝트 개요

### 1.1 시스템 구성
| 모듈 | 파일 | 설명 |
|------|------|------|
| API 클라이언트 | `src/api/upbit_client.py` | 업비트 거래소 API 연동 |
| 시장 분석기 | `src/analyzer/market_analyzer.py` | 기술적 지표 계산 및 분석 |
| AI 예측기 | `src/analyzer/ai_predictor.py` | AI/규칙 기반 가격 예측 |
| 강화학습 에이전트 | `src/models/rl_agent.py` | PPO 알고리즘 기반 에이전트 |
| 트레이딩 환경 | `src/environment/trading_env.py` | Gymnasium 커스텀 환경 |
| 백테스팅 | `src/backtesting/backtest.py` | 전략 성능 검증 |
| 트레이딩 봇 | `src/bot/trading_bot.py` | 실시간 자동 매매 |
| 뉴스 수집기 | `src/news/news_collector.py` | NewsAPI 뉴스 수집 |
| 감정 분석기 | `src/news/sentiment_analyzer.py` | TextBlob 기반 감정 분석 |
| 신호 생성기 | `src/news/signal_generator.py` | 뉴스 기반 매매 신호 |
| 웹 애플리케이션 | `webapp.py` | FastAPI 기반 웹 UI |
| 메인 실행 | `main.py` | CLI 메인 진입점 |
| 추천 시스템 | `recommend.py` | 종목 추천 기능 |

### 1.2 사용 기술 스택
- **백엔드**: Python 3.10+, FastAPI, uvicorn
- **AI/ML**: stable-baselines3 (PPO), Gymnasium, PyTorch
- **분석**: pandas, numpy, ta (Technical Analysis)
- **감정분석**: TextBlob, NewsAPI
- **인증**: JWT (PyJWT)

---

## 2. 심각한 버그 (Critical)

### 2.1 [수정됨] 매수 로직 이중 공제 문제
**파일**: `trading_env.py:190-198`, `backtest.py:61-77`

**문제**: 잔액의 95%를 계산한 후 수수료를 다시 적용하여 이중 공제 발생
```python
# 수정 전 (잘못된 로직)
max_buy_amount = self.balance * 0.95  # 95% 계산
cost = max_buy_amount * (1 + self.commission)  # 수수료 추가 -> 이중 공제!

# 수정 후 (올바른 로직)
max_buy_amount = (self.balance * 0.95) / (1 + self.commission)
cost = max_buy_amount * (1 + self.commission)  # 정확히 잔액의 95%
```

**영향**: 실제 매수 가능 금액보다 적게 매수되어 수익률 저하

**상태**: ✅ 수정 완료

---

### 2.2 [수정됨] 보상 계산 타이밍 오류
**파일**: `trading_env.py:211-219`

**문제**: step 이동 전 가격으로 보상을 계산하여 행동의 결과를 반영하지 못함
```python
# 수정 전 (잘못된 로직)
current_net_worth = self.balance + self.crypto_held * current_price  # 같은 가격 사용
self.current_step += 1  # 이후에 이동

# 수정 후 (올바른 로직)
self.current_step += 1  # 먼저 이동
next_price = self.df.iloc[self.current_step]['close']  # 다음 가격
current_net_worth = self.balance + self.crypto_held * next_price
```

**영향**: 강화학습 모델이 잘못된 보상 신호를 받아 학습 품질 저하

**상태**: ✅ 수정 완료

---

### 2.3 [수정됨] API 응답 타입 불일치
**파일**: `upbit_client.py:137-151, 153-193`

**문제**: `get_accounts()`, `get_orders()` 메서드가 성공 시 `List[Dict]`, 실패 시 `Dict` 반환
```python
# 수정 전 (타입 불일치)
if isinstance(result, dict) and 'error' in result:
    return result  # Dict 반환
return result  # List[Dict] 반환

# 수정 후 (일관된 타입)
if isinstance(result, dict) and 'error' in result:
    print(f"[UPBIT] 조회 실패: {result}")
    return []  # 항상 List 반환
return result
```

**영향**: 호출하는 코드에서 타입 체크 누락 시 런타임 에러

**상태**: ✅ 수정 완료

---

## 3. 논리적 오류 (Logic Errors)

### 3.1 [수정됨] 매수/매도 임계값 불균형
**파일**: `ai_predictor.py:162-170`

**문제**: 매수 임계값(2)이 매도 임계값(4)보다 낮아 매수 편향 발생
```python
# 수정 전 (불균형)
if buy_score >= 2:  # 매수 쉬움
    return 1, buy_score / (total_score + 1)
elif sell_score >= 4:  # 매도 어려움
    return 2, sell_score / (total_score + 1)

# 수정 후 (균형)
THRESHOLD = 3  # 동일한 임계값
if buy_score > sell_score and buy_score >= THRESHOLD:
    return 1, buy_score / (total_score + 1)
elif sell_score > buy_score and sell_score >= THRESHOLD:
    return 2, sell_score / (total_score + 1)
```

**영향**: 예측 시스템이 매수 신호를 과도하게 생성

**상태**: ✅ 수정 완료

---

### 3.2 [수정됨] 승률 계산에서 수수료 미반영
**파일**: `backtest.py:113-123`

**문제**: 단순 가격 비교로 승률 계산, 수수료로 인한 실제 손실 미반영
```python
# 수정 전 (수수료 무시)
if sell_trades[i]['price'] > buy_trades[i]['price']:
    profitable_trades += 1

# 수정 후 (수수료 반영)
buy_cost = buy_trades[i]['price'] * (1 + self.commission)
sell_revenue = sell_trades[i]['price'] * (1 - self.commission)
if sell_revenue > buy_cost:
    profitable_trades += 1
```

**영향**: 백테스트 결과가 실제보다 과대평가됨

**상태**: ✅ 수정 완료

---

### 3.3 [수정됨] 뉴스 수량 파라미터 미전달
**파일**: `signal_generator.py:117, 306`

**문제**: `_determine_signal()` 호출 시 실제 뉴스 수량을 전달하지 않음
```python
# 수정 전 (기본값 사용)
signal, confidence = self._determine_signal(positive_ratio)  # news_count=10 기본값

# 수정 후 (실제 값 전달)
signal, confidence = self._determine_signal(positive_ratio, news_count=summary['total_articles'])
```

**영향**: 신뢰도 계산이 부정확함 (뉴스 수량이 많아도 기본 신뢰도 적용)

**상태**: ✅ 수정 완료

---

### 3.4 [수정됨] NaN 값 전파 가능성
**파일**: `market_analyzer.py:108-116`

**문제**: 데이터 부족 시 인덱스 에러 또는 NaN 발생 가능

**수정 내용**: 모든 가격 변동 계산에 0 나누기 및 NaN 보호 추가
```python
# 변동률 계산 (Division by zero 및 NaN 보호)
prev_close_1d = df.iloc[-2]['close'] if len(df) >= 2 else 0
prev_close_7d = df.iloc[-8]['close'] if len(df) >= 8 else 0
prev_close_30d = df.iloc[0]['close'] if len(df) >= 1 else 0
current_close = df.iloc[-1]['close']

price_change_1d = ((current_close - prev_close_1d) / prev_close_1d * 100) if prev_close_1d > 0 else 0.0
price_change_7d = ((current_close - prev_close_7d) / prev_close_7d * 100) if prev_close_7d > 0 else 0.0
price_change_30d = ((current_close - prev_close_30d) / prev_close_30d * 100) if prev_close_30d > 0 else 0.0
```

**상태**: ✅ 수정 완료

---

### 3.5 [수정됨] 백테스트 데이터 인덱스 불일치
**파일**: `main.py:128-155`

**문제**: 환경(env)과 백테스터(backtester)가 다른 시작 인덱스 사용

**수정 내용**: backtester에 env와 동일한 인덱스 범위의 데이터 전달
```python
# env.reset()은 current_step을 60으로 설정
start_index = 60

for _ in range(len(df) - start_index - 1):
    action, _ = agent.predict(obs, deterministic=True)
    actions.append(int(action))
    ...

# 백테스트 - env와 동일한 인덱스 범위 사용
backtest_df = df.iloc[start_index:].reset_index(drop=True)
backtester = Backtester(backtest_df, initial_balance=1000000)
results = backtester.run_backtest(actions)
```

**상태**: ✅ 수정 완료

---

## 4. 보안 취약점 (Security)

### 4.1 [수정됨] CORS 정책 과도한 허용
**파일**: `webapp.py:55-69`

**문제**: 모든 도메인에서의 접근 허용 (`allow_origins=["*"]`)
```python
# 수정 전 (취약)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용
    ...
)

# 수정 후 (제한)
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    ...
)
```

**상태**: ✅ 수정 완료

---

### 4.2 [수정됨] API 키 검증 미흡
**파일**: `webapp.py:87-99`

**문제**: 단순 존재 여부만 확인, 형식 검증 없음
```python
# 수정 전 (미흡)
if not access_key or not secret_key:
    return None

# 수정 후 (검증 강화)
def is_valid_api_key(key: str) -> bool:
    if not key:
        return False
    placeholders = ['', 'your_access_key_here', ...]
    if key in placeholders:
        return False
    if len(key) < 30:
        return False
    api_key_pattern = r'^[a-zA-Z0-9\-]+$'
    return bool(re.match(api_key_pattern, key))
```

**상태**: ✅ 수정 완료

---

### 4.3 [경고] 민감 정보 로깅
**파일**: 여러 파일

**문제**: 에러 메시지에 민감한 정보가 포함될 수 있음
```python
# 잠재적 위험
print(f"[UPBIT] 계좌 조회 실패: {result}")  # API 에러에 키 정보 포함 가능성
return {'error': {'message': f'요청 실패: {str(e)}'}}  # 상세 에러 노출
```

**권장**: 프로덕션 환경에서는 상세 에러 메시지 대신 일반적인 에러 코드 사용

**상태**: ⚠️ 검토 필요

---

### 4.4 [수정됨] Rate Limiting 미구현
**파일**: `webapp.py`

**문제**: API 엔드포인트에 Rate Limiting이 없어 DoS 공격에 취약

**수정 내용**: 커스텀 RateLimiter 클래스 구현 및 미들웨어 추가
```python
class RateLimiter:
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        ...

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(status_code=429, content={"error": "Too Many Requests"})
    return await call_next(request)
```

**상태**: ✅ 수정 완료

---

### 4.5 [수정됨] WebSocket 연결 제한 없음
**파일**: `webapp.py`

**문제**: WebSocket 연결 수 제한이 없어 리소스 고갈 가능

**수정 내용**: MAX_CONNECTIONS 상수 추가 및 연결 제한 구현
```python
class ConnectionManager:
    MAX_CONNECTIONS = 100

    async def connect(self, websocket: WebSocket) -> bool:
        async with self._lock:
            if len(self.active_connections) >= self.MAX_CONNECTIONS:
                await websocket.close(code=1013)
                return False
            await websocket.accept()
            self.active_connections.append(websocket)
            return True
```

**상태**: ✅ 수정 완료

---

## 5. 성능 이슈 (Performance)

### 5.1 [수정됨] 환경 객체 반복 생성
**파일**: `trading_bot.py:59-79`

**문제**: 매 예측마다 CryptoTradingEnv 객체를 새로 생성
```python
# 수정 전 (비효율적)
def _prepare_observation(self, candles: pd.DataFrame) -> np.ndarray:
    env = CryptoTradingEnv(candles, initial_balance=1000000)  # 매번 생성
    ...

# 수정 후 (캐싱)
def _prepare_observation(self, candles: pd.DataFrame) -> np.ndarray:
    if self._cached_env is None:
        self._cached_env = CryptoTradingEnv(candles, initial_balance=1000000)
    else:
        self._cached_env.df = candles.reset_index(drop=True)
        self._cached_env._add_technical_indicators()
    ...
```

**영향**: 매 트레이딩 주기마다 불필요한 메모리 할당 및 지표 재계산

**상태**: ✅ 수정 완료

---

### 5.2 [수정됨] 신뢰도 계산 과도한 샘플링
**파일**: `ai_predictor.py:88-95`

**문제**: 신뢰도 측정을 위해 10회 예측 수행
```python
# 수정 전 (비효율적)
for _ in range(9):  # 10회 샘플링
    sample_action, _ = self.agent.predict(obs, deterministic=False)
    ...
confidence = consistent_count / 10

# 수정 후 (효율적)
for _ in range(4):  # 5회 샘플링
    sample_action, _ = self.agent.predict(obs, deterministic=False)
    ...
confidence = consistent_count / 5
```

**상태**: ✅ 수정 완료

---

### 5.3 [경고] 동기 API 호출
**파일**: `upbit_client.py`

**문제**: 모든 API 호출이 동기 방식 (`requests`)으로 구현됨
```python
# 현재: 동기 방식
response = requests.get(url, params=params, timeout=10)

# 권장: 비동기 방식 (aiohttp)
async with aiohttp.ClientSession() as session:
    async with session.get(url, params=params) as response:
        result = await response.json()
```

**영향**: webapp.py의 async 엔드포인트에서 블로킹 발생

**상태**: ⚠️ 리팩토링 고려

---

### 5.4 [수정됨] 메모리 누수 가능성
**파일**: `signal_generator.py`

**문제**: 캐시 크기 제한 및 정리 로직 없음

**수정 내용**: MAX_CACHE_SIZE 상수 추가 및 자동 정리 로직 구현
```python
class NewsSignalGenerator:
    MAX_CACHE_SIZE = 50  # 최대 캐시 항목 수

    def _set_cache(self, key: str, data: Dict):
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            self._cleanup_old_cache()
        self._cache[key] = data

    def _cleanup_old_cache(self):
        # 만료된 항목 삭제
        now = datetime.now()
        expired_keys = [k for k, v in self._cache.items()
                       if (now - datetime.fromisoformat(v['timestamp'])).total_seconds() > self._cache_ttl]
        for key in expired_keys:
            del self._cache[key]

        # 아직 꽉 찼으면 가장 오래된 항목 삭제
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            oldest_key = min(self._cache.keys(),
                           key=lambda k: datetime.fromisoformat(self._cache[k].get('timestamp', '2000-01-01')))
            del self._cache[oldest_key]
```

**상태**: ✅ 수정 완료

---

## 6. 잠재적 버그 (Potential Bugs)

### 6.1 [수정됨] 티커 조회 빈 응답 처리
**파일**: `trading_bot.py:259-267`

**문제**: `get_ticker()` 반환값이 빈 리스트일 때 IndexError 발생
```python
# 수정 전 (취약)
ticker = self.client.get_ticker([self.market])
current_price = ticker[0]['trade_price']  # IndexError 가능

# 수정 후 (안전)
ticker_list = self.client.get_ticker([self.market])
if ticker_list and len(ticker_list) > 0:
    current_price = ticker_list[0]['trade_price']
    final_value = final_balance + crypto * current_price
else:
    print("티커 조회 실패, 코인 가치를 계산할 수 없습니다.")
    final_value = final_balance
```

**상태**: ✅ 수정 완료

---

### 6.2 [수정됨] 0으로 나누기 오류
**파일**: `market_analyzer.py:112-117`

**문제**: `volume_sma`가 0일 때 ZeroDivisionError
```python
# 수정 전 (취약)
volume_change = (df.iloc[-1]['volume'] - volume_sma.iloc[-1]) / volume_sma.iloc[-1] * 100

# 수정 후 (안전)
volume_sma_val = volume_sma.iloc[-1]
if volume_sma_val > 0:
    volume_change = (df.iloc[-1]['volume'] - volume_sma_val) / volume_sma_val * 100
else:
    volume_change = 0.0
```

**상태**: ✅ 수정 완료

---

### 6.3 [수정됨] 동시성 문제
**파일**: `webapp.py`

**문제**: 자동매매 상태가 글로벌 딕셔너리로 관리되어 동시 접근 시 데이터 불일치

**수정 내용**: asyncio.Lock 추가 및 주요 함수에 Lock 적용
```python
trading_status_lock = asyncio.Lock()
auto_trading_status_lock = asyncio.Lock()

@app.post("/api/auto-trading/start")
async def start_auto_trading(request: AutoTradingStartRequest):
    async with auto_trading_status_lock:
        if auto_trading_status['is_running']:
            return {"success": False, "error": "이미 실행 중"}
        auto_trading_status['is_running'] = True
        ...
```

**상태**: ✅ 수정 완료

---

### 6.4 [수정됨] 예외 처리 불완전
**파일**: `main.py`, `recommend.py`

**문제**: API 호출 실패 시 빈 데이터프레임으로 진행될 수 있음

**수정 내용**: train_model, run_backtest, recommend 함수에 API 실패 체크 추가
```python
candles = client.get_candles_day(market, count=200)

# API 실패 체크
if not candles:
    print("❌ 데이터 수집 실패: API 응답이 비어있습니다.")
    return None

# 필수 컬럼 확인
if not all(col in df.columns for col in required_columns):
    print("❌ 데이터 수집 실패: 필수 컬럼이 없습니다.")
    return None

# 최소 데이터 확인
if len(df) < 100:
    print(f"❌ 데이터 부족: {len(df)}일치 (최소 100일 필요)")
    return None
```

**상태**: ✅ 수정 완료

---

### 6.5 [수정됨] WebSocket 브로드캐스트 에러 무시
**파일**: `webapp.py`

**문제**: 브로드캐스트 실패 시 예외를 무시하고 연결이 리스트에 남음

**수정 내용**: 죽은 연결 자동 정리 로직 추가
```python
async def broadcast(self, message: dict):
    dead_connections = []
    for connection in self.active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            dead_connections.append(connection)

    # 죽은 연결 정리
    if dead_connections:
        async with self._lock:
            for conn in dead_connections:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
```

**상태**: ✅ 수정 완료

---

## 7. 코드 품질 개선 제안

### 7.1 타입 힌트 불완전
여러 함수에서 반환 타입이 명시되지 않음

```python
# 현재
def analyze_market(self, market: str, days: int = 30):
    ...

# 권장
def analyze_market(self, market: str, days: int = 30) -> Optional[Dict[str, Any]]:
    ...
```

### 7.2 매직 넘버 사용
코드 내 하드코딩된 숫자들을 상수로 정의 권장

```python
# 현재
if self.balance > 5000:  # 5000이 뭔지?
    ...
max_buy_amount = self.balance * 0.95  # 0.95가 뭔지?

# 권장
MIN_TRADE_AMOUNT = 5000  # KRW
MAX_BALANCE_RATIO = 0.95  # 최대 매수 비율
```

### 7.3 로깅 표준화
`print()` 대신 `logging` 모듈 사용 권장

```python
# 현재
print(f"[UPBIT] 조회 실패: {result}")

# 권장
import logging
logger = logging.getLogger(__name__)
logger.error(f"조회 실패: {result}")
```

### 7.4 설정 관리
하드코딩된 설정값들을 설정 파일로 분리 권장

```python
# config.py
class Config:
    COMMISSION_RATE = 0.0005
    INITIAL_BALANCE = 1_000_000
    MIN_TRADE_AMOUNT = 5000
    API_TIMEOUT = 10
```

---

## 8. 수정 완료 항목

### 8.1 1차 수정 (기존)
| # | 파일 | 이슈 | 심각도 |
|---|------|------|--------|
| 1 | trading_env.py | NaN 처리 방식 개선 (ffill/bfill) | 중간 |
| 2 | trading_env.py | 매수 로직 이중 공제 수정 | 높음 |
| 3 | trading_env.py | 보상 계산 타이밍 수정 | 높음 |
| 4 | upbit_client.py | API 응답 타입 일관성 | 높음 |
| 5 | ai_predictor.py | 매수/매도 임계값 균형 | 중간 |
| 6 | ai_predictor.py | 신뢰도 계산 최적화 | 낮음 |
| 7 | backtest.py | 빈 데이터 처리 추가 | 중간 |
| 8 | backtest.py | 승률 계산에 수수료 반영 | 중간 |
| 9 | backtest.py | 매수 로직 이중 공제 수정 | 높음 |
| 10 | market_analyzer.py | 0으로 나누기 방지 | 중간 |
| 11 | signal_generator.py | news_count 파라미터 전달 | 중간 |
| 12 | webapp.py | CORS 정책 강화 | 중간 |
| 13 | webapp.py | API 키 검증 강화 | 중간 |
| 14 | trading_bot.py | 환경 캐싱 구현 | 낮음 |
| 15 | trading_bot.py | 빈 티커 응답 처리 | 중간 |

### 8.2 2차 수정 (추가)
| # | 파일 | 이슈 | 심각도 |
|---|------|------|--------|
| 16 | market_analyzer.py | NaN/Division by zero 보호 강화 | 중간 |
| 17 | main.py | 백테스트 인덱스 불일치 수정 | 높음 |
| 18 | main.py | API 실패 시 예외 처리 추가 | 중간 |
| 19 | webapp.py | Rate Limiting 구현 | 중간 |
| 20 | webapp.py | WebSocket 연결 수 제한 | 중간 |
| 21 | webapp.py | 동시성 Lock 추가 | 중간 |
| 22 | webapp.py | WebSocket 브로드캐스트 에러 처리 | 낮음 |
| 23 | signal_generator.py | 캐시 크기 제한 추가 | 낮음 |
| 24 | recommend.py | API 실패 시 예외 처리 추가 | 중간 |

**총 24개 이슈 수정 완료**

---

## 9. 권장 조치 사항 (장기 개선)

모든 긴급/단기 이슈가 수정되었습니다. 아래는 장기적으로 고려할 개선 사항입니다.

### 9.1 코드 품질 개선
1. **비동기 API 클라이언트 전환** - `upbit_client.py`에서 `requests` 대신 `aiohttp` 사용
2. **로깅 시스템 표준화** - `print()` 대신 `logging` 모듈 사용
3. **설정 파일 분리** - 하드코딩된 상수를 config 파일로 분리
4. **타입 힌트 보강** - 반환 타입 명시 추가

### 9.2 테스트 및 품질 보증
1. **단위 테스트 추가** - pytest 기반 테스트 코드 작성
2. **통합 테스트** - API 호출 및 트레이딩 로직 테스트
3. **CI/CD 파이프라인** - GitHub Actions 등 자동화

### 9.3 보안 강화
1. **민감 정보 로깅 제거** - 에러 메시지에서 API 키 등 제거
2. **환경 변수 검증** - 시작 시 필수 환경 변수 체크
3. **입력 값 검증** - 사용자 입력 sanitization

---

## 부록: 파일별 코드 라인 수

| 파일 | 라인 수 | 함수 수 |
|------|---------|---------|
| webapp.py | ~1500+ | 30+ |
| trading_env.py | 256 | 8 |
| market_analyzer.py | 530 | 12 |
| ai_predictor.py | 424 | 10 |
| upbit_client.py | 382 | 15 |
| backtest.py | 279 | 7 |
| trading_bot.py | 283 | 10 |
| signal_generator.py | 365 | 15 |
| sentiment_analyzer.py | 220 | 8 |
| news_collector.py | 183 | 8 |
| rl_agent.py | 204 | 8 |
| main.py | 228 | 4 |
| recommend.py | 258 | 4 |

---

**문서 종료**
