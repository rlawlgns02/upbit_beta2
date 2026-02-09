"""
uvicorn webapp:app --reload --host 127.0.0.1 --port 8000
업비트 실시간 종목 분석 웹앱
FastAPI + 현대적인 UI/UX + 자동매매
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
import os
import sys
from typing import List, Optional, Dict
import json
import math
from datetime import datetime

# optional psutil for resource monitoring (type ignored to allow running without installation)
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None
except Exception:
    psutil = None
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from collections import defaultdict
import time

# .env 파일 로드
load_dotenv()

# 프로젝트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.api.upbit_client import UpbitClient
from src.analyzer.market_analyzer import MarketAnalyzer
from src.analyzer.ai_predictor import AIPredictor
from src.news.signal_generator import NewsSignalGenerator
from src.models.rl_agent import TradingAgent
from src.environment.trading_env import CryptoTradingEnv

# FastAPI 앱 생성
app = FastAPI(
    title="업비트 AI 트레이딩 분석",
    description="실시간 종목 분석 및 AI 예측",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행"""
    print("[STARTUP] 서버 시작 중...")

    # 저장된 자동매매 상태 복원
    if load_auto_trading_state():
        print("[STARTUP] 자동매매 상태가 복원되었습니다.")
        print("[STARTUP] 자동매매를 재시작하려면 웹 UI에서 '자동매매 시작'을 클릭하세요.")
        print("[STARTUP] (이전 포지션 정보가 유지됩니다)")
    else:
        print("[STARTUP] 새로운 세션을 시작합니다.")
    
    # LSTM 큐 처리 시작
    asyncio.create_task(process_lstm_queue())
    print("[STARTUP] LSTM 학습 큐 처리 시작")

@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행"""
    print("[SHUTDOWN] 서버 종료 중...")
    # Rate limiter 정리
    rate_limiter.cleanup()
    print("[SHUTDOWN] 서버 종료 완료")

# CORS 설정 (보안 강화: 허용 도메인 제한)
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

# Rate Limiting 미들웨어
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """API 요청 Rate Limiting"""
    # 정적 파일과 WebSocket은 제외
    if request.url.path.startswith("/static") or request.url.path.startswith("/ws"):
        return await call_next(request)

    # 클라이언트 IP 가져오기
    client_ip = request.client.host if request.client else "unknown"

    # Rate Limit 체크
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=429,
            content={"success": False, "error": "Too Many Requests. 잠시 후 다시 시도해주세요."}
        )

    return await call_next(request)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 글로벌 객체
client = UpbitClient("", "")
analyzer = MarketAnalyzer(client)
predictor = AIPredictor()
news_signal_generator = NewsSignalGenerator()  # 뉴스 신호 생성기

# LSTM 관련 전역 상태: 서버에서 학습/로드/예측 관리를 위한 객체
lstm_predictors: Dict[str, object] = {}
lstm_statuses: Dict[str, Dict] = {}
lstm_tasks: Dict[str, asyncio.Task] = {}
lstm_queue: List[Dict] = []  # 대기열
lstm_max_concurrent: int = 3  # 동시에 학습할 최대 코인 수

# LSTM 단타 자동매매 관련 전역 상태
lstm_scalping_bot = None
lstm_scalping_task: Optional[asyncio.Task] = None
lstm_scalping_status = {
    "is_running": False,
    "markets": [],
    "start_time": None,
    "positions": {},
    "stats": {},
    "trade_history": [],
    "config": {},
    "use_unified_model": False,
    "signal_log": []  # 실시간 신호 로그
}
lstm_scalping_ws_clients: List[WebSocket] = []  # WebSocket 클라이언트 목록

# LSTM 모듈 로드 (실패해도 서버는 동작하도록 예외 처리)
try:
    from src.models.lstm_predictor import LSTMPredictor, get_device
except Exception as e:
    LSTMPredictor = None
    def get_device():
        return 'cpu'
    print(f"[WARNING] LSTM 모듈 로드 실패: {e}")

# LSTM 단타 봇 모듈 로드
try:
    from src.bot.lstm_scalping_bot import LSTMScalpingBot, LSTMScalpingConfig
    LSTM_SCALPING_AVAILABLE = True
except Exception as e:
    LSTMScalpingBot = None
    LSTMScalpingConfig = None
    LSTM_SCALPING_AVAILABLE = False
    print(f"[WARNING] LSTM 단타 봇 모듈 로드 실패: {e}")

# 통합 LSTM 모델 로드
try:
    from src.models.unified_lstm_predictor import UnifiedLSTMPredictor, UnifiedModelConfig, unified_predictor
    UNIFIED_LSTM_AVAILABLE = True
except Exception as e:
    UnifiedLSTMPredictor = None
    UnifiedModelConfig = None
    unified_predictor = None
    UNIFIED_LSTM_AVAILABLE = False
    print(f"[WARNING] 통합 LSTM 모듈 로드 실패: {e}")

# 자동매매 관련 글로벌 객체
UPBIT_ACCESS_KEY = os.getenv('UPBIT_ACCESS_KEY', '')
UPBIT_SECRET_KEY = os.getenv('UPBIT_SECRET_KEY', '')

# API 키가 placeholder가 아닌 실제 값인지 확인 (검증 강화)
import re

def is_valid_api_key(key: str) -> bool:
    if not key:
        return False
    # 플레이스홀더 체크
    placeholders = ['', 'your_access_key_here', 'your_secret_key_here', 'YOUR_ACCESS_KEY', 'YOUR_SECRET_KEY']
    if key in placeholders:
        return False
    # 최소 길이 및 형식 검증 (업비트 API 키는 보통 36자 이상의 영숫자-하이픈 조합)
    if len(key) < 30:
        return False
    # 기본적인 형식 검증
    api_key_pattern = r'^[a-zA-Z0-9\-]+$'
    return bool(re.match(api_key_pattern, key))

trading_client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY) if is_valid_api_key(UPBIT_ACCESS_KEY) and is_valid_api_key(UPBIT_SECRET_KEY) else None

# 자동매매 상태 (단일 코인)
trading_bot_task: Optional[asyncio.Task] = None
trading_status = {
    "is_running": False,
    "market": "KRW-BTC",
    "interval": 60,
    "max_trade_amount": 100000,
    "start_time": None,
    "trade_count": 0,
    "current_position": None,
    "start_balance": 0,
    "current_balance": 0,
    "profit": 0,
    "profit_rate": 0,
    "last_action": None,
    "last_action_time": None,
    "last_price": 0,
    "trade_history": []
}

# ========== 원클릭 자동매매 (다중 코인) ==========
auto_trading_task: Optional[asyncio.Task] = None
auto_trading_status = {
    "is_running": False,
    "mode": "auto",
    "total_investment": 50000,
    "coin_count": 3,
    "analysis_mode": "volume_top50",
    "coin_category": "normal",  # 'safe', 'normal', 'meme', 'all'
    "trading_interval": 60,
    "allocation_mode": "weighted",  # 'equal' (균등배분) or 'weighted' (점수기반)
    "target_profit_percent": 10.0,  # 목표가 (+%)
    "stop_loss_percent": 10.0,      # 손절가 (-%)
    "start_time": None,
    "start_balance": 0,
    "current_balance": 0,
    "profit": 0,
    "profit_rate": 0,
    "positions": {},
    "selected_coins": [],
    "trade_history": []
}

# 상태 저장 파일 경로
AUTO_TRADING_STATE_FILE = "auto_trading_state.json"
BLACKLIST_FILE = "coin_blacklist.json"

# 코인 블랙리스트 (제외할 코인)
coin_blacklist: set = set()

def load_blacklist():
    """블랙리스트 파일에서 로드"""
    global coin_blacklist
    try:
        if os.path.exists(BLACKLIST_FILE):
            with open(BLACKLIST_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                coin_blacklist = set(data.get('blacklist', []))
                print(f"[BLACKLIST] 로드 완료: {len(coin_blacklist)}개 코인 제외")
    except Exception as e:
        print(f"[BLACKLIST] 로드 실패: {e}")
        coin_blacklist = set()

def save_blacklist():
    """블랙리스트를 파일에 저장"""
    try:
        with open(BLACKLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump({'blacklist': list(coin_blacklist)}, f, ensure_ascii=False, indent=2)
        print(f"[BLACKLIST] 저장 완료: {len(coin_blacklist)}개 코인")
    except Exception as e:
        print(f"[BLACKLIST] 저장 실패: {e}")

# 시작 시 블랙리스트 로드
load_blacklist()

def save_auto_trading_state():
    """자동매매 상태를 파일에 저장"""
    try:
        state_to_save = {
            'is_running': auto_trading_status['is_running'],
            'total_investment': auto_trading_status['total_investment'],
            'coin_count': auto_trading_status['coin_count'],
            'analysis_mode': auto_trading_status['analysis_mode'],
            'coin_category': auto_trading_status['coin_category'],
            'trading_interval': auto_trading_status['trading_interval'],
            'allocation_mode': auto_trading_status['allocation_mode'],
            'target_profit_percent': auto_trading_status['target_profit_percent'],
            'stop_loss_percent': auto_trading_status['stop_loss_percent'],
            'start_time': auto_trading_status['start_time'],
            'start_balance': auto_trading_status['start_balance'],
            'current_balance': auto_trading_status['current_balance'],
            'profit': auto_trading_status['profit'],
            'profit_rate': auto_trading_status['profit_rate'],
            'positions': auto_trading_status['positions'],
            'selected_coins': auto_trading_status['selected_coins'],
            'trade_history': auto_trading_status['trade_history'][-50:],  # 최근 50개만
            'saved_at': datetime.now().isoformat()
        }

        with open(AUTO_TRADING_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=2)

        print(f"[STATE] 자동매매 상태 저장 완료: {AUTO_TRADING_STATE_FILE}")
    except Exception as e:
        print(f"[STATE] 상태 저장 실패: {e}")

def load_auto_trading_state():
    """파일에서 자동매매 상태 복원"""
    global auto_trading_status

    try:
        if not os.path.exists(AUTO_TRADING_STATE_FILE):
            print("[STATE] 저장된 상태 파일이 없습니다.")
            return False

        with open(AUTO_TRADING_STATE_FILE, 'r', encoding='utf-8') as f:
            saved_state = json.load(f)

        # 실행 중이었던 경우만 복원
        if saved_state.get('is_running'):
            auto_trading_status.update({
                'total_investment': saved_state['total_investment'],
                'coin_count': saved_state['coin_count'],
                'analysis_mode': saved_state['analysis_mode'],
                'coin_category': saved_state['coin_category'],
                'trading_interval': saved_state['trading_interval'],
                'allocation_mode': saved_state['allocation_mode'],
                'target_profit_percent': saved_state['target_profit_percent'],
                'stop_loss_percent': saved_state['stop_loss_percent'],
                'start_time': saved_state['start_time'],
                'start_balance': saved_state['start_balance'],
                'current_balance': saved_state['current_balance'],
                'profit': saved_state['profit'],
                'profit_rate': saved_state['profit_rate'],
                'positions': saved_state['positions'],
                'selected_coins': saved_state['selected_coins'],
                'trade_history': saved_state['trade_history']
            })

            print(f"[STATE] ✅ 자동매매 상태 복원 완료")
            print(f"[STATE] - 시작 시간: {saved_state['start_time']}")
            print(f"[STATE] - 포지션: {len(saved_state['positions'])}개")
            print(f"[STATE] - 수익률: {saved_state['profit_rate']:+.2f}%")
            return True
        else:
            print("[STATE] 이전에 실행 중이 아니었으므로 복원하지 않습니다.")
            return False

    except Exception as e:
        print(f"[STATE] 상태 복원 실패: {e}")
        return False

# AI 트레이딩 에이전트
trading_agent: Optional[TradingAgent] = None

# 동시성 제어용 Lock
trading_status_lock = asyncio.Lock()
auto_trading_status_lock = asyncio.Lock()

# Rate Limiting 설정
class RateLimiter:
    """간단한 Rate Limiter 구현"""
    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """요청 허용 여부 확인"""
        now = time.time()
        # 오래된 요청 제거
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.window_seconds
        ]
        # 제한 확인
        if len(self.requests[client_ip]) >= self.max_requests:
            return False
        # 요청 기록
        self.requests[client_ip].append(now)
        return True

    def cleanup(self):
        """오래된 데이터 정리"""
        now = time.time()
        empty_keys = [
            ip for ip, times in self.requests.items()
            if all(now - t >= self.window_seconds for t in times)
        ]
        for ip in empty_keys:
            del self.requests[ip]

rate_limiter = RateLimiter(max_requests=180, window_seconds=60)  # 분당 180회 제한 (여러 실시간 polling 고려)

# WebSocket 연결 관리
class ConnectionManager:
    MAX_CONNECTIONS = 100  # 최대 연결 수

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> bool:
        """WebSocket 연결 (연결 수 제한 적용)"""
        async with self._lock:
            if len(self.active_connections) >= self.MAX_CONNECTIONS:
                await websocket.close(code=1013)  # Try Again Later
                return False
            await websocket.accept()
            self.active_connections.append(websocket)
            return True

    async def disconnect(self, websocket: WebSocket):
        """WebSocket 연결 해제"""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """모든 연결에 메시지 브로드캐스트 (에러 발생 시 연결 정리)"""
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

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)

manager = ConnectionManager()


def clean_dict(d):
    """NaN/Infinity 값을 None으로 변환, numpy 타입을 Python 기본 타입으로 변환"""
    if isinstance(d, dict):
        return {k: clean_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_dict(item) for item in d]
    elif isinstance(d, (np.floating, np.float64, np.float32)):
        # numpy float 타입 처리
        if np.isnan(d) or np.isinf(d):
            return None
        return float(d)
    elif isinstance(d, (np.integer, np.int64, np.int32)):
        # numpy int 타입 처리
        return int(d)
    elif isinstance(d, np.ndarray):
        # numpy 배열을 리스트로 변환
        return clean_dict(d.tolist())
    elif isinstance(d, float):
        if math.isnan(d) or math.isinf(d):
            return None
        return d
    elif isinstance(d, np.bool_):
        return bool(d)
    return d


def normalize_score_to_100(score):
    """점수를 100점 만점으로 정규화

    Args:
        score: 원래 점수 (대략 -10 ~ +12 범위)

    Returns:
        0~100 사이의 정규화된 점수
    """
    # -10 ~ +12 범위를 0 ~ 100으로 매핑
    # -10 = 0점, 0 = 45점, +12 = 100점
    normalized = ((score + 10) / 22.0) * 100
    return max(0, min(100, round(normalized, 1)))


def calculate_trade_prices(tech_data):
    """매수/매도/손절 가격 계산

    Args:
        tech_data: 기술적 분석 데이터

    Returns:
        매수가, 매도가, 손절가 딕셔너리
    """
    current_price = tech_data.get('current_price', 0)
    rsi = tech_data.get('rsi', 50)
    bb_low = tech_data.get('bb_low', current_price * 0.95)
    bb_high = tech_data.get('bb_high', current_price * 1.05)
    recommendation = tech_data.get('recommendation', '중립')

    # 매수가 계산 - 현재가 기준으로 실제 매수 가능한 가격
    if rsi < 30:  # 과매도 - 적극 매수
        buy_price = current_price * 1.00  # 현재가에 바로 매수
        target_profit = 0.12  # 12% 목표
    elif rsi < 40:  # 저평가
        buy_price = current_price * 0.99  # 현재가의 99%
        target_profit = 0.10  # 10% 목표
    elif recommendation in ['강력 매수', '매수']:
        buy_price = current_price * 0.98  # 현재가의 98%
        target_profit = 0.08  # 8% 목표
    else:
        buy_price = current_price * 0.97  # 현재가의 97% (관망 시 조금 더 낮게)
        target_profit = 0.05  # 5% 목표

    # 매도가 계산 (목표가) - 매수가 기준으로 계산
    if recommendation in ['강력 매수', '매수', '약한 매수']:
        sell_price = buy_price * (1 + target_profit)  # 목표 수익률 적용
    else:
        sell_price = buy_price * 1.05  # 매수가 대비 5% 이익

    # 매도가는 반드시 매수가보다 높아야 함 (최소 2% 이익 보장)
    min_sell_price = buy_price * 1.02
    if sell_price < min_sell_price:
        sell_price = min_sell_price

    # 손절가 계산 - 매수가 기준으로 계산
    if rsi < 30:
        stop_loss = buy_price * 0.95  # 5% 손절
    elif recommendation in ['강력 매수', '매수']:
        stop_loss = buy_price * 0.96  # 4% 손절
    else:
        stop_loss = buy_price * 0.97  # 3% 손절

    # 손절가는 반드시 매수가보다 낮아야 함 (최대 2% 손실 이하로 설정 방지)
    max_stop_loss = buy_price * 0.98
    if stop_loss > max_stop_loss:
        stop_loss = max_stop_loss

    return {
        'buy_price': round(buy_price, 2 if buy_price < 1000 else 0),
        'sell_price': round(sell_price, 2 if sell_price < 1000 else 0),
        'stop_loss': round(stop_loss, 2 if stop_loss < 1000 else 0),
        'expected_profit_rate': round(((sell_price - buy_price) / buy_price) * 100, 2),
        'risk_rate': round(((buy_price - stop_loss) / buy_price) * 100, 2)
    }


@app.get('/favicon.ico')
async def favicon():
    """Serve favicon"""
    return FileResponse('static/favicon.svg', media_type='image/svg+xml')


@app.post('/api/client-log')
async def api_client_log(payload: dict):
    """클라이언트에서 전송한 콘솔/에러 로그를 수집합니다.

    Expects JSON: { level: 'error'|'warn'|'info', message: str, stack?: str, url?: str, userAgent?: str }
    """
    try:
        level = payload.get('level', 'info')
        message = payload.get('message', '')
        stack = payload.get('stack')
        url = payload.get('url')
        ua = payload.get('userAgent')

        log_entry = {
            'time': datetime.utcnow().isoformat(),
            'level': level,
            'message': message,
            'stack': stack,
            'url': url,
            'userAgent': ua
        }

        # 간단히 stdout에 기록하고 models/logs 폴더에 파일로 저장
        print(f"[CLIENT_LOG] {log_entry}")
        os.makedirs('logs', exist_ok=True)
        with open(os.path.join('logs', 'client_logs.txt'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get('/.well-known/appspecific/com.chrome.devtools.json')
async def chrome_devtools_probe():
    """브라우저 개발자 도구가 요청하는 잘 알려진 경로를 204로 응답하여 404 로그를 줄입니다."""
    return JSONResponse(status_code=204, content=None)


@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지"""
    return FileResponse("templates/index.html")


@app.get("/api/markets")
async def get_markets():
    """전체 KRW 마켓 조회"""
    try:
        markets = analyzer.get_all_krw_markets()

        # 모든 마켓을 100개씩 분할해서 조회 (업비트 API 제한)
        result = []
        batch_size = 100

        for i in range(0, len(markets), batch_size):
            batch = markets[i:i + batch_size]
            market_codes = [m['market'] for m in batch]
            tickers = client.get_ticker(market_codes)

            if not tickers:
                continue

            # 티커를 마켓 코드로 매핑
            ticker_map = {t['market']: t for t in tickers}

            for market in batch:
                ticker = ticker_map.get(market['market'])
                if ticker:
                    result.append({
                        'market': market['market'],
                        'korean_name': market['korean_name'],
                        'english_name': market['english_name'],
                        'current_price': ticker.get('trade_price', 0),
                        'change_rate': ticker.get('signed_change_rate', 0) * 100,
                        'trade_volume': ticker.get('acc_trade_price_24h', 0)  # 거래대금 (KRW)
                    })

        return {"success": True, "data": clean_dict(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/analysis/{market}")
async def get_analysis(market: str):
    """특정 종목 상세 분석"""
    try:
        # 기술적 분석
        tech_result = analyzer.analyze_market(market, days=30)
        if not tech_result:
            return {"success": False, "error": "분석 데이터 없음"}

        # 100점 만점 점수 추가
        tech_result['score_100'] = normalize_score_to_100(tech_result['score'])

        # 매수/매도/손절 가격 계산
        trade_prices = calculate_trade_prices(tech_result)
        tech_result['trade_prices'] = trade_prices

        # AI 예측
        df = analyzer.get_market_data(market, days=30)
        ai_result = None
        if df is not None:
            ai_result = predictor.predict_market(df, market)

        # 뉴스 신호 (코인별)
        news_signal = None
        try:
            news_signal = news_signal_generator.generate_coin_signal(market=market, page_size=100)
        except Exception as e:
            print(f"[WARNING] 뉴스 신호 생성 실패: {e}")

        # 실시간 티커 정보
        ticker_list = client.get_ticker([market])
        if not ticker_list or len(ticker_list) == 0:
            return {"success": False, "error": "티커 정보를 가져올 수 없습니다."}
        ticker = ticker_list[0]

        # 추가 시장 정보
        market_info = {
            "opening_price": ticker.get('opening_price'),
            "high_price": ticker.get('high_price'),
            "low_price": ticker.get('low_price'),
            "prev_closing_price": ticker.get('prev_closing_price'),
            "acc_trade_price": ticker.get('acc_trade_price'),
            "acc_trade_price_24h": ticker.get('acc_trade_price_24h'),
            "acc_trade_volume_24h": ticker.get('acc_trade_volume_24h'),
            "highest_52_week_price": ticker.get('highest_52_week_price'),
            "highest_52_week_date": ticker.get('highest_52_week_date'),
            "lowest_52_week_price": ticker.get('lowest_52_week_price'),
            "lowest_52_week_date": ticker.get('lowest_52_week_date'),
            "timestamp": ticker.get('timestamp')
        }

        result_data = {
            "market": market,
            "timestamp": datetime.now().isoformat(),
            "technical": tech_result,
            "ai_prediction": ai_result,
            "news_signal": news_signal,
            "market_info": market_info
        }

        return {
            "success": True,
            "data": clean_dict(result_data)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ------------------ LSTM 학습 대기열 처리 ------------------
async def process_lstm_queue():
    """LSTM 학습 대기열을 처리합니다."""
    while True:
        try:
            # 현재 진행 중인 작업 수 계산
            running_count = sum(1 for m in lstm_tasks.values() if not m.done())
            
            # 동시 실행 제한 내에서 새 작업 시작
            while running_count < lstm_max_concurrent and lstm_queue:
                job = lstm_queue.pop(0)
                market = job['market']
                
                if market not in lstm_tasks or lstm_tasks[market].done():
                    task = asyncio.create_task(train_lstm_background(
                        market,
                        job['epochs'],
                        job['batch_size'],
                        job['learning_rate'],
                        job['seq_length'],
                        job['days'],
                        resume=job.get('resume', False),
                        checkpoint_interval=job.get('checkpoint_interval', 5)
                    ))
                    lstm_tasks[market] = task
                    lstm_statuses[market] = {"status": "running", "message": "학습 시작", "queue_position": -1}
                    running_count += 1
                    print(f"[LSTM Queue] 학습 시작: {market} (진행 중: {running_count}/{lstm_max_concurrent})")
                else:
                    print(f"[LSTM Queue] 이미 진행 중: {market}")
            
            # 대기 중인 작업의 위치 업데이트
            for i, job in enumerate(lstm_queue):
                market = job['market']
                if market in lstm_statuses:
                    lstm_statuses[market]["queue_position"] = i
            
            await asyncio.sleep(2)  # 2초마다 체크
        except Exception as e:
            print(f"[LSTM Queue Error] {e}")
            await asyncio.sleep(5)


# ------------------ LSTM 학습 및 예측 API ------------------
class LSTMTrainRequest(BaseModel):
    market: Optional[str] = None  # 단일 코인 (이전 호환성)
    markets: Optional[List[str]] = None  # 여러 코인
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    seq_length: int = 60
    days: int = 200
    resume: bool = False
    checkpoint_interval: int = 5


async def train_lstm_background(market: str, epochs: int, batch_size: int, learning_rate: float, seq_length: int, days: int, resume: bool = False, checkpoint_interval: int = 5):
    """백그라운드에서 LSTM 학습을 실행하고 상태를 업데이트합니다."""
    coin = market.replace('KRW-', '').lower()
    model_path = f"models/lstm_{coin}"

    # 초기 상태
    lstm_statuses[market] = {"status": "running", "message": "데이터 수집 중", "logs": [], "resource": {}}

    try:
        df = analyzer.get_market_data(market, days=days)
        if df is None or len(df) < max(seq_length + 30, 100):
            lstm_statuses[market] = {"status": "error", "message": "데이터가 부족합니다."}
            return

        lstm_statuses[market]["message"] = "학습 시작"

        if LSTMPredictor is None:
            lstm_statuses[market] = {"status": "error", "message": "LSTM 모듈을 사용할 수 없습니다."}
            return

        predictor = LSTMPredictor(model_path=model_path, seq_length=seq_length, device=get_device())
        lstm_predictors[market] = predictor

        # progress callback: update status and append logs; include resource usage if psutil available
        def progress_cb(info: dict):
            msg = f"Epoch {info.get('epoch')} - train: {info.get('train_loss'):.6f}, val: {info.get('val_loss'):.6f}"
            lstm_statuses[market]['message'] = msg
            lstm_statuses[market].setdefault('logs', []).append(msg)
            # resource monitoring
            try:
                if psutil:
                    p = psutil.Process()
                    mem = p.memory_info().rss
                    cpu = psutil.cpu_percent(interval=None)
                    lstm_statuses[market]['resource'] = {'memory_rss': mem, 'cpu_percent': cpu}
            except Exception:
                pass

        # 학습은 CPU/GPU 바운드 작업이므로 스레드에서 실행
        results = await asyncio.to_thread(
            predictor.train,
            df,
            epochs,
            batch_size,
            learning_rate,
            0.2,
            15,
            True,
            progress_cb,
            resume,
            checkpoint_interval
        )

        # 학습 취소 체크
        if isinstance(results, dict) and results.get('cancelled'):
            lstm_statuses[market] = {"status": "cancelled", "message": "학습이 취소되었습니다.", "result": results}
            return

        await asyncio.to_thread(predictor.save)
        lstm_predictors[market] = predictor

        # 학습 완료 상태에 저장된 모델 이름을 포함
        saved_model_name = f"lstm_{coin}.pt"
        lstm_statuses[market] = {"status": "done", "message": "학습 완료", "result": results, 'logs': lstm_statuses[market].get('logs', []), 'saved_model': saved_model_name}

    except Exception as e:
        lstm_statuses[market] = {"status": "error", "message": str(e), 'logs': lstm_statuses[market].get('logs', [])}


@app.post("/api/lstm/train")
async def api_lstm_train(req: LSTMTrainRequest):
    # markets 배열 또는 market 문자열 지원 (이전 호환성)
    markets = req.markets if req.markets else ([req.market] if req.market else [])
    
    if not markets:
        return {"success": False, "error": "학습할 코인을 선택해주세요"}
    
    # 각 마켓이 이미 완료되었거나 대기열에 있는지 확인
    already_in_queue = [m for m in markets if any(job['market'] == m for job in lstm_queue)]
    already_running = [m for m in markets if m in lstm_tasks and not lstm_tasks[m].done()]
    
    if already_running:
        return {"success": False, "error": f"이미 학습이 진행 중인 마켓: {', '.join(already_running[:5])}"}
    
    # 대기열에 작업 추가
    added_count = 0
    for market in markets:
        # 이미 진행 중이 아니면 대기열에 추가
        if market not in lstm_tasks or lstm_tasks[market].done():
            job = {
                "market": market,
                "epochs": req.epochs,
                "batch_size": req.batch_size,
                "learning_rate": req.learning_rate,
                "seq_length": req.seq_length,
                "days": req.days,
                "resume": req.resume,
                "checkpoint_interval": req.checkpoint_interval
            }
            lstm_queue.append(job)
            lstm_statuses[market] = {
                "status": "queued", 
                "message": f"대기 중 (위치: {len(lstm_queue)})",
                "queue_position": len(lstm_queue) - 1
            }
            added_count += 1
    
    running_count = sum(1 for m in lstm_tasks.values() if not m.done())
    
    return {
        "success": True, 
        "message": f"{added_count}개 코인을 학습 대기열에 추가했습니다. (현재 진행: {running_count}/{lstm_max_concurrent}, 대기: {len(lstm_queue)})",
        "markets": markets
    }


@app.get('/api/lstm/status')
async def api_lstm_status(market: str):
    st = lstm_statuses.get(market, {"status": "idle", "message": "대기 중"})
    running = market in lstm_tasks and not lstm_tasks[market].done()
    if running:
        st["status"] = "running"
    st["model_loaded"] = market in lstm_predictors
    return {"success": True, "data": st}


@app.get('/api/lstm/status-all')
async def api_lstm_status_all():
    """전체 LSTM 학습 상태를 조회합니다."""
    running_count = sum(1 for m in lstm_tasks.values() if not m.done())
    completed_count = sum(1 for st in lstm_statuses.values() if st.get("status") == "done")
    error_count = sum(1 for st in lstm_statuses.values() if st.get("status") == "error")
    queued_count = len(lstm_queue)
    
    return {
        "success": True,
        "data": {
            "running": running_count,
            "completed": completed_count,
            "error": error_count,
            "queued": queued_count,
            "max_concurrent": lstm_max_concurrent,
            "queue_position": queued_count,
            "statuses": lstm_statuses
        }
    }


@app.get('/api/lstm/models')
async def api_lstm_models(market: str):
    """해당 마켓으로 저장된 LSTM 모델 파일 목록을 반환합니다."""
    try:
        coin = market.replace('KRW-', '').lower()
        models_dir = 'models'
        models_list = []
        if os.path.exists(models_dir):
            for fname in os.listdir(models_dir):
                if not fname.startswith(f'lstm_{coin}'):
                    continue
                is_best = '_best' in fname
                is_latest = fname == f'lstm_{coin}.pt'
                epoch = None
                m = re.search(r'_epoch(\d+)', fname)
                if m:
                    try:
                        epoch = int(m.group(1))
                    except Exception:
                        epoch = None
                models_list.append({'file': fname, 'epoch': epoch, 'is_best': is_best, 'is_latest': is_latest})
        models_list.sort(key=lambda x: (x['epoch'] if x['epoch'] else 0), reverse=True)
        return {"success": True, "data": models_list}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post('/api/lstm/load')
async def api_lstm_load(payload: dict):
    """지정된 모델 파일을 로드하여 메모리 내에 배치합니다."""
    market = payload.get('market')
    model_file = payload.get('model')
    seq_length = int(payload.get('seq_length', 60))
    if not market or not model_file:
        return {"success": False, "error": "market and model parameters required"}
    coin = market.replace('KRW-', '').lower()
    models_dir = 'models'
    model_path = os.path.join(models_dir, model_file)
    # 모델 파일이 없으면 가장 최신 epoch 체크포인트를 찾아 폴백 시도
    if not os.path.exists(model_path):
        try:
            # 검색 패턴: lstm_{coin}_epoch*.pt
            candidates = []
            if os.path.exists(models_dir):
                for fname in os.listdir(models_dir):
                    if fname.startswith(f"lstm_{coin}_epoch") and fname.endswith('.pt'):
                        try:
                            epoch_str = fname.split('_epoch')[-1].split('.pt')[0]
                            epoch = int(epoch_str)
                            candidates.append((epoch, fname))
                        except Exception:
                            continue
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                fallback_fname = candidates[0][1]
                model_file = fallback_fname
                model_path = os.path.join(models_dir, model_file)
                # 안내 메시지를 포함해 계속 진행
                fallback_used_msg = f"Requested model not found; using latest epoch checkpoint: {model_file}"
            else:
                return {"success": False, "error": "모델 파일이 없습니다. 학습을 먼저 실행하여 모델을 생성하세요."}
        except Exception as e:
            return {"success": False, "error": f"모델 파일 조회 중 오류: {e}"}

    if LSTMPredictor is None:
        return {"success": False, "error": "LSTM 모듈을 사용할 수 없습니다."}
    try:
        predictor = LSTMPredictor(model_path=f"models/lstm_{coin}", seq_length=seq_length, device=get_device())
        if model_file != f"lstm_{coin}.pt":
            await asyncio.to_thread(predictor._load_checkpoint, model_path)
        else:
            ok = await asyncio.to_thread(predictor.load)
            if not ok:
                return {"success": False, "error": "모델 로드 실패"}
        lstm_predictors[market] = predictor
        resp = {"success": True, "message": "모델 로드 완료", "model": model_file}
        if 'fallback_used_msg' in locals():
            resp['warning'] = fallback_used_msg
        return resp
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.delete('/api/lstm/models')
async def api_lstm_delete(payload: dict):
    """저장된 모델 파일을 삭제합니다."""
    market = payload.get('market')
    model_file = payload.get('model')
    if not market or not model_file:
        return {"success": False, "error": "market and model parameters required"}
    path = os.path.join('models', model_file)
    if not os.path.exists(path):
        return {"success": False, "error": "model file not found"}
    try:
        # 만약 기본 파일(lstm_{coin}.pt)을 삭제하면 메모리에서 언로드
        coin = market.replace('KRW-', '').lower()
        base_name = f"lstm_{coin}.pt"
        if model_file == base_name and market in lstm_predictors:
            try:
                del lstm_predictors[market]
            except Exception:
                pass
        os.remove(path)
        return {"success": True, "message": "모델 삭제 완료"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post('/api/lstm/predict')
async def api_lstm_predict(payload: dict):
    market = payload.get('market')
    seq_length = int(payload.get('seq_length', 60))

    if not market:
        return {"success": False, "error": "market 파라미터가 필요합니다."}

    df = analyzer.get_market_data(market, days=max(seq_length + 30, 120))
    if df is None:
        return {"success": False, "error": "데이터를 가져오지 못했습니다."}

    predictor = lstm_predictors.get(market)
    if predictor is None:
        # 저장된 모델이 있는지 시도 로드
        coin = market.replace('KRW-', '').lower()
        model_path = f'models/lstm_{coin}'
        if LSTMPredictor is None:
            return {"success": False, "error": "LSTM 모듈을 사용할 수 없습니다."}
        predictor = LSTMPredictor(model_path=model_path, seq_length=seq_length, device=get_device())
        if not predictor.load():
            return {"success": False, "error": "모델 파일이 없습니다."}
        lstm_predictors[market] = predictor

    try:
        pred_price, change_rate, direction = await asyncio.to_thread(predictor.predict, df)
        return {"success": True, "data": {"predicted_price": pred_price, "change_rate": change_rate, "direction": direction}}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post('/api/lstm/cancel')
async def api_lstm_cancel(payload: dict):
    market = payload.get('market')
    if not market:
        return {"success": False, "error": "market 파라미터가 필요합니다."}

    predictor = lstm_predictors.get(market)
    if predictor is None:
        # 모델이 로드되어 있지 않더라도 작업 중인 태스크가 있으면 취소 요청
        task = lstm_tasks.get(market)
        if task and not task.done():
            task.cancel()
            lstm_statuses[market] = {"status": "cancel_requested", "message": "취소 요청됨"}
            return {"success": True, "message": "취소 요청되었습니다."}
        return {"success": False, "error": "해당 마켓으로 진행중인 학습이 없습니다."}

    # 요청된 모델이 있을 경우 요청으로 중단시키기
    try:
        if hasattr(predictor, 'request_stop'):
            predictor.request_stop()
            lstm_statuses[market] = {"status": "cancel_requested", "message": "취소 요청됨"}
            return {"success": True, "message": "취소 요청되었습니다."}
        else:
            return {"success": False, "error": "이 모델은 중단을 지원하지 않습니다."}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get('/api/lstm/stream')
async def api_lstm_stream(market: str):
    """Server-Sent Events로 로그/상태 스트리밍 제공"""
    async def event_generator():
        last_index = 0
        while True:
            st = lstm_statuses.get(market, {})
            logs = st.get('logs', [])

            # 새로운 로그 전송
            if last_index < len(logs):
                for msg in logs[last_index:]:
                    payload = json.dumps({'type': 'log', 'message': msg, 'status': st.get('status')})
                    yield f"data: {payload}\n\n"
                last_index = len(logs)

            # 상태 전송
            payload = json.dumps({'type': 'status', 'status': st.get('status'), 'message': st.get('message'), 'resource': st.get('resource')})
            yield f"data: {payload}\n\n"

            # 종료 조건
            if st.get('status') in ('done', 'error', 'cancelled') and last_index >= len(logs):
                break

            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type='text/event-stream')


# ========== LSTM 단타 자동매매 API ==========

class LSTMScalpingStartRequest(BaseModel):
    """LSTM 단타 시작 요청"""
    markets: List[str] = []
    trade_amount: float = 50000
    max_positions: int = 5
    trading_interval: int = 30
    lstm_weight: float = 0.6
    use_dynamic_stops: bool = True
    crash_detection: bool = True
    base_profit_percent: float = 0.5
    base_loss_percent: float = 0.3
    use_unified_model: bool = False  # 통합 LSTM 모델 사용 여부
    auto_discover: bool = False  # 자동 코인 탐색 모드
    top_coin_count: int = 10  # 자동 탐색 시 모니터링할 상위 코인 수
    scan_interval: int = 300  # 자동 탐색 간격 (초)


class UnifiedPredictorWrapper:
    """통합 LSTM 모델을 개별 predictor 인터페이스로 감싸는 래퍼 클래스"""
    def __init__(self, unified_pred, market: str):
        self.unified_predictor = unified_pred
        self.market = market

    def predict(self, df):
        """
        LSTMScalpingSignal이 기대하는 인터페이스로 예측 수행
        Returns: (predicted_price, change_rate, direction)
        """
        try:
            # 통합 모델의 predict는 DataFrame을 받음
            result = self.unified_predictor.predict(df)
            if result and result.get('success'):
                return (
                    result.get('predicted_price', df['close'].iloc[-1]),
                    result.get('predicted_change_rate', 0),
                    (result.get('direction_en') or 'NEUTRAL').lower()  # UP/DOWN/NEUTRAL -> up/down/neutral
                )
        except Exception as e:
            print(f"[UnifiedWrapper] {self.market} 예측 오류: {e}")

        # 예측 실패 시 기본값 반환
        return (df['close'].iloc[-1] if len(df) > 0 else 0, 0, 'neutral')


class AutoDiscoverPredictorDict(dict):
    """자동 탐색 모드용 동적 predictor 딕셔너리

    어떤 마켓에 대해서도 동적으로 UnifiedPredictorWrapper를 생성하여 반환
    """
    def __init__(self, unified_predictor):
        super().__init__()
        self.unified_predictor = unified_predictor
        self._cache = {}

    def get(self, market, default=None):
        """마켓에 대한 predictor 반환 (없으면 동적 생성)"""
        if market not in self._cache:
            self._cache[market] = UnifiedPredictorWrapper(self.unified_predictor, market)
        return self._cache[market]

    def __getitem__(self, market):
        return self.get(market)

    def __contains__(self, market):
        # 모든 마켓에 대해 predictor 제공 가능
        return True

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()


@app.post("/api/lstm-scalping/start")
async def start_lstm_scalping(req: LSTMScalpingStartRequest):
    """LSTM 단타 자동매매 시작"""
    global lstm_scalping_bot, lstm_scalping_task, lstm_scalping_status

    if not LSTM_SCALPING_AVAILABLE:
        return {"success": False, "error": "LSTM 단타 봇 모듈이 로드되지 않았습니다."}

    if lstm_scalping_status["is_running"]:
        return {"success": False, "error": "이미 LSTM 단타 자동매매가 실행 중입니다."}

    # API 키 확인
    if not UPBIT_ACCESS_KEY or not UPBIT_SECRET_KEY:
        return {"success": False, "error": "API 키가 설정되지 않았습니다. .env 파일을 확인하세요."}

    # 마켓 확인 (자동 탐색 모드가 아닌 경우에만 필수)
    if not req.markets and not req.auto_discover:
        return {"success": False, "error": "거래할 코인을 선택하거나 자동 탐색 모드를 활성화해주세요."}

    loaded_predictors = {}
    use_unified = req.use_unified_model or req.auto_discover  # 자동 탐색은 통합 모델 필수

    # 통합 LSTM 모델 사용
    if use_unified:
        if not UNIFIED_LSTM_AVAILABLE or unified_predictor is None:
            return {"success": False, "error": "통합 LSTM 모델이 로드되지 않았습니다."}

        # 통합 모델 로드 확인
        if not unified_predictor.load_model():
            return {"success": False, "error": "통합 LSTM 모델 로드에 실패했습니다. 먼저 모델을 학습해주세요."}

        if req.auto_discover:
            print(f"[LSTM-SCALPING] 자동 탐색 모드 (통합 모델)")
            # 자동 탐색 모드: 동적으로 마켓에 대한 래퍼 생성 가능하도록 설정
            # 초기에는 빈 상태로 시작, 봇이 자동으로 마켓 탐색
        else:
            print(f"[LSTM-SCALPING] 통합 모델 사용: {req.markets}")

        # 선택된 마켓에 대해 통합 모델 래퍼 생성
        for market in req.markets:
            loaded_predictors[market] = UnifiedPredictorWrapper(unified_predictor, market)

    # 개별 LSTM 모델 사용
    else:
        for market in req.markets:
            coin = market.replace('KRW-', '').lower()
            model_path = f'models/lstm_{coin}'

            # 이미 로드된 예측기 확인
            if market in lstm_predictors and lstm_predictors[market] is not None:
                loaded_predictors[market] = lstm_predictors[market]
            else:
                # 모델 파일 존재 확인
                if os.path.exists(f'{model_path}_best.pt') or os.path.exists(f'{model_path}.pt'):
                    try:
                        predictor_obj = LSTMPredictor(model_path=model_path)
                        if predictor_obj.load():
                            loaded_predictors[market] = predictor_obj
                            lstm_predictors[market] = predictor_obj
                        else:
                            print(f"[LSTM-SCALPING] {market} 모델 로드 실패")
                    except Exception as e:
                        print(f"[LSTM-SCALPING] {market} 모델 로드 오류: {e}")
                else:
                    print(f"[LSTM-SCALPING] {market} 모델 파일 없음")

    # 자동 탐색 모드에서는 predictor 없이도 시작 가능
    if not loaded_predictors and not req.auto_discover:
        if use_unified:
            return {"success": False, "error": "통합 모델로 예측기를 생성할 수 없습니다."}
        return {"success": False, "error": "로드된 LSTM 모델이 없습니다. 먼저 모델을 학습해주세요."}

    # 실제 거래 클라이언트 생성
    trading_client = UpbitClient(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)

    # 설정 생성
    config = LSTMScalpingConfig(
        trade_amount=req.trade_amount,
        max_positions=req.max_positions,
        lstm_weight=req.lstm_weight,
        technical_weight=1.0 - req.lstm_weight,
        use_dynamic_stops=req.use_dynamic_stops,
        crash_detection_enabled=req.crash_detection,
        base_profit_percent=req.base_profit_percent,
        base_loss_percent=req.base_loss_percent
    )

    # 봇 생성
    lstm_scalping_bot = LSTMScalpingBot(
        client=trading_client,
        lstm_predictors=loaded_predictors,
        config=config
    )

    # 자동 탐색 모드 설정
    if req.auto_discover:
        lstm_scalping_bot.auto_discover = True
        lstm_scalping_bot.top_coin_count = req.top_coin_count
        lstm_scalping_bot.scan_interval = req.scan_interval
        # 자동 탐색용 동적 predictor 생성 함수 설정
        lstm_scalping_bot.lstm_predictors = AutoDiscoverPredictorDict(unified_predictor)

    # 콜백 설정
    lstm_scalping_bot.on_trade_callback = broadcast_lstm_scalping_trade
    lstm_scalping_bot.on_status_update_callback = broadcast_lstm_scalping_status
    lstm_scalping_bot.on_signal_callback = broadcast_lstm_scalping_signal  # 신호 콜백 추가

    # 상태 업데이트
    lstm_scalping_status["is_running"] = True
    lstm_scalping_status["markets"] = list(loaded_predictors.keys()) if not req.auto_discover else []
    lstm_scalping_status["start_time"] = datetime.now().isoformat()
    lstm_scalping_status["use_unified_model"] = use_unified
    lstm_scalping_status["auto_discover"] = req.auto_discover
    lstm_scalping_status["signal_log"] = []  # 신호 로그 초기화
    lstm_scalping_status["config"] = {
        "trade_amount": req.trade_amount,
        "max_positions": req.max_positions,
        "lstm_weight": req.lstm_weight,
        "crash_detection": req.crash_detection,
        "dynamic_stops": req.use_dynamic_stops,
        "use_unified_model": use_unified,
        "auto_discover": req.auto_discover,
        "top_coin_count": req.top_coin_count
    }

    # 비동기 루프 시작
    lstm_scalping_task = asyncio.create_task(
        lstm_scalping_bot.run_loop(
            markets=list(loaded_predictors.keys()),
            interval=req.trading_interval
        )
    )

    # Task 완료 감시 시작
    asyncio.create_task(monitor_lstm_scalping_task())

    if req.auto_discover:
        return {
            "success": True,
            "message": f"LSTM 단타 자동매매 시작 (자동 탐색 모드, 상위 {req.top_coin_count}개 코인)",
            "markets": [],
            "auto_discover": True,
            "use_unified_model": True
        }

    model_type = "통합 모델" if use_unified else "개별 모델"
    return {
        "success": True,
        "message": f"LSTM 단타 자동매매 시작 ({model_type}, {len(loaded_predictors)}개 코인)",
        "markets": list(loaded_predictors.keys()),
        "use_unified_model": use_unified
    }


@app.post("/api/lstm-scalping/stop")
async def stop_lstm_scalping():
    """LSTM 단타 자동매매 중지"""
    global lstm_scalping_bot, lstm_scalping_task, lstm_scalping_status

    if not lstm_scalping_status["is_running"]:
        return {"success": False, "error": "실행 중인 LSTM 단타 자동매매가 없습니다."}

    if lstm_scalping_bot:
        lstm_scalping_bot.stop()

    if lstm_scalping_task:
        lstm_scalping_task.cancel()
        try:
            await lstm_scalping_task
        except asyncio.CancelledError:
            pass

    lstm_scalping_status["is_running"] = False

    return {"success": True, "message": "LSTM 단타 자동매매가 중지되었습니다."}


@app.get("/api/lstm-scalping/status")
async def get_lstm_scalping_status():
    """LSTM 단타 자동매매 상태 조회"""
    global lstm_scalping_bot, lstm_scalping_status, lstm_scalping_task

    # Task 실제 상태 확인 (완료되었으면 is_running을 False로)
    if lstm_scalping_task and lstm_scalping_task.done():
        lstm_scalping_status["is_running"] = False

    if lstm_scalping_bot and lstm_scalping_status["is_running"]:
        bot_status = lstm_scalping_bot.get_status()
        lstm_scalping_status["positions"] = bot_status.get("positions", {})
        lstm_scalping_status["stats"] = bot_status.get("stats", {})
        lstm_scalping_status["trade_history"] = lstm_scalping_bot.trade_history[-50:]

    return lstm_scalping_status


@app.post("/api/lstm-scalping/emergency-sell")
async def emergency_sell_all_positions():
    """긴급 전체 매도"""
    global lstm_scalping_bot

    if not lstm_scalping_bot or not lstm_scalping_status["is_running"]:
        return {"success": False, "error": "실행 중인 LSTM 단타 자동매매가 없습니다."}

    try:
        results = await lstm_scalping_bot.emergency_sell_all()
        return {
            "success": True,
            "message": f"{len(results)}개 포지션 긴급 매도 완료",
            "results": results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/lstm-scalping/positions")
async def get_lstm_scalping_positions():
    """현재 포지션 조회"""
    global lstm_scalping_bot

    if not lstm_scalping_bot:
        return {"success": True, "data": {}}

    return {"success": True, "data": lstm_scalping_bot.get_status().get("positions", {})}


@app.get("/api/lstm-scalping/history")
async def get_lstm_scalping_history(limit: int = 50):
    """거래 내역 조회"""
    global lstm_scalping_bot

    if not lstm_scalping_bot:
        return {"success": True, "data": []}

    history = lstm_scalping_bot.trade_history[-limit:] if lstm_scalping_bot.trade_history else []
    return {"success": True, "data": history}


@app.get("/api/lstm-scalping/trained-models")
async def get_trained_lstm_models():
    """학습된 LSTM 모델 목록 조회"""
    models = []
    models_dir = "models"

    if os.path.exists(models_dir):
        for filename in os.listdir(models_dir):
            if filename.startswith("lstm_") and filename.endswith("_best.pt"):
                coin = filename.replace("lstm_", "").replace("_best.pt", "").upper()
                market = f"KRW-{coin}"
                models.append({
                    "market": market,
                    "coin": coin,
                    "filename": filename,
                    "path": os.path.join(models_dir, filename)
                })

    return {"success": True, "data": models}


@app.websocket("/ws/lstm-scalping")
async def websocket_lstm_scalping(websocket: WebSocket):
    """LSTM 단타 실시간 상태 WebSocket"""
    await websocket.accept()
    lstm_scalping_ws_clients.append(websocket)

    try:
        while True:
            # 상태 전송
            if lstm_scalping_bot and lstm_scalping_status["is_running"]:
                status = lstm_scalping_bot.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": {
                        "is_running": lstm_scalping_status["is_running"],
                        "positions": status.get("positions", {}),
                        "stats": status.get("stats", {}),
                        "markets": lstm_scalping_status.get("markets", []),
                        "signal_log": lstm_scalping_status.get("signal_log", [])[-20:]
                    }
                })

            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] LSTM 단타 WebSocket 오류: {e}")
    finally:
        if websocket in lstm_scalping_ws_clients:
            lstm_scalping_ws_clients.remove(websocket)


async def broadcast_lstm_scalping_trade(trade_data: Dict):
    """거래 발생 시 모든 WebSocket 클라이언트에 전송"""
    message = {"type": "trade", "data": trade_data}
    for ws in lstm_scalping_ws_clients[:]:
        try:
            await ws.send_json(message)
        except Exception:
            if ws in lstm_scalping_ws_clients:
                lstm_scalping_ws_clients.remove(ws)


async def broadcast_lstm_scalping_status(status_data: Dict):
    """상태 업데이트 시 모든 WebSocket 클라이언트에 전송"""
    global lstm_scalping_status
    lstm_scalping_status["positions"] = status_data.get("positions", {})
    lstm_scalping_status["stats"] = status_data.get("stats", {})

    message = {"type": "status_update", "data": lstm_scalping_status}
    for ws in lstm_scalping_ws_clients[:]:
        try:
            await ws.send_json(message)
        except Exception:
            if ws in lstm_scalping_ws_clients:
                lstm_scalping_ws_clients.remove(ws)


async def broadcast_lstm_scalping_signal(signal_data: Dict):
    """신호 발생 시 로그 추가 및 WebSocket 전송"""
    global lstm_scalping_status

    # 로그에 추가 (최대 100개 유지)
    lstm_scalping_status["signal_log"].append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "market": signal_data.get("market") or "UNKNOWN",
        "signal": signal_data.get("signal") or "UNKNOWN",
        "confidence": signal_data.get("confidence", 0),
        "reason": signal_data.get("reason", "")
    })

    # 최신 100개만 유지
    if len(lstm_scalping_status["signal_log"]) > 100:
        lstm_scalping_status["signal_log"] = lstm_scalping_status["signal_log"][-100:]

    # WebSocket 전송
    message = {"type": "signal", "data": signal_data}
    for ws in lstm_scalping_ws_clients[:]:
        try:
            await ws.send_json(message)
        except Exception:
            if ws in lstm_scalping_ws_clients:
                lstm_scalping_ws_clients.remove(ws)


async def monitor_lstm_scalping_task():
    """Task 완료 감시 및 상태 동기화"""
    global lstm_scalping_task, lstm_scalping_status
    try:
        if lstm_scalping_task:
            await lstm_scalping_task
    except asyncio.CancelledError:
        print("[LSTM-SCALPING] Task가 취소되었습니다")
    except Exception as e:
        print(f"[LSTM-SCALPING] Task 오류: {e}")
    finally:
        lstm_scalping_status["is_running"] = False
        print("[LSTM-SCALPING] Task 완료, 상태를 is_running=False로 변경")
        # WebSocket으로 클라이언트에 알림
        try:
            await broadcast_lstm_scalping_status(lstm_scalping_status)
        except Exception as e:
            print(f"[LSTM-SCALPING] 상태 브로드캐스트 오류: {e}")


# ========== 통합 LSTM 모델 API ==========

# 통합 모델 학습 상태
unified_lstm_training_status = {
    "is_training": False,
    "progress": 0,
    "total_epochs": 0,
    "current_epoch": 0,
    "loss": 0,
    "accuracy": 0,
    "message": ""
}


@app.post("/api/unified-lstm/train")
async def train_unified_lstm(request: Request):
    """통합 LSTM 모델 학습 (여러 코인 데이터로 하나의 모델)"""
    global unified_lstm_training_status

    if not UNIFIED_LSTM_AVAILABLE:
        return JSONResponse({"success": False, "error": "통합 LSTM 모듈이 로드되지 않았습니다"}, status_code=500)

    if unified_predictor.is_training:
        return JSONResponse({"success": False, "error": "이미 학습 중입니다"}, status_code=400)

    try:
        data = await request.json()
        markets = data.get("markets", [])
        epochs = data.get("epochs", 100)
        min_coins = data.get("min_coins", 5)

        if len(markets) < min_coins:
            return {"success": False, "error": f"최소 {min_coins}개 코인이 필요합니다"}

        # 데이터 수집
        all_data = {}
        unified_lstm_training_status["message"] = "데이터 수집 중..."

        for market in markets:
            candles = client.get_candles_minute(market, unit=1, count=200)
            if candles and len(candles) >= 100:
                # DataFrame으로 변환
                df = pd.DataFrame(candles)
                # 컬럼명 통일
                df = df.rename(columns={
                    'opening_price': 'open',
                    'high_price': 'high',
                    'low_price': 'low',
                    'trade_price': 'close',
                    'candle_acc_trade_volume': 'volume'
                })
                # 시간순 정렬 (오래된 데이터가 먼저)
                df = df.sort_values('candle_date_time_kst').reset_index(drop=True)
                all_data[market] = df

        if len(all_data) < min_coins:
            return {"success": False, "error": f"충분한 데이터가 있는 코인이 {min_coins}개 미만입니다"}

        # 비동기 학습 시작
        unified_lstm_training_status["is_training"] = True
        unified_lstm_training_status["total_epochs"] = epochs

        async def training_callback(progress):
            unified_lstm_training_status["current_epoch"] = progress["epoch"]
            unified_lstm_training_status["loss"] = progress["train_loss"]
            unified_lstm_training_status["accuracy"] = progress["val_accuracy"]
            unified_lstm_training_status["message"] = f"학습 중... {progress['epoch']}/{progress['total_epochs']}"

        # 백그라운드 학습
        asyncio.create_task(_run_unified_training(all_data, epochs, training_callback))

        return {
            "success": True,
            "message": f"{len(all_data)}개 코인 데이터로 학습을 시작합니다",
            "num_coins": len(all_data)
        }

    except Exception as e:
        unified_lstm_training_status["is_training"] = False
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


async def _run_unified_training(all_data: Dict, epochs: int, callback):
    """백그라운드에서 통합 모델 학습"""
    global unified_lstm_training_status

    try:
        result = await unified_predictor.train(all_data, epochs=epochs, callback=callback)

        if result["success"]:
            unified_lstm_training_status["message"] = f"학습 완료! 정확도: {result['final_accuracy']:.2%}"
        else:
            unified_lstm_training_status["message"] = f"학습 실패: {result.get('error', '알 수 없는 오류')}"

    except Exception as e:
        unified_lstm_training_status["message"] = f"학습 오류: {str(e)}"
    finally:
        unified_lstm_training_status["is_training"] = False


@app.get("/api/unified-lstm/status")
async def get_unified_lstm_status():
    """통합 LSTM 학습 상태 조회"""
    if not UNIFIED_LSTM_AVAILABLE:
        return {"available": False, "error": "통합 LSTM 모듈이 로드되지 않았습니다"}

    # 모델 파일 확인
    model_exists = os.path.exists("models/unified_lstm_latest.pt")

    return {
        "available": True,
        "model_loaded": unified_predictor.model is not None,
        "model_exists": model_exists,
        "is_training": unified_lstm_training_status["is_training"],
        "current_epoch": unified_lstm_training_status["current_epoch"],
        "total_epochs": unified_lstm_training_status["total_epochs"],
        "loss": unified_lstm_training_status["loss"],
        "accuracy": unified_lstm_training_status["accuracy"],
        "message": unified_lstm_training_status["message"]
    }


@app.post("/api/unified-lstm/predict")
async def predict_unified_lstm(request: Request):
    """통합 모델로 예측"""
    if not UNIFIED_LSTM_AVAILABLE:
        return JSONResponse({"success": False, "error": "통합 LSTM 모듈이 로드되지 않았습니다"}, status_code=500)

    try:
        data = await request.json()
        market = data.get("market", "KRW-BTC")

        # 데이터 가져오기
        candles = client.get_candles_minute(market, unit=1, count=100)
        if not candles or len(candles) < 60:
            return {"success": False, "error": "데이터를 가져올 수 없습니다"}

        # DataFrame으로 변환
        df = pd.DataFrame(candles)
        df = df.rename(columns={
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        })
        df = df.sort_values('candle_date_time_kst').reset_index(drop=True)

        # 예측
        result = unified_predictor.predict(df)

        if result["success"]:
            # numpy 타입을 Python 기본 타입으로 변환
            return {
                "success": True,
                "market": market,
                "direction": result["direction"],
                "direction_en": result["direction_en"],
                "predicted_change_rate": float(result["predicted_change_rate"]),
                "current_price": float(result["current_price"]),
                "predicted_price": float(result["predicted_price"]),
                "confidence": float(result["confidence"]),
                "probabilities": {k: float(v) for k, v in result["probabilities"].items()}
            }
        else:
            return result

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/unified-lstm/load")
async def load_unified_lstm_model():
    """통합 모델 로드"""
    if not UNIFIED_LSTM_AVAILABLE:
        return JSONResponse({"success": False, "error": "통합 LSTM 모듈이 로드되지 않았습니다"}, status_code=500)

    try:
        success = unified_predictor.load_model()
        if success:
            return {"success": True, "message": "모델이 로드되었습니다"}
        else:
            return {"success": False, "error": "모델 파일을 찾을 수 없습니다"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ========== 모델 파일 관리 API ==========

@app.get("/api/models/list")
async def list_all_models():
    """모든 모델 파일 목록 조회"""
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            return {"success": True, "models": [], "total_size": 0}

        models = []
        total_size = 0

        for filename in os.listdir(models_dir):
            if filename.endswith('.pt'):
                filepath = os.path.join(models_dir, filename)
                stat = os.stat(filepath)
                size_mb = stat.st_size / (1024 * 1024)
                total_size += stat.st_size

                # 모델 타입 분류
                if filename.startswith('unified_lstm'):
                    model_type = 'unified'
                elif 'epoch' in filename.lower():
                    model_type = 'checkpoint'
                elif 'best' in filename.lower():
                    model_type = 'best'
                else:
                    model_type = 'individual'

                # 코인 이름 추출
                coin = None
                if model_type == 'individual':
                    parts = filename.replace('.pt', '').split('_')
                    if len(parts) >= 2:
                        coin = parts[1] if parts[0] == 'lstm' else parts[0]

                models.append({
                    "filename": filename,
                    "filepath": filepath,
                    "size_mb": round(size_mb, 2),
                    "size_bytes": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "model_type": model_type,
                    "coin": coin
                })

        # 수정 시간 기준 정렬 (최신순)
        models.sort(key=lambda x: x['modified'], reverse=True)

        return {
            "success": True,
            "models": models,
            "total_count": len(models),
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        }

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/models/delete")
async def delete_model(request: Request):
    """모델 파일 삭제"""
    try:
        data = await request.json()
        filename = data.get("filename")

        if not filename:
            return {"success": False, "error": "파일명이 필요합니다"}

        # 보안: 경로 탐색 방지
        if '..' in filename or '/' in filename or '\\' in filename:
            return {"success": False, "error": "잘못된 파일명입니다"}

        filepath = os.path.join("models", filename)

        if not os.path.exists(filepath):
            return {"success": False, "error": "파일을 찾을 수 없습니다"}

        os.remove(filepath)
        return {"success": True, "message": f"{filename} 삭제 완료"}

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/models/delete-bulk")
async def delete_models_bulk(request: Request):
    """여러 모델 파일 일괄 삭제"""
    try:
        data = await request.json()
        filenames = data.get("filenames", [])

        if not filenames:
            return {"success": False, "error": "삭제할 파일이 없습니다"}

        deleted = []
        errors = []

        for filename in filenames:
            # 보안: 경로 탐색 방지
            if '..' in filename or '/' in filename or '\\' in filename:
                errors.append(f"{filename}: 잘못된 파일명")
                continue

            filepath = os.path.join("models", filename)

            if not os.path.exists(filepath):
                errors.append(f"{filename}: 파일 없음")
                continue

            try:
                os.remove(filepath)
                deleted.append(filename)
            except Exception as e:
                errors.append(f"{filename}: {str(e)}")

        return {
            "success": True,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "errors": errors
        }

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/models/cleanup-checkpoints")
async def cleanup_checkpoint_models():
    """체크포인트 모델 정리 (최신 best만 남김)"""
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            return {"success": True, "deleted": 0, "saved_space_mb": 0}

        # 코인별로 그룹화
        coin_models = {}
        for filename in os.listdir(models_dir):
            if not filename.endswith('.pt'):
                continue

            if 'unified' in filename:
                continue  # 통합 모델은 제외

            parts = filename.replace('.pt', '').split('_')
            if len(parts) >= 2:
                coin = parts[1] if parts[0] == 'lstm' else parts[0]
                if coin not in coin_models:
                    coin_models[coin] = []
                coin_models[coin].append(filename)

        deleted = []
        saved_size = 0

        for coin, files in coin_models.items():
            if len(files) <= 1:
                continue

            # 최신 파일만 유지
            files_with_time = []
            for f in files:
                filepath = os.path.join(models_dir, f)
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                is_best = 'best' in f.lower()
                files_with_time.append((f, mtime, size, is_best))

            # best 파일 우선, 그 다음 최신순
            files_with_time.sort(key=lambda x: (not x[3], -x[1]))

            # 첫 번째 파일만 유지, 나머지 삭제
            for f, _, size, _ in files_with_time[1:]:
                if 'epoch' in f.lower():  # 체크포인트만 삭제
                    filepath = os.path.join(models_dir, f)
                    try:
                        os.remove(filepath)
                        deleted.append(f)
                        saved_size += size
                    except:
                        pass

        return {
            "success": True,
            "deleted": deleted,
            "deleted_count": len(deleted),
            "saved_space_mb": round(saved_size / (1024 * 1024), 2)
        }

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ----------------------------------------------------------


@app.get("/api/top-recommendations")
async def get_top_recommendations(limit: int = 10):
    """상위 추천 종목"""
    try:
        markets = analyzer.get_all_krw_markets()[:50]
        results = []

        for market_info in markets:
            market = market_info['market']
            result = analyzer.analyze_market(market, days=30)

            if result and result['score'] > 0:
                df = analyzer.get_market_data(market, days=30)
                if df is not None:
                    ai_result = predictor.predict_market(df, market)
                    result['ai_action'] = ai_result['action']
                    result['ai_confidence'] = ai_result['confidence']
                else:
                    result['ai_action'] = 0
                    result['ai_confidence'] = 0

                # 100점 만점 점수 추가
                result['score_100'] = normalize_score_to_100(result['score'])

                # 매수/매도/손절 가격 계산
                trade_prices = calculate_trade_prices(result)
                result['trade_prices'] = trade_prices

                # NaN/Infinity 제거
                result = clean_dict(result)
                results.append(result)

        results.sort(key=lambda x: x.get('score_100', 0), reverse=True)
        return {"success": True, "data": clean_dict(results[:limit])}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/top-recommendations-stream")
async def get_top_recommendations_stream(limit: int = 10, mode: str = "all"):
    """상위 추천 종목 (진행률 스트리밍)

    Args:
        limit: 결과 개수
        mode: 분석 모드
            - all: 전체 235개 분석 (기본값)
            - volume_top50: 거래량 상위 50개 분석
            - volume_top100: 거래량 상위 100개 분석
    """
    async def generate():
        try:
            all_markets = analyzer.get_all_krw_markets()

            # 모드에 따라 분석할 종목 선택
            if mode == "volume_top50":
                # 거래량 상위 50개
                tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
                sorted_markets = sorted(
                    zip(all_markets[:100], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:50]
                markets = [m[0] for m in sorted_markets]
            elif mode == "volume_top100":
                # 거래량 상위 100개
                tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
                sorted_markets = sorted(
                    zip(all_markets[:150], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:100]
                markets = [m[0] for m in sorted_markets]
            else:
                # 전체 분석
                markets = all_markets

            total = len(markets)
            results = []

            for i, market_info in enumerate(markets, 1):
                market = market_info['market']

                # 진행률 전송
                progress = {
                    "type": "progress",
                    "current": i,
                    "total": total,
                    "market": market
                }
                yield f"data: {json.dumps(progress)}\n\n"

                # 버퍼 플러시를 위한 작은 대기
                await asyncio.sleep(0.01)

                result = analyzer.analyze_market(market, days=30)

                if result and result['score'] > 0:
                    df = analyzer.get_market_data(market, days=30)
                    if df is not None:
                        ai_result = predictor.predict_market(df, market)
                        result['ai_action'] = ai_result['action']
                        result['ai_confidence'] = ai_result['confidence']
                    else:
                        result['ai_action'] = 0
                        result['ai_confidence'] = 0

                    # 100점 만점 점수 추가
                    result['score_100'] = normalize_score_to_100(result['score'])

                    # 매수/매도/손절 가격 계산
                    trade_prices = calculate_trade_prices(result)
                    result['trade_prices'] = trade_prices

                    result = clean_dict(result)
                    results.append(result)

            # 정렬 및 최종 결과 전송 (score_100 기준으로 정렬하여 UI와 일치)
            results.sort(key=lambda x: x.get('score_100', 0), reverse=True)
            final_data = {
                "type": "complete",
                "data": results[:limit]
            }
            yield f"data: {json.dumps(clean_dict(final_data))}\n\n"

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@app.get("/api/news-signal")
async def get_news_signal(query: str = "cryptocurrency"):
    """뉴스 감정 분석 기반 트레이딩 신호 (시장 전체)

    Args:
        query: 검색 쿼리 (기본: cryptocurrency)

    Returns:
        신호 정보 (BUY, SELL, HOLD)
    """
    try:
        signal_data = news_signal_generator.generate_signal(query=query, page_size=100)
        return {"success": True, "data": clean_dict(signal_data)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news-signal/{market}")
async def get_coin_news_signal(market: str):
    """특정 코인의 뉴스 감정 분석 신호

    Args:
        market: 마켓 코드 (예: KRW-BTC, KRW-ETH)

    Returns:
        해당 코인의 뉴스 신호
    """
    try:
        signal_data = news_signal_generator.generate_coin_signal(market=market, page_size=100)
        return {"success": True, "data": clean_dict(signal_data)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news-signal/combined")
async def get_combined_news_signal():
    """종합 뉴스 신호 (비트코인 + 시장 분석)"""
    try:
        signal_data = news_signal_generator.get_combined_signal()
        return {"success": True, "data": clean_dict(signal_data)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/news-feed")
async def get_news_feed(page_size: int = 10):
    """최신 암호화폐 뉴스 피드

    Args:
        page_size: 가져올 뉴스 개수

    Returns:
        뉴스 리스트와 감정 분석 결과
    """
    try:
        from src.news.news_collector import NewsCollector
        from src.news.sentiment_analyzer import SentimentAnalyzer

        collector = NewsCollector()
        analyzer_news = SentimentAnalyzer()

        # 뉴스 수집
        articles = collector.get_crypto_news(page_size=page_size)

        # 감정 분석
        analysis_result = analyzer_news.analyze_news_batch(articles)

        return {
            "success": True,
            "data": {
                "articles": analysis_result['articles'],
                "summary": analysis_result['summary'],
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """실시간 데이터 스트리밍"""
    await manager.connect(websocket)

    # 기본 종목 리스트
    default_markets = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE']
    user_markets = default_markets.copy()
    is_connected = True

    try:
        while is_connected:
            # 클라이언트에서 종목 설정 메시지 수신 (non-blocking)
            try:
                # 0.1초 타임아웃으로 메시지 확인
                message = await asyncio.wait_for(websocket.receive_json(), timeout=0.1)
                if message.get('type') == 'set_markets':
                    new_markets = message.get('markets', [])
                    if new_markets and isinstance(new_markets, list):
                        user_markets = new_markets[:10]  # 최대 10개로 제한
                        print(f"[WS] 종목 설정 변경: {user_markets}")
            except asyncio.TimeoutError:
                pass  # 메시지 없으면 계속 진행
            except WebSocketDisconnect:
                is_connected = False
                break
            except Exception:
                pass

            # 연결 상태 확인 후 데이터 전송
            if not is_connected:
                break

            try:
                if user_markets:
                    tickers = client.get_ticker(user_markets)
                    data = []

                    for ticker in tickers:
                        data.append({
                            'market': ticker['market'],
                            'price': ticker['trade_price'],
                            'change_rate': ticker['signed_change_rate'] * 100,
                            'volume': ticker['acc_trade_volume_24h'],
                            'timestamp': datetime.now().isoformat()
                        })

                    await websocket.send_json({
                        'type': 'price_update',
                        'data': data
                    })
            except WebSocketDisconnect:
                is_connected = False
                break
            except Exception as e:
                # API 오류 등 일시적 오류는 로그만 남기고 계속
                if "websocket" not in str(e).lower():
                    print(f"실시간 데이터 오류: {e}")
                else:
                    is_connected = False
                    break

            await asyncio.sleep(3)

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await manager.disconnect(websocket)
        except Exception:
            pass


# ========== 자동매매 API ==========

class TradingStartRequest(BaseModel):
    market: str = "KRW-BTC"
    interval: int = 60
    max_trade_amount: float = 100000


def get_market_data_for_trading(market: str, count: int = 200) -> pd.DataFrame:
    """트레이딩을 위한 시장 데이터 가져오기"""
    # trading_client가 없으면 공개 API client 사용
    api_client = trading_client if trading_client else client
    candles = api_client.get_candles_minute(market, unit=1, count=count)

    if not candles:
        raise ValueError(f"캔들 데이터를 가져올 수 없습니다: {market}")

    df = pd.DataFrame(candles)
    df = df.rename(columns={
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    })
    df = df.sort_values('candle_date_time_kst').reset_index(drop=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def prepare_observation(candles: pd.DataFrame) -> np.ndarray:
    """관측값 준비"""
    env = CryptoTradingEnv(candles, initial_balance=1000000)
    env.reset()
    return env._get_observation()


async def trading_loop():
    """비동기 트레이딩 루프"""
    global trading_status, trading_agent, trading_client

    print(f"[TRADING] 자동매매 루프 시작 - {trading_status['market']}")

    # AI 모델 로드
    try:
        dummy_data = get_market_data_for_trading(trading_status['market'], 200)
        dummy_env = CryptoTradingEnv(dummy_data)
        trading_agent = TradingAgent(dummy_env)
        trading_agent.load('models/crypto_trader')
        print("[TRADING] AI 모델 로드 완료")
    except Exception as e:
        print(f"[TRADING] AI 모델 로드 실패: {e}")
        trading_agent = None

    # 초기 잔고 기록
    try:
        start_balance = trading_client.get_balance('KRW')
        async with trading_status_lock:
            trading_status['start_balance'] = start_balance
            trading_status['current_balance'] = start_balance
    except Exception as e:
        print(f"[TRADING] 잔고 조회 실패: {e}")

    while trading_status['is_running']:
        try:
            # 시장 데이터 가져오기
            market = trading_status['market']
            df = get_market_data_for_trading(market, 200)
            current_price = float(df.iloc[-1]['close'])

            # 현재 잔고 조회
            krw_balance = trading_client.get_balance('KRW')
            crypto_symbol = market.split('-')[1]
            crypto_balance = trading_client.get_balance(crypto_symbol)
            total_value = krw_balance + crypto_balance * current_price

            # AI 액션 결정
            action = 0  # 기본: Hold
            action_text = "HOLD"

            if trading_agent:
                obs = prepare_observation(df)
                action, _ = trading_agent.predict(obs, deterministic=True)
                action = int(action)
                action_text = ['HOLD', 'BUY', 'SELL'][action]

            # 상태 업데이트 (Lock 보호)
            async with trading_status_lock:
                trading_status['last_price'] = current_price
                trading_status['current_balance'] = total_value
                trading_status['profit'] = total_value - trading_status['start_balance']
                if trading_status['start_balance'] > 0:
                    trading_status['profit_rate'] = (trading_status['profit'] / trading_status['start_balance']) * 100
                else:
                    trading_status['profit_rate'] = 0
                trading_status['last_action'] = action_text
                trading_status['last_action_time'] = datetime.now().isoformat()

            # 액션 실행
            current_position = trading_status['current_position']
            max_trade_amount = trading_status['max_trade_amount']

            if action == 1:  # Buy
                if current_position is None and krw_balance > 5000:
                    trade_amount = min(krw_balance * 0.5, max_trade_amount)
                    if trade_amount >= 5000:
                        try:
                            result = trading_client.buy_market_order(market, trade_amount)
                            if 'error' not in result:
                                trade_record = {
                                    'time': datetime.now().isoformat(),
                                    'action': 'BUY',
                                    'price': current_price,
                                    'amount': trade_amount,
                                    'uuid': result.get('uuid', 'N/A')
                                }
                                async with trading_status_lock:
                                    trading_status['current_position'] = 'long'
                                    trading_status['trade_count'] += 1
                                    trading_status['trade_history'].append(trade_record)
                                print(f"[TRADING] 매수 체결: {trade_amount:,.0f} KRW @ {current_price:,.0f}")

                                # WebSocket으로 브로드캐스트
                                await manager.broadcast({
                                    'type': 'trading_update',
                                    'data': {'action': 'BUY', 'price': current_price, 'amount': trade_amount}
                                })
                        except Exception as e:
                            print(f"[TRADING] 매수 오류: {e}")

            elif action == 2:  # Sell
                if current_position == 'long' and crypto_balance > 0:
                    try:
                        result = trading_client.sell_market_order(market, crypto_balance)
                        if 'error' not in result:
                            trade_record = {
                                'time': datetime.now().isoformat(),
                                'action': 'SELL',
                                'price': current_price,
                                'volume': crypto_balance,
                                'uuid': result.get('uuid', 'N/A')
                            }
                            async with trading_status_lock:
                                trading_status['current_position'] = None
                                trading_status['trade_count'] += 1
                                trading_status['trade_history'].append(trade_record)
                            print(f"[TRADING] 매도 체결: {crypto_balance:.8f} @ {current_price:,.0f}")

                            # WebSocket으로 브로드캐스트
                            await manager.broadcast({
                                'type': 'trading_update',
                                'data': {'action': 'SELL', 'price': current_price, 'volume': crypto_balance}
                            })
                    except Exception as e:
                        print(f"[TRADING] 매도 오류: {e}")

            print(f"[TRADING] {datetime.now().strftime('%H:%M:%S')} - 가격: {current_price:,.0f}, 액션: {action_text}, 수익률: {trading_status['profit_rate']:+.2f}%")

        except Exception as e:
            print(f"[TRADING] 루프 오류: {e}")

        # 지정된 간격만큼 대기
        await asyncio.sleep(trading_status['interval'])

    print("[TRADING] 자동매매 루프 종료")


@app.get("/api/account/check")
async def check_api_connection():
    """API 키 유효성 확인"""
    if not trading_client or not is_valid_api_key(UPBIT_ACCESS_KEY) or not is_valid_api_key(UPBIT_SECRET_KEY):
        return {
            "success": False,
            "connected": False,
            "error": "API 키가 설정되지 않았습니다. .env 파일을 확인하세요."
        }

    try:
        accounts = trading_client.get_accounts()
        if isinstance(accounts, dict) and 'error' in accounts:
            return {
                "success": False,
                "connected": False,
                "error": accounts['error'].get('message', '알 수 없는 오류')
            }

        return {
            "success": True,
            "connected": True,
            "message": "API 연결 성공"
        }
    except Exception as e:
        return {
            "success": False,
            "connected": False,
            "error": str(e)
        }


@app.get("/api/account/balance")
async def get_account_balance():
    """계좌 잔고 조회"""
    if not trading_client:
        return {"success": False, "error": "API 키가 설정되지 않았습니다."}

    try:
        accounts = trading_client.get_accounts()
        if isinstance(accounts, dict) and 'error' in accounts:
            return {"success": False, "error": accounts['error'].get('message', '알 수 없는 오류')}

        balances = []
        for account in accounts:
            balance = float(account['balance'])
            locked = float(account['locked'])
            if balance > 0 or locked > 0:
                balances.append({
                    'currency': account['currency'],
                    'balance': balance,
                    'locked': locked,
                    'avg_buy_price': float(account['avg_buy_price']),
                    'unit_currency': account['unit_currency']
                })

        return {"success": True, "data": balances}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/trading/start")
async def start_trading(request: TradingStartRequest):
    """자동매매 시작"""
    global trading_bot_task, trading_status

    if not trading_client:
        return {"success": False, "error": "API 키가 설정되지 않았습니다."}

    if trading_status['is_running']:
        return {"success": False, "error": "이미 자동매매가 실행 중입니다."}

    # 상태 초기화
    trading_status['is_running'] = True
    trading_status['market'] = request.market
    trading_status['interval'] = request.interval
    trading_status['max_trade_amount'] = request.max_trade_amount
    trading_status['start_time'] = datetime.now().isoformat()
    trading_status['trade_count'] = 0
    trading_status['current_position'] = None
    trading_status['profit'] = 0
    trading_status['profit_rate'] = 0
    trading_status['trade_history'] = []

    # 비동기 트레이딩 루프 시작
    trading_bot_task = asyncio.create_task(trading_loop())

    return {
        "success": True,
        "message": f"자동매매 시작 - {request.market}",
        "data": {
            "market": request.market,
            "interval": request.interval,
            "max_trade_amount": request.max_trade_amount
        }
    }


@app.post("/api/trading/stop")
async def stop_trading():
    """자동매매 중지"""
    global trading_bot_task, trading_status

    if not trading_status['is_running']:
        return {"success": False, "error": "실행 중인 자동매매가 없습니다."}

    trading_status['is_running'] = False

    if trading_bot_task:
        trading_bot_task.cancel()
        try:
            await trading_bot_task
        except asyncio.CancelledError:
            pass
        trading_bot_task = None

    return {
        "success": True,
        "message": "자동매매 중지됨",
        "data": {
            "trade_count": trading_status['trade_count'],
            "profit": trading_status['profit'],
            "profit_rate": trading_status['profit_rate']
        }
    }


@app.get("/api/trading/status")
async def get_trading_status():
    """자동매매 상태 조회"""
    return {
        "success": True,
        "data": clean_dict(trading_status)
    }


@app.get("/api/trading/history")
async def get_trading_history():
    """거래 내역 조회"""
    return {
        "success": True,
        "data": clean_dict(trading_status['trade_history'])
    }


@app.get("/api/trading/realtime-price")
async def get_realtime_price(market: str = "KRW-BTC"):
    """실시간 현재가 조회"""
    try:
        ticker = client.get_ticker([market])
        if ticker and len(ticker) > 0:
            return {
                "success": True,
                "price": ticker[0]['trade_price'],
                "change_rate": ticker[0]['signed_change_rate'] * 100,
                "timestamp": datetime.now().isoformat()
            }
        return {"success": False, "error": "가격 정보 없음"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/api/trading/chart-data")
async def get_chart_data(market: str = "KRW-BTC", count: int = 60):
    """실시간 차트 데이터 조회 (최근 N개 분봉)"""
    try:
        candles = client.get_candles_minute(market, unit=1, count=count)

        chart_data = []
        for candle in reversed(candles):  # 시간순 정렬
            chart_data.append({
                'time': candle['candle_date_time_kst'],
                'price': candle['trade_price'],
                'open': candle['opening_price'],
                'high': candle['high_price'],
                'low': candle['low_price'],
                'close': candle['trade_price'],
                'volume': candle['candle_acc_trade_volume']
            })

        return {"success": True, "data": chart_data}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========== 원클릭 자동매매 API ==========

class AutoTradingStartRequest(BaseModel):
    total_investment: float = 50000
    coin_count: int = 3
    analysis_mode: str = "volume_top50"
    trading_interval: int = 60
    coin_category: str = "normal"  # 'safe', 'normal', 'meme', 'all'
    allocation_mode: str = "weighted"  # 'equal' (균등배분) or 'weighted' (점수기반)
    target_profit_percent: float = 10.0  # 목표가 (+%)
    stop_loss_percent: float = 10.0      # 손절가 (-%, 양수로 입력)


# 밈코인 및 고위험 코인 리스트
MEME_COINS = [
    'DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'MEME', 'BABYDOGE',
    'ELON', 'AKITA', 'KISHU', 'SAMO', 'CATE', 'LEASH', 'BONE',
    'TURBO', 'LADYS', 'AIDOGE', 'BOB', 'WOJAK', 'CHAD', 'TOSHI',
    'COQ', 'MYRO', 'SMOG', 'SLERF', 'BOME', 'MEW', 'POPCAT', 'BRETT',
    '0G', 'NOM', 'HIPPO', 'PNUT', 'ACT', 'VIRTUAL', 'PENGU', 'TRUMP'
]

# 스테이블 코인 리스트 (추천/자동매매에서 제외)
STABLE_COINS = [
    'USDT',   # 테더
    'USDC',   # USD 코인
    'DAI',    # 다이
    'BUSD',   # 바이낸스 USD
    'TUSD',   # 트루 USD
    'USDP',   # 팍스 달러
    'GUSD',   # 제미니 달러
    'FRAX',   # 프랙스
    'USDD',   # USDD
    'UST',    # 테라 USD (디페그됨)
]

# 대형 안전 코인 리스트 (시가총액 상위 6개)
SAFE_COINS = [
    'BTC',   # 비트코인
    'ETH',   # 이더리움
    'XRP',   # 리플
    'SOL',   # 솔라나
    'ADA',   # 카르다노
    'AVAX',  # 아발란체
]


def filter_coins_by_category(markets: list, category: str) -> list:
    """카테고리에 따라 코인 필터링"""
    if category == 'all':
        # 전체 코인 (필터링 없음)
        return markets
    elif category == 'safe':
        # 안전 코인만
        return [m for m in markets if m['market'].split('-')[1] in SAFE_COINS]
    elif category == 'meme':
        # 밈코인만
        return [m for m in markets if m['market'].split('-')[1] in MEME_COINS]
    else:  # 'normal'
        # 밈코인 제외
        return [m for m in markets if m['market'].split('-')[1] not in MEME_COINS]


async def select_top_coins(coin_count: int, mode: str = "volume_top50", category: str = "normal") -> list:
    """상위 N개 코인 선택

    Args:
        coin_count: 선택할 코인 수
        mode: 분석 범위 (volume_top50, volume_top100)
        category: 코인 카테고리 (safe, normal, meme, all)
    """
    global coin_blacklist

    try:
        all_markets = analyzer.get_all_krw_markets()

        # 블랙리스트 제외
        if coin_blacklist:
            all_markets = [m for m in all_markets if m['market'] not in coin_blacklist]
            print(f"[AUTO-TRADING] 블랙리스트 {len(coin_blacklist)}개 제외, 남은 마켓: {len(all_markets)}개")

        # 스테이블 코인 제외
        all_markets = [m for m in all_markets if m['market'].split('-')[1] not in STABLE_COINS]
        print(f"[AUTO-TRADING] 스테이블 코인 제외, 남은 마켓: {len(all_markets)}개")

        # 사용자가 이미 보유 중인 코인 제외 (자동매매가 건들지 않음)
        if trading_client:
            try:
                user_holdings = []
                accounts = trading_client.get_accounts()
                if isinstance(accounts, list):
                    for acc in accounts:
                        currency = acc.get('currency', '')
                        balance = float(acc.get('balance', 0))
                        if currency != 'KRW' and balance > 0:
                            user_holdings.append(f"KRW-{currency}")

                if user_holdings:
                    all_markets = [m for m in all_markets if m['market'] not in user_holdings]
                    print(f"[AUTO-TRADING] 기존 보유 코인 {len(user_holdings)}개 제외: {user_holdings}")
            except Exception as e:
                print(f"[AUTO-TRADING] 보유 코인 조회 실패: {e}")

        # 'safe' 카테고리는 SAFE_COINS에서 직접 선택
        if category == 'safe':
            # SAFE_COINS에 해당하는 마켓만 선택
            safe_markets = []
            for m in all_markets:
                coin_symbol = m['market'].split('-')[1]
                if coin_symbol in SAFE_COINS:
                    safe_markets.append(m)
            markets = safe_markets
            print(f"[AUTO-TRADING] 안전 코인 {len(markets)}개 발견")
        elif category == 'meme':
            # 밈코인만 선택 (거래량 상위)
            tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
            sorted_markets = sorted(
                zip(all_markets[:150], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )
            # 밈코인만 필터링
            markets = [m[0] for m in sorted_markets if m[0]['market'].split('-')[1] in MEME_COINS]
            print(f"[AUTO-TRADING] 밈코인 {len(markets)}개 발견")
        else:
            # 모드에 따라 분석할 종목 선택
            if mode == "volume_top50":
                tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
                sorted_markets = sorted(
                    zip(all_markets[:100], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:50]
                markets = [m[0] for m in sorted_markets]
            elif mode == "volume_top100":
                tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
                sorted_markets = sorted(
                    zip(all_markets[:150], tickers),
                    key=lambda x: x[1].get('acc_trade_price_24h', 0),
                    reverse=True
                )[:100]
                markets = [m[0] for m in sorted_markets]
            else:
                markets = all_markets[:50]  # 기본값

            # 카테고리에 따라 코인 필터링 (normal, all)
            markets = filter_coins_by_category(markets, category)

        print(f"[AUTO-TRADING] 카테고리 '{category}' 필터링 후 {len(markets)}개 코인")

        # 안전 코인 / 밈코인은 간단하게 처리 (분석 실패해도 선택)
        if category in ['safe', 'meme']:
            results = []
            market_codes = [m['market'] for m in markets]
            tickers = client.get_ticker(market_codes)

            # 거래량 기준으로 정렬
            market_ticker_pairs = []
            for market_info in markets:
                market = market_info['market']
                ticker = next((t for t in tickers if t['market'] == market), None)
                if ticker:
                    market_ticker_pairs.append((market_info, ticker))

            # 거래량 내림차순 정렬
            market_ticker_pairs.sort(key=lambda x: x[1].get('acc_trade_price_24h', 0), reverse=True)

            # 순위 기반 점수 부여
            label = '안전 투자' if category == 'safe' else '밈코인'
            base_score = 70 if category == 'safe' else 60

            for i, (market_info, ticker) in enumerate(market_ticker_pairs[:coin_count]):
                # 순위에 따라 점수 차등 부여
                if coin_count <= 3:
                    score_diff = [0, -15, -30]  # 1위, 2위, 3위
                elif coin_count == 4:
                    score_diff = [0, -10, -20, -30]
                else:  # 5개
                    score_diff = [0, -8, -16, -24, -32]

                score = base_score + score_diff[i] if i < len(score_diff) else base_score - 32
                market = market_info['market']

                # AI 기술적 분석 수행
                try:
                    result = analyzer.analyze_market(market, days=30)
                    if result and result.get('current_price', 0) > 0:
                        # 기술적 분석 성공 - AI 기반 가격 계산
                        trade_prices = calculate_trade_prices(result)
                        current_price = result['current_price']
                        recommendation = result.get('recommendation', label)
                        print(f"[AUTO-TRADING] {market} AI 분석 완료: 매수가 ₩{trade_prices['buy_price']:,}")
                    else:
                        # 분석 실패 시 기본값
                        current_price = ticker.get('trade_price', 0)
                        trade_prices = {
                            'buy_price': round(current_price * 0.99),
                            'sell_price': round(current_price * 1.08),
                            'stop_loss': round(current_price * 0.96),
                            'expected_profit_rate': 8.0,
                            'risk_rate': 4.0
                        }
                        recommendation = label
                except Exception as e:
                    print(f"[AUTO-TRADING] {market} 분석 실패: {e}, 기본값 사용")
                    current_price = ticker.get('trade_price', 0)
                    trade_prices = {
                        'buy_price': round(current_price * 0.99),
                        'sell_price': round(current_price * 1.08),
                        'stop_loss': round(current_price * 0.96),
                        'expected_profit_rate': 8.0,
                        'risk_rate': 4.0
                    }
                    recommendation = label

                results.append({
                    'market': market,
                    'name': market_info.get('korean_name', market.split('-')[1]),
                    'current_price': current_price,
                    'score': score,
                    'score_100': score,
                    'recommendation': recommendation,
                    'trade_prices': trade_prices,
                    'ai_action': 0,
                    'ai_confidence': 0
                })

            print(f"[AUTO-TRADING] {label} 점수: {[c['score'] for c in results]}")
            return results

        # 일반/전체 카테고리는 분석 진행
        results = []
        for market_info in markets:
            market = market_info['market']
            try:
                result = analyzer.analyze_market(market, days=30)

                if result and result.get('current_price', 0) > 0:
                    # score가 음수여도 포함 (조건 완화)
                    df = analyzer.get_market_data(market, days=30)
                    if df is not None:
                        ai_result = predictor.predict_market(df, market)
                        result['ai_action'] = ai_result['action']
                        result['ai_confidence'] = ai_result['confidence']
                    else:
                        result['ai_action'] = 0
                        result['ai_confidence'] = 0

                    result['score_100'] = normalize_score_to_100(result.get('score', 0))
                    trade_prices = calculate_trade_prices(result)
                    result['trade_prices'] = trade_prices
                    result = clean_dict(result)
                    results.append(result)
                    print(f"[AUTO-TRADING] {market} 분석 완료: 점수 {result.get('score_100', 0)}")
            except Exception as e:
                print(f"[AUTO-TRADING] 코인 분석 실패 {market}: {e}")
                continue

            # 충분한 코인이 모이면 중단 (성능 최적화)
            if len(results) >= coin_count * 2:
                break

        # score_100 (정규화된 점수) 기준으로 정렬 - UI에 표시되는 점수와 일치
        results.sort(key=lambda x: x.get('score_100', 0), reverse=True)
        print(f"[AUTO-TRADING] 총 {len(results)}개 분석 완료, 상위 {coin_count}개 선택")
        print(f"[AUTO-TRADING] 선택된 코인 점수: {[(r['market'], r.get('score_100', 0)) for r in results[:coin_count]]}")
        return results[:coin_count]
    except Exception as e:
        print(f"[AUTO-TRADING] 코인 선택 실패: {e}")
        return []


async def select_and_allocate_coins():
    """코인 선택 및 투자금 배분"""
    global auto_trading_status

    print("[AUTO-TRADING] 상위 코인 선택 중...")
    coins = await select_top_coins(
        auto_trading_status['coin_count'],
        auto_trading_status.get('analysis_mode', 'volume_top50'),
        auto_trading_status.get('coin_category', 'normal')
    )

    if not coins:
        print("[AUTO-TRADING] 선택된 코인 없음")
        return False

    total = auto_trading_status['total_investment']
    allocation_mode = auto_trading_status.get('allocation_mode', 'weighted')

    # 투자금 배분 계산
    if allocation_mode == 'weighted':
        # 점수 기반 가중치 배분
        # 점수가 0이거나 없는 경우 최소값 10 사용
        scores = [max(coin.get('score_100', 0), 10) for coin in coins]
        total_score = sum(scores)

        # 가중치 비율 계산
        weights = [score / total_score for score in scores]
        allocations = [total * weight for weight in weights]

        print(f"[AUTO-TRADING] 점수 기반 배분: {dict(zip([c['market'] for c in coins], [f'{a:,.0f}원 ({w*100:.1f}%)' for a, w in zip(allocations, weights)]))}")
    else:
        # 균등 배분
        per_coin = total / len(coins)
        allocations = [per_coin] * len(coins)
        print(f"[AUTO-TRADING] 균등 배분: 각 {per_coin:,.0f}원")

    auto_trading_status['positions'] = {}
    auto_trading_status['selected_coins'] = []

    for i, coin in enumerate(coins):
        market = coin['market']
        allocated = allocations[i]
        allocation_percent = (allocated / total) * 100

        auto_trading_status['positions'][market] = {
            'market': market,
            'name': coin.get('name', market.split('-')[1]),
            'allocated_amount': allocated,
            'allocation_percent': round(allocation_percent, 1),
            'entry_price': None,
            'current_price': coin.get('current_price', 0),
            'volume': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'status': 'none',
            'score': coin.get('score_100', 0),
            'recommendation': coin.get('recommendation', ''),
            'trade_prices': coin.get('trade_prices', {}),
            'last_action': None,
            'trade_history': []
        }
        auto_trading_status['selected_coins'].append({
            'market': market,
            'name': coin.get('name', market.split('-')[1]),
            'score': coin.get('score_100', 0),
            'recommendation': coin.get('recommendation', ''),
            'allocated_amount': allocated,
            'allocation_percent': round(allocation_percent, 1)
        })

    print(f"[AUTO-TRADING] 선택된 코인: {[c['market'] for c in auto_trading_status['selected_coins']]}")
    print(f"[AUTO-TRADING] 배분 상세:")
    for coin in auto_trading_status['selected_coins']:
        print(f"  - {coin['market']}: {coin['allocation_percent']}% (₩{coin['allocated_amount']:,.0f})")
    return True


async def replace_sold_coin(sold_market: str):
    """매도된 코인을 새로운 코인으로 교체"""
    global auto_trading_status, coin_blacklist

    print(f"[AUTO-TRADING] {sold_market} 교체 시작...")

    # 기존 코인 목록 (교체할 코인 제외)
    existing_markets = [m for m in auto_trading_status['positions'].keys() if m != sold_market]

    # 새 코인 선택 (기존 코인 + 블랙리스트 제외)
    try:
        all_coins = await select_top_coins(
            coin_count=20,  # 여유있게 선택
            mode=auto_trading_status.get('analysis_mode', 'volume_top50'),
            category=auto_trading_status.get('coin_category', 'normal')
        )

        # 기존 코인 + 블랙리스트 제외하고 새 코인 찾기
        new_coin = None
        for coin in all_coins:
            if coin['market'] not in existing_markets and coin['market'] not in coin_blacklist:
                new_coin = coin
                break

        if not new_coin:
            print(f"[AUTO-TRADING] 새로운 코인을 찾을 수 없습니다. {sold_market} 제거만 진행")
            # 매도된 코인 제거
            del auto_trading_status['positions'][sold_market]
            auto_trading_status['selected_coins'] = [
                c for c in auto_trading_status['selected_coins'] if c['market'] != sold_market
            ]
            return False

        # 투자금 배분 (매도된 코인의 배분금액 사용)
        allocated_amount = auto_trading_status['positions'][sold_market]['allocated_amount']
        allocation_percent = auto_trading_status['positions'][sold_market]['allocation_percent']

        # 새 코인으로 교체
        market = new_coin['market']
        auto_trading_status['positions'][market] = {
            'market': market,
            'name': new_coin.get('name', market.split('-')[1]),
            'allocated_amount': allocated_amount,
            'allocation_percent': allocation_percent,
            'entry_price': None,
            'current_price': new_coin.get('current_price', 0),
            'volume': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'status': 'none',
            'score': new_coin.get('score_100', 0),
            'recommendation': new_coin.get('recommendation', ''),
            'trade_prices': new_coin.get('trade_prices', {}),
            'last_action': None,
            'trade_history': []
        }

        # selected_coins 업데이트
        auto_trading_status['selected_coins'] = [
            c for c in auto_trading_status['selected_coins'] if c['market'] != sold_market
        ]
        auto_trading_status['selected_coins'].append({
            'market': market,
            'name': new_coin.get('name', market.split('-')[1]),
            'score': new_coin.get('score_100', 0),
            'recommendation': new_coin.get('recommendation', ''),
            'allocated_amount': allocated_amount,
            'allocation_percent': allocation_percent
        })

        # 매도된 코인 제거
        del auto_trading_status['positions'][sold_market]

        print(f"[AUTO-TRADING] ✅ 교체 완료: {sold_market} → {market} (점수: {new_coin.get('score_100', 0)})")
        save_auto_trading_state()  # 상태 저장
        return True

    except Exception as e:
        print(f"[AUTO-TRADING] 코인 교체 실패: {e}")
        return False


async def process_auto_coin_position(market: str, position: dict):
    """개별 코인 포지션 처리 (지정가 매수 방식)"""
    global auto_trading_status, trading_client

    if not trading_client:
        print(f"[AUTO-TRADING] {market} 처리 생략: API 클라이언트 없음")
        return

    # 목표가/손절가 설정 (상태에서 가져오기)
    TARGET_PROFIT_PERCENT = auto_trading_status.get('target_profit_percent', 10.0)
    STOP_LOSS_PERCENT = -abs(auto_trading_status.get('stop_loss_percent', 10.0))  # 음수로 변환

    try:
        # 시장 데이터 가져오기
        df = get_market_data_for_trading(market, 200)
        current_price = float(df.iloc[-1]['close'])
        position['current_price'] = current_price

        # 기술적 분석으로 trade_prices 갱신 (미보유 & 주문 없을 때만)
        if position['status'] == 'none' and not position.get('pending_order'):
            try:
                tech_result = analyzer.analyze_market(market, days=30)
                if tech_result and tech_result.get('current_price', 0) > 0:
                    new_trade_prices = calculate_trade_prices(tech_result)
                    position['trade_prices'] = new_trade_prices
                    print(f"[AUTO-TRADING] {market} trade_prices 갱신: 매수가 ₩{new_trade_prices['buy_price']:,}")
            except Exception as e:
                print(f"[AUTO-TRADING] {market} trade_prices 갱신 실패: {e}")

        # AI 예측
        action = 0  # 기본: Hold
        action_text = "HOLD"

        if trading_agent:
            obs = prepare_observation(df)
            action, _ = trading_agent.predict(obs, deterministic=True)
            action = int(action)
            action_text = ['HOLD', 'BUY', 'SELL'][action]

        position['last_action'] = action_text

        # ========== 미체결 주문 확인 ==========
        pending_order = position.get('pending_order')
        if pending_order:
            order_uuid = pending_order.get('uuid')
            order_price = pending_order.get('price', 0)

            try:
                order_info = trading_client.get_order(order_uuid)

                if isinstance(order_info, dict) and 'error' not in order_info:
                    order_state = order_info.get('state', '')

                    if order_state == 'done':
                        # 체결 완료
                        executed_volume = float(order_info.get('executed_volume', 0))
                        avg_price = float(order_info.get('avg_price', order_price))

                        position['status'] = 'long'
                        position['entry_price'] = avg_price
                        position['volume'] = executed_volume
                        position['pending_order'] = None

                        trade_record = {
                            'time': datetime.now().isoformat(),
                            'market': market,
                            'action': 'BUY (지정가 체결)',
                            'price': avg_price,
                            'volume': executed_volume,
                            'uuid': order_uuid
                        }
                        position['trade_history'].append(trade_record)
                        auto_trading_status['trade_history'].append(trade_record)
                        print(f"[AUTO-TRADING] ✅ 지정가 매수 체결: {market} - {executed_volume:.8f} @ ₩{avg_price:,.0f}")
                        save_auto_trading_state()

                    elif order_state == 'cancel':
                        # 취소된 주문
                        position['pending_order'] = None
                        print(f"[AUTO-TRADING] ⚠️ {market} 주문 취소됨")

                    elif order_state == 'wait':
                        # 대기 중 - 가격 변동 체크 (5% 이상 변동 시 재주문)
                        price_diff_percent = abs(order_price - current_price) / order_price * 100

                        if price_diff_percent > 5:
                            # 주문 취소 후 재주문
                            cancel_result = trading_client.cancel_order(order_uuid)
                            if cancel_result and 'error' not in cancel_result:
                                position['pending_order'] = None
                                print(f"[AUTO-TRADING] 🔄 {market} 가격 변동 {price_diff_percent:.1f}% → 재주문 예정")
                        else:
                            print(f"[AUTO-TRADING] ⏳ {market} 지정가 대기 중: ₩{order_price:,.0f} (현재가: ₩{current_price:,.0f})")
            except Exception as e:
                print(f"[AUTO-TRADING] {market} 주문 상태 확인 실패: {e}")

        # ========== 목표가/손절가 체크 (보유 중일 때) ==========
        should_sell_by_target = False
        sell_reason = ""

        if position['status'] == 'long' and position['entry_price']:
            profit_rate = ((current_price - position['entry_price']) / position['entry_price']) * 100

            if profit_rate >= TARGET_PROFIT_PERCENT:
                should_sell_by_target = True
                sell_reason = f"목표가 도달 (+{profit_rate:.1f}%)"
                print(f"[AUTO-TRADING] {market} 목표가 도달! 수익률: +{profit_rate:.1f}%")
            elif profit_rate <= STOP_LOSS_PERCENT:
                should_sell_by_target = True
                sell_reason = f"손절가 도달 ({profit_rate:.1f}%)"
                print(f"[AUTO-TRADING] {market} 손절가 도달! 손실률: {profit_rate:.1f}%")

        # ========== 지정가 매수 주문 생성 ==========
        buy_price = position.get('trade_prices', {}).get('buy_price', 0)

        # 주문이 없고, 미보유 상태일 때 지정가 주문
        if position['status'] == 'none' and not position.get('pending_order') and buy_price > 0:
            krw_balance = trading_client.get_balance('KRW')
            trade_amount = min(position['allocated_amount'], krw_balance * 0.95)

            if trade_amount >= 5000:
                # 지정가 매수를 위한 수량 계산
                buy_volume = trade_amount / buy_price

                print(f"[AUTO-TRADING] {market} 지정가 매수 주문: ₩{buy_price:,.0f} x {buy_volume:.8f}")

                try:
                    result = trading_client.buy_limit_order(market, buy_price, buy_volume)

                    if isinstance(result, dict) and 'error' in result:
                        error_msg = result.get('error', {}).get('message', str(result))
                        print(f"[AUTO-TRADING] ❌ 지정가 주문 실패 {market}: {error_msg}")
                    elif result and result.get('uuid'):
                        # 주문 성공 - pending_order에 저장 (체결 대기)
                        position['pending_order'] = {
                            'uuid': result.get('uuid'),
                            'price': buy_price,
                            'volume': buy_volume,
                            'amount': trade_amount,
                            'created_at': datetime.now().isoformat()
                        }
                        print(f"[AUTO-TRADING] 📝 지정가 주문 등록: {market} @ ₩{buy_price:,.0f} (체결 대기 중)")
                        save_auto_trading_state()
                    else:
                        print(f"[AUTO-TRADING] ❌ 주문 응답 없음 {market}")
                except Exception as e:
                    print(f"[AUTO-TRADING] ❌ 주문 오류 {market}: {e}")
            else:
                print(f"[AUTO-TRADING] {market} 매수 금액 부족: {trade_amount:,.0f} < 5,000 KRW (잔고: {krw_balance:,.0f})")

        # 매도 실행 (AI 신호 또는 목표가/손절가 도달)
        elif (action == 2 or should_sell_by_target) and position['status'] == 'long':
            crypto_symbol = market.split('-')[1]
            crypto_balance = trading_client.get_balance(crypto_symbol)

            sell_reason_text = sell_reason if should_sell_by_target else 'AI 매도 신호'
            print(f"[AUTO-TRADING] {market} 매도 시도: {crypto_balance:.8f} {crypto_symbol}, 사유: {sell_reason_text}")

            if crypto_balance > 0:
                try:
                    result = trading_client.sell_market_order(market, crypto_balance)

                    if isinstance(result, dict) and 'error' in result:
                        error_msg = result.get('error', {}).get('message', str(result))
                        print(f"[AUTO-TRADING] ❌ 매도 실패 {market}: {error_msg}")
                    elif result:
                        entry_price = position.get('entry_price') or current_price
                        realized_pnl = (current_price - entry_price) * position.get('volume', 0)
                        position['realized_pnl'] += realized_pnl

                        # 매도 사유 결정
                        action_label = sell_reason if should_sell_by_target else 'SELL (AI)'

                        trade_record = {
                            'time': datetime.now().isoformat(),
                            'market': market,
                            'action': action_label,
                            'price': current_price,
                            'volume': crypto_balance,
                            'realized_pnl': realized_pnl,
                            'uuid': result.get('uuid', 'N/A')
                        }
                        position['trade_history'].append(trade_record)
                        auto_trading_status['trade_history'].append(trade_record)

                        position['status'] = 'sold'  # 매도 완료 표시 (교체 대상)
                        position['entry_price'] = None
                        position['volume'] = 0
                        position['unrealized_pnl'] = 0
                        position['sold_time'] = datetime.now().isoformat()

                        pnl_emoji = "📈" if realized_pnl >= 0 else "📉"
                        print(f"[AUTO-TRADING] ✅ 매도 체결: {market} - {crypto_balance:.8f} @ {current_price:,.0f}")
                        print(f"[AUTO-TRADING] {pnl_emoji} 실현 손익: {realized_pnl:+,.0f} KRW [{action_label}]")
                        print(f"[AUTO-TRADING] {market} 자리에 새로운 코인 선택 예정")
                        save_auto_trading_state()
                    else:
                        print(f"[AUTO-TRADING] ❌ 매도 응답 없음 {market}")
                except Exception as e:
                    print(f"[AUTO-TRADING] ❌ 매도 오류 {market}: {e}")
            else:
                print(f"[AUTO-TRADING] {market} 매도할 잔고 없음: {crypto_balance}")

        # 미실현 손익 업데이트
        if position['status'] == 'long' and position['entry_price']:
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['volume']

    except Exception as e:
        print(f"[AUTO-TRADING] 포지션 처리 오류 {market}: {e}")


async def update_auto_portfolio_status():
    """포트폴리오 전체 상태 업데이트"""
    global auto_trading_status, trading_client

    try:
        # KRW 잔고
        krw_balance = trading_client.get_balance('KRW') if trading_client else 0

        # 코인 평가액 계산
        total_crypto_value = 0
        total_unrealized_pnl = 0
        total_realized_pnl = 0

        for market, position in auto_trading_status['positions'].items():
            if position['status'] == 'long':
                total_crypto_value += position['current_price'] * position['volume']
                total_unrealized_pnl += position['unrealized_pnl']
            total_realized_pnl += position['realized_pnl']

        # 전체 상태 업데이트
        auto_trading_status['current_balance'] = krw_balance + total_crypto_value
        auto_trading_status['profit'] = auto_trading_status['current_balance'] - auto_trading_status['start_balance']

        if auto_trading_status['start_balance'] > 0:
            auto_trading_status['profit_rate'] = (auto_trading_status['profit'] / auto_trading_status['start_balance']) * 100
        else:
            auto_trading_status['profit_rate'] = 0

    except Exception as e:
        print(f"[AUTO-TRADING] 포트폴리오 상태 업데이트 오류: {e}")


async def auto_trading_loop():
    """다중 코인 자동매매 루프"""
    global auto_trading_status, trading_agent, trading_client

    print("[AUTO-TRADING] 자동매매 루프 시작")

    # AI 모델 로드
    try:
        sample_market = list(auto_trading_status['positions'].keys())[0] if auto_trading_status['positions'] else 'KRW-BTC'
        dummy_data = get_market_data_for_trading(sample_market, 200)
        dummy_env = CryptoTradingEnv(dummy_data)
        trading_agent = TradingAgent(dummy_env)
        trading_agent.load('models/crypto_trader')
        print("[AUTO-TRADING] AI 모델 로드 완료")
    except Exception as e:
        print(f"[AUTO-TRADING] AI 모델 로드 실패: {e}")
        trading_agent = None

    # 초기 잔고 기록
    try:
        auto_trading_status['start_balance'] = trading_client.get_balance('KRW')
        auto_trading_status['current_balance'] = auto_trading_status['start_balance']
    except Exception as e:
        print(f"[AUTO-TRADING] 잔고 조회 실패: {e}")

    while auto_trading_status['is_running']:
        try:
            # 각 코인 포지션 처리
            markets_to_process = list(auto_trading_status['positions'].keys())
            for market in markets_to_process:
                if not auto_trading_status['is_running']:
                    break
                position = auto_trading_status['positions'].get(market)
                if position:
                    await process_auto_coin_position(market, position)
                await asyncio.sleep(0.5)  # API 제한 방지

            # 매도된 코인 교체
            sold_coins = [
                market for market, pos in auto_trading_status['positions'].items()
                if pos.get('status') == 'sold'
            ]

            for sold_market in sold_coins:
                if not auto_trading_status['is_running']:
                    break
                print(f"[AUTO-TRADING] {sold_market} 매도 완료, 새 코인으로 교체 중...")
                await replace_sold_coin(sold_market)
                await asyncio.sleep(1)  # 교체 후 대기

            # 포트폴리오 상태 업데이트
            await update_auto_portfolio_status()

            # WebSocket 브로드캐스트
            await manager.broadcast({
                'type': 'auto_trading_update',
                'data': clean_dict(auto_trading_status)
            })

            print(f"[AUTO-TRADING] {datetime.now().strftime('%H:%M:%S')} - 수익률: {auto_trading_status['profit_rate']:+.2f}%")

            # 주기적으로 상태 저장 (매 루프마다)
            save_auto_trading_state()

        except Exception as e:
            print(f"[AUTO-TRADING] 루프 오류: {e}")

        await asyncio.sleep(auto_trading_status.get('trading_interval', 60))

    # 종료 시 최종 상태 저장
    save_auto_trading_state()
    print("[AUTO-TRADING] 자동매매 루프 종료")


@app.post("/api/auto-trading/start")
async def start_auto_trading(request: AutoTradingStartRequest):
    """원클릭 자동매매 시작"""
    global auto_trading_task, auto_trading_status

    # Lock으로 동시 접근 방지
    async with auto_trading_status_lock:
        if auto_trading_status['is_running']:
            return {"success": False, "error": "이미 자동매매가 실행 중입니다."}

        # 최소 투자금 확인
        if request.total_investment < 50000:
            return {"success": False, "error": "최소 투자금은 50,000원입니다."}

        # 상태 초기화
        auto_trading_status['is_running'] = True
        auto_trading_status['total_investment'] = request.total_investment
        auto_trading_status['coin_count'] = min(max(request.coin_count, 1), 5)  # 1-5 제한
        auto_trading_status['analysis_mode'] = request.analysis_mode
        auto_trading_status['coin_category'] = request.coin_category
        auto_trading_status['trading_interval'] = request.trading_interval
        auto_trading_status['allocation_mode'] = request.allocation_mode
        auto_trading_status['target_profit_percent'] = request.target_profit_percent
        auto_trading_status['stop_loss_percent'] = request.stop_loss_percent
        auto_trading_status['start_time'] = datetime.now().isoformat()
        auto_trading_status['start_balance'] = 0
        auto_trading_status['current_balance'] = 0
        auto_trading_status['profit'] = 0
        auto_trading_status['profit_rate'] = 0
        auto_trading_status['trade_history'] = []

    # 초기 코인 선택 (Lock 밖에서 실행 - 시간이 걸릴 수 있음)
    success = await select_and_allocate_coins()
    if not success:
        async with auto_trading_status_lock:
            auto_trading_status['is_running'] = False
        return {"success": False, "error": "코인 선택에 실패했습니다."}

    # API 키가 있으면 트레이딩 루프 시작
    if trading_client:
        auto_trading_task = asyncio.create_task(auto_trading_loop())

    return {
        "success": True,
        "message": f"원클릭 자동매매 시작 - {len(auto_trading_status['selected_coins'])}개 코인",
        "data": {
            "total_investment": request.total_investment,
            "coin_count": auto_trading_status['coin_count'],
            "selected_coins": auto_trading_status['selected_coins']
        }
    }


class AutoTradingPreviewRequest(BaseModel):
    coin_count: int = 3
    analysis_mode: str = "volume_top50"
    coin_category: str = "safe"
    allocation_mode: str = "weighted"  # 'equal' or 'weighted'
    total_investment: float = 50000  # 투자금 (배분 계산용)


async def get_preview_coins_fast(coin_count: int, category: str) -> list:
    """빠른 코인 미리보기 (간단한 정보만)"""
    try:
        all_markets = analyzer.get_all_krw_markets()
        print(f"[PREVIEW] 전체 마켓 수: {len(all_markets)}")

        if category == 'safe':
            # 안전 코인만 선택
            markets = []
            for m in all_markets:
                coin_symbol = m['market'].split('-')[1]
                if coin_symbol in SAFE_COINS:
                    markets.append(m)
                    print(f"[PREVIEW] 안전 코인 발견: {m['market']}")
            print(f"[PREVIEW] 안전 코인 총 {len(markets)}개 발견")
        elif category == 'meme':
            # 밈코인만 선택 (거래량 상위)
            tickers = client.get_ticker([m['market'] for m in all_markets[:150]])
            sorted_markets = sorted(
                zip(all_markets[:150], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )
            # 밈코인만 필터링
            markets = [m[0] for m in sorted_markets if m[0]['market'].split('-')[1] in MEME_COINS]
            print(f"[PREVIEW] 밈코인 {len(markets)}개 발견")
        elif category == 'normal':
            # 밈코인 제외, 거래량 상위
            tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
            sorted_markets = sorted(
                zip(all_markets[:100], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )[:50]
            markets = [m[0] for m in sorted_markets]
            markets = [m for m in markets if m['market'].split('-')[1] not in MEME_COINS]
        else:  # 'all'
            # 거래량 상위
            tickers = client.get_ticker([m['market'] for m in all_markets[:100]])
            sorted_markets = sorted(
                zip(all_markets[:100], tickers),
                key=lambda x: x[1].get('acc_trade_price_24h', 0),
                reverse=True
            )[:50]
            markets = [m[0] for m in sorted_markets]

        if not markets:
            print(f"[PREVIEW] 마켓이 비어있음!")
            return []

        # 선택된 마켓의 현재가 조회
        market_codes = [m['market'] for m in markets[:coin_count * 2]]  # 여유있게 조회
        print(f"[PREVIEW] 조회할 마켓: {market_codes}")

        tickers = client.get_ticker(market_codes)
        print(f"[PREVIEW] 조회된 티커 수: {len(tickers) if tickers else 0}")

        results = []
        for i, market_info in enumerate(markets):
            if len(results) >= coin_count:
                break

            market = market_info['market']
            ticker = next((t for t in tickers if t['market'] == market), None)

            if ticker:
                change_rate = ticker.get('signed_change_rate', 0) * 100
                current_price = ticker.get('trade_price', 0)

                # 순위 기반 점수 계산 (1위: 100점, 순위가 내려갈수록 감소)
                # coin_count에 따라 점수 범위 조정
                if coin_count <= 3:
                    score_range = [100, 70, 50]  # 3개: 큰 차이
                elif coin_count == 4:
                    score_range = [100, 80, 60, 40]  # 4개: 중간 차이
                else:  # 5개
                    score_range = [100, 85, 70, 55, 40]  # 5개: 단계적 차이

                score = score_range[i] if i < len(score_range) else 40

                # AI 기술적 분석 기반 trade_prices 계산
                try:
                    # 간단한 분석 데이터 생성 (현재가 기준)
                    tech_data = {
                        'current_price': current_price,
                        'rsi': 50,  # 중립
                        'bb_low': current_price * 0.95,
                        'bb_high': current_price * 1.05,
                        'recommendation': '매수' if change_rate > 0 else '중립'
                    }
                    trade_prices = calculate_trade_prices(tech_data)
                except Exception as e:
                    print(f"[PREVIEW] {market} 가격 계산 실패: {e}")
                    trade_prices = {
                        'buy_price': round(current_price * 0.99),
                        'sell_price': round(current_price * 1.08),
                        'stop_loss': round(current_price * 0.96),
                        'expected_profit_rate': 8.0,
                        'risk_rate': 4.0
                    }

                results.append({
                    'market': market,
                    'name': market_info.get('korean_name', market.split('-')[1]),
                    'current_price': current_price,
                    'change_rate': change_rate,
                    'score': score,
                    'score_100': score,  # UI 표시용 점수 추가
                    'recommendation': '분석 대기',
                    'trade_prices': trade_prices
                })
            else:
                print(f"[PREVIEW] 티커 없음: {market}")

        print(f"[PREVIEW] 최종 결과: {len(results)}개, 점수: {[c['score_100'] for c in results]}")
        return results
    except Exception as e:
        print(f"[AUTO-TRADING] 빠른 미리보기 실패: {e}")
        return []


@app.post("/api/auto-trading/preview")
async def preview_auto_trading(request: AutoTradingPreviewRequest):
    """코인 선택 미리보기 (빠른 버전)"""
    try:
        coin_count = min(max(request.coin_count, 1), 5)
        category = request.coin_category
        allocation_mode = request.allocation_mode
        total_investment = request.total_investment

        print(f"=" * 50)
        print(f"[PREVIEW API] 요청 받음")
        print(f"[PREVIEW API] 카테고리: '{category}'")
        print(f"[PREVIEW API] 코인 수: {coin_count}")
        print(f"[PREVIEW API] 배분 방식: '{allocation_mode}'")
        print(f"=" * 50)

        # 빠른 미리보기 사용
        coins = await get_preview_coins_fast(coin_count, category)

        if not coins:
            return {"success": False, "error": "선택된 코인이 없습니다."}

        # 배분 비율 계산
        if allocation_mode == 'weighted' and len(coins) > 0:
            # 점수 기반 가중치 배분
            scores = [max(coin.get('score', 50), 10) for coin in coins]
            total_score = sum(scores)
            for i, coin in enumerate(coins):
                weight = scores[i] / total_score
                coin['allocation_percent'] = round(weight * 100, 1)
                coin['allocated_amount'] = round(total_investment * weight)
        else:
            # 균등 배분
            per_coin = total_investment / len(coins) if coins else 0
            percent_per_coin = 100 / len(coins) if coins else 0
            for coin in coins:
                coin['allocation_percent'] = round(percent_per_coin, 1)
                coin['allocated_amount'] = round(per_coin)

        return {
            "success": True,
            "data": {
                "coin_count": len(coins),
                "coin_category": request.coin_category,
                "allocation_mode": allocation_mode,
                "selected_coins": coins
            }
        }
    except Exception as e:
        print(f"[AUTO-TRADING] 미리보기 실패: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/auto-trading/stop")
async def stop_auto_trading():
    """원클릭 자동매매 중지 및 전체 청산"""
    global auto_trading_task, auto_trading_status, trading_client

    # Lock으로 동시 접근 방지
    async with auto_trading_status_lock:
        if not auto_trading_status['is_running']:
            return {"success": False, "error": "실행 중인 자동매매가 없습니다."}

        auto_trading_status['is_running'] = False

    # 태스크 취소 (Lock 밖에서 실행)
    if auto_trading_task:
        auto_trading_task.cancel()
        try:
            await auto_trading_task
        except asyncio.CancelledError:
            pass
        auto_trading_task = None

    # 모든 포지션 청산 (Lock 밖에서 실행 - API 호출 포함)
    print("[AUTO-TRADING] 전체 포지션 청산 중...")
    if trading_client:
        # positions 복사본 사용
        positions_snapshot = dict(auto_trading_status['positions'])
        for market, position in positions_snapshot.items():
            if position['status'] == 'long':
                crypto_symbol = market.split('-')[1]
                try:
                    crypto_balance = trading_client.get_balance(crypto_symbol)
                    if crypto_balance > 0:
                        result = trading_client.sell_market_order(market, crypto_balance)
                        if result and 'error' not in result:
                            print(f"[AUTO-TRADING] 청산 완료: {market}")

                            trade_record = {
                                'time': datetime.now().isoformat(),
                                'market': market,
                                'action': 'SELL (청산)',
                                'price': position['current_price'],
                                'volume': crypto_balance
                            }
                            async with auto_trading_status_lock:
                                auto_trading_status['trade_history'].append(trade_record)
                except Exception as e:
                    print(f"[AUTO-TRADING] 청산 오류 {market}: {e}")
    else:
        print("[AUTO-TRADING] API 클라이언트 없음, 청산 생략")

    # 최종 상태 업데이트
    await update_auto_portfolio_status()

    return {
        "success": True,
        "message": "자동매매 중지 및 전체 청산 완료",
        "data": clean_dict(auto_trading_status)
    }


@app.get("/api/auto-trading/status")
async def get_auto_trading_status():
    """원클릭 자동매매 상태 조회"""
    # 현재 KRW 잔고 조회 (자동매매 시작 전에도 표시하기 위함)
    available_balance = 0
    if trading_client:
        try:
            available_balance = trading_client.get_balance('KRW')
        except Exception as e:
            print(f"[AUTO-TRADING] 잔고 조회 실패: {e}")

    result = dict(auto_trading_status)
    result['available_balance'] = available_balance

    return {
        "success": True,
        "data": clean_dict(result)
    }


@app.get("/api/auto-trading/mini-charts")
async def get_mini_charts():
    """선택된 코인들의 미니 차트 데이터 조회"""
    if not auto_trading_status['selected_coins']:
        return {"success": True, "data": {}}

    charts = {}
    for coin in auto_trading_status['selected_coins']:
        market = coin['market']
        try:
            # 최근 30개 분봉 데이터
            candles = client.get_candles_minute(market, unit=5, count=30)
            prices = [c['trade_price'] for c in reversed(candles)]
            charts[market] = prices
        except Exception as e:
            print(f"[MINI-CHART] {market} 데이터 조회 실패: {e}")
            charts[market] = []

    return {"success": True, "data": charts}


# ========== 블랙리스트 및 코인 제외/대체 API ==========

class ExcludeCoinRequest(BaseModel):
    market: str  # 제외할 코인 (예: KRW-BTC)
    add_to_blacklist: bool = True  # 블랙리스트에 추가할지 여부


@app.get("/api/blacklist")
async def get_blacklist():
    """블랙리스트 조회"""
    return {
        "success": True,
        "data": {
            "blacklist": list(coin_blacklist),
            "count": len(coin_blacklist)
        }
    }


@app.post("/api/blacklist/add")
async def add_to_blacklist(market: str):
    """블랙리스트에 코인 추가"""
    global coin_blacklist

    # 마켓 코드 정규화 (KRW-BTC 또는 BTC 둘 다 허용)
    if not market.startswith('KRW-'):
        market = f"KRW-{market.upper()}"
    else:
        market = market.upper()

    coin_blacklist.add(market)
    save_blacklist()

    return {
        "success": True,
        "message": f"{market}이(가) 블랙리스트에 추가되었습니다.",
        "data": {"blacklist": list(coin_blacklist)}
    }


@app.delete("/api/blacklist/remove")
async def remove_from_blacklist(market: str):
    """블랙리스트에서 코인 제거"""
    global coin_blacklist

    # 마켓 코드 정규화
    if not market.startswith('KRW-'):
        market = f"KRW-{market.upper()}"
    else:
        market = market.upper()

    if market in coin_blacklist:
        coin_blacklist.remove(market)
        save_blacklist()
        return {
            "success": True,
            "message": f"{market}이(가) 블랙리스트에서 제거되었습니다.",
            "data": {"blacklist": list(coin_blacklist)}
        }
    else:
        return {
            "success": False,
            "error": f"{market}은(는) 블랙리스트에 없습니다."
        }


@app.delete("/api/blacklist/clear")
async def clear_blacklist():
    """블랙리스트 전체 삭제"""
    global coin_blacklist
    coin_blacklist = set()
    save_blacklist()

    return {
        "success": True,
        "message": "블랙리스트가 초기화되었습니다.",
        "data": {"blacklist": []}
    }


@app.post("/api/auto-trading/exclude-coin")
async def exclude_and_replace_coin(request: ExcludeCoinRequest):
    """현재 포지션에서 코인 제외 및 다른 코인으로 대체

    1. 해당 코인이 보유 중이면 매도
    2. 블랙리스트에 추가 (옵션)
    3. 다음 유망한 코인으로 대체
    """
    global auto_trading_status, coin_blacklist, trading_client

    market = request.market.upper()
    if not market.startswith('KRW-'):
        market = f"KRW-{market}"

    # 자동매매 실행 중인지 확인
    if not auto_trading_status['is_running']:
        # 블랙리스트에만 추가
        if request.add_to_blacklist:
            coin_blacklist.add(market)
            save_blacklist()
            return {
                "success": True,
                "message": f"{market}이(가) 블랙리스트에 추가되었습니다. (자동매매 미실행 중)",
                "data": {"blacklist": list(coin_blacklist)}
            }
        return {"success": False, "error": "자동매매가 실행 중이 아닙니다."}

    # 포지션에 해당 코인이 있는지 확인
    if market not in auto_trading_status['positions']:
        # 블랙리스트에만 추가
        if request.add_to_blacklist:
            coin_blacklist.add(market)
            save_blacklist()
        return {
            "success": True,
            "message": f"{market}은(는) 현재 포지션에 없습니다. 블랙리스트에 추가됨.",
            "data": {"blacklist": list(coin_blacklist)}
        }

    position = auto_trading_status['positions'][market]
    result_message = []

    # 1. 보유 중이면 매도
    if position['status'] == 'long' and trading_client:
        crypto_symbol = market.split('-')[1]
        try:
            crypto_balance = trading_client.get_balance(crypto_symbol)
            if crypto_balance > 0:
                sell_result = trading_client.sell_market_order(market, crypto_balance)
                if sell_result and 'error' not in sell_result:
                    current_price = position.get('current_price', 0)
                    entry_price = position.get('entry_price', current_price)
                    realized_pnl = (current_price - entry_price) * position.get('volume', 0)

                    trade_record = {
                        'time': datetime.now().isoformat(),
                        'market': market,
                        'action': 'SELL (사용자 제외)',
                        'price': current_price,
                        'volume': crypto_balance,
                        'realized_pnl': realized_pnl
                    }
                    auto_trading_status['trade_history'].append(trade_record)
                    result_message.append(f"✅ {market} 매도 완료 (손익: {realized_pnl:+,.0f} KRW)")
                    print(f"[EXCLUDE] {market} 매도 체결: {crypto_balance:.8f} @ {current_price:,.0f}")
                else:
                    result_message.append(f"❌ {market} 매도 실패")
        except Exception as e:
            result_message.append(f"❌ {market} 매도 오류: {e}")
            print(f"[EXCLUDE] {market} 매도 오류: {e}")

    # 2. 블랙리스트에 추가
    if request.add_to_blacklist:
        coin_blacklist.add(market)
        save_blacklist()
        result_message.append(f"🚫 {market} 블랙리스트 추가됨")

    # 3. 새 코인으로 대체
    old_allocation = position.get('allocated_amount', 0)
    old_percent = position.get('allocation_percent', 0)

    # 기존 포지션 목록에서 제거
    del auto_trading_status['positions'][market]
    auto_trading_status['selected_coins'] = [
        c for c in auto_trading_status['selected_coins'] if c['market'] != market
    ]

    # 새 코인 선택 (블랙리스트 및 기존 코인 제외)
    try:
        existing_markets = list(auto_trading_status['positions'].keys())
        all_coins = await select_top_coins(
            coin_count=20,
            mode=auto_trading_status.get('analysis_mode', 'volume_top50'),
            category=auto_trading_status.get('coin_category', 'normal')
        )

        # 블랙리스트 및 기존 코인 제외
        new_coin = None
        for coin in all_coins:
            if coin['market'] not in existing_markets and coin['market'] not in coin_blacklist:
                new_coin = coin
                break

        if new_coin:
            new_market = new_coin['market']
            auto_trading_status['positions'][new_market] = {
                'market': new_market,
                'name': new_coin.get('name', new_market.split('-')[1]),
                'allocated_amount': old_allocation,
                'allocation_percent': old_percent,
                'entry_price': None,
                'current_price': new_coin.get('current_price', 0),
                'volume': 0,
                'unrealized_pnl': 0,
                'realized_pnl': 0,
                'status': 'none',
                'score': new_coin.get('score_100', 0),
                'recommendation': new_coin.get('recommendation', ''),
                'trade_prices': new_coin.get('trade_prices', {}),
                'last_action': None,
                'trade_history': []
            }

            auto_trading_status['selected_coins'].append({
                'market': new_market,
                'name': new_coin.get('name', new_market.split('-')[1]),
                'score': new_coin.get('score_100', 0),
                'recommendation': new_coin.get('recommendation', ''),
                'allocated_amount': old_allocation,
                'allocation_percent': old_percent
            })

            result_message.append(f"✅ {new_market} (으)로 대체 완료 (점수: {new_coin.get('score_100', 0)})")
            print(f"[EXCLUDE] {market} → {new_market} 대체 완료")
        else:
            result_message.append("⚠️ 대체할 코인을 찾지 못했습니다.")

    except Exception as e:
        result_message.append(f"❌ 코인 대체 실패: {e}")
        print(f"[EXCLUDE] 코인 대체 실패: {e}")

    save_auto_trading_state()

    return {
        "success": True,
        "message": " / ".join(result_message),
        "data": {
            "excluded_market": market,
            "blacklist": list(coin_blacklist),
            "current_positions": list(auto_trading_status['positions'].keys())
        }
    }


@app.get("/api/auto-trading/replaceable-coins")
async def get_replaceable_coins(count: int = 10):
    """대체 가능한 코인 목록 조회 (블랙리스트 및 현재 포지션 제외)"""
    try:
        existing_markets = list(auto_trading_status.get('positions', {}).keys())

        all_coins = await select_top_coins(
            coin_count=count + len(existing_markets) + len(coin_blacklist),
            mode=auto_trading_status.get('analysis_mode', 'volume_top50'),
            category=auto_trading_status.get('coin_category', 'normal')
        )

        # 블랙리스트 및 현재 포지션 제외
        available_coins = [
            coin for coin in all_coins
            if coin['market'] not in existing_markets and coin['market'] not in coin_blacklist
        ][:count]

        return {
            "success": True,
            "data": {
                "available_coins": available_coins,
                "excluded_by_blacklist": list(coin_blacklist),
                "current_positions": existing_markets
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ========== 코인 편집 API (블랙리스트 없이 간단한 편집) ==========

class SwapCoinRequest(BaseModel):
    old_market: str  # 제거할 코인
    new_market: str  # 추가할 코인


@app.post("/api/auto-trading/remove-coin")
async def remove_coin_from_portfolio(market: str):
    """포트폴리오에서 코인 제거 (매도 후 제거, 블랙리스트 추가 안 함)

    투자금은 다른 코인에 재배분되지 않고, 코인 수만 줄어듭니다.
    """
    global auto_trading_status, trading_client

    market = market.upper()
    if not market.startswith('KRW-'):
        market = f"KRW-{market}"

    if market not in auto_trading_status.get('positions', {}):
        return {"success": False, "error": f"{market}은(는) 현재 포지션에 없습니다."}

    position = auto_trading_status['positions'][market]
    result_message = []

    # 보유 중이면 매도
    if position['status'] == 'long' and trading_client:
        crypto_symbol = market.split('-')[1]
        try:
            crypto_balance = trading_client.get_balance(crypto_symbol)
            if crypto_balance > 0:
                sell_result = trading_client.sell_market_order(market, crypto_balance)
                if sell_result and 'error' not in sell_result:
                    current_price = position.get('current_price', 0)
                    entry_price = position.get('entry_price', current_price)
                    realized_pnl = (current_price - entry_price) * position.get('volume', 0)

                    trade_record = {
                        'time': datetime.now().isoformat(),
                        'market': market,
                        'action': 'SELL (사용자 제거)',
                        'price': current_price,
                        'volume': crypto_balance,
                        'realized_pnl': realized_pnl
                    }
                    auto_trading_status['trade_history'].append(trade_record)
                    result_message.append(f"✅ 매도 완료 (손익: {realized_pnl:+,.0f} KRW)")
                else:
                    result_message.append("❌ 매도 실패")
        except Exception as e:
            result_message.append(f"❌ 매도 오류: {e}")

    # 포지션에서 제거
    del auto_trading_status['positions'][market]
    auto_trading_status['selected_coins'] = [
        c for c in auto_trading_status['selected_coins'] if c['market'] != market
    ]
    auto_trading_status['coin_count'] = len(auto_trading_status['positions'])

    save_auto_trading_state()
    result_message.append(f"🗑️ {market} 제거됨")

    return {
        "success": True,
        "message": " / ".join(result_message),
        "data": {
            "removed": market,
            "remaining_coins": list(auto_trading_status['positions'].keys())
        }
    }


@app.post("/api/auto-trading/cancel-order")
async def cancel_pending_order(market: str):
    """특정 코인의 지정가 주문 취소"""
    global auto_trading_status, trading_client

    if not trading_client:
        return {"success": False, "error": "API 키가 설정되지 않았습니다."}

    market = market.upper()
    if not market.startswith('KRW-'):
        market = f"KRW-{market}"

    if market not in auto_trading_status.get('positions', {}):
        return {"success": False, "error": f"{market}은(는) 포지션에 없습니다."}

    position = auto_trading_status['positions'][market]
    pending_order = position.get('pending_order')

    if not pending_order:
        return {"success": False, "error": f"{market}에 대기 중인 주문이 없습니다."}

    order_uuid = pending_order.get('uuid')

    try:
        # 주문 취소
        cancel_result = trading_client.cancel_order(order_uuid)

        if cancel_result and 'error' not in cancel_result:
            position['pending_order'] = None
            save_auto_trading_state()

            return {
                "success": True,
                "message": f"✅ {market} 주문 취소 완료",
                "data": {"market": market, "cancelled_uuid": order_uuid}
            }
        else:
            error_msg = cancel_result.get('error', {}).get('message', '알 수 없는 오류')
            return {"success": False, "error": f"주문 취소 실패: {error_msg}"}

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/auto-trading/add-coin")
async def add_coin_to_portfolio(market: str, amount: float = 0):
    """포트폴리오에 코인 수동 추가

    Args:
        market: 추가할 코인 (예: KRW-BTC 또는 BTC)
        amount: 배분 금액 (0이면 균등 배분)
    """
    global auto_trading_status

    market = market.upper()
    if not market.startswith('KRW-'):
        market = f"KRW-{market}"

    # 이미 포지션에 있는지 확인
    if market in auto_trading_status.get('positions', {}):
        return {"success": False, "error": f"{market}은(는) 이미 포지션에 있습니다."}

    # 코인 정보 조회
    try:
        # 기술적 분석
        tech_result = analyzer.analyze_market(market, days=30)
        if not tech_result:
            return {"success": False, "error": f"{market} 분석 실패"}

        ticker = client.get_ticker([market])
        if not ticker:
            return {"success": False, "error": f"{market} 가격 조회 실패"}

        current_price = ticker[0].get('trade_price', 0)
        score = normalize_score_to_100(tech_result.get('score', 0))
        trade_prices = calculate_trade_prices(tech_result)

        # 배분 금액 계산
        if amount <= 0:
            # 기존 코인들의 평균 배분 금액
            existing_amounts = [p.get('allocated_amount', 0) for p in auto_trading_status['positions'].values()]
            amount = sum(existing_amounts) / len(existing_amounts) if existing_amounts else 50000

        total_investment = auto_trading_status.get('total_investment', 0) + amount
        allocation_percent = (amount / total_investment) * 100 if total_investment > 0 else 0

        # 마켓 정보 조회
        all_markets = analyzer.get_all_krw_markets()
        market_info = next((m for m in all_markets if m['market'] == market), {})

        # 포지션 추가
        auto_trading_status['positions'][market] = {
            'market': market,
            'name': market_info.get('korean_name', market.split('-')[1]),
            'allocated_amount': amount,
            'allocation_percent': round(allocation_percent, 1),
            'entry_price': None,
            'current_price': current_price,
            'volume': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'status': 'none',
            'score': score,
            'recommendation': tech_result.get('recommendation', ''),
            'trade_prices': trade_prices,
            'last_action': None,
            'trade_history': []
        }

        auto_trading_status['selected_coins'].append({
            'market': market,
            'name': market_info.get('korean_name', market.split('-')[1]),
            'score': score,
            'recommendation': tech_result.get('recommendation', ''),
            'allocated_amount': amount,
            'allocation_percent': round(allocation_percent, 1)
        })

        auto_trading_status['total_investment'] = total_investment
        auto_trading_status['coin_count'] = len(auto_trading_status['positions'])

        save_auto_trading_state()

        return {
            "success": True,
            "message": f"✅ {market} 추가됨 (점수: {score}, 배분: ₩{amount:,.0f})",
            "data": {
                "added": market,
                "score": score,
                "allocated_amount": amount,
                "current_coins": list(auto_trading_status['positions'].keys())
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/auto-trading/swap-coin")
async def swap_coin_in_portfolio(request: SwapCoinRequest):
    """포트폴리오에서 코인 교체 (A코인 → B코인)

    기존 코인을 매도하고, 새 코인으로 교체합니다.
    배분 금액은 그대로 유지됩니다.
    """
    global auto_trading_status, trading_client

    old_market = request.old_market.upper()
    new_market = request.new_market.upper()

    if not old_market.startswith('KRW-'):
        old_market = f"KRW-{old_market}"
    if not new_market.startswith('KRW-'):
        new_market = f"KRW-{new_market}"

    # 기존 코인 확인
    if old_market not in auto_trading_status.get('positions', {}):
        return {"success": False, "error": f"{old_market}은(는) 현재 포지션에 없습니다."}

    # 새 코인이 이미 있는지 확인
    if new_market in auto_trading_status.get('positions', {}):
        return {"success": False, "error": f"{new_market}은(는) 이미 포지션에 있습니다."}

    old_position = auto_trading_status['positions'][old_market]
    result_message = []

    # 1. 기존 코인 매도
    if old_position['status'] == 'long' and trading_client:
        crypto_symbol = old_market.split('-')[1]
        try:
            crypto_balance = trading_client.get_balance(crypto_symbol)
            if crypto_balance > 0:
                sell_result = trading_client.sell_market_order(old_market, crypto_balance)
                if sell_result and 'error' not in sell_result:
                    current_price = old_position.get('current_price', 0)
                    entry_price = old_position.get('entry_price', current_price)
                    realized_pnl = (current_price - entry_price) * old_position.get('volume', 0)

                    trade_record = {
                        'time': datetime.now().isoformat(),
                        'market': old_market,
                        'action': 'SELL (교체)',
                        'price': current_price,
                        'volume': crypto_balance,
                        'realized_pnl': realized_pnl
                    }
                    auto_trading_status['trade_history'].append(trade_record)
                    result_message.append(f"✅ {old_market} 매도 (손익: {realized_pnl:+,.0f})")
        except Exception as e:
            result_message.append(f"⚠️ {old_market} 매도 오류: {e}")

    # 2. 새 코인 정보 조회
    try:
        tech_result = analyzer.analyze_market(new_market, days=30)
        ticker = client.get_ticker([new_market])

        if not tech_result or not ticker:
            return {"success": False, "error": f"{new_market} 분석 실패"}

        current_price = ticker[0].get('trade_price', 0)
        score = normalize_score_to_100(tech_result.get('score', 0))
        trade_prices = calculate_trade_prices(tech_result)

        # 마켓 정보 조회
        all_markets = analyzer.get_all_krw_markets()
        market_info = next((m for m in all_markets if m['market'] == new_market), {})

        # 기존 배분 금액 유지
        allocated_amount = old_position.get('allocated_amount', 0)
        allocation_percent = old_position.get('allocation_percent', 0)

        # 3. 기존 포지션 제거
        del auto_trading_status['positions'][old_market]
        auto_trading_status['selected_coins'] = [
            c for c in auto_trading_status['selected_coins'] if c['market'] != old_market
        ]

        # 4. 새 포지션 추가
        auto_trading_status['positions'][new_market] = {
            'market': new_market,
            'name': market_info.get('korean_name', new_market.split('-')[1]),
            'allocated_amount': allocated_amount,
            'allocation_percent': allocation_percent,
            'entry_price': None,
            'current_price': current_price,
            'volume': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'status': 'none',
            'score': score,
            'recommendation': tech_result.get('recommendation', ''),
            'trade_prices': trade_prices,
            'last_action': None,
            'trade_history': []
        }

        auto_trading_status['selected_coins'].append({
            'market': new_market,
            'name': market_info.get('korean_name', new_market.split('-')[1]),
            'score': score,
            'recommendation': tech_result.get('recommendation', ''),
            'allocated_amount': allocated_amount,
            'allocation_percent': allocation_percent
        })

        save_auto_trading_state()
        result_message.append(f"✅ {new_market} 추가 (점수: {score})")

        return {
            "success": True,
            "message": " / ".join(result_message),
            "data": {
                "removed": old_market,
                "added": new_market,
                "score": score,
                "allocated_amount": allocated_amount,
                "current_coins": list(auto_trading_status['positions'].keys())
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/auto-trading/rebalance")
async def rebalance_portfolio():
    """포트폴리오 배분 비율 재조정 (균등 배분)"""
    global auto_trading_status

    positions = auto_trading_status.get('positions', {})
    if not positions:
        return {"success": False, "error": "포지션이 없습니다."}

    total_investment = auto_trading_status.get('total_investment', 0)
    if total_investment <= 0:
        # 현재 배분 금액 합계로 계산
        total_investment = sum(p.get('allocated_amount', 0) for p in positions.values())

    per_coin = total_investment / len(positions)
    percent_per_coin = 100 / len(positions)

    for _, position in positions.items():
        position['allocated_amount'] = per_coin
        position['allocation_percent'] = round(percent_per_coin, 1)

    # selected_coins도 업데이트
    for coin in auto_trading_status['selected_coins']:
        coin['allocated_amount'] = per_coin
        coin['allocation_percent'] = round(percent_per_coin, 1)

    auto_trading_status['total_investment'] = total_investment
    save_auto_trading_state()

    return {
        "success": True,
        "message": f"✅ {len(positions)}개 코인 균등 배분 완료 (각 ₩{per_coin:,.0f})",
        "data": {
            "total_investment": total_investment,
            "per_coin": per_coin,
            "coins": list(positions.keys())
        }
    }


# ========== 스캘핑(단타) 모드 ==========
scalping_status = {
    "is_running": False,
    "markets": [],
    "config": {
        "min_profit_percent": 0.15,
        "max_loss_percent": 0.5,
        "target_profit_percent": 0.3,
        "trade_amount": 50000,
        "max_positions": 3,
        "cooldown_seconds": 30
    },
    "stats": {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_profit": 0,
        "win_rate": 0
    },
    "positions": {},
    "trade_history": []
}
scalping_task: Optional[asyncio.Task] = None
scalping_lock = asyncio.Lock()


class ScalpingStartRequest(BaseModel):
    markets: List[str] = ["KRW-BTC"]
    trade_amount: float = 50000
    target_profit_percent: float = 0.3
    max_loss_percent: float = 0.5
    max_positions: int = 3


async def scalping_loop():
    """스캘핑 메인 루프"""
    global scalping_status, trading_client

    print("[SCALPING] 스캘핑 모드 시작")

    from src.bot.scalping_bot import ScalpingBot, ScalpingConfig

    config = ScalpingConfig(
        min_profit_percent=scalping_status['config']['min_profit_percent'],
        max_loss_percent=scalping_status['config']['max_loss_percent'],
        target_profit_percent=scalping_status['config']['target_profit_percent'],
        trade_amount=scalping_status['config']['trade_amount'],
        max_positions=scalping_status['config']['max_positions'],
        cooldown_seconds=scalping_status['config']['cooldown_seconds']
    )

    bot = ScalpingBot(trading_client, config)

    while scalping_status['is_running']:
        try:
            markets = scalping_status['markets']

            for market in markets:
                if not scalping_status['is_running']:
                    break

                result = bot.run_single_check(market)

                if result:
                    async with scalping_lock:
                        scalping_status['trade_history'].append(result)
                        scalping_status['positions'] = {
                            m: {
                                'entry_price': p['entry_price'],
                                'entry_time': p['entry_time'].isoformat(),
                                'amount': p['amount']
                            }
                            for m, p in bot.positions.items()
                        }

                    # WebSocket 브로드캐스트
                    await manager.broadcast({
                        'type': 'scalping_trade',
                        'data': result
                    })

                    print(f"[SCALPING] {result.get('action', 'TRADE')} - {market}")

                await asyncio.sleep(1)  # 빠른 체크

            # 통계 업데이트
            stats = bot.get_stats()
            async with scalping_lock:
                scalping_status['stats'] = stats

            # 5초 대기 후 다음 사이클
            await asyncio.sleep(5)

        except Exception as e:
            print(f"[SCALPING] 루프 오류: {e}")
            await asyncio.sleep(5)

    print("[SCALPING] 스캘핑 모드 종료")


@app.post("/api/scalping/start")
async def start_scalping(request: ScalpingStartRequest):
    """스캘핑 모드 시작"""
    global scalping_status, scalping_task, trading_client

    if not trading_client:
        return {"success": False, "error": "API 키가 설정되지 않았습니다."}

    async with scalping_lock:
        if scalping_status['is_running']:
            return {"success": False, "error": "이미 실행 중입니다."}

        scalping_status['is_running'] = True
        scalping_status['markets'] = request.markets
        scalping_status['config']['trade_amount'] = request.trade_amount
        scalping_status['config']['target_profit_percent'] = request.target_profit_percent
        scalping_status['config']['max_loss_percent'] = request.max_loss_percent
        scalping_status['config']['max_positions'] = request.max_positions
        scalping_status['trade_history'] = []
        scalping_status['positions'] = {}

    scalping_task = asyncio.create_task(scalping_loop())

    return {
        "success": True,
        "message": f"스캘핑 모드 시작 - {len(request.markets)}개 마켓",
        "data": {
            "markets": request.markets,
            "config": scalping_status['config']
        }
    }


@app.post("/api/scalping/stop")
async def stop_scalping():
    """스캘핑 모드 중지"""
    global scalping_status, scalping_task

    async with scalping_lock:
        if not scalping_status['is_running']:
            return {"success": False, "error": "실행 중이 아닙니다."}

        scalping_status['is_running'] = False

    if scalping_task:
        scalping_task.cancel()

    return {
        "success": True,
        "message": "스캘핑 모드 중지",
        "stats": scalping_status['stats']
    }


@app.get("/api/scalping/status")
async def get_scalping_status():
    """스캘핑 상태 조회"""
    return {
        "success": True,
        "data": {
            "is_running": scalping_status['is_running'],
            "markets": scalping_status['markets'],
            "config": scalping_status['config'],
            "stats": scalping_status['stats'],
            "positions": scalping_status['positions'],
            "recent_trades": scalping_status['trade_history'][-20:]
        }
    }


if __name__ == "__main__":
    print("="*60)
    print("[START] 업비트 실시간 분석 웹앱 시작")
    print("="*60)
    print()
    print("[WEB] 웹 브라우저에서 접속하세요:")
    print("   http://localhost:8000")
    print()
    print("[EXIT] 종료: Ctrl+C")
    print("="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
