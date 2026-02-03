# 업비트 AI 자동 매매 & 종목 추천 시스템

강화학습(PPO) 기반 업비트 가상화폐 자동 매매 및 AI 종목 추천 시스템입니다.

## 주요 기능

### 🌐 실시간 웹앱 (NEW! ⭐)
- **현대적인 UI/UX**: Tailwind CSS 기반 다크모드 디자인
- **실시간 모니터링**: WebSocket으로 3초마다 가격 업데이트
- **대시보드**: TOP 10 추천 종목, AI 예측, 전체 종목 목록
- **상세 분석**: 클릭 한 번으로 종목 분석 + AI 예측
- **반응형 디자인**: 모바일/태블릿/데스크톱 완벽 대응

### 🎯 종목 추천 시스템
- **기술적 분석 추천**: 20개 이상의 기술적 지표로 전체 시장 스캔
- **AI 예측 추천**: 강화학습 모델 기반 상승 예상 종목 추천
- **특정 종목 검색**: 원하는 코인 검색 및 상세 분석
- **API 키 불필요**: 공개 API만으로 모든 분석 가능

### 🤖 자동 매매 시스템
- **AI 모델 학습**: PPO(Proximal Policy Optimization) 알고리즘 사용
- **백테스팅**: 과거 데이터로 전략 성능 검증
- **실시간 자동 매매**: 학습된 모델을 사용한 실시간 트레이딩 (API 키 필요)
- **기술적 지표**: 20개 이상의 기술적 지표 활용 (SMA, EMA, MACD, RSI, 볼린저밴드, 스토캐스틱 등)

## 프로젝트 구조

```
up_bit/
├── src/
│   ├── api/
│   │   └── upbit_client.py          # 업비트 API 클라이언트
│   ├── analyzer/                    # 📊 종목 분석 (NEW!)
│   │   ├── market_analyzer.py       # 기술적 분석
│   │   └── ai_predictor.py          # AI 예측
│   ├── models/
│   │   └── rl_agent.py              # 강화학습 에이전트 (PPO)
│   ├── environment/
│   │   └── trading_env.py           # 트레이딩 환경 (Gymnasium)
│   ├── backtesting/
│   │   └── backtest.py              # 백테스팅 시스템
│   └── bot/
│       └── trading_bot.py           # 실시간 트레이딩 봇
├── config/
├── data/
├── logs/
├── models/
├── main.py                          # 자동매매 메인 프로그램
├── recommend.py                     # 🎯 종목 추천 메인 (NEW!)
├── quick_recommend.py               # 🚀 빠른 추천 데모 (NEW!)
├── quick_start.py                   # 빠른 시작 데모
├── test_installation.py             # 설치 확인 테스트 (NEW!)
├── setup_conda.bat                  # Windows 자동 설치 (NEW!)
├── setup_conda.sh                   # macOS/Linux 자동 설치 (NEW!)
├── requirements.txt                 # 의존성 패키지
├── environment.yml                  # Conda 환경 파일 (NEW!)
├── .env.example                     # 환경 변수 예제
├── README.md                        # 프로젝트 소개
├── QUICKSTART.md                    # 🚀 빠른 시작 가이드 (NEW!)
├── INSTALL_CONDA.md                 # 📘 Conda 설치 가이드 (NEW!)
└── GUIDE.md                         # 📖 사용 가이드
```

## 설치 방법

**🚀 5분 빠른 시작**: [QUICKSTART.md](QUICKSTART.md) 참고
**📘 Conda 상세 가이드**: [INSTALL_CONDA.md](INSTALL_CONDA.md) 참고

### 방법 A: 자동 설치 (가장 빠름!)

#### Windows
```bash
setup_conda.bat
```

#### macOS/Linux
```bash
bash setup_conda.sh
```

설치가 완료되면:
```bash
conda activate upbit
python quick_recommend.py
```

### 방법 B: 수동 설치

#### 1. Python 환경 준비

Python 3.8 이상이 필요합니다.

```bash
python --version
```

#### 2. 가상환경 생성

##### 옵션 1: Conda 사용 (권장 ⭐)

```bash
# 1. Conda 가상환경 생성
conda create -n upbit python=3.10 -y

# 2. 가상환경 활성화
conda activate upbit

# 3. 패키지 설치
pip install -r requirements.txt
```

**📘 Conda 상세 가이드**: [INSTALL_CONDA.md](INSTALL_CONDA.md) 참고

##### 옵션 2: venv 사용

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

#### 4. 설치 확인

```bash
python test_installation.py
```

모든 패키지에 ✅가 표시되면 설치 완료!

## 빠른 시작 (API 키 불필요!)

### 🌐 실시간 웹앱 (추천! ⭐)

```bash
# 웹앱 패키지 설치 (최초 1회)
pip install -r webapp_requirements.txt

# 웹앱 실행
python webapp.py

# 브라우저에서 접속
http://localhost:8000
```

**웹앱 기능:**
- ✨ 실시간 주요 코인 모니터링 (3초마다 업데이트)
- 📊 TOP 10 추천 종목 대시보드
- 🔍 종목 검색 및 상세 분석
- 📱 모바일/태블릿 지원

**📘 상세 가이드**: [WEBAPP_GUIDE.md](WEBAPP_GUIDE.md)

### 🚀 종목 추천 (CLI)

```bash
# 주요 코인 TOP 5 빠른 분석
python quick_recommend.py
```

몇 초 만에 주요 코인(BTC, ETH, XRP, SOL, DOGE)의 기술적 분석과 AI 예측을 확인할 수 있습니다!

### 🎯 상세 종목 추천

```bash
# 전체 종목 추천 프로그램
python recommend.py
```

**사용 예시:**

1. **기술적 분석 추천**: 전체 KRW 마켓 스캔 후 점수 순으로 추천
2. **AI 예측 추천**: AI 모델 기반 상승 예상 종목 추천
3. **특정 종목 검색**: 원하는 코인 입력 (예: BTC, ETH) → 상세 분석 + AI 예측

**모두 API 키 없이 바로 사용 가능합니다!**

## 업비트 API 키 설정 (자동 매매용)

### 1. API 키 발급

1. [업비트 사이트](https://upbit.com) 로그인
2. 마이페이지 → Open API 관리
3. Open API Key 발급
4. 필요한 권한 선택:
   - ✅ **자산조회** (필수)
   - ✅ **주문조회** (필수)
   - ✅ **주문하기** (실시간 매매 시 필수)
5. IP 주소 등록 (선택사항)
6. **Secret Key를 안전하게 보관** (최초 1회만 확인 가능)

### 2. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 API 키를 입력합니다.

```bash
# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
```

`.env` 파일 내용:

```
UPBIT_ACCESS_KEY=발급받은_Access_Key
UPBIT_SECRET_KEY=발급받은_Secret_Key
```

## 사용 방법

### 📊 종목 추천 (추천!)

#### 1. 빠른 추천
```bash
python quick_recommend.py
```

주요 코인 5개의 기술적 분석 + AI 예측을 빠르게 확인합니다.

#### 2. 전체 시장 스캔 추천
```bash
python recommend.py
```

**메뉴:**
1. **기술적 분석 기반 추천**: RSI, MACD, 이동평균 등으로 점수 산정
2. **AI 예측 기반 추천**: 강화학습 모델로 상승 예상 종목 추천
3. **특정 종목 검색**: 원하는 코인 검색 및 종합 분석

**예시:**
```
선택: 3
종목 코드: BTC
```
→ 비트코인 기술적 분석 + AI 예측 + 종합 판단 출력

### 🤖 자동 매매 (API 키 필요)

#### 메인 프로그램 실행

```bash
python main.py
```

실행하면 다음 메뉴가 표시됩니다:

```
1. AI 모델 학습
2. 백테스팅
3. 실시간 자동 매매 (주의!)
4. 종료
```

### 1. AI 모델 학습

과거 데이터를 사용하여 AI 모델을 학습합니다.

- 기본 학습 데이터: 최근 200일 일봉
- 학습 알고리즘: PPO (Proximal Policy Optimization)
- 학습 결과: `models/crypto_trader` 디렉토리에 저장

**예시:**
```
선택: 1
마켓 코드: KRW-BTC
학습 스텝 수: 100000
```

### 2. 백테스팅

학습된 모델의 성능을 과거 데이터로 검증합니다.

- 백테스팅 결과 출력
- 수익률, MDD, Sharpe Ratio 계산
- 차트 저장: `logs/backtest_result.png`

**예시:**
```
선택: 2
마켓 코드: KRW-BTC
```

### 3. 실시간 자동 매매

**⚠️ 주의: 실제 자금이 사용됩니다!**

- API 키가 필요합니다
- 소액으로 테스트 권장
- 1분마다 시장 데이터 분석 및 매매 실행

**예시:**
```
선택: 3
마켓 코드: KRW-BTC
```

## API만으로 사용 가능한가요?

**네, 완벽하게 가능합니다!**

### 🆓 API 키 불필요 (공개 API 사용)

다음 기능은 **API 키 없이** 바로 사용할 수 있습니다:

1. ✅ **종목 추천 시스템** (NEW!)
   - 전체 시장 기술적 분석
   - AI 기반 상승 예상 종목 추천
   - 특정 종목 검색 및 예측

2. ✅ **AI 모델 학습**
   - 과거 데이터로 모델 학습
   - 백테스팅

### 🔑 API 키 필요 (인증 API 사용)

다음 기능만 API 키가 필요합니다:

3. ✅ **실시간 자동 매매**
   - 자산조회 권한
   - 주문조회 권한
   - 주문하기 권한

**💡 추천 사용법:**
1. 먼저 `python quick_recommend.py`로 종목 추천 받기 (API 키 불필요)
2. 마음에 드는 종목 발견 시 직접 거래 또는 API 키 설정 후 자동 매매


## 기술 스택

- **언어**: Python 3.8+
- **AI 프레임워크**:
  - Stable-Baselines3 (강화학습)
  - PyTorch (딥러닝 백엔드)
  - Gymnasium (환경 인터페이스)
- **데이터 처리**: Pandas, NumPy
- **기술적 분석**: TA-Lib
- **시각화**: Matplotlib, Seaborn

## AI 모델 설명

### PPO (Proximal Policy Optimization)

본 시스템은 **PPO** 알고리즘을 사용합니다.

**선택 이유:**
1. **안정적인 학습**: On-policy 알고리즘으로 안정적
2. **샘플 효율성**: 적은 데이터로도 효과적 학습
3. **검증된 성능**: OpenAI가 개발한 최신 알고리즘
4. **금융 데이터 적합**: 연속적인 의사결정에 강점

### 액션 스페이스

- **0**: HOLD (보유)
- **1**: BUY (매수)
- **2**: SELL (매도)

### 관측 스페이스 (27차원)

1. OHLCV 데이터 (5개)
2. 이동평균선 (5개)
3. MACD 지표 (3개)
4. RSI (1개)
5. 볼린저 밴드 (3개)
6. 스토캐스틱 (2개)
7. ATR (1개)
8. 계좌 상태 (4개)

### 보상 함수

```python
reward = (현재 자산 - 이전 자산) / 이전 자산
```

자산 증가율을 보상으로 사용하여 수익을 극대화합니다.

## 주의사항

### ⚠️ 투자 위험 경고

- 본 시스템은 **교육 및 연구 목적**으로 제작되었습니다
- 가상화폐 투자는 **고위험 투자**이며 원금 손실 가능성이 있습니다
- AI 모델의 성능은 **과거 성과가 미래를 보장하지 않습니다**
- 실제 자금 투자 전 **충분한 백테스팅과 소액 테스트**를 권장합니다
- 투자 손실에 대한 책임은 **사용자 본인**에게 있습니다

### 보안 주의사항

- API Secret Key는 **절대 공유하지 마세요**
- `.env` 파일을 Git에 **커밋하지 마세요**
- API 키에는 **IP 주소 제한**을 설정하세요
- 정기적으로 **API 키를 갱신**하세요

## 성능 개선 팁

1. **더 많은 데이터**: 분봉 데이터 사용 (1분, 5분, 15분)
2. **앙상블**: 여러 모델의 결과를 조합
3. **하이퍼파라미터 튜닝**: Learning rate, batch size 조정
4. **추가 지표**: 더 많은 기술적 지표 활용
5. **리스크 관리**: 손절매, 익절매 로직 추가

## 문제 해결

### PyJWT 오류
```bash
pip install PyJWT==2.8.0
```

### TA-Lib 설치 오류 (Windows)
```bash
pip install ta
# 또는 conda 사용
conda install -c conda-forge ta-lib
```

### CUDA 오류 (GPU 사용 시)
```bash
# PyTorch 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 라이선스

MIT License

## 기여

Pull Request는 언제나 환영합니다!

## 연락처

문의사항이 있으시면 Issue를 등록해주세요.

---

**Happy Trading! 📈**
