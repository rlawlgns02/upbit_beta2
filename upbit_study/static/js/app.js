function app() {
    return {
        // 상태
        activeTab: 'recommendations',
        loading: false,
        analysisLoading: false,
        searchQuery: '',
        currentTime: '',
        analysisMode: 'all', // 분석 모드: all, volume_top100, volume_top50

        // 정렬
        sortBy: 'market', // 정렬 기준: market, current_price, change_rate, trade_volume
        sortDirection: 'asc', // 정렬 방향: asc, desc

        // 진행률
        progressCurrent: 0,
        progressTotal: 0,
        progressMarket: '',

        // 데이터
        realtimeData: [],
        recommendations: [],
        markets: [],
        selectedMarket: null,
        analysisData: null,

        // 뉴스 데이터
        newsSignal: null,
        newsLoading: false,

        // 자동매매 데이터 (수동 선택)
        tradingStatus: null,
        tradingHistory: [],
        tradingSettings: {
            market: 'KRW-BTC',
            interval: 60,
            max_trade_amount: 100000
        },
        apiConnected: false,
        tradingLoading: false,
        tradingStatusInterval: null,

        // 원클릭 자동매매 데이터
        tradingMode: 'auto',  // 'auto' or 'manual'
        autoTradingStatus: null,
        autoTradingSettings: {
            total_investment: 50000,
            coin_count: 3,
            analysis_mode: 'volume_top50',
            trading_interval: 60,
            coin_category: 'safe',  // 'safe', 'normal', 'meme', 'all'
            allocation_mode: 'weighted',  // 'equal' (균등) or 'weighted' (점수기반)
            target_profit_percent: 10,  // 목표가 (+%)
            stop_loss_percent: 10       // 손절가 (-%)
        },
        showConditionSettings: false,  // 조건 설정 패널 토글
        previewCoins: [],  // 미리보기 코인 목록
        previewLoading: false,
        autoTradingLoading: false,
        autoTradingStatusInterval: null,
        balanceRefreshing: false,
        balanceRefreshDone: false,
        miniCharts: {},  // 미니 차트 데이터
        selectedCoinDetail: null,  // 선택된 코인 상세 정보
        showCoinDetailModal: false,  // 코인 상세 모달 표시 여부
        miniChartsInterval: null,

        // 코인 상세 차트 모달 관련
        selectedCoinForChart: null,
        coinChartData: null,
        coinChartLoading: false,
        coinDetailChart: null,
        coinChartPeriod: 60,
        coinChartUpdateInterval: null,

        // 차트 관련
        tradingChart: null,
        chartData: [],
        chartAnnotations: [],
        chartUpdateInterval: null,

        // WebSocket
        ws: null,

        // 인터벌
        analysisInterval: null,

        // 주요 종목 설정
        watchlistMarkets: [],
        defaultWatchlist: ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', 'KRW-SOL', 'KRW-DOGE'],
        showMarketSelector: false,
        marketSelectorSearch: '',

        // 초기화
        async init() {
            this.loadWatchlist();  // 저장된 관심 종목 로드
            this.connectWebSocket();
            this.loadRecommendationsWithProgress();
            this.loadMarkets();
            this.updateTime();
            setInterval(() => this.updateTime(), 1000);

            // 자동매매 상태 로드 및 미니 차트 시작
            await this.loadAutoTradingStatus();
            if (this.autoTradingStatus?.is_running) {
                this.startAutoTradingStatusPolling();
                this.startMiniChartsPolling();
            }
        },

        // 관심 종목 localStorage에서 로드
        loadWatchlist() {
            const saved = localStorage.getItem('watchlistMarkets');
            if (saved) {
                try {
                    this.watchlistMarkets = JSON.parse(saved);
                } catch (e) {
                    this.watchlistMarkets = [...this.defaultWatchlist];
                }
            } else {
                this.watchlistMarkets = [...this.defaultWatchlist];
            }
        },

        // 관심 종목 저장
        saveWatchlist() {
            localStorage.setItem('watchlistMarkets', JSON.stringify(this.watchlistMarkets));
            // WebSocket으로 변경 사항 전송
            this.sendWatchlistToServer();
        },

        // WebSocket으로 관심 종목 전송
        sendWatchlistToServer() {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'set_markets',
                    markets: this.watchlistMarkets
                }));
            }
        },

        // 관심 종목에 추가
        addToWatchlist(market) {
            if (this.watchlistMarkets.includes(market)) {
                // 이미 있으면 제거
                this.removeFromWatchlist(market);
            } else if (this.watchlistMarkets.length < 10) {
                this.watchlistMarkets.push(market);
                this.saveWatchlist();
            }
        },

        // 관심 종목에서 제거
        removeFromWatchlist(market) {
            const index = this.watchlistMarkets.indexOf(market);
            if (index > -1) {
                this.watchlistMarkets.splice(index, 1);
                this.saveWatchlist();
            }
        },

        // 관심 종목 순서 변경
        moveWatchlistItem(index, direction) {
            const newIndex = index + direction;
            if (newIndex >= 0 && newIndex < this.watchlistMarkets.length) {
                const item = this.watchlistMarkets.splice(index, 1)[0];
                this.watchlistMarkets.splice(newIndex, 0, item);
                this.saveWatchlist();
            }
        },

        // 기본값으로 복원
        resetWatchlist() {
            this.watchlistMarkets = [...this.defaultWatchlist];
            this.saveWatchlist();
        },

        // 마켓 이름 가져오기
        getMarketName(marketCode) {
            const market = this.markets.find(m => m.market === marketCode);
            return market ? market.korean_name : '';
        },

        // 종목 선택 모달용 필터링된 마켓
        get filteredMarketsForSelector() {
            if (!this.marketSelectorSearch) {
                return this.markets;
            }
            const query = this.marketSelectorSearch.toLowerCase();
            return this.markets.filter(m =>
                m.market.toLowerCase().includes(query) ||
                m.korean_name.toLowerCase().includes(query) ||
                m.english_name.toLowerCase().includes(query)
            );
        },

        // 시간 업데이트
        updateTime() {
            const now = new Date();
            this.currentTime = now.toLocaleTimeString('ko-KR');
        },

        // WebSocket 연결
        connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/realtime`;

            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket 연결됨');
                // 연결 후 저장된 관심 종목 전송
                this.sendWatchlistToServer();
            };

            this.ws.onmessage = (event) => {
                const message = JSON.parse(event.data);

                if (message.type === 'price_update') {
                    this.realtimeData = message.data;
                }
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket 오류:', error);
            };

            this.ws.onclose = () => {
                console.log('WebSocket 연결 종료, 5초 후 재연결');
                setTimeout(() => this.connectWebSocket(), 5000);
            };
        },

        // 추천 종목 로드 (진행률 포함)
        async loadRecommendationsWithProgress() {
            this.loading = true;
            this.recommendations = [];
            this.progressCurrent = 0;
            this.progressTotal = 0;
            this.progressMarket = '';

            try {
                const response = await fetch(`/api/top-recommendations-stream?mode=${this.analysisMode}`);
                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const text = decoder.decode(value);
                    const lines = text.split('\n\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.substring(6));

                            if (data.type === 'progress') {
                                this.progressCurrent = data.current;
                                this.progressTotal = data.total;
                                this.progressMarket = data.market;
                            } else if (data.type === 'complete') {
                                this.recommendations = data.data;
                            } else if (data.type === 'error') {
                                console.error('분석 오류:', data.error);
                            }
                        }
                    }
                }
            } catch (error) {
                console.error('추천 종목 로드 실패:', error);
            } finally {
                this.loading = false;
            }
        },

        // 전체 마켓 로드
        async loadMarkets() {
            try {
                const response = await fetch('/api/markets');
                const data = await response.json();

                if (data.success) {
                    this.markets = data.data;
                }
            } catch (error) {
                console.error('마켓 로드 실패:', error);
            }
        },

        // 종목 선택
        async selectMarket(market) {
            this.selectedMarket = market;
            this.analysisLoading = true;
            this.analysisData = null;

            // 초기 데이터 로드
            await this.loadAnalysisData(market);

            // 5초마다 실시간 업데이트
            if (this.analysisInterval) {
                clearInterval(this.analysisInterval);
            }

            this.analysisInterval = setInterval(async () => {
                if (this.selectedMarket === market) {
                    await this.loadAnalysisData(market, false); // 로딩 표시 없이
                } else {
                    clearInterval(this.analysisInterval);
                }
            }, 5000);
        },

        // 분석 데이터 로드
        async loadAnalysisData(market, showLoading = true) {
            if (showLoading) {
                this.analysisLoading = true;
            }

            try {
                const response = await fetch(`/api/analysis/${market}`);
                const data = await response.json();

                if (data.success) {
                    this.analysisData = data.data;
                }
            } catch (error) {
                console.error('분석 데이터 로드 실패:', error);
            } finally {
                if (showLoading) {
                    this.analysisLoading = false;
                }
            }
        },

        // 필터링된 마켓
        get filteredMarkets() {
            let filtered = this.markets;

            // 검색 필터
            if (this.searchQuery) {
                const query = this.searchQuery.toLowerCase();
                filtered = filtered.filter(m =>
                    m.market.toLowerCase().includes(query) ||
                    m.korean_name.toLowerCase().includes(query) ||
                    m.english_name.toLowerCase().includes(query)
                );
            }

            // 정렬
            if (this.sortBy) {
                filtered = [...filtered].sort((a, b) => {
                    let aValue = a[this.sortBy];
                    let bValue = b[this.sortBy];

                    // 문자열 비교 (종목명)
                    if (this.sortBy === 'market') {
                        return this.sortDirection === 'asc'
                            ? aValue.localeCompare(bValue)
                            : bValue.localeCompare(aValue);
                    }

                    // 숫자 비교 (가격, 변동률, 거래량)
                    if (this.sortDirection === 'asc') {
                        return aValue - bValue;
                    } else {
                        return bValue - aValue;
                    }
                });
            }

            return filtered;
        },

        // 정렬 토글
        toggleSort(field) {
            if (this.sortBy === field) {
                // 같은 필드 클릭 시 방향 전환
                this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                // 새로운 필드 클릭 시 내림차순으로 시작 (가격, 거래량 등은 큰 값부터 보는게 일반적)
                this.sortBy = field;
                this.sortDirection = field === 'market' ? 'asc' : 'desc';
            }
        },

        // 진행률 퍼센트
        get progressPercent() {
            if (this.progressTotal === 0) return 0;
            return Math.round((this.progressCurrent / this.progressTotal) * 100);
        },

        // 유틸리티 함수들
        formatPrice(price) {
            if (!price) return '-';

            if (price >= 1000) {
                return price.toLocaleString('ko-KR', { maximumFractionDigits: 0 });
            } else if (price >= 1) {
                return price.toLocaleString('ko-KR', { maximumFractionDigits: 2 });
            } else {
                return price.toLocaleString('ko-KR', { maximumFractionDigits: 4 });
            }
        },

        formatVolume(volume) {
            if (!volume) return '-';

            if (volume >= 1000000000000) {
                // 1조 이상 (Trillion)
                return (volume / 1000000000000).toFixed(2) + 'T';
            } else if (volume >= 1000000000) {
                // 10억 이상 (Billion)
                return (volume / 1000000000).toFixed(2) + 'B';
            } else if (volume >= 1000000) {
                // 100만 이상 (Million)
                return (volume / 1000000).toFixed(1) + 'M';
            } else if (volume >= 1000) {
                // 1천 이상 (Thousand)
                return (volume / 1000).toFixed(1) + 'K';
            }
            return volume.toFixed(0);
        },

        getRecommendationClass(recommendation) {
            if (!recommendation) return 'bg-gray-100 text-gray-600';

            if (recommendation.includes('강력 매수')) {
                return 'bg-red-100 text-red-700';
            } else if (recommendation.includes('매수')) {
                return 'bg-orange-100 text-orange-700';
            } else if (recommendation.includes('매도')) {
                return 'bg-blue-100 text-blue-700';
            }
            return 'bg-gray-100 text-gray-600';
        },

        getRecommendationTextClass(recommendation) {
            if (!recommendation) return 'text-gray-600';

            if (recommendation.includes('강력 매수') || recommendation.includes('매수')) {
                return 'text-red-600';
            } else if (recommendation.includes('매도')) {
                return 'text-blue-600';
            }
            return 'text-gray-600';
        },

        // 모달 닫기
        closeModal() {
            this.selectedMarket = null;
            this.analysisData = null;
            if (this.analysisInterval) {
                clearInterval(this.analysisInterval);
                this.analysisInterval = null;
            }
        },

        // 뉴스 신호 로드
        async loadNewsSignal() {
            if (this.newsLoading) return;

            this.newsLoading = true;

            try {
                const response = await fetch('/api/news-signal');
                const data = await response.json();

                if (data.success) {
                    this.newsSignal = data.data;
                } else {
                    console.error('뉴스 신호 로드 실패:', data.error);
                }
            } catch (error) {
                console.error('뉴스 신호 로드 실패:', error);
            } finally {
                this.newsLoading = false;
            }
        },

        // 뉴스 신호 색상 클래스
        getNewsSignalClass(signal) {
            if (signal === 'BUY') return 'text-red-600';
            if (signal === 'SELL') return 'text-blue-600';
            return 'text-gray-600';
        },

        // 뉴스 감정 아이콘
        getNewsSentimentIcon(sentiment) {
            if (sentiment === 'positive') return 'fa-face-smile text-green-500';
            if (sentiment === 'negative') return 'fa-face-frown text-red-500';
            return 'fa-face-meh text-gray-400';
        },

        // 긍정 비율에 따른 그라데이션 색상 계산 (파스텔톤)
        // 0% = 연한 빨간색, 50% = 연한 노란색, 100% = 연한 초록색
        getSentimentColor(ratio) {
            if (ratio === null || ratio === undefined) return 'rgb(180, 180, 180)'; // gray

            // 0~1 범위로 정규화
            const r = Math.max(0, Math.min(1, ratio));

            let red, green, blue;

            if (r < 0.5) {
                // 0~0.5: 연한 빨간색 → 연한 노란색
                const t = r * 2;
                red = Math.round(220 - (220 - 220) * t);   // 220 유지
                green = Math.round(120 + (190 - 120) * t); // 120 → 190
                blue = Math.round(120 + (130 - 120) * t);  // 120 → 130
            } else {
                // 0.5~1: 연한 노란색 → 연한 초록색
                const t = (r - 0.5) * 2;
                red = Math.round(220 - (220 - 130) * t);   // 220 → 130
                green = Math.round(190 + (195 - 190) * t); // 190 → 195
                blue = Math.round(130 + (140 - 130) * t);  // 130 → 140
            }

            return `rgb(${red}, ${green}, ${blue})`;
        },

        // 긍정 비율에 따른 텍스트 색상 (채도 낮춤)
        getSentimentTextColor(ratio) {
            if (ratio === null || ratio === undefined) return '#6b7280'; // gray
            if (ratio > 0.6) return '#059669'; // green (유지)
            if (ratio < 0.4) return '#b91c1c'; // 더 어두운 빨간색
            return '#a16207'; // 더 어두운 노란색
        },

        // ========== 자동매매 관련 함수 ==========

        // API 연결 확인
        async checkApiConnection() {
            try {
                const response = await fetch('/api/account/check');
                const data = await response.json();
                this.apiConnected = data.connected || false;

                if (this.apiConnected) {
                    this.loadTradingStatus();
                }

                // 차트 초기화 및 실시간 업데이트 시작 (API 연결 여부와 관계없이)
                setTimeout(() => {
                    this.initTradingChart();
                    this.startChartUpdate();
                }, 100);
            } catch (error) {
                console.error('API 연결 확인 실패:', error);
                this.apiConnected = false;
            }
        },

        // 자동매매 시작 (수동 선택)
        async startTrading() {
            if (this.tradingLoading) return;

            this.tradingLoading = true;

            // 잔고 확인
            const balance = await this.checkBalance();
            if (balance <= 0) {
                alert('⚠️ 잔고가 없습니다.\n\n업비트 계좌에 KRW를 입금한 후 다시 시도해주세요.');
                this.tradingLoading = false;
                return;
            }

            // 최대 거래금액이 잔고보다 큰 경우 경고
            if (this.tradingSettings.max_trade_amount > balance) {
                alert(`⚠️ 설정한 최대 거래금액(${this.formatPrice(this.tradingSettings.max_trade_amount)}원)이 현재 잔고(${this.formatPrice(balance)}원)보다 큽니다.\n\n거래금액을 조정하거나 잔고를 확인해주세요.`);
                this.tradingLoading = false;
                return;
            }

            if (!confirm(`${this.tradingSettings.market} 자동매매를 시작합니다.\n\n현재 잔고: ${this.formatPrice(balance)}원\n최대 거래금액: ${this.formatPrice(this.tradingSettings.max_trade_amount)}원\n\n계속하시겠습니까?`)) {
                this.tradingLoading = false;
                return;
            }

            try {
                const response = await fetch('/api/trading/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.tradingSettings)
                });

                const data = await response.json();

                if (data.success) {
                    console.log('자동매매 시작:', data.message);
                    this.startTradingStatusPolling();
                    // 차트 리셋 후 새로 로드
                    this.resetChart();
                    this.loadChartData();
                } else {
                    alert('자동매매 시작 실패: ' + data.error);
                }
            } catch (error) {
                console.error('자동매매 시작 실패:', error);
                alert('자동매매 시작 중 오류가 발생했습니다.');
            } finally {
                this.tradingLoading = false;
                await this.loadTradingStatus();
            }
        },

        // 자동매매 중지
        async stopTrading() {
            if (this.tradingLoading) return;

            this.tradingLoading = true;

            try {
                const response = await fetch('/api/trading/stop', {
                    method: 'POST'
                });

                const data = await response.json();

                if (data.success) {
                    console.log('자동매매 중지:', data.message);
                    this.stopTradingStatusPolling();
                    // 차트는 계속 업데이트 유지
                } else {
                    alert('자동매매 중지 실패: ' + data.error);
                }
            } catch (error) {
                console.error('자동매매 중지 실패:', error);
                alert('자동매매 중지 중 오류가 발생했습니다.');
            } finally {
                this.tradingLoading = false;
                await this.loadTradingStatus();
            }
        },

        // 자동매매 상태 조회
        async loadTradingStatus() {
            try {
                const response = await fetch('/api/trading/status');
                const data = await response.json();

                if (data.success) {
                    const prevTradeCount = this.tradingHistory?.length || 0;
                    this.tradingStatus = data.data;
                    this.tradingHistory = data.data.trade_history || [];

                    // 새 거래가 있으면 차트에 마커 추가
                    if (this.tradingHistory.length > prevTradeCount) {
                        const newTrades = this.tradingHistory.slice(prevTradeCount);
                        for (const trade of newTrades) {
                            this.addTradeAnnotation(trade.action, trade.price, trade.time);
                        }
                    }

                    // 실행 중이면 상태 폴링 시작
                    if (this.tradingStatus?.is_running && !this.tradingStatusInterval) {
                        this.startTradingStatusPolling();
                    }
                }
            } catch (error) {
                console.error('자동매매 상태 조회 실패:', error);
            }
        },

        // 거래 내역 조회
        async loadTradingHistory() {
            try {
                const response = await fetch('/api/trading/history');
                const data = await response.json();

                if (data.success) {
                    this.tradingHistory = data.data || [];
                }
            } catch (error) {
                console.error('거래 내역 조회 실패:', error);
            }
        },

        // 상태 폴링 시작
        startTradingStatusPolling() {
            if (this.tradingStatusInterval) {
                clearInterval(this.tradingStatusInterval);
            }

            this.tradingStatusInterval = setInterval(async () => {
                await this.loadTradingStatus();

                // 실행 중지되면 폴링 중지
                if (!this.tradingStatus?.is_running) {
                    this.stopTradingStatusPolling();
                }
            }, 3000);
        },

        // 상태 폴링 중지
        stopTradingStatusPolling() {
            if (this.tradingStatusInterval) {
                clearInterval(this.tradingStatusInterval);
                this.tradingStatusInterval = null;
            }
        },

        // 거래 시간 포맷
        formatTradeTime(isoString) {
            if (!isoString) return '-';
            const date = new Date(isoString);
            return date.toLocaleString('ko-KR', {
                month: '2-digit',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        },

        // ========== 차트 관련 함수 ==========

        // 차트 초기화
        initTradingChart() {
            const chartEl = document.querySelector('#trading-chart');
            if (!chartEl) return;

            // 기존 차트 제거
            if (this.tradingChart) {
                this.tradingChart.destroy();
            }

            const options = {
                series: [{
                    name: '가격',
                    data: []
                }],
                chart: {
                    type: 'area',
                    height: 350,
                    animations: {
                        enabled: true,
                        easing: 'linear',
                        dynamicAnimation: {
                            speed: 1000
                        }
                    },
                    toolbar: {
                        show: true,
                        tools: {
                            download: false,
                            selection: true,
                            zoom: true,
                            zoomin: true,
                            zoomout: true,
                            pan: true,
                            reset: true
                        }
                    },
                    zoom: {
                        enabled: true
                    }
                },
                dataLabels: {
                    enabled: false
                },
                stroke: {
                    curve: 'smooth',
                    width: 2
                },
                fill: {
                    type: 'gradient',
                    gradient: {
                        shadeIntensity: 1,
                        opacityFrom: 0.4,
                        opacityTo: 0.1,
                        stops: [0, 90, 100]
                    }
                },
                colors: ['#3b82f6'],
                xaxis: {
                    type: 'datetime',
                    labels: {
                        datetimeUTC: false,
                        format: 'HH:mm:ss'
                    }
                },
                yaxis: {
                    labels: {
                        formatter: (value) => {
                            if (value >= 1000000) {
                                return (value / 1000000).toFixed(1) + 'M';
                            } else if (value >= 1000) {
                                return value.toLocaleString();
                            }
                            return value.toFixed(2);
                        }
                    }
                },
                tooltip: {
                    x: {
                        format: 'yyyy-MM-dd HH:mm:ss'
                    },
                    y: {
                        formatter: (value) => '₩ ' + value.toLocaleString()
                    }
                },
                annotations: {
                    points: []
                },
                grid: {
                    borderColor: '#e5e7eb',
                    strokeDashArray: 4
                }
            };

            this.tradingChart = new ApexCharts(chartEl, options);
            this.tradingChart.render();

            // 초기 데이터 로드
            this.loadChartData();
        },

        // 차트 데이터 로드
        async loadChartData() {
            try {
                const market = this.tradingSettings.market;
                const response = await fetch(`/api/trading/chart-data?market=${market}`);
                const data = await response.json();

                if (data.success && data.data) {
                    this.chartData = data.data.map(item => ({
                        x: new Date(item.time).getTime(),
                        y: item.price
                    }));

                    if (this.tradingChart) {
                        this.tradingChart.updateSeries([{
                            name: '가격',
                            data: this.chartData
                        }]);
                    }
                }
            } catch (error) {
                console.error('차트 데이터 로드 실패:', error);
            }
        },

        // 차트에 새 데이터 추가
        addChartDataPoint(price) {
            const now = new Date().getTime();
            this.chartData.push({ x: now, y: price });

            // 최대 100개 데이터 유지
            if (this.chartData.length > 100) {
                this.chartData.shift();
            }

            if (this.tradingChart) {
                this.tradingChart.updateSeries([{
                    name: '가격',
                    data: this.chartData
                }]);
            }
        },

        // 차트에 매매 포인트 추가
        addTradeAnnotation(action, price, time) {
            const annotation = {
                x: new Date(time).getTime(),
                y: price,
                marker: {
                    size: 8,
                    fillColor: action === 'BUY' ? '#ef4444' : '#3b82f6',
                    strokeColor: '#fff',
                    strokeWidth: 2
                },
                label: {
                    borderColor: action === 'BUY' ? '#ef4444' : '#3b82f6',
                    style: {
                        color: '#fff',
                        background: action === 'BUY' ? '#ef4444' : '#3b82f6'
                    },
                    text: action === 'BUY' ? '매수' : '매도'
                }
            };

            this.chartAnnotations.push(annotation);

            if (this.tradingChart) {
                this.tradingChart.updateOptions({
                    annotations: {
                        points: this.chartAnnotations
                    }
                });
            }
        },

        // 차트 업데이트 시작 (실시간 가격 폴링)
        startChartUpdate() {
            if (this.chartUpdateInterval) {
                clearInterval(this.chartUpdateInterval);
            }

            this.chartUpdateInterval = setInterval(async () => {
                try {
                    // 현재 선택된 마켓의 실시간 가격 조회
                    const market = this.tradingSettings.market;
                    const response = await fetch(`/api/trading/realtime-price?market=${market}`);
                    const data = await response.json();

                    if (data.success && data.price) {
                        this.addChartDataPoint(data.price);
                    }
                } catch (error) {
                    console.error('실시간 가격 조회 실패:', error);
                }
            }, 2000); // 2초마다 업데이트
        },

        // 차트 업데이트 중지
        stopChartUpdate() {
            if (this.chartUpdateInterval) {
                clearInterval(this.chartUpdateInterval);
                this.chartUpdateInterval = null;
            }
        },

        // 차트 리셋
        resetChart() {
            this.chartData = [];
            this.chartAnnotations = [];
            if (this.tradingChart) {
                this.tradingChart.updateSeries([{ name: '가격', data: [] }]);
                this.tradingChart.updateOptions({ annotations: { points: [] } });
            }
        },

        // ========== 원클릭 자동매매 함수 ==========

        // 잔고 확인
        async checkBalance() {
            try {
                const response = await fetch('/api/account/balance');
                const data = await response.json();

                if (data.success && data.data) {
                    const krwBalance = data.data.find(b => b.currency === 'KRW');
                    return krwBalance ? krwBalance.balance : 0;
                }
                return 0;
            } catch (error) {
                console.error('잔고 확인 실패:', error);
                return 0;
            }
        },

        // 자동매매 시작
        async startAutoTrading() {
            if (this.autoTradingLoading) return;

            // 현재 잔고 확인
            const balance = await this.checkBalance();

            // 총 투자금액이 현재 잔고보다 많은지 확인
            if (this.autoTradingSettings.total_investment > balance) {
                alert(`⚠️ 투자금액이 잔고를 초과합니다.\n\n설정 투자금액: ${this.formatPrice(this.autoTradingSettings.total_investment)}원\n현재 잔고: ${this.formatPrice(balance)}원\n\n투자금액을 현재 잔고 이하로 설정해주세요.`);
                return;
            }

            // 미리보기가 없으면 먼저 로드
            if (!this.previewCoins || this.previewCoins.length === 0) {
                this.previewLoading = true;
                await this.loadCoinPreview();
                this.previewLoading = false;
            }

            // 선택될 코인 목록 표시
            let coinListText = this.previewCoins.map((coin, i) => {
                const name = coin.name || coin.market.split('-')[1];
                const score = coin.score_100 || coin.score || 0;
                return `${i + 1}. ${name} (${coin.market}) - 점수: ${score}점`;
            }).join('\n');

            const confirmMessage = `🤖 AI 자동매매를 시작합니다.\n\n` +
                `📊 선택된 코인 (점수 높은 순):\n${coinListText}\n\n` +
                `💰 투자금액: ${this.formatPrice(this.autoTradingSettings.total_investment)}원\n` +
                `💵 현재 잔고: ${this.formatPrice(balance)}원\n\n` +
                `위 코인으로 자동매매를 시작하시겠습니까?`;

            if (!confirm(confirmMessage)) {
                return;
            }

            this.autoTradingLoading = true;

            try {
                const response = await fetch('/api/auto-trading/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.autoTradingSettings)
                });

                const data = await response.json();

                if (data.success) {
                    console.log('원클릭 자동매매 시작:', data.message);
                    this.startAutoTradingStatusPolling();
                    this.startMiniChartsPolling();  // 미니 차트 폴링 시작

                    // 잔고 확인 후 알림
                    const balance = await this.checkBalance();
                    if (balance <= 0) {
                        alert('⚠️ 계좌에 잔고가 없습니다.\n\nAI가 선택한 코인을 확인할 수 있습니다.\n실제 자동매매를 진행하려면 업비트 계좌에 KRW를 입금해주세요.');
                    }
                } else {
                    alert('자동매매 시작 실패: ' + data.error);
                }
            } catch (error) {
                console.error('자동매매 시작 실패:', error);
                alert('자동매매 시작 중 오류가 발생했습니다.');
            } finally {
                this.autoTradingLoading = false;
                await this.loadAutoTradingStatus();
            }
        },

        // 자동매매 중지
        async stopAutoTrading() {
            if (!confirm('자동매매를 중지하면 모든 포지션이 청산됩니다.\n\n계속하시겠습니까?')) {
                return;
            }

            this.autoTradingLoading = true;

            try {
                const response = await fetch('/api/auto-trading/stop', {
                    method: 'POST'
                });

                const data = await response.json();

                if (data.success) {
                    console.log('자동매매 중지:', data.message);
                    this.stopAutoTradingStatusPolling();
                } else {
                    alert('자동매매 중지 실패: ' + data.error);
                }
            } catch (error) {
                console.error('자동매매 중지 실패:', error);
                alert('자동매매 중지 중 오류가 발생했습니다.');
            } finally {
                this.autoTradingLoading = false;
                await this.loadAutoTradingStatus();
            }
        },

        // 포트폴리오에서 코인 제거
        async removeCoinFromPortfolio(market) {
            if (!confirm(`${market}을(를) 포트폴리오에서 제거하시겠습니까?\n\n보유 중이면 시장가로 매도됩니다.`)) {
                return;
            }

            try {
                const response = await fetch(`/api/auto-trading/remove-coin?market=${market}`, {
                    method: 'POST'
                });

                const data = await response.json();

                if (data.success) {
                    console.log('코인 제거:', data.message);
                    await this.loadAutoTradingStatus();
                } else {
                    alert('코인 제거 실패: ' + data.error);
                }
            } catch (error) {
                console.error('코인 제거 실패:', error);
                alert('코인 제거 중 오류가 발생했습니다.');
            }
        },

        // 지정가 주문 취소
        async cancelPendingOrder(market) {
            if (!confirm(`${market}의 지정가 주문을 취소하시겠습니까?`)) {
                return;
            }

            try {
                const response = await fetch(`/api/auto-trading/cancel-order?market=${market}`, {
                    method: 'POST'
                });

                const data = await response.json();

                if (data.success) {
                    console.log('주문 취소:', data.message);
                    await this.loadAutoTradingStatus();
                } else {
                    alert('주문 취소 실패: ' + data.error);
                }
            } catch (error) {
                console.error('주문 취소 실패:', error);
                alert('주문 취소 중 오류가 발생했습니다.');
            }
        },

        // 코인 미리보기 로드
        async loadCoinPreview(category = null) {
            // 이미 실행 중이면 미리보기 안 함
            if (this.autoTradingStatus?.is_running) return;

            this.previewLoading = true;

            // 명시적으로 전달된 카테고리 사용, 없으면 현재 설정값 사용
            const effectiveCategory = category || this.autoTradingSettings.coin_category;

            console.log('미리보기 요청 - 카테고리:', effectiveCategory, '코인수:', this.autoTradingSettings.coin_count, '배분:', this.autoTradingSettings.allocation_mode);

            try {
                const response = await fetch('/api/auto-trading/preview', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        coin_count: parseInt(this.autoTradingSettings.coin_count),
                        analysis_mode: this.autoTradingSettings.analysis_mode,
                        coin_category: effectiveCategory,
                        allocation_mode: this.autoTradingSettings.allocation_mode,
                        total_investment: parseFloat(this.autoTradingSettings.total_investment)
                    })
                });

                const data = await response.json();

                if (data.success) {
                    this.previewCoins = data.data.selected_coins;
                    console.log('코인 미리보기 로드:', this.previewCoins.length + '개');
                } else {
                    console.error('코인 미리보기 실패:', data.error);
                    this.previewCoins = [];
                }
            } catch (error) {
                console.error('코인 미리보기 실패:', error);
                this.previewCoins = [];
            } finally {
                this.previewLoading = false;
            }
        },

        // 카테고리 변경 시 미리보기 업데이트
        async onCategoryChange() {
            await this.loadCoinPreview();
        },

        // 코인 수 변경 시 미리보기 업데이트
        async onCoinCountChange() {
            await this.loadCoinPreview();
        },

        // 자동매매 상태 조회
        async loadAutoTradingStatus() {
            try {
                const response = await fetch('/api/auto-trading/status');
                const data = await response.json();

                if (data.success) {
                    this.autoTradingStatus = data.data;
                }
            } catch (error) {
                console.error('자동매매 상태 조회 실패:', error);
            }
        },

        // 잔고 새로고침 (버튼 클릭용)
        async fetchAutoTradingStatus() {
            if (this.balanceRefreshing) return;

            this.balanceRefreshing = true;
            this.balanceRefreshDone = false;

            try {
                await this.loadAutoTradingStatus();
                // 성공 시 체크 아이콘 표시
                this.balanceRefreshing = false;
                this.balanceRefreshDone = true;

                // 1.5초 후 원래대로
                setTimeout(() => {
                    this.balanceRefreshDone = false;
                }, 1500);
            } catch (error) {
                this.balanceRefreshing = false;
                console.error('잔고 새로고침 실패:', error);
            }
        },

        // 자동매매 상태 폴링 시작
        startAutoTradingStatusPolling() {
            if (this.autoTradingStatusInterval) {
                clearInterval(this.autoTradingStatusInterval);
            }

            this.autoTradingStatusInterval = setInterval(async () => {
                await this.loadAutoTradingStatus();

                if (!this.autoTradingStatus?.is_running) {
                    this.stopAutoTradingStatusPolling();
                }
            }, 3000);
        },

        // 자동매매 상태 폴링 중지
        stopAutoTradingStatusPolling() {
            if (this.autoTradingStatusInterval) {
                clearInterval(this.autoTradingStatusInterval);
                this.autoTradingStatusInterval = null;
            }
            this.stopMiniChartsPolling();
        },

        // 미니 차트 데이터 로드
        async loadMiniCharts() {
            try {
                const response = await fetch('/api/auto-trading/mini-charts');
                const data = await response.json();
                if (data.success) {
                    this.miniCharts = data.data;
                }
            } catch (error) {
                console.error('미니 차트 데이터 로드 실패:', error);
            }
        },

        // 미니 차트 폴링 시작
        startMiniChartsPolling() {
            this.loadMiniCharts();
            if (this.miniChartsInterval) {
                clearInterval(this.miniChartsInterval);
            }
            this.miniChartsInterval = setInterval(() => {
                this.loadMiniCharts();
            }, 10000);  // 10초마다 업데이트
        },

        // 미니 차트 폴링 중지
        stopMiniChartsPolling() {
            if (this.miniChartsInterval) {
                clearInterval(this.miniChartsInterval);
                this.miniChartsInterval = null;
            }
        },

        // SVG 스파크라인 경로 생성
        getSparklinePath(market) {
            const prices = this.miniCharts[market];
            if (!prices || prices.length < 2) return '';

            const width = 120;
            const height = 40;
            const padding = 2;

            const min = Math.min(...prices);
            const max = Math.max(...prices);
            const range = max - min || 1;

            const points = prices.map((price, i) => {
                const x = padding + (i / (prices.length - 1)) * (width - padding * 2);
                const y = height - padding - ((price - min) / range) * (height - padding * 2);
                return `${x},${y}`;
            });

            return `M ${points.join(' L ')}`;
        },

        // 스파크라인 색상 (상승/하락)
        getSparklineColor(market) {
            const prices = this.miniCharts[market];
            if (!prices || prices.length < 2) return '#9CA3AF';
            return prices[prices.length - 1] >= prices[0] ? '#EF4444' : '#3B82F6';
        },

        // ========== 코인 상세 차트 관련 함수 ==========

        // 코인 차트 모달 열기
        async openCoinChart(coin) {
            this.selectedCoinForChart = coin;
            this.coinChartPeriod = 60;
            this.coinChartData = null;
            this.coinChartLoading = true;

            // 차트 초기화 대기 후 데이터 로드
            await this.$nextTick();
            setTimeout(() => {
                this.initCoinDetailChart();
                this.loadCoinChartData(coin.market, this.coinChartPeriod, true);
                this.startCoinChartUpdate();
            }, 100);
        },

        // 코인 차트 모달 닫기
        closeCoinChart() {
            this.stopCoinChartUpdate();
            this.selectedCoinForChart = null;
            this.coinChartData = null;

            if (this.coinDetailChart) {
                this.coinDetailChart.destroy();
                this.coinDetailChart = null;
            }
        },

        // 코인 상세 차트 초기화
        initCoinDetailChart() {
            const chartEl = document.querySelector('#coin-detail-chart');
            if (!chartEl) return;

            // 기존 차트 제거
            if (this.coinDetailChart) {
                this.coinDetailChart.destroy();
            }

            const options = {
                series: [{
                    name: '가격',
                    data: []
                }],
                chart: {
                    type: 'area',
                    height: 256,
                    animations: {
                        enabled: true,
                        easing: 'linear',
                        dynamicAnimation: {
                            speed: 1000
                        }
                    },
                    toolbar: {
                        show: true,
                        tools: {
                            download: false,
                            selection: true,
                            zoom: true,
                            zoomin: true,
                            zoomout: true,
                            pan: true,
                            reset: true
                        }
                    },
                    zoom: {
                        enabled: true
                    }
                },
                dataLabels: {
                    enabled: false
                },
                stroke: {
                    curve: 'smooth',
                    width: 2
                },
                fill: {
                    type: 'gradient',
                    gradient: {
                        shadeIntensity: 1,
                        opacityFrom: 0.4,
                        opacityTo: 0.1,
                        stops: [0, 90, 100]
                    }
                },
                colors: ['#3b82f6'],
                xaxis: {
                    type: 'datetime',
                    labels: {
                        datetimeUTC: false,
                        format: 'HH:mm'
                    }
                },
                yaxis: {
                    labels: {
                        formatter: (value) => {
                            if (value >= 1000000) {
                                return (value / 1000000).toFixed(1) + 'M';
                            } else if (value >= 1000) {
                                return value.toLocaleString();
                            }
                            return value.toFixed(2);
                        }
                    }
                },
                tooltip: {
                    x: {
                        format: 'yyyy-MM-dd HH:mm:ss'
                    },
                    y: {
                        formatter: (value) => '₩ ' + value.toLocaleString()
                    }
                },
                grid: {
                    borderColor: '#e5e7eb',
                    strokeDashArray: 4
                }
            };

            this.coinDetailChart = new ApexCharts(chartEl, options);
            this.coinDetailChart.render();
        },

        // 코인 차트 데이터 로드
        // showLoading: true면 로딩 스피너 표시, false면 백그라운드 업데이트
        async loadCoinChartData(market, count = 60, showLoading = false) {
            if (!market) return;

            // 초기 로딩 시에만 스피너 표시
            if (showLoading) {
                this.coinChartLoading = true;
            }
            this.coinChartPeriod = count;

            try {
                const response = await fetch(`/api/trading/chart-data?market=${market}&count=${count}`);
                const data = await response.json();

                if (data.success && data.data) {
                    const chartData = data.data.map(item => ({
                        x: new Date(item.time).getTime(),
                        y: item.price
                    }));

                    // 현재가 및 변동률 업데이트
                    if (data.data.length > 0) {
                        const latestPrice = data.data[data.data.length - 1].price;
                        const firstPrice = data.data[0].price;
                        const changeRate = ((latestPrice - firstPrice) / firstPrice) * 100;

                        this.coinChartData = {
                            currentPrice: latestPrice,
                            changeRate: changeRate
                        };
                    }

                    if (this.coinDetailChart) {
                        this.coinDetailChart.updateSeries([{
                            name: '가격',
                            data: chartData
                        }]);
                    }
                }
            } catch (error) {
                console.error('코인 차트 데이터 로드 실패:', error);
            } finally {
                if (showLoading) {
                    this.coinChartLoading = false;
                }
            }
        },

        // 코인 차트 실시간 업데이트 시작
        startCoinChartUpdate() {
            if (this.coinChartUpdateInterval) {
                clearInterval(this.coinChartUpdateInterval);
            }

            this.coinChartUpdateInterval = setInterval(async () => {
                if (this.selectedCoinForChart) {
                    await this.loadCoinChartData(this.selectedCoinForChart.market, this.coinChartPeriod, false);
                }
            }, 10000); // 10초마다 백그라운드 업데이트
        },

        // 코인 차트 실시간 업데이트 중지
        stopCoinChartUpdate() {
            if (this.coinChartUpdateInterval) {
                clearInterval(this.coinChartUpdateInterval);
                this.coinChartUpdateInterval = null;
            }
        },

        // 특정 코인의 거래 내역 필터링
        getCoinTradeHistory(market) {
            if (!market || !this.autoTradingStatus?.trade_history) {
                return [];
            }
            return this.autoTradingStatus.trade_history
                .filter(trade => trade.market === market)
                .reverse();
        }
    };
}
