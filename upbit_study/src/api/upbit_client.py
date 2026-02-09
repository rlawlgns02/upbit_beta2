"""
업비트 API 클라이언트
공식 문서: https://docs.upbit.com/reference
"""
import jwt
import hashlib
import requests
import uuid
from urllib.parse import urlencode, unquote
from typing import Dict, List, Optional


def get_tick_size(price: float) -> float:
    """업비트 호가 단위 계산

    Args:
        price: 주문 가격

    Returns:
        해당 가격대의 호가 단위
    """
    if price >= 2000000:
        return 1000
    elif price >= 1000000:
        return 500
    elif price >= 500000:
        return 100
    elif price >= 100000:
        return 50
    elif price >= 10000:
        return 10
    elif price >= 1000:
        return 5
    elif price >= 100:
        return 1
    elif price >= 10:
        return 0.1
    elif price >= 1:
        return 0.01
    elif price >= 0.1:
        return 0.001
    else:
        return 0.0001


def adjust_price_to_tick(price: float, is_buy: bool = True) -> float:
    """가격을 호가 단위에 맞게 조정

    Args:
        price: 원래 가격
        is_buy: 매수 여부 (True: 내림, False: 올림)

    Returns:
        호가 단위에 맞춘 가격
    """
    tick_size = get_tick_size(price)

    if is_buy:
        # 매수: 내림 (더 낮은 가격으로)
        adjusted = (price // tick_size) * tick_size
    else:
        # 매도: 올림 (더 높은 가격으로)
        import math
        adjusted = math.ceil(price / tick_size) * tick_size

    # 소수점 정밀도 문제 해결
    if tick_size >= 1:
        return int(adjusted)
    else:
        # 소수점 자릿수 결정
        decimals = len(str(tick_size).split('.')[-1])
        return round(adjusted, decimals)


class UpbitClient:
    """업비트 거래소 API 클라이언트"""

    def __init__(self, access_key: str, secret_key: str):
        """
        Args:
            access_key: 업비트 Open API Access Key
            secret_key: 업비트 Open API Secret Key
        """
        self.access_key = access_key
        self.secret_key = secret_key
        self.server_url = "https://api.upbit.com"

    def _get_headers(self, query: Optional[Dict] = None) -> Dict[str, str]:
        """JWT 토큰이 포함된 헤더 생성"""
        payload = {
            'access_key': self.access_key,
            'nonce': str(uuid.uuid4()),
        }

        if query:
            query_string = unquote(urlencode(query, doseq=True)).encode("utf-8")
            m = hashlib.sha512()
            m.update(query_string)
            query_hash = m.hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'

        jwt_token = jwt.encode(payload, self.secret_key)
        return {'Authorization': f'Bearer {jwt_token}'}

    # ========== 공개 API (인증 불필요) ==========

    def get_market_all(self) -> List[Dict]:
        """마켓 코드 조회 (전체 종목 리스트)"""
        url = f"{self.server_url}/v1/market/all"
        params = {'isDetails': 'true'}
        try:
            response = requests.get(url, params=params, timeout=10)

            # HTTP 상태 코드 체크
            if response.status_code != 200:
                print(f"[UPBIT] 마켓 조회 실패 - HTTP {response.status_code}: {response.text}")
                return []

            result = response.json()
            if isinstance(result, dict) and 'error' in result:
                print(f"[UPBIT] 마켓 조회 실패: {result}")
                return []
            return result
        except requests.exceptions.Timeout:
            print(f"[UPBIT] 마켓 조회 타임아웃")
            return []
        except Exception as e:
            print(f"[UPBIT] 마켓 조회 예외: {e}")
            return []

    def get_ticker(self, markets: List[str]) -> List[Dict]:
        """현재가 정보 조회

        Args:
            markets: 마켓 코드 리스트 (예: ['KRW-BTC', 'KRW-ETH'])
        """
        if not markets:
            return []
        url = f"{self.server_url}/v1/ticker"
        params = {'markets': ','.join(markets)}
        try:
            response = requests.get(url, params=params, timeout=10)

            # HTTP 상태 코드 체크
            if response.status_code != 200:
                print(f"[UPBIT] 티커 조회 실패 - HTTP {response.status_code}: {response.text}")
                return []

            result = response.json()
            if isinstance(result, dict) and 'error' in result:
                print(f"[UPBIT] 티커 조회 실패: {result}")
                return []
            return result
        except requests.exceptions.Timeout:
            print(f"[UPBIT] 티커 조회 타임아웃")
            return []
        except Exception as e:
            print(f"[UPBIT] 티커 조회 예외: {e}")
            return []

    def get_orderbook(self, markets: List[str]) -> List[Dict]:
        """호가 정보 조회"""
        if not markets:
            return []
        url = f"{self.server_url}/v1/orderbook"
        params = {'markets': ','.join(markets)}
        try:
            response = requests.get(url, params=params, timeout=10)

            # HTTP 상태 코드 체크
            if response.status_code != 200:
                print(f"[UPBIT] 호가 조회 실패 - HTTP {response.status_code}: {response.text}")
                return []

            result = response.json()
            if isinstance(result, dict) and 'error' in result:
                print(f"[UPBIT] 호가 조회 실패: {result}")
                return []
            return result
        except requests.exceptions.Timeout:
            print(f"[UPBIT] 호가 조회 타임아웃")
            return []
        except Exception as e:
            print(f"[UPBIT] 호가 조회 예외: {e}")
            return []

    def get_candles_minute(self, market: str, unit: int = 1, count: int = 200) -> List[Dict]:
        """분봉 데이터 조회

        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            unit: 분 단위 (1, 3, 5, 15, 10, 30, 60, 240)
            count: 캔들 개수 (최대 200)
        """
        url = f"{self.server_url}/v1/candles/minutes/{unit}"
        params = {'market': market, 'count': count}
        try:
            response = requests.get(url, params=params, timeout=15)

            # HTTP 상태 코드 체크
            if response.status_code != 200:
                print(f"[UPBIT] 분봉 조회 실패 - HTTP {response.status_code}: {response.text}")
                return []

            result = response.json()
            if isinstance(result, dict) and 'error' in result:
                print(f"[UPBIT] 분봉 조회 실패: {result}")
                return []
            return result
        except requests.exceptions.Timeout:
            print(f"[UPBIT] 분봉 조회 타임아웃")
            return []
        except Exception as e:
            print(f"[UPBIT] 분봉 조회 예외: {e}")
            return []

    def get_candles_day(self, market: str, count: int = 200) -> List[Dict]:
        """일봉 데이터 조회"""
        url = f"{self.server_url}/v1/candles/days"
        params = {'market': market, 'count': count}
        try:
            response = requests.get(url, params=params, timeout=15)

            # HTTP 상태 코드 체크
            if response.status_code != 200:
                print(f"[UPBIT] 일봉 조회 실패 - HTTP {response.status_code}: {response.text}")
                return []

            result = response.json()
            if isinstance(result, dict) and 'error' in result:
                print(f"[UPBIT] 일봉 조회 실패: {result}")
                return []
            return result
        except requests.exceptions.Timeout:
            print(f"[UPBIT] 일봉 조회 타임아웃")
            return []
        except Exception as e:
            print(f"[UPBIT] 일봉 조회 예외: {e}")
            return []

    # ========== 인증 필요 API ==========

    def get_accounts(self) -> List[Dict]:
        """전체 계좌 조회 (자산 확인)"""
        url = f"{self.server_url}/v1/accounts"
        headers = self._get_headers()
        try:
            response = requests.get(url, headers=headers, timeout=10)

            # HTTP 상태 코드 체크 (인증 실패 등)
            if response.status_code == 401:
                print(f"[UPBIT] 계좌 조회 실패 - 인증 실패: API 키를 확인하세요")
                return []
            elif response.status_code != 200:
                print(f"[UPBIT] 계좌 조회 실패 - HTTP {response.status_code}: {response.text}")
                return []

            result = response.json()
            # API 에러 응답 체크
            if isinstance(result, dict) and 'error' in result:
                print(f"[UPBIT] 계좌 조회 실패: {result}")
                return []
            return result
        except requests.exceptions.Timeout:
            print(f"[UPBIT] 계좌 조회 타임아웃")
            return []
        except Exception as e:
            print(f"[UPBIT] 계좌 조회 예외: {e}")
            return []  # 타입 일관성을 위해 빈 리스트 반환

    def get_orders(self, market: Optional[str] = None,
                   state: str = 'wait',
                   states: Optional[List[str]] = None,
                   page: int = 1,
                   limit: int = 100,
                   order_by: str = 'desc') -> List[Dict]:
        """주문 리스트 조회

        Args:
            market: 마켓 코드
            state: 주문 상태 (wait, watch, done, cancel)
            states: 주문 상태 리스트
            page: 페이지 수
            limit: 요청 개수
            order_by: 정렬 방식 (asc, desc)
        """
        query = {
            'state': state,
            'page': page,
            'limit': limit,
            'order_by': order_by
        }

        if market:
            query['market'] = market
        if states:
            query['states[]'] = states

        url = f"{self.server_url}/v1/orders"
        headers = self._get_headers(query)
        try:
            response = requests.get(url, params=query, headers=headers, timeout=10)

            # HTTP 상태 코드 체크
            if response.status_code == 401:
                print(f"[UPBIT] 주문 목록 조회 실패 - 인증 실패")
                return []
            elif response.status_code != 200:
                print(f"[UPBIT] 주문 목록 조회 실패 - HTTP {response.status_code}: {response.text}")
                return []

            result = response.json()
            # API 에러 응답 체크
            if isinstance(result, dict) and 'error' in result:
                print(f"[UPBIT] 주문 목록 조회 실패: {result}")
                return []
            return result
        except requests.exceptions.Timeout:
            print(f"[UPBIT] 주문 목록 조회 타임아웃")
            return []
        except Exception as e:
            print(f"[UPBIT] 주문 목록 조회 예외: {e}")
            return []  # 타입 일관성을 위해 빈 리스트 반환

    def buy_limit_order(self, market: str, price: float, volume: float) -> Dict:
        """지정가 매수

        Args:
            market: 마켓 코드 (예: 'KRW-BTC')
            price: 주문 가격
            volume: 주문 수량
        """
        # 호가 단위에 맞게 가격 조정 (매수: 내림)
        adjusted_price = adjust_price_to_tick(price, is_buy=True)

        # 가격에 따라 정수 또는 소수점 처리
        if adjusted_price >= 1:
            price_str = str(int(adjusted_price))
        else:
            price_str = str(adjusted_price)

        query = {
            'market': market,
            'side': 'bid',
            'ord_type': 'limit',
            'price': price_str,
            'volume': str(volume)
        }

        url = f"{self.server_url}/v1/orders"
        headers = self._get_headers(query)

        try:
            response = requests.post(url, json=query, headers=headers, timeout=10)
            result = response.json()

            if response.status_code != 201:
                print(f"[UPBIT] 지정가 매수 실패 - 상태코드: {response.status_code}, 응답: {result}")

            return result
        except requests.exceptions.Timeout:
            return {'error': {'message': '요청 타임아웃'}}
        except requests.exceptions.RequestException as e:
            return {'error': {'message': f'요청 실패: {str(e)}'}}

    def buy_market_order(self, market: str, price: float) -> Dict:
        """시장가 매수

        Args:
            market: 마켓 코드
            price: 매수 총액 (KRW)
        """
        query = {
            'market': market,
            'side': 'bid',
            'ord_type': 'price',
            'price': str(int(price))  # 정수로 변환
        }

        url = f"{self.server_url}/v1/orders"
        headers = self._get_headers(query)

        try:
            response = requests.post(url, json=query, headers=headers, timeout=10)
            result = response.json()

            if response.status_code != 201:
                print(f"[UPBIT] 매수 주문 실패 - 상태코드: {response.status_code}, 응답: {result}")

            return result
        except requests.exceptions.Timeout:
            return {'error': {'message': '요청 타임아웃'}}
        except requests.exceptions.RequestException as e:
            return {'error': {'message': f'요청 실패: {str(e)}'}}

    def sell_limit_order(self, market: str, price: float, volume: float) -> Dict:
        """지정가 매도

        Args:
            market: 마켓 코드
            price: 주문 가격
            volume: 주문 수량
        """
        # 호가 단위에 맞게 가격 조정 (매도: 올림)
        adjusted_price = adjust_price_to_tick(price, is_buy=False)

        # 가격에 따라 정수 또는 소수점 처리
        if adjusted_price >= 1:
            price_str = str(int(adjusted_price))
        else:
            price_str = str(adjusted_price)

        query = {
            'market': market,
            'side': 'ask',
            'ord_type': 'limit',
            'price': price_str,
            'volume': str(volume)
        }

        url = f"{self.server_url}/v1/orders"
        headers = self._get_headers(query)

        try:
            response = requests.post(url, json=query, headers=headers, timeout=10)
            result = response.json()

            if response.status_code != 201:
                print(f"[UPBIT] 지정가 매도 실패 - 상태코드: {response.status_code}, 응답: {result}")

            return result
        except requests.exceptions.Timeout:
            return {'error': {'message': '요청 타임아웃'}}
        except requests.exceptions.RequestException as e:
            return {'error': {'message': f'요청 실패: {str(e)}'}}

    def sell_market_order(self, market: str, volume: float) -> Dict:
        """시장가 매도

        Args:
            market: 마켓 코드
            volume: 매도 수량
        """
        query = {
            'market': market,
            'side': 'ask',
            'ord_type': 'market',
            'volume': str(volume)
        }

        url = f"{self.server_url}/v1/orders"
        headers = self._get_headers(query)

        try:
            response = requests.post(url, json=query, headers=headers, timeout=10)
            result = response.json()

            if response.status_code != 201:
                print(f"[UPBIT] 매도 주문 실패 - 상태코드: {response.status_code}, 응답: {result}")

            return result
        except requests.exceptions.Timeout:
            return {'error': {'message': '요청 타임아웃'}}
        except requests.exceptions.RequestException as e:
            return {'error': {'message': f'요청 실패: {str(e)}'}}

    def cancel_order(self, uuid: str) -> Dict:
        """주문 취소

        Args:
            uuid: 주문 UUID
        """
        query = {'uuid': uuid}

        url = f"{self.server_url}/v1/order"
        headers = self._get_headers(query)
        try:
            response = requests.delete(url, params=query, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            return {'error': {'message': f'주문 취소 실패: {str(e)}'}}

    def get_order(self, uuid: str) -> Dict:
        """개별 주문 조회

        Args:
            uuid: 주문 UUID

        Returns:
            주문 정보 (state: wait/watch/done/cancel)
        """
        query = {'uuid': uuid}

        url = f"{self.server_url}/v1/order"
        headers = self._get_headers(query)
        try:
            response = requests.get(url, params=query, headers=headers, timeout=10)
            return response.json()
        except Exception as e:
            return {'error': {'message': f'주문 조회 실패: {str(e)}'}}

    def get_balance(self, currency: str = 'KRW') -> float:
        """특정 화폐 잔고 조회

        Args:
            currency: 화폐 코드 (KRW, BTC, ETH 등)

        Returns:
            잔고 (float)
        """
        try:
            accounts = self.get_accounts()

            # API 에러 응답 체크
            if isinstance(accounts, dict) and 'error' in accounts:
                print(f"[UPBIT] 잔고 조회 실패: {accounts.get('error', {}).get('message', 'Unknown error')}")
                return 0.0

            if not isinstance(accounts, list):
                print(f"[UPBIT] 잔고 조회 실패: 예상치 못한 응답 형식")
                return 0.0

            for account in accounts:
                if account.get('currency') == currency:
                    return float(account.get('balance', 0))
            return 0.0
        except Exception as e:
            print(f"[UPBIT] 잔고 조회 예외: {e}")
            return 0.0
