"""
강화학습 트레이딩 에이전트
PPO(Proximal Policy Optimization) 알고리즘 사용
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
import torch


class TradingAgent:
    """PPO 기반 트레이딩 에이전트"""

    def __init__(self,
                 env,
                 model_name: str = "crypto_trader",
                 learning_rate: float = 0.0003,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 device: str = 'auto',
                 use_tensorboard: bool = False):
        """
        Args:
            env: Gymnasium 환경
            model_name: 모델 이름
            learning_rate: 학습률
            n_steps: 업데이트 전 수집할 스텝 수
            batch_size: 배치 크기
            n_epochs: 에포크 수
            gamma: 할인 계수
            device: 'cpu', 'cuda', 'auto'
            use_tensorboard: TensorBoard 로깅 사용 여부
        """
        self.env = DummyVecEnv([lambda: env])
        self.model_name = model_name

        # TensorBoard 로깅 설정 (선택적)
        tensorboard_log = "./logs/tensorboard/" if use_tensorboard else None

        # PPO 모델 생성
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            verbose=1,
            device=device,
            tensorboard_log=tensorboard_log
        )

    def train(self, total_timesteps: int = 100000, callback: Optional[BaseCallback] = None):
        """모델 학습

        Args:
            total_timesteps: 총 학습 스텝
            callback: 콜백 함수
        """
        print(f"🎯 학습 시작: {total_timesteps:,} 스텝")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        print("✅ 학습 완료!")

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """액션 예측

        Args:
            observation: 현재 관측값
            deterministic: 결정적 예측 여부

        Returns:
            action, _states
        """
        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: Optional[str] = None):
        """모델 저장

        Args:
            path: 저장 경로 (None이면 기본 경로)
        """
        if path is None:
            os.makedirs("models", exist_ok=True)
            path = f"models/{self.model_name}"

        self.model.save(path)
        print(f"💾 모델 저장: {path}")

    def load(self, path: Optional[str] = None):
        """모델 로드

        Args:
            path: 로드 경로 (None이면 기본 경로)
        """
        if path is None:
            path = f"models/{self.model_name}"

        self.model = PPO.load(path, env=self.env)
        print(f"📂 모델 로드: {path}")

    @staticmethod
    def get_device() -> str:
        """사용 가능한 디바이스 반환"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"


class TrainingCallback(BaseCallback):
    """학습 중 성능 모니터링 콜백"""

    def __init__(self, check_freq: int = 1000, save_path: str = "logs/"):
        """
        Args:
            check_freq: 체크 빈도
            save_path: 로그 저장 경로
        """
        super().__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        """각 스텝마다 호출"""
        if self.n_calls % self.check_freq == 0:
            # 에피소드 보상 기록
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])

                self.episode_rewards.append(mean_reward)
                self.episode_lengths.append(mean_length)

                print(f"\n📊 Step {self.n_calls:,}")
                print(f"   평균 보상: {mean_reward:.4f}")
                print(f"   평균 길이: {mean_length:.0f}")

        return True


def evaluate_agent(agent: TradingAgent, env, n_episodes: int = 10) -> dict:
    """에이전트 성능 평가

    Args:
        agent: 트레이딩 에이전트
        env: 평가 환경
        n_episodes: 평가 에피소드 수

    Returns:
        평가 결과 딕셔너리
    """
    episode_rewards = []
    episode_profits = []
    episode_trades = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)
        episode_profits.append(info['total_profit'])
        episode_trades.append(info['total_trades'])

    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_profit': np.mean(episode_profits),
        'std_profit': np.std(episode_profits),
        'mean_trades': np.mean(episode_trades),
        'win_rate': sum(1 for p in episode_profits if p > 0) / n_episodes
    }

    print("\n📈 평가 결과:")
    print(f"   평균 보상: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"   평균 수익: {results['mean_profit']:,.0f} ± {results['std_profit']:,.0f} KRW")
    print(f"   평균 거래 횟수: {results['mean_trades']:.0f}")
    print(f"   승률: {results['win_rate']*100:.1f}%")

    return results
