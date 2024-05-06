# rl_models5.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import gymnasium as gym
from gym import spaces
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input
import math
import matplotlib.pyplot as plt

# Importar las funciones del archivo indicators.py
from indicators import calculate_multiple_rsi, calculate_multiple_williams_r


class TradingEnv(gym.Env):
    def __init__(self, df, lookback_window=100, initial_cash=10000, commission_per_trade=0.001):
        super(TradingEnv, self).__init__()
        self.df = df.dropna()
        # Asegurarse de que solo las columnas numéricas están incluidas
        self.df = self.df.select_dtypes(include=[np.number])  # Esto excluye columnas no numéricas
        self.lookback_window = lookback_window
        self.features = self.df.shape[1]
        self.max_steps = len(self.df) - lookback_window
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # Acciones: mantener, comprar, vender
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features,), dtype=np.float32)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission = commission_per_trade
        self.shares_held = 0
        self.current_price = 0
        self.entry_price = 0
        self.total_profit = 0
        self.stop_loss = 0.01
        self.take_profit = 0.01

    def _next_observation(self):
        # Simplemente devolvemos los valores numéricos como un array de flotantes
        return self.df.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        self.current_price = self.df.iloc[self.current_step, :].Close
        reward = 0
        done = False

        if action == 1 and self.position == 0:  # Comprar
            shares_to_buy = self.cash // self.current_price
            potential_cost = shares_to_buy * self.current_price * (1 + self.commission)
            if self.cash >= potential_cost:
                self.shares_held += shares_to_buy
                self.cash -= potential_cost
                self.entry_price = self.current_price
                self.position = 1

        elif action == 2 and self.position == 1:  # Vender
            if self.shares_held > 0:
                self.cash += self.shares_held * self.current_price * (1 - self.commission)
                profit = (self.current_price - self.entry_price) * self.shares_held
                self.total_profit += profit
                reward = profit
                self.shares_held = 0
                self.position = 0

        # Check for stop loss or take profit
        if self.position == 1:
            loss = (self.entry_price - self.current_price) / self.entry_price
            gain = (self.current_price - self.entry_price) / self.entry_price
            if loss >= self.stop_loss or gain >= self.take_profit:
                self.cash += self.shares_held * self.current_price * (1 - self.commission)
                profit = (self.current_price - self.entry_price) * self.shares_held
                self.total_profit += profit
                reward = profit if profit > 0 else -abs(profit)
                self.shares_held = 0
                self.position = 0
                done = True  # Optionally end the episode when a target is hit

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._next_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        self.total_profit = 0
        self.position = 0
        return self._next_observation()

    def render(self):
        print(f'Step: {self.current_step}, Price: {self.current_price}, Position: {self.position}, Profit: {self.total_profit}')

class DQAgent:
    def __init__(self, env, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.01, batch_size=32, episodes=100):
        self.env = env
        self.action_space = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.episodes = episodes
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_buffer = deque(maxlen=10000)
        # Calcular decay rate de epsilon basado en el número de episodios
        self.epsilon_decay = math.pow((self.epsilon_min / self.epsilon), (1 / self.episodes))

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size, 1)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='sparse-categorical-crossentropy', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=100):
        total_rewards_per_episode = []  # Lista para almacenar las recompensas totales por episodio
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0  # Reiniciar la recompensa total al inicio de cada episodio

            for time in range(500):  # o el número máximo de pasos por episodio que desees
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state

                if done:
                    break
            total_rewards_per_episode.append(total_reward)
            print(f"Episode: {e + 1}/{episodes}, Total reward: {total_reward}, Epsilon: {self.epsilon:.2f}")

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if e % 10 == 0:
                self.target_model.set_weights(self.model.get_weights())
        return total_rewards_per_episode


if __name__ == "__main__":
    data = pd.read_csv('aapl_5m_train.csv')
    # Ejemplo de diferentes ventanas para RSI y períodos para Williams %R
    windows = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
    periods = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
    data = calculate_multiple_rsi(data, windows)
    data = calculate_multiple_williams_r(data, periods)
    env = TradingEnv(data)
    agent = DQAgent(env)
    print("Agent Created")
    agent.train(episodes=50)
    print("Agent Train finished")



rewards = agent.train(episodes=50)
plt.plot(rewards)
plt.title('Recompensa Acumulada por Episodio')
plt.xlabel('Episodio')
plt.ylabel('Recompensa Acumulada')
plt.show()
