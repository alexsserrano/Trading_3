# rl_models4.py

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


# Importar las funciones del archivo indicators.py
from indicators import calculate_multiple_rsi, calculate_multiple_williams_r


class TradingEnv(gym.Env):
    def __init__(self, df, lookback_window=100):
        # Asegúrate de que el DataFrame ya tenga los indicadores calculados
        super(TradingEnv, self).__init__()
        self.df = df.dropna()
        self.df2 = self.df.copy().drop(["Unnamed: 0", "Timestamp", "Gmtoffset", "Datetime", "Volume"], axis=1)
        print(self.df2)
        print(self.df2.iloc[0].values)
        print(self.df2.iloc[0].values.shape)
        self.lookback_window = lookback_window
        # Actualiza para incluir todas las columnas de indicadores relevantes
        self.features = len(self.df2.columns)
        self.max_steps = len(df) - lookback_window
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # Acciones: mantener, comprar, vender
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features, 1), dtype=np.float32)
        self.position = 0
        self.current_price = 0
        self.entry_price = 0
        self.total_profit = 0

    def _next_observation(self):
        # start = self.current_step
        # end = self.current_step + self.lookback_window
        # frames = []
        # for feature_name in self.features:
        #     frames.append(self.df.loc[start:end, feature_name].values.flatten())
        # frame = np.concatenate(frames)
        # scaled_frame = self.scaler.fit_transform(frame.reshape(-1, 1)).flatten()
        # return scaled_frame
        return self.df2.iloc[self.current_step].values.astype("float32")

    def step(self, action):
        self.current_price = self.df.iloc[self.current_step, :].Close
        reward = 0

        if action == 1 and self.position == 0:  # Comprar
            self.position = 1
            self.entry_price = self.current_price
        elif action == 2 and self.position == 1:  # Vender
            self.position = 0
            profit = self.current_price - self.entry_price * (1 - .001)
            self.total_profit += profit
            reward = profit

        self.current_step += 1

        done = self.current_step >= self.max_steps
        reward = 1 if reward > 0 else -1 if reward < 0 else 0

        return self._next_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.total_profit = 0
        self.entry_price = 0
        return self._next_observation()

    def render(self):
        print(f'Step: {self.current_step}, Price: {self.current_price}, Position: {self.position}, Profit: {self.total_profit}')

class DQAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.epsilon = 1.0  # Exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        self.replay_buffer = deque(maxlen=10000)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,1)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_space, activation='softmax'))
        model.compile(loss='sparse-categorical-crossentropy', optimizer=Adam(learning_rate=0.01))
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
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.model.predict(state)
            print(target_f)
            target_f[0][action] = target
            states.append(state[0])
            targets_f.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets_f), batch_size=self.batch_size, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, episodes=50):
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [self.state_size, 1])
            for time in range(2000):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [self.state_size, 1])
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                self.replay()
                if done:
                    print(f"Episode: {e+1}/{episodes}, Total Profit: {self.env.total_profit}")
                    break
            self.update_target_model()

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
