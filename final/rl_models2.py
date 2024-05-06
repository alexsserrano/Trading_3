import numpy as np
import tensorflow as tf
from collections import deque
import gymnasium as gym
from gym import spaces
import pandas as pd
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

class TradingEnv(gym.Env):
    """Un entorno de trading personalizado que simula la operación en mercados financieros."""
    def __init__(self, df, lookback_window=200):
        super(TradingEnv, self).__init__()

        self.df = df
        self.lookback_window = lookback_window
        self.features = ['Close', 'RSI', 'W%R'] * lookback_window
        self.max_steps = len(df) - lookback_window
        self.current_step = 0

        # Espacios de acción y observación
        self.action_space = spaces.Discrete(3)  # 0: mantener, 1: comprar, 2: vender
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features),), dtype=np.float32)

        self.position = 0  # 1 para comprado, 0 para no comprado, -1 para vendido
        self.current_price = 0
        self.cash = 10000  # Balance inicial de efectivo
        self.total_profit = 0
        self.scaler = StandardScaler()

    def _next_observation(self):
        # Agrega RSI y Williams %R como indicadores técnicos para las últimas `lookback_window` observaciones
        frame = np.concatenate([self.df.loc[self.current_step:self.current_step+self.lookback_window, 'Close'].values,
                                self.df.loc[self.current_step:self.current_step+self.lookback_window, 'RSI'].values,
                                self.df.loc[self.current_step:self.current_step+self.lookback_window, 'W%R'].values])
        return self.scaler.transform([frame])[0]

    def step(self, action):
        self.current_price = self.df.loc[self.current_step, 'Close']
        reward = 0

        if action == 1:  # Comprar
            if self.position == 0:  # No comprado aún
                self.position = 1
                self.entry_price = self.current_price
        elif action == 2:  # Vender
            if self.position == 1:  # Solo vender si previamente comprado
                self.position = 0
                profit = self.current_price - self.entry_price
                self.total_profit += profit
                reward = profit

        self.current_step += 1

        # Se termina el episodio si se alcanza el máximo de pasos
        done = self.current_step >= self.max_steps

        # La recompensa puede ser más compleja dependiendo del resultado
        reward = 3 if reward > 0 else -1 if reward < 0 else 0

        return self._next_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.total_profit = 0
        self.position = 0
        self.entry_price = 0
        return self._next_observation()

    def render(self, mode='human'):
        profit = self.current_price - self.entry_price if self.position == 1 else 0
        print(f"Step: {self.current_step}, Price: {self.current_price}, Position: {self.position}, Profit: {profit}, Total Profit: {self.total_profit}")

class DQAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space.n  # Asegúrate de que esto se define antes de construir el modelo
        self.q_network = self.build_model()
        self.q_target_network = self.build_model()
        self.q_target_network.set_weights(self.q_network.get_weights())
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.batch_size = 32
        self.optimizer = Adam(learning_rate=0.001)
        self.replay_buffer = deque(maxlen=100000)

    def build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.q_network.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.q_target_network.predict(next_state)[0]))
            target_f = self.q_network.predict(state)
            target_f[0][action] = target
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, episodes=1000):
        for e in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            for time in range(2000):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                self.replay_buffer.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, episodes, time, self.epsilon))
                    break
            self.replay()
            if e % 10 == 0:
                self.q_target_network.set_weights(self.q_network.get_weights())


def calculate_multiple_rsi(data, windows):
    """ Calcula múltiples RSI para diferentes ventanas de tiempo """
    rsis = pd.DataFrame(index=data.index)
    for window in windows:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsis[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    return rsis

def calculate_multiple_williams_r(data, periods):
    """ Calcula múltiples Williams %R para diferentes periodos """
    williams_rs = pd.DataFrame(index=data.index)
    for period in periods:
        high = data['High'].rolling(window=period).max()
        low = data['Low'].rolling(window=period).min()
        williams_rs[f'W%R_{period}'] = -100 * ((high - data['Close']) / (high - low))
    return williams_rs




# Código para ejecutar el entorno y entrenar el agente
if __name__ == "__main__":
    data = pd.read_csv('aapl_5m_train.csv')
    # Asegúrate de tener las columnas 'Close', 'RSI' y 'W%R' correctamente calculadas en tu dataframe
    env = TradingEnv(data)
    agent = DQAgent(env)
    agent.train(episodes=50)
