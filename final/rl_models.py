# rl_models.py
import gymnasium as gym
from gym import spaces
import pandas as pd
import numpy as np


class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.max_steps = len(df)
        self.current_step = 0

        # Definir el espacio de acción (0: mantener, 1: comprar, 2: vender)
        self.action_space = spaces.Discrete(3)

        # Definir el espacio de observación (precios e indicadores técnicos)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(3,), dtype=np.float32)

        self.positions = []
        self.total_profit = 0
        self.current_price = 0

    def _next_observation(self):
        # Obtener los datos de precios e indicadores técnicos
        frame = np.array([
            self.df.loc[self.current_step, 'Close'],
            self.df.loc[self.current_step, 'RSI'],
            self.df.loc[self.current_step, 'W%R']
        ])
        return frame

    def step(self, action):
        self.current_price = self.df.loc[self.current_step, 'Close']
        self.current_step += 1

        previous_total = self.total_profit
        if action == 1:  # Comprar
            self.positions.append(self.current_price)
        elif action == 2 and self.positions:  # Vender
            buy_price = self.positions.pop(0)
            profit = self.current_price - buy_price
            self.total_profit += profit

        # Actualizar el estado
        done = self.current_step >= self.max_steps
        reward = self.total_profit - previous_total
        reward = 3 if reward > 0 else -1 if reward < 0 else 0

        return self._next_observation(), reward, done, {}

    def reset(self):
        self.current_step = 0
        self.total_profit = 0
        self.positions = []
        return self._next_observation()

    def render(self):
        # Puede expandirse para incluir visualizaciones más detalladas de la actividad de trading
        print(f'Step: {self.current_step}, Price: {self.current_price}, Total Profit: {self.total_profit}')


class DQAgent:
    def __init__(self, env, max_iters=10, max_steps=1000,
                 gamma=0.9, epsilon=1, epsilon_min=0.1, epsilon_max=1,
                 batch_size=32, learning_rate=0.00025, history_len=100_000):
        self.env = env
        self.action_space = env.action_space.n

        self.max_iters = max_iters
        self.max_steps = max_steps

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_range = (epsilon_max - epsilon_min)
        self.batch_size = batch_size

        self.q_network = self.init_q_network()
        self.q_target_network = self.init_q_network()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
        self.loss = tf.keras.losses.Huber()
        self.replay_buffer = deque(maxlen=history_len)
        self.epoch_reward_history = []

    def init_q_network(self):
        return tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_dim=self.env.observation_space.shape[0]),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_space, activation='linear')
        ])

    # Métodos 'train', 'load_model', y 'run_single_game' deberán ser ajustados de forma similar.


def train(self):
    running_reward = 0
    episode_count = 0
    frame_count = 0

    while episode_count < self.max_iters:
        state = self.env.reset()
        episode_reward = 0

        for timestep in range(self.max_steps):
            frame_count += 1

            # Epsilon-greedy para exploración
            if np.random.rand() < self.epsilon:
                action = self.env.action_space.sample()  # Acción aleatoria
            else:
                action_probs = self.q_network.predict(np.array([state]))
                action = np.argmax(action_probs[0])  # Mejor acción

            # Decremento de epsilon
            self.epsilon -= self.epsilon_range / self.max_steps
            self.epsilon = max(self.epsilon, self.epsilon_min)

            # Ejecutar acción
            next_state, reward, done, _ = self.env.step(action)
            episode_reward += reward

            # Guardar en el buffer
            self.replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            # Entrenamiento
            if len(self.replay_buffer) > self.batch_size:
                self._replay()

            if done:
                break

        # Actualizar la red objetivo cada ciertos pasos
        if episode_count % 10 == 0:
            self.q_target_network.set_weights(self.q_network.get_weights())

        self.epoch_reward_history.append(episode_reward)
        running_reward = np.mean(self.epoch_reward_history[-100:])

        episode_count += 1
        print(
            f'Episode: {episode_count}, Total Reward: {episode_reward}, Running Reward: {running_reward:.2f}, Epsilon: {self.epsilon:.2f}')


def _replay(self):
    minibatch = random.sample(self.replay_buffer, self.batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = self.q_network.predict(np.array([state]))[0]
        if done:
            target[action] = reward
        else:
            t = self.q_target_network.predict(np.array([next_state]))[0]
            target[action] = reward + self.gamma * np.max(t)

        self.q_network.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)

def load_model(self, model_name):
    self.q_network = tf.keras.models.load_model(model_name)
    self.q_target_network.set_weights(self.q_network.get_weights())
    print("Model loaded successfully.")


def run_single_game(self):
    state = self.env.reset()
    done = False
    total_reward = 0

    while not done:
        action_probs = self.q_network.predict(np.array([state]))
        action = np.argmax(action_probs[0])
        state, reward, done, _ = self.env.step(action)
        total_reward += reward
        self.env.render()

    return total_reward
