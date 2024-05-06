# indicators.py

import pandas as pd
import numpy as np

def calculate_multiple_rsi(data, windows):
    """ Calcula múltiples RSI para diferentes ventanas de tiempo """
    for window in windows:
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
    return data

def calculate_multiple_williams_r(data, periods):
    """ Calcula múltiples Williams %R para diferentes periodos """
    for period in periods:
        high = data['High'].rolling(window=period).max()
        low = data['Low'].rolling(window=period).min()
        data[f'W%R_{period}'] = -100 * ((high - data['Close']) / (high - low))
    return data


data = pd.read_csv('aapl_5m_train.csv')
windows = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]  # Ejemplo de diferentes ventanas para RSI
periods = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]  # Ejemplo de diferentes períodos para Williams %R
data = calculate_multiple_rsi(data, windows)
data = calculate_multiple_williams_r(data, periods)

print(data)