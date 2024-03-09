# generate_sell_signals.py
import pandas as pd
import numpy as np
import ta  # Importamos la biblioteca para análisis técnico


def generate_sell_signals(data: pd.DataFrame, strategies: list) -> pd.DataFrame:
    """
    Genera señales de venta para cada estrategia en los datos proporcionados, basándose en SMA, RSI, Bandas de Bollinger y Oscilador de Volumen.

    Parameters:
    data (pd.DataFrame): DataFrame que contiene los datos de precios y otros necesarios para calcular indicadores.
    strategies (list): Lista de estrategias, cada una con sus indicadores y parámetros específicos.

    Returns:
    pd.DataFrame: DataFrame con las señales de venta para cada estrategia.
    """
    signals = pd.DataFrame(index=data.index)

    for strategy in strategies:
        signal_name = 'Signal_' + str(strategy['id'])
        signals[signal_name] = np.ones(len(data))  # Inicializamos todas las señales a 1 (venta)

        if 'SMA' in strategy['indicators']:
            # Para SMA, una señal de venta podría ser cuando el SMA de corto plazo cruza por debajo del de largo plazo
            window_short = strategy['params'].get('short_window', 5)
            window_long = strategy['params'].get('long_window', 21)
            short_ma = ta.trend.SMAIndicator(data['Close'], window=window_short).sma_indicator()
            long_ma = ta.trend.SMAIndicator(data['Close'], window=window_long).sma_indicator()
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (short_ma < long_ma)

        if 'RSI' in strategy['indicators']:
            # Para RSI, una señal de venta podría ser cuando el RSI está sobrecomprado (RSI > 70)
            rsi_period = strategy['params'].get('rsi_period', 14)
            rsi = ta.momentum.RSIIndicator(data['Close'], window=rsi_period).rsi()
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (rsi > 70)

        if 'Bollinger Bands' in strategy['indicators']:
            # Para Bandas de Bollinger, una señal de venta podría ser cuando el precio cierra por encima de la banda superior
            bb_window = strategy['params'].get('bb_window', 20)
            bb_std = strategy['params'].get('bb_std', 2)
            bb = ta.volatility.BollingerBands(data['Close'], window=bb_window, window_dev=bb_std)
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (data['Close'] > bb.bollinger_hband())

        if 'Volume Oscillator' in strategy['indicators']:
            # Para el Oscilador de Volumen, una señal de venta podría ser cuando el oscilador de volumen es negativo (indicando disminución del volumen)
            short_vol_window = strategy['params'].get('short_vol_window', 5)
            long_vol_window = strategy['params'].get('long_vol_window', 10)
            short_vol_ma = data['Volume'].rolling(window=short_vol_window).mean()
            long_vol_ma = data['Volume'].rolling(window=long_vol_window).mean()
            vol_osc = short_vol_ma - long_vol_ma
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (vol_osc < 0)

    # Convertir las señales a binario (1 para venta, 0 para no venta)
    # signals = signals.applymap(lambda x: 1 if x else 0)
    for col in signals.columns:
        signals[col] = signals[col].astype(int)

    return signals
