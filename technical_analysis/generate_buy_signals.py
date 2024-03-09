# generate_buy_signals.py
import pandas as pd
import numpy as np
import ta  # Importamos la biblioteca para análisis técnico


def generate_buy_signals(data: pd.DataFrame, strategies: list) -> pd.DataFrame:
    """
    Genera señales de compra para cada estrategia en los datos proporcionados, basándose en SMA, RSI, Bandas de Bollinger y Oscilador de Volumen.

    Parameters:
    data (pd.DataFrame): DataFrame que contiene los datos de precios y otros necesarios para calcular indicadores.
    strategies (list): Lista de estrategias, cada una con sus indicadores y parámetros específicos.

    Returns:
    pd.DataFrame: DataFrame con las señales de compra para cada estrategia.
    """
    signals = pd.DataFrame(index=data.index)

    for strategy in strategies:
        signal_name = 'Signal_' + str(strategy['id'])
        signals[signal_name] = np.ones(len(data))  # Inicializamos todas las señales a 1 (compra)

        if 'SMA' in strategy['indicators']:
            # Para SMA, una señal de compra podría ser cuando el SMA de corto plazo cruza por encima del de largo plazo
            window_short = strategy['params'].get('short_window', 5)
            window_long = strategy['params'].get('long_window', 21)
            short_ma = ta.trend.SMAIndicator(data['Close'], window=window_short).sma_indicator()
            long_ma = ta.trend.SMAIndicator(data['Close'], window=window_long).sma_indicator()
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (short_ma > long_ma)

        if 'RSI' in strategy['indicators']:
            # Ejemplo: Generar señal de compra basada en RSI
            rsi_period = strategy['params'].get('rsi_period', 14)
            rsi = ta.momentum.RSIIndicator(data['Close'], window=rsi_period).rsi()
            # Señal de compra cuando RSI < 30 (condición de sobreventa)
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (rsi < 30)

        if 'Bollinger Bands' in strategy['indicators']:
            # Ejemplo: Generar señal de compra basada en Bandas de Bollinger
            bb_window = strategy['params'].get('bb_window', 20)
            bb_std = strategy['params'].get('bb_std', 2)
            bb = ta.volatility.BollingerBands(data['Close'], window=bb_window, window_dev=bb_std)
            # Señal de compra cuando el precio cierra por debajo de la banda inferior (potencial rebote)
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (data['Close'] < bb.bollinger_lband())

        if 'Volume Oscillator' in strategy['indicators']:
            # Ejemplo: Generar señal de compra basada en Oscilador de Volumen
            short_vol_window = strategy['params'].get('short_vol_window', 5)
            long_vol_window = strategy['params'].get('long_vol_window', 10)
            short_vol_ma = data['Volume'].rolling(window=short_vol_window).mean()
            long_vol_ma = data['Volume'].rolling(window=long_vol_window).mean()
            vol_osc = short_vol_ma - long_vol_ma
            # Señal de compra cuando el oscilador de volumen es positivo (indicando aumento del volumen)
            signals[signal_name] = signals[signal_name].astype(bool)
            signals[signal_name] &= (vol_osc > 0)

    # Convertir las señales a binario (1 para compra, 0 para no compra)
    # signals = signals.applymap(lambda x: 1 if x else 0)
    for col in signals.columns:
        signals[col] = signals[col].astype(int)

    return signals
