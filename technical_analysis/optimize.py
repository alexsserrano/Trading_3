# optimize.py
import optuna
import pandas as pd
from technical_analysis.backtest import backtest
from technical_analysis.generate_buy_signals import generate_buy_signals
from technical_analysis.generate_sell_signals import generate_sell_signals


def optimize(data: pd.DataFrame, n_trials: int = 100):
    """
    Optimiza los parámetros de las estrategias de trading utilizando Optuna.

    Parameters:
    - data (pd.DataFrame): DataFrame que contiene los datos de precios.
    - n_trials (int): Número de ensayos de optimización a realizar.

    Returns:
    - El estudio de Optuna después de completar la optimización.
    """

    def objective(trial):
        # Define las estrategias base y sus posibles parámetros para optimización
        strategies = [{
            'id': 1,
            'indicators': ['SMA', 'RSI', 'Bollinger Bands', 'Volume Oscillator'],
            'params': {}
        }]  # Ajusta según sea necesario

        # Modificar estrategias con parámetros sugeridos
        for strategy in strategies:
            if 'SMA' in strategy['indicators']:
                strategy['params']['short_window'] = trial.suggest_int('short_window', 5, 50)
                strategy['params']['long_window'] = trial.suggest_int('long_window', 51, 200)
            if 'RSI' in strategy['indicators']:
                strategy['params']['rsi_period'] = trial.suggest_int('rsi_period', 10, 30)
            if 'Bollinger Bands' in strategy['indicators']:
                strategy['params']['bb_window'] = trial.suggest_int('bb_window', 20, 50)
                strategy['params']['bb_std'] = trial.suggest_int('bb_std', 1, 3)
            if 'Volume Oscillator' in strategy['indicators']:
                strategy['params']['short_vol_window'] = trial.suggest_int('short_vol_window', 5, 20)
                strategy['params']['long_vol_window'] = trial.suggest_int('long_vol_window', 21, 50)

        # Generar señales de compra y venta usando las estrategias optimizadas
        buy_signals = generate_buy_signals(data, strategies)
        sell_signals = generate_sell_signals(data, strategies)

        # Realizar backtesting y obtener el rendimiento de la estrategia
        backtest_results = backtest(data, buy_signals, sell_signals, initial_cash=100000, commission_per_trade=0.00125)
        total_return = backtest_results['total_return']

        return total_return  # Objetivo a maximizar

    # Crear y ejecutar el estudio de optimización
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    return study

# Nota: Antes de ejecutar optimize, asegúrate de que todas las funciones necesarias
# (generate_buy_signals, generate_sell_signals, backtest, y cualquier otra función relevante)
# estén definidas correctamente y de que 'data' esté cargado con tus datos de precios.
# def optimize_strategy():
# return None
