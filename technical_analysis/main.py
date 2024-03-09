# main.py
import pandas as pd
from matplotlib import pyplot as plt

from optimize import optimize
from get_strategies import get_strategies
from generate_buy_signals import generate_buy_signals
from generate_sell_signals import generate_sell_signals
from technical_analysis.backtest import backtest
from indicators import calculate_rsi, calculate_sma, calculate_bollinger, calculate_volume_oscillator
from set_params import set_params

# Cargar los conjuntos de datos
datasets = {
    "5m": pd.read_csv("data/aapl_5m_train.csv"),
    "1m": pd.read_csv("data/aapl_1m_train.csv"),
    "1d": pd.read_csv("data/aapl_1d_train.csv"),
    "1h": pd.read_csv("data/aapl_1h_train.csv"),
}

# Obtener estrategias
strategies = get_strategies()

# Optimizar estrategias para cada conjunto de datos
optimized_strategies = {timeframe: optimize(data, strategies) for timeframe, data in datasets.items()}

# Definir los resultados finales
final_results = []

# Iterar sobre cada conjunto de datos
for timeframe, data in datasets.items():
    print(f"Procesando datos de {timeframe}...")

    # Obtener la estrategia óptima para el timeframe actual
    optimal_strategy = optimized_strategies[timeframe]

    # Aplicar los indicadores técnicos necesarios utilizando la estrategia óptima
    data['RSI'] = calculate_rsi(data, window=optimal_strategy['params']['RSI']['window'])
    data['SMA_short'], data['SMA_long'] = calculate_sma(data['close'], optimal_strategy['params']['SMA']['short_window']), calculate_sma(data['close'], optimal_strategy['params']['SMA']['long_window'])
    data['BB_mavg'], data['BB_hband'], data['BB_lband'] = calculate_bollinger(data, optimal_strategy['params']['BB']['window'], optimal_strategy['params']['BB']['std_dev'])
    data['Volume_Osc'] = calculate_volume_oscillator(data, optimal_strategy['params']['Volume_Osc']['short_window'], optimal_strategy['params']['Volume_Osc']['long_window'])

    # Generar señales de compra y venta usando la estrategia óptima
    buy_signals = generate_buy_signals(data, optimal_strategy)
    sell_signals = generate_sell_signals(data, optimal_strategy)

    # Realizar backtesting con la estrategia óptima
    results = backtest(data, buy_signals, sell_signals, 1000000, 0.001, 100, 0.01, 0.01)

    # Guardar los resultados
    final_results.append({"timeframe": timeframe, "results": results})

# Imprimir un resumen de los resultados
for result in final_results:
    print(f"Timeframe: {result['timeframe']}, Total Return: {result['results']['total_return']}")

# Asumiendo que final_results es una lista de diccionarios con los resultados del backtesting,
# incluyendo un DataFrame para cada timeframe que contiene una columna 'Portfolio_Value'.

for result in final_results:
    timeframe = result['timeframe']
    data = datasets[timeframe]  # Obtiene el DataFrame original para este timeframe
    backtest_results = result['results']  # Supongamos que esto es un DataFrame que incluye 'Portfolio_Value'

    # Calcula el valor del portafolio para una estrategia de "comprar y mantener"
    initial_investment = data['close'].iloc[0] * 100  # Asumiendo 100 acciones como ejemplo
    buy_and_hold_value = data['close'] / data['close'].iloc[0] * initial_investment

    # Visualización
    plt.figure(figsize=(14, 7))
    plt.plot(backtest_results.index, backtest_results['Portfolio_Value'], label='Estrategia de Trading')
    plt.plot(data.index, buy_and_hold_value, label='Comprar y Mantener', alpha=0.7)
    plt.title(f'Rendimiento de la Estrategia de Trading vs. Comprar y Mantener - {timeframe}')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio')
    plt.legend()
    plt.show()
