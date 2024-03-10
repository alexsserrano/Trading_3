# main.py
def create_target_column(data):
    """
    Crea una columna 'Target' en el DataFrame basada en una lógica de trading.

    Parámetros:
    - data (pd.DataFrame): DataFrame con los datos históricos e indicadores técnicos.

    Retorna:
    - pd.DataFrame: El mismo DataFrame con una nueva columna 'Target'.
    """
    # Ejemplo de lógica de trading: comprar si el precio de cierre supera el SMA de 20 , vender si está por debajo.
    data['SMA_20'] = data['Close'].rolling(window=20).mean()  # Calcular el SMA de 20
    data['Target'] = 0  # Inicializar la columna 'Target' con 0 (mantener)
    data.loc[data['Close'] > data['SMA_20'], 'Target'] = 1  # 1 para comprar
    data.loc[data['Close'] < data['SMA_20'], 'Target'] = -1  # -1 para vender

    return data


import matplotlib.pyplot as plt

from technical_analysis.backtest import backtest


def main():
    # Cargar los datos de entrenamiento y prueba
    datasets = {

        train_5m : pd.read_csv('data/aapl_5m_train.csv'),
        train_1m : pd.read_csv('data/aapl_1m_train.csv'),
        train_1d : pd.read_csv('data/aapl_1d_train.csv'),
        train_1h : pd.read_csv('data/aapl_1h_train.csv')
    }

    # Entrenar y guardar modelos para cada conjunto de datos de entrenamiento
    for key, data in datasets.items():
        if 'train' in key:  # Verifica si es un conjunto de entrenamiento
            model_paths = {
                'svc': f"models/svc_{key}.joblib",
                'xgb': f"models/xgb_{key}.joblib",
                'lr': f"models/lr_{key}.joblib"
            }
            train_and_save_models(data, features_columns, target_column, model_paths)

    # Cargar modelos y realizar predicciones para cada conjunto de datos de prueba
    predictions = {}
    for key, data in datasets.items():
        if 'test' in key:  # Verifica si es un conjunto de prueba
            model_paths = {
                'svc': f"models/svc_train_{key.split('_')[1]}.joblib",
                'xgb': f"models/xgb_train_{key.split('_')[1]}.joblib",
                'lr': f"models/lr_train_{key.split('_')[1]}.joblib"
            }
            predictions[key] = load_models_and_predict(data, features_columns, model_paths)



    # Generar señales de compra/venta con los modelos entrenados en los datos de prueba
    buy_signals, sell_signals = generate_buy_sell_signals(datasets["test"], features_columns,
                                                          [logistic_model, svm_model, xgboost_model])

    # Realizar backtesting con las señales generadas
    initial_capital = 10000  # Ajustar según sea necesario
    backtest_results = backtest(datasets["test"], buy_signals, sell_signals, initial_capital)

    # Optimizar la estrategia de trading (si es necesario)
    optimized_results = optimize_strategy(backtest_results)  # Ajustar esta función según tus necesidades

    # Visualizar los resultados del backtesting y comparar con comprar y mantener
    plt.figure(figsize=(14, 7))
    plt.plot(backtest_results.index, backtest_results['Portfolio_Value'], label='Estrategia de Trading')
    buy_and_hold_value = datasets["test"]['close'] / datasets["test"]['close'].iloc[0] * initial_capital
    plt.plot(datasets["test"].index, buy_and_hold_value, label='Comprar y Mantener', alpha=0.7)
    plt.title('Rendimiento de la Estrategia de Trading vs. Comprar y Mantener')
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Portafolio')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
