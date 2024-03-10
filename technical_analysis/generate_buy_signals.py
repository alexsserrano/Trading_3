import pandas as pd
from ml_models import prepare_data_and_train_models

def generate_buy_signals(data, features_columns, target_column):
    """
    Utiliza las predicciones de los modelos entrenados para generar señales de compra.

    Parameters:
    - data (pd.DataFrame): DataFrame que contiene los datos del mercado.
    - features_columns (list): Lista de columnas utilizadas como características para el entrenamiento.
    - target_column (str): La columna objetivo usada para el entrenamiento.

    Returns:
    - pd.Series: Serie con señales de compra (1 para compra, 0 para no compra).
    """

    # Obtener las predicciones de los modelos entrenados
    svc_predictions, xgb_predictions, lr_predictions = prepare_data_and_train_models(data, features_columns, target_column)

    # Generar señales de compra basadas en la mayoría de votos de los modelos
    # Una señal de compra se activa si al menos dos de los tres modelos sugieren comprar
    buy_signals = (svc_predictions + xgb_predictions + lr_predictions) >= 2

    return buy_signals.astype(int)

# Ejemplo de uso:
# Asumiendo que 'data' es tu DataFrame que incluye tanto las características como el objetivo
# features_columns = ['feature1', 'feature2', ...] # Tus columnas de características
# target_column = 'target' # Tu columna objetivo
# buy_signals = generate_buy_signals(data, features_columns, target_column)
