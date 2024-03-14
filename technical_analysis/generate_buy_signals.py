# generate_buy_signals.py
import pandas as pd
from technical_analysis.ml_models import load_models_and_predict

def generate_buy_signals(data, features_columns, model_paths):
    """
    Genera señales de compra basadas en las predicciones de los modelos SVC, XGBoost, y Logistic Regression.
    Una señal de compra se genera si al menos dos de los tres modelos predicen comprar.

    Parameters:
    - data (pd.DataFrame): El DataFrame con los nuevos datos para generar las señales.
    - features_columns (list of str): Lista de columnas usadas como características.
    - model_paths (dict): Diccionario con las rutas de los modelos entrenados a cargar.

    Returns:
    - pd.Series: Serie con señales de compra, donde 1 indica una señal de compra y 0 indica no comprar.
    """
    # Cargar modelos y obtener predicciones para el nuevo conjunto de datos
    svc_predictions, xgb_predictions, lr_predictions = load_models_and_predict(data, features_columns, model_paths)

    # Calcula las señales de compra basadas en la mayoría de votos de los modelos
    buy_signals = (svc_predictions + xgb_predictions + lr_predictions) >= 2

    return buy_signals.astype(int)
