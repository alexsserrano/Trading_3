# generate_sell_signals.py
import pandas as pd
from technical_analysis.ml_models import load_models_and_predict

def generate_sell_signals(data, features_columns, model_paths):
    """
    Genera señales de venta basadas en las predicciones de los modelos SVC, XGBoost, y Logistic Regression.
    Una señal de venta se genera si al menos dos de los tres modelos predicen vender.

    Parameters:
    - data (pd.DataFrame): El DataFrame con los nuevos datos para generar las señales.
    - features_columns (list of str): Lista de columnas usadas como características.
    - model_paths (dict): Diccionario con las rutas de los modelos entrenados a cargar.

    Returns:
    - pd.Series: Serie con señales de venta, donde 1 indica una señal de venta y 0 indica no vender.
    """
    # Cargar modelos y obtener predicciones para el nuevo conjunto de datos
    svc_predictions, xgb_predictions, lr_predictions = load_models_and_predict(data, features_columns, model_paths)

    # Calcula las señales de venta basadas en la mayoría de votos de los modelos
    sell_signals = (svc_predictions + xgb_predictions + lr_predictions) >= 2

    return sell_signals.astype(int)
