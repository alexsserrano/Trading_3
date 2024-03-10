# Suponiendo que ml_models.py contiene una función que devuelve las predicciones
# de los tres modelos para un nuevo conjunto de datos.
from ml_models import prepare_data_and_train_models

def generate_sell_signals(data, features_columns):
    """
    Genera señales de venta basadas en las predicciones de los modelos SVC, XGBoost y Logistic Regression.
    Una señal de venta se genera si al menos dos de los tres modelos predicen vender.

    Parameters:
    - data (pd.DataFrame): El DataFrame con los datos nuevos para los cuales generar las señales.
    - features_columns (list of str): Lista de columnas usadas como características.

    Returns:
    - pd.Series: Serie con señales de venta, donde 1 indica una señal de venta y 0 indica no vender.
    """
    # Obtiene las predicciones de los modelos para el conjunto de datos actual
    svc_predictions, xgb_predictions, lr_predictions = prepare_data_and_train_models(data, features_columns, 'target_column_name')

    # Calcula las señales de venta basadas en la mayoría de votos de los modelos
    # Nota: Asegúrate de que 'target_column_name' sea la columna en tu DataFrame que quieres predecir
    sell_signals = (svc_predictions + xgb_predictions + lr_predictions) >= 2

    return sell_signals.astype(int)

