from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def prepare_data_and_train_models(data, features_columns, target_column):
    """
    Entrena los modelos SVC, XGBoost, y Logistic Regression con el dataset proporcionado
    y devuelve las predicciones de cada modelo.

    Parameters:
    - data (pd.DataFrame): El DataFrame con los datos.
    - features_columns (list of str): Lista de columnas usadas como características.
    - target_column (str): La columna objetivo.

    Returns:
    Tuple de Series con predicciones de SVC, XGBoost, y Logistic Regression.
    """
    # Dividir el dataset en características y objetivo
    X = data[features_columns]
    y = data[target_column]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar SVC
    svc = SVC(C=1.0, kernel='rbf', gamma='scale')
    svc.fit(X_train, y_train)
    svc_predictions = svc.predict(X_test)

    # Entrenar XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, use_label_encoder=False,
                                  eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    # Entrenar Logistic Regression
    lr = LogisticRegression(C=1.0, solver='liblinear')
    lr.fit(X_train, y_train)
    lr_predictions = lr.predict(X_test)

    # Convertir a Series para facilitar operaciones posteriores
    svc_predictions_series = pd.Series(svc_predictions, index=X_test.index)
    xgb_predictions_series = pd.Series(xgb_predictions, index=X_test.index)
    lr_predictions_series = pd.Series(lr_predictions, index=X_test.index)

    return svc_predictions_series, xgb_predictions_series, lr_predictions_series
