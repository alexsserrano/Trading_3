from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# finnn

def train_svc_model(X, y, C=1.0, kernel='rbf', gamma='scale', test_size=0.2, random_state=None):
    """
    Entrena un modelo SVC con los datos proporcionados y los parámetros específicos.

    Parameters:
    - X (pd.DataFrame): Variables independientes del dataset.
    - y (pd.Series): Variable dependiente del dataset.
    - C (float): Parámetro de regularización.
    - kernel (str): Tipo de kernel usado en el algoritmo.
    - gamma (str or float): Coeficiente del kernel.
    - test_size (float): Proporción del dataset a incluir en el conjunto de prueba.
    - random_state (int): Semilla para reproducir los mismos resultados.

    Returns:
    - model (SVC): El modelo SVC entrenado.
    - test_metrics (dict): Métricas de evaluación sobre el conjunto de prueba.
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Inicializar y entrenar el modelo SVC
    model = SVC(C=C, kernel=kernel, gamma=gamma)
    model.fit(X_train, y_train)

    # Hacer predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    test_metrics = {
        'accuracy_score': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    return model, test_metrics




def train_xgboost_model(X, y, n_estimators=100, max_depth=3, max_leaves=0, learning_rate=0.1, booster='gbtree',
                        gamma=0, reg_alpha=0, reg_lambda=1, test_size=0.2, random_state=None):
    """
    Entrena un modelo XGBoost con los datos proporcionados y los parámetros específicos.

    Parameters:
    - X (pd.DataFrame): Variables independientes del dataset.
    - y (pd.Series): Variable dependiente del dataset.
    - n_estimators (int): Número de árboles a construir.
    - max_depth (int): Profundidad máxima de los árboles.
    - max_leaves (int): Número máximo de hojas en los árboles.
    - learning_rate (float): Tasa de aprendizaje.
    - booster (str): Tipo de booster a utilizar (gbtree, gblinear o dart).
    - gamma (float): Parámetro de regularización gamma.
    - reg_alpha (float): Parámetro de regularización L1 (alpha).
    - reg_lambda (float): Parámetro de regularización L2 (lambda).
    - test_size (float): Proporción del dataset a incluir en el conjunto de prueba.
    - random_state (int): Semilla para reproducir los mismos resultados.

    Returns:
    - model (XGBClassifier): El modelo XGBoost entrenado.
    - test_metrics (dict): Métricas de evaluación sobre el conjunto de prueba.
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Configurar y entrenar el modelo XGBoost
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                              tree_method='hist' if max_leaves > 0 else 'exact',
                              learning_rate=learning_rate, booster=booster, gamma=gamma, reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Hacer predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    test_metrics = {
        'accuracy_score': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }

    return model, test_metrics


def train_logistic_regression(X, y, C=1.0, fit_intercept=True, l1_ratio=None, penalty='l2', test_size=0.2,
                              random_state=None, solver='saga'):
    """
    Entrena un modelo de regresión logística con los datos proporcionados y los parámetros específicos.

    Parameters:
    - X (pd.DataFrame): Variables independientes del dataset.
    - y (pd.Series): Variable dependiente del dataset.
    - C (float): Parámetro de regularización inversa.
    - fit_intercept (bool): Especifica si se debe agregar un término de intercepción.
    - l1_ratio (float): El parámetro de mezcla de la regularización elástica net (0 = regresión L2, 1 = L1).
    - penalty (str): Tipo de penalización, puede ser 'l1', 'l2', 'elasticnet'.
    - test_size (float): Proporción del dataset a incluir en el conjunto de prueba.
    - random_state (int): Semilla para reproducir los mismos resultados.
    - solver (str): Algoritmo a utilizar en el problema de optimización.

    Returns:
    - model (LogisticRegression): El modelo de regresión logística entrenado.
    - test_metrics (dict): Métricas de evaluación sobre el conjunto de prueba.
    """

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Inicializar y entrenar el modelo de regresión logística
    model = LogisticRegression(C=C, fit_intercept=fit_intercept, penalty=penalty, l1_ratio=l1_ratio, solver=solver)
    model.fit(X_train, y_train)

    # Hacer predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    test_metrics = {
        'accuracy_score': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }

    return model, test_metrics