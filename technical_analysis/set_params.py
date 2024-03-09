# set_params.py
def set_params(strategy: dict, **kwargs) -> dict:
    """
    Establece o ajusta los parámetros para los indicadores de una estrategia de trading.
    Si el indicador ya tiene parámetros, estos se actualizarán con los nuevos valores,
    permitiendo ajustes granulares sin sobrescribir completamente los parámetros existentes.

    Parameters:
    - strategy (dict): Estrategia de trading a modificar. Debe contener un campo 'params' para cada indicador.
    - **kwargs: Parámetros específicos de los indicadores a ajustar, donde cada clave corresponde a un indicador
      y cada valor es un diccionario de los parámetros a ajustar para ese indicador.

    Returns:
    - dict: Estrategia de trading con los parámetros actualizados.
    """
    if 'params' not in strategy:
        strategy['params'] = {}

    for indicator, params in kwargs.items():
        if indicator in strategy['indicators']:
            if indicator not in strategy['params']:
                strategy['params'][indicator] = params
            else:
                # Fusiona los nuevos parámetros con los existentes para el indicador
                strategy['params'][indicator].update(params)

    return strategy

