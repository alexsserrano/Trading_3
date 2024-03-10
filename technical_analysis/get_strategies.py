# get_strategies.py
def  get_strategies() -> list:
    """
    Genera una lista de todas las posibles estrategias de trading basadas en combinaciones de indicadores técnicos.

    Returns:
    list: Lista de todas las estrategias posibles, cada una representada por un diccionario.
    """
    strategies = []
    indicators = ['svc', 'xgb', 'lr']
    n = len(indicators)

    # Genera todas las combinaciones posibles de indicadores
    for i in range(1, 2 ** n ):
        strategy_indicators = [indicators[j] for j in range(n) if (i >> j) & 1]
        strategy = {
            'id': i,
            'indicators': strategy_indicators,
            'params': {}
        }
        strategies.append(strategy)

    return strategies


# Generamos todas las estrategias posibles
all_strategies = get_strategies()
# Por motivos de espacio, mostraré solo las ultimas 5 estrategias generadas
print(all_strategies[:5])
