# backtest.py
import pandas as pd


class Operation:
    def __init__(self, operation_type, bought_at, shares, stop_loss=None, take_profit=None, initial_margin=0, strategy_id=None):
        self.operation_type = operation_type  # "long" o "short"
        self.bought_at = bought_at
        self.shares = shares
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.initial_margin = initial_margin  # Margen inicial para operaciones cortas
        self.current_margin = initial_margin  # Margen actual, ajustado para operaciones cortas
        self.strategy_id = strategy_id  # Número de estrategia
        self.closed = False  # Indica si la operación está cerrada


def adjust_margin_and_handle_margin_calls(data, active_operations, cash, commission_per_trade):
    total_margin_required = 0
    # Ajustar el margen requerido para cada operación corta y calcular el total requerido
    for op in active_operations:
        if op.operation_type == "short" and not op.closed:
            current_price = data['Close'].iloc[-1]  # Precio actual del activo
            op.current_margin = max(op.initial_margin,
                                    op.shares * current_price * 0.25)  # Ejemplo: 25% del valor de mercado
            total_margin_required += op.current_margin

    # Calcular el valor total de la cartera (efectivo + valor de las acciones activas)
    total_shares_value = sum(
        op.shares * data['Close'].iloc[-1] for op in active_operations if op.operation_type == "long" and not op.closed)
    portfolio_value = cash + total_shares_value

    # Manejar llamadas de margen si el total requerido supera el valor de la cartera
    if portfolio_value < total_margin_required:
        for op in sorted(active_operations, key=lambda x: x.current_margin, reverse=True):
            if op.operation_type == "short" and not op.closed:
                current_price = data['Close'].iloc[-1]
                # Cerrar la operación corta
                cash += (op.bought_at * op.shares - current_price * op.shares) * (1 - commission_per_trade)
                op.closed = True
                total_margin_required -= op.current_margin
                if cash + total_shares_value >= total_margin_required:
                    break  # No más acciones requeridas para satisfacer el margen

    # Actualizar la lista de operaciones activas excluyendo las cerradas
    active_operations = [op for op in active_operations if not op.closed]
    return active_operations, cash


def backtest(data: pd.DataFrame, buy_signals: pd.DataFrame, sell_signals: pd.DataFrame, initial_cash: float = 10000,
             commission_per_trade: float = 0.001, shares_to_operate: int = 10, stop_loss: float = 0.01,
             take_profit: float = 0.01):
    cash = initial_cash
    active_operations = []  # Lista para almacenar todas las operaciones activas
    portfolio_value = []

    for i in range(len(data)):
        current_price = data['Close'].iloc[i]

        # Revisar y cerrar operaciones activas si es necesario, basado en stop_loss y take_profit
        for op in active_operations:
            profit_loss_ratio = (current_price / op.bought_at - 1) if op.operation_type == "long" else (
                        op.bought_at / current_price - 1)
            if not op.closed:
                if op.operation_type == "long" and (
                        (profit_loss_ratio <= -stop_loss) or (profit_loss_ratio >= take_profit)):
                    cash += current_price * op.shares * (1 - commission_per_trade)
                    op.closed = True
                elif op.operation_type == "short" and (
                        (profit_loss_ratio <= -stop_loss) or (profit_loss_ratio >= take_profit)):
                    cash += (op.bought_at * op.shares - current_price * op.shares) * (1 - commission_per_trade)
                    op.closed = True

        # Ajustar margen y manejar llamadas de margen para operaciones cortas
        active_operations, cash = adjust_margin_and_handle_margin_calls(data.iloc[:i + 1], active_operations, cash,
                                                                        commission_per_trade)

        # Iterar sobre cada estrategia en buy_signals
        for strategy_column in buy_signals.columns:
            # Verificar la señal de compra para la estrategia actual y abrir nuevas operaciones basadas en señales de compra y venta
            if buy_signals[strategy_column].iloc[i] and cash >= current_price * shares_to_operate * (
                    1 + commission_per_trade):
                cash -= current_price * shares_to_operate * (1 + commission_per_trade)

                # Extraer el número de estrategia desde el nombre de la columna
                #strategy_number = int(strategy_column.split('_')[1])

                # Agregar una nueva operación basada en la estrategia actual
                active_operations.append(Operation("long", current_price, shares_to_operate,
                                                                            stop_loss, take_profit,
                                                                            initial_margin=current_price * shares_to_operate * 0.25))
                                                                            #strategy_id=strategy_number))
        # Iterar sobre cada estrategia en sell_signals para operaciones de venta
        for strategy_column in sell_signals.columns:
            # Verificar la señal de venta para la estrategia actual y abrir nuevas operaciones basadas en señales de venta
            if sell_signals[strategy_column].iloc[i] and cash >= current_price * shares_to_operate * (
                    1 + commission_per_trade):
                cash -= current_price * shares_to_operate * (1 + commission_per_trade)
        
                # Extraer el número de estrategia desde el nombre de la columna
                #strategy_number = int(strategy_column.split('_')[1])
        
                # Agregar una nueva operación de venta basada en la estrategia actual
                active_operations.append(Operation("short", current_price, shares_to_operate,
                                                   stop_loss, take_profit,
                                                   initial_margin=current_price * shares_to_operate * 0.25))
                                                   #strategy_id=strategy_number))



        # Calcular el valor total de la cartera
        total_shares_value = sum(current_price * op.shares for op in active_operations if not op.closed)
        portfolio_value.append(cash + total_shares_value)

    final_portfolio_value = portfolio_value[-1]
    total_return = (final_portfolio_value - initial_cash) / initial_cash

    return {
        'initial_cash': initial_cash,
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return,
        'portfolio_value_over_time': portfolio_value
    }
