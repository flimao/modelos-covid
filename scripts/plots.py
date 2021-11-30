import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_percentage_error as fmape, mean_squared_error as fmse, mean_absolute_error as fmae, r2_score as fr2, f1_score as ff1, precision_score as fprecision, recall_score as frecall, accuracy_score as faccuracy

def plot_true_pred(
    modelos, 
    X_test: np.array or pd.Series or pd.DataFrame,
    y_test: np.array or pd.Series or pd.DataFrame,
) -> plt.Figure:
    
    # Vamos criar um gráfico para comparar os Valores Reais com os Preditos
    fig = plt.figure(figsize=(10, 8), dpi = 120)

    for i, modelo in enumerate(modelos):
        y_pred = modelo.predict(X_test)    

        # metricas
        metricas_kwargs = dict(
            y_true = y_test,
            y_pred = y_pred
        )

        r2 = fr2(**metricas_kwargs)
        mape = fmape(**metricas_kwargs)
        rmse = fmse(squared = False, **metricas_kwargs)
        mae = fmae(**metricas_kwargs)

        label = str(modelo)
        label += f'\n(R² = {r2:.3%}, MAPE = {mape:.3%}, RMSE = {rmse:.2f}, MAE = {mae:.2f})'

        cor = f'C{i}'
        plt.plot(y_pred, y_test, 'bo', markersize = 10, markerfacecolor = cor, markeredgecolor = cor, alpha = 0.3, label = label)

    plt.ylabel("True Value", fontsize=12)
    plt.xlabel("Predict Value", fontsize=12)

    plt.title('Comparação Valor Predito x Valor Real', fontsize=12)

    # mostra os valores preditos e originais
    xl = np.arange(min(y_test), 1.2*max(y_test),(max(y_test)-min(y_test))/10)
    yl = xl
    plt.plot(xl, yl, 'r--')

    plt.legend()
    plt.show()
    return fig