import os

from lstm import Lstm
from utilities import get_data, preprocess


def load(n_days=60):
    if not os.path.isdir('save_model'):
        os.makedirs('save_model')

    files = os.listdir('save_model')

    models = {
        'lstm_close': Lstm('save_model/lstm_close.h5', ['close'], n_days),
        # 'lstm_close_poc': Lstm('save_model/lstm_close_poc.h5', ['close', 'poc'], n_days),
        # 'lstm_close_rsi': Lstm('save_model/lstm_close_rsi.h5', ['close', 'rsi'], n_days),
        # 'lstm_close_poc_rsi': Lstm('save_model/lstm_close_poc_rsi.h5', ['close', 'poc', 'rsi'], n_days)
    }

    stocks = ['aapl']

    for stock in stocks:
        df = get_data(stock, start='2010-01-01', end='2021-01-01')
        for model in models:
            _, X, Y = preprocess(df, models[model].features, models[model].n_days)

            if not models[model].path.split('/')[-1] in files:
                models[model].fit(X[:-1], Y)

            models[model].predict(X[:models[model].n_days])

    return models
