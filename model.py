import os

from lstm import Lstm
from utilities import get_data, preprocess


def load():
    if not os.path.isdir('save_model'):
        os.makedirs('save_model')

    files = os.listdir('save_model')

    models = [
        Lstm('save_model/lstm_close.h5', ['close']),
        # Lstm('save_model/lstm_close_poc.h5', ['close', 'poc']),
        # Lstm('save_model/lstm_close_rsi.h5', ['close', 'rsi']),
        # Lstm('save_model/lstm_close_poc_rsi.h5', ['close', 'poc', 'rsi'])
    ]

    stocks = ['aapl']

    for stock in stocks:
        df = get_data(stock, start='2010-01-01', end='2021-01-01')
        for model in models:
            if model.path.split('/')[-1] in files:
                continue

            features = model.features
            _, X, Y = preprocess(df, features)
            print(X.shape)
            print(model.model.input_shape)
            model.fit(X[:-1], Y)

    return models
