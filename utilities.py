import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.preprocessing import MinMaxScaler


TOKEN = 'a7213dd9f3ba445a808a3310c5031dd5021adffd'
COLUMNS = [
    'close',
    'open',
    'high',
    'low',
    'volume',
    'adjClose',
    'adjOpen',
    'adjHigh',
    'adjLow',
    'adjVolume',
    'divCash',
    'splitFactor'
]


def get_data(
        stock='aapl',
        freq='daily',
        start=None,
        end=None
):
    params = {
        'format': 'csv',
        'sort': 'date',
        'token': TOKEN,
        'columns': COLUMNS,
        'startDate': start,
        'endDate': end,
    }
    if start:
        params['startDate'] = start
    if end:
        params['endDate'] = end
    url = f'https://api.tiingo.com/tiingo/{freq}/{stock}/prices'

    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception('Something wrong!')

    csv = r.content.decode('utf-8')
    return pd.read_csv(StringIO(csv))


# features=['Close','PoC','RSI']
def preprocess(df, features=None, n_days=60):
    if features is None:
        features = ['close']

    data = df[['close']]
    if 'poc' in features:
        data = data.assign(PoC=data.pct_change())
    if 'rsi' in features:
        delta = data['poc'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down

        data = data.assign(RSI=100 - (100 / (1 + rs)))

    data = data.dropna()

    # Transform
    Y = data[n_days:][['close']].to_numpy()
    data = data[features].to_numpy()

    n = len(data) - n_days + 1
    X = np.empty((n, len(features) * n_days))
    for i in range(n):
        X[i] = data[i:i + n_days].reshape(-1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = X.reshape((n, 1, len(features) * n_days))
    Y = scaler.fit_transform(Y)

    return scaler, X, Y
