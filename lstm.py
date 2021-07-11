from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation
import os


class Lstm:
    def __init__(self, path, features=None, n_days=60):
        if not features:
            features = ['Close']

        self.path = path
        self.features = features
        self.n_days = n_days

        if os.path.isfile(path):
            self.model = load_model(path)
        else:
            self.model = Sequential()
            self.model.add(LSTM(50, input_shape=(1, len(features)*n_days), return_sequences=True))
            self.model.add(LSTM(50))
            self.model.add(Dense(1))
            self.model.add(Activation('linear'))
            self.model.compile(loss='mean_squared_error', optimizer='adam')

    def fit(self, X, Y, epochs=10):
        self.model.fit(X, Y, epochs=epochs, verbose=1)
        self.model.save(self.path)

    def predict(self, X):
        return self.model.predict(X)
