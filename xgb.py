import xgboost as xgb
import os


class Xgb:
    def __init__(self, path, features=None):
        if not features:
            features = ['Close']

        self.path = path
        self.features = features
        self.n_days = n_days

        self.model = xgb.Booster()
        if os.path.isfile(path):
            self.model.load_model(path)
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=8,
                gamma=0.01,
                objective='reg:squarederror'
            )

    def fit(self, X, Y):
        self.model.fit(X, Y)
        self.model.save_model(self.path)

    def predict(self, X):
        return self.model.predict(X)
