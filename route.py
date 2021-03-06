from flask import Blueprint, request, jsonify
import numpy as np

from utilities import get_data, preprocess, ErrorAPI


def create_route(models: list):
    bp = Blueprint('bp', __name__)

    @bp.route('/predict')
    def predict():
        data = request.args
        # String
        stock = data.get('stock')
        # String
        model = data.get('model')
        # Start date
        start = data.get('start')
        # End date
        end = data.get('end')

        if model not in models:
            raise ErrorAPI(400, 'Model not exist!')

        df = get_data(stock=stock, start=start, end=end)
        scaler, X, _ = preprocess(
            df,
            models[model].features,
            models[model].n_days
        )

        result = models[model].predict(X)
        result = scaler.inverse_transform(result)
        result = result.reshape(-1)
        empty = [None] * models[model].n_days
        result = np.concatenate((empty, result))

        tomorrow = result[-1]

        df = df.assign(predict=result[:-1])
        df['predict'] = df['predict'].astype(float)
        df = df.dropna()

        return jsonify({
            'csv': df.to_csv(),
            'tomorrow': tomorrow
        })

    return bp
