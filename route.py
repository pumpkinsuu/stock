from flask import Blueprint, request, jsonify
from utilities import get_data, preprocess


def create_route(models: list):
    bp = Blueprint('bp', __name__)

    @bp.route('/predict')
    def predict():
        data = request.json
        # String
        stock = data['stock']
        # String
        model = data['save_model']
        # List
        features = data['features']
        # Int
        n_days = data['n_days']
        # Start date
        start = data['start']
        # End date
        end = data['end']
        # Frequency
        freq = data['freq']

        df = get_data(stock=stock, freq=freq, start=start, end=end)
        scaler, X, _ = preprocess(df, features, n_days)

        result = models[model].predict(X)
        result = scaler.inverse_transform(result)
        tomorrow = result[-1]

        df = df.assign(Predict=result[:-1])
        df['predict'] = df['predict'].astype(float)
        df = df.dropna()

        return jsonify({
            'csv': df.to_csv(),
            'tomorrow': tomorrow
        })
