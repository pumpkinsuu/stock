from flask import Flask, jsonify
from flask_cors import CORS

from route import create_route
from model import load
from utilities import ErrorAPI

app = Flask(__name__)
CORS(app)


@app.errorhandler(ErrorAPI)
def exception(e: ErrorAPI):
    return jsonify(e.detail()), 500


@app.errorhandler(Exception)
def exception(e: Exception):
    return jsonify({'error': str(e)}), 500


models = load()


bp = create_route(models)
CORS(bp)
app.register_blueprint(bp, url_prefix='/api')


if __name__ == '__main__':
    app.run(host='localhost', port='5000')
