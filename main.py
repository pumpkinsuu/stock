from flask import Flask
from flask_cors import CORS

from route import create_route
from model import load

app = Flask(__name__)
CORS(app)

models = load()

bp = create_route(models)
CORS(bp)
app.register_blueprint(bp, url_prefix='/api')


if __name__ == '__main__':
    app.run(host='localhost', port='5000')
