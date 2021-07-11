from flask import Flask
from flask_cors import CORS

from route import create_route
from model import load

app = Flask(__name__)
CORS(app)

models = load()

admin_bp = create_route(models)
CORS(admin_bp)
app.register_blueprint(admin_bp, url_prefix='/api')


if __name__ == '__main__':
    app.run(host='localhost', port='5000')
