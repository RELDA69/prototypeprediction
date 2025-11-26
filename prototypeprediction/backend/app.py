from flask import Flask
from flask_cors import CORS
from api.routes import api

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Register the API blueprint at root path
    app.register_blueprint(api, url_prefix='/')
    
    return app

if __name__ == '__main__':
    app = create_app()
    # debug=True for development; set to False in production
    app.run(host='0.0.0.0', port=5000, debug=True)