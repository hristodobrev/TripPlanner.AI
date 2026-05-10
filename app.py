from flask import Flask

from routes.description_routes import description_bp
from routes.recommendation_routes import recommendation_bp

app = Flask(__name__)

app.register_blueprint(description_bp)
app.register_blueprint(recommendation_bp)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)