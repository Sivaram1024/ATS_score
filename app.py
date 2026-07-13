import logging
import os

from flask import Flask

from config import Config

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    missing = Config.validate()
    if missing:
        logging.warning(
            "Missing required environment variables: %s. "
            "Copy .env.example to .env and fill them in before analyzing resumes.",
            ", ".join(missing),
        )

    from routes.main_routes import main_bp
    app.register_blueprint(main_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
