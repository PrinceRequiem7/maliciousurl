

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly (`python app/app.py`) as well as as a
# module (`python -m app.app` from the project root) by ensuring the
# project root is importable either way.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from flask import Flask, render_template, request  # noqa: E402

from src.predict import predict_url  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """Render the homepage with the URL input form."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Run a prediction for the submitted URL and render the result page.

    Reads `url` from the submitted form. On an invalid URL, re-renders
    the homepage with an inline error message rather than a raw 500
    error, so the "keep the interface simple and professional"
    requirement holds even for bad input.
    """
    submitted_url = (request.form.get("url") or "").strip()

    if not submitted_url:
        return render_template("index.html", error="Please enter a URL to analyze.")

    try:
        result = predict_url(submitted_url)
    except ValueError as exc:
        logger.warning("Invalid URL submitted: %s", submitted_url)
        return render_template("index.html", error=str(exc), previous_url=submitted_url)
    except FileNotFoundError:
        logger.error("Model artifacts not found; has `python -m src.train` been run?")
        return render_template(
            "index.html",
            error=(
                "The prediction model is not available yet. Please run the "
                "training pipeline (`python -m src.train`) before using this app."
            ),
        )
    except Exception:  # noqa: BLE001 — surface as a friendly error, log details
        logger.exception("Unexpected error while predicting URL: %s", submitted_url)
        return render_template(
            "index.html",
            error="Something went wrong analyzing that URL. Please try again.",
        )

    return render_template("result.html", result=result)


@app.route("/healthz", methods=["GET"])
def healthz():
    """Simple health check endpoint."""
    return {"status": "ok"}, 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
