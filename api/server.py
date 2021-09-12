import os
import sys
from flask import Flask
from tests.cli_test import test_sample
from filters.cli import main_for_server
from filters.config.directories import directories
from filters.config.constants import _BASE_CONFIG_FILE
app = Flask(__name__)


@app.route("/")
def index():
    results = main_for_server()
    return str(results)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)