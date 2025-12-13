import json
import os

PARAMS_FILE = "params.json"


def load_params():
    if not os.path.exists(PARAMS_FILE):
        raise FileNotFoundError("params.json not found")

    with open(PARAMS_FILE, "r") as f:
        return json.load(f)


def save_params(params: dict):
    with open(PARAMS_FILE, "w") as f:
        json.dump(params, f, indent=4)