import os
from typing import Dict, Any
import yaml


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_od_config() -> Struct:
    # Load the existing YAML config
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    config['local_data_path'] = os.path.join(os.getenv("HOME"), "tensorleap", "data", config['BUCKET_NAME'])

    return Struct(**config)

cnf = load_od_config()
