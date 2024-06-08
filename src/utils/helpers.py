import logging.config
import yaml


def setup_logging():
    """Sets up logger using yml config file."""
    with open("config/logging.yml", "r") as f:
        yaml_config = yaml.safe_load(f.read())
        logging.config.dictConfig(yaml_config)
