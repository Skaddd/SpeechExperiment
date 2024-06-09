import logging.config
import yaml
import re
import logging


logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Sets up logger using yml config file."""
    with open("config/logging.yml", "r") as f:
        yaml_config = yaml.safe_load(f.read())
        logging.config.dictConfig(yaml_config)


def display_linear_modules(model) -> None:
    """Display model modules that contains Linear Layers.

    Args:
        model (_type_): HuggingFace model.
    """
    pattern = r"\((\w+)\): Linear"
    linear_layers = re.findall(pattern, str(model.modules))
    target_modules = set(linear_layers)
    logger.info(f"--- Fine-tuning modules : {target_modules}---")
