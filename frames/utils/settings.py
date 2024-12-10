import pandas as pd
import yaml


def load_models(filepath: str = "models.yaml") -> pd.DataFrame:
    """Load all models from a YAML file into a pandas DataFrame."""
    return pd.DataFrame(yaml.safe_load(open(filepath))).fillna(value="auto")
