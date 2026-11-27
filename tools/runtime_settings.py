"""Explicit private TOML loading for maintenance commands; never auto-load .env."""
import os
from pathlib import Path
import tomllib


def read_settings(path):
    """Read an explicit file without exporting credentials to process environment."""
    path = Path(path)
    if path.suffix.lower() != '.toml':
        raise ValueError('Use an explicit TOML settings file; .env is not loaded.')
    try:
        settings = tomllib.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        raise ValueError('Could not read private TOML settings. Check the file locally.') from None
    accepted = {key: value for key, value in settings.items()
                if key.startswith(('SENTO_', 'QDRANT_', 'CHROMA_', 'OPENAI_', 'FIREWORKS_'))}
    if any(not isinstance(value, (str, int, float, bool)) for value in accepted.values()):
        raise ValueError('Provider settings must be root-level scalar TOML values.')
    return {key: str(value) for key, value in accepted.items()}


def load_settings(path):
    if path is not None:
        os.environ.update(read_settings(path))
