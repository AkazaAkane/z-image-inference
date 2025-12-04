"""i18n helper module for runtime translations."""

from pathlib import Path

import yaml

_translations = None
_translations_path = Path(__file__).parent / "translations.yaml"


def load_translations() -> dict:
    """Load translations from YAML file (cached)."""
    global _translations
    if _translations is None:
        with open(_translations_path, encoding="utf-8") as f:
            _translations = yaml.safe_load(f)
    return _translations


def get_text(key: str, lang: str = "zh", **kwargs) -> str:
    """Get translated text with optional variable substitution.

    Args:
        key: Translation key
        lang: Language code (zh, en)
        **kwargs: Variables to substitute in the string

    Returns:
        Translated string with variables substituted
    """
    translations = load_translations()
    text = translations.get(lang, {}).get(key, key)
    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass
    return text
