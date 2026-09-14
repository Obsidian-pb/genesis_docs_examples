"""
Подпакет-адаптер для совместимости импортов.

Позволяет обращаться к функционалу как через старые импорты:
    `from lists import ...`
так и через единое пространство имён:
    `from genesis.lists import ...`
"""

from lists._api import *  # noqa: F403
from lists._version import __version__  # noqa: F401

