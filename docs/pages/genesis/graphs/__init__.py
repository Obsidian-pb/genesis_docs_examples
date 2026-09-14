"""
Подпакет-адаптер для совместимости импортов.

Позволяет обращаться к функционалу как через старые импорты:
    `from graphs.speeds import ...`
так и через единое пространство имён:
    `from genesis.graphs.speeds import ...`
"""

from graphs._version import __version__  # noqa: F401

