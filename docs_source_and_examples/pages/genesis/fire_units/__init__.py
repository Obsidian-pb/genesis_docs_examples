"""
Подпакет-адаптер для совместимости импортов.

Позволяет обращаться к функционалу как через старые импорты:
    `from fire_units.settings import ...`
так и через единое пространство имён:
    `from genesis.fire_units.settings import ...`
"""

# У `fire_units` нет явного __version__ в __init__, но модуль существует.
from fire_units._version import __version__  # noqa: F401

