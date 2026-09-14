"""
genesis
========

genesis - это библиотека для проведения расчетов в области решения 
задач пространственной оптимизации
"""



# from ._api import *
from ._version import __version__


# from . import algorithms
# from . import tools

# Единое пространство имён для модулей-надстроек (не ломает старые импорты
# вида `import graphs`, но позволяет `import genesis.graphs`).
from . import graphs as graphs  # noqa: F401
from . import fire_units as fire_units  # noqa: F401
from . import lists as lists  # noqa: F401

