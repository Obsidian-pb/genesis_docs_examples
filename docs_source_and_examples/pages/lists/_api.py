'''
Подключение наиболее важных функций
'''

# from lists.buildings import BUILDINGS as BUILDINGS
# from lists.allotments import ALLOTMENTS as ALLOTMENTS
# from lists.kfpo import KFPO as KFPO
from .buildings import BUILDINGS as BUILDINGS
from .allotments import ALLOTMENTS as ALLOTMENTS
from .kfpo import KFPO as KFPO




BUILDINGS_CLASS = {k:v[0] for k,v in BUILDINGS.items()}
BUILDINGS_KFPO = {v[0]:v[1] for _,v in BUILDINGS.items()}
KFPO_DEMAND = {k:v[2] for k,v in KFPO.items()}
