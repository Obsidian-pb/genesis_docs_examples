'''
Швейцарский нож.

Думаю не нужен!

Глобальные инструменты и настройки системы, которые могут быть изменены пользователем

## Функции
`multi_source_forward (msf)`: алгоритм расчета кратчайшей цены
    (время или расстояние или еще что-то) и маршрута следования
    прямого пути во все узлы графа от множества источников.
    По умолчанию - `nx.multi_source_dijkstra`

    
## Константы
`DELAY_TIME`: Время обработки вызова. 
    (то время которое проходит с момента поступления сообщения до момента выезда сил и средств)
'''

import networkx as nx
from matplotlib.colors import LinearSegmentedColormap


# Функции используемые при проведении расчетов
MSF = nx.multi_source_dijkstra
# ssf = nx.single_source_dijkstra
# ssfpl = nx.single_source_dijkstra_path_length
# msfpl = nx.multi_source_dijkstra_path_length
# sppl = nx.dijkstra_path_length



# Цветовые схемы
CMAP_GrGldRd = LinearSegmentedColormap.from_list("mycmap", ["green", "gold", "red"])
CMAP_RdGldGr = LinearSegmentedColormap.from_list("mycmap", ["red", "gold", "green"]) 

DELAY_TIME = 1
