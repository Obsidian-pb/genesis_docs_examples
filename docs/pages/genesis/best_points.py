"""
Алгоритмы поиска оптимальных узлов графа.
"""

import random
import math

from collections import Counter

import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd

from .core import BestPointsBase, MetricBase, StateBase
from .metrics import ArrivalTime
from .tools import get_all_neighbor_nodes



class NodeMetric(BestPointsBase):
    """
    Расчет метрики для конкретного узла графа.

    Применяется строго к графам!

    Args:
        state_function (StateBase): функция расчета состояния окружения.
        metric_function (MetricBase): функция расчета метрики.
        appr_val (float, optional): Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики. По умолчанию 0.95.
        err_val (any, optional): Значение, возвращаемое если из точки node невозможно достичь требуемой доли узлов графа.
        **kwargs: Дополнительные параметры.
    """
    def __init__(self, state_function: StateBase,
                 metric_function: MetricBase,
                 appr_val = 0.95,
                 err_val=None,
                 **kwargs) -> None:
        '''
            `state_function`: StateBase
                функция расчета состояния окружения
            `metric_function`: MetricBase
                Функция расчета метрики
            `appr_val`: = 0.95
                Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики. 
                Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
                то такой узел не рассматривается.
            `err_val`: any = None
                Значение которое будет возвращено в случае если из точки `node`
                невозможно будет достичь требуемой доли узлов графа.
        '''
        self.appr_val = appr_val
        self.err_val = err_val
        super().__init__(state_function, metric_function, **kwargs)

    def __call__(self, env:nx.Graph, node:int, area=None, **kwargs):
        """
        Вычисляет метрику для заданного узла графа.

        Args:
            env (nx.Graph): Граф дорожной сети
            node (int): Идентификатор узла графа для которого происходит расчет
            area (pd.Series, optional): Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.

        Returns:
            any: Значение метрики или err_val, если покрытие недостаточно.
        """

        # Если маска приемлемых узлов графа не передана,
        # то приемлемое количество узлов считается от количества узлов в графе
        # Иначе - от количества True в маске
        if area is None:
            appr_nodes_count = int(env.number_of_nodes() * self.appr_val)
        else:
            appr_nodes_count = int(sum(area) * self.appr_val)
        # # Приемлемое количество узлов считается от количества узлов в графе
        # appr_nodes_count = int(env.number_of_nodes() * self.appr_val)

        # Собственно расчет
        times, _ = self.state_function(env=env, points=[node], area=area, **kwargs)
        if len(times)>=appr_nodes_count:
            cur_val = self.metric_function(times, **kwargs)
        else:
            # print(node, len(times))
            cur_val = self.err_val
        return cur_val


class BestNodesFull(BestPointsBase):
    """
    Поиск лучших узлов графа полным перебором. 
    Могут быть возвращены только узлы из которых можно попасть в большую часть других узлов графа.
    Если таковых узлов нет, возвращается ошибка некорректности графа. 
    (граф должен быть проверен на корректность прежде чем будет передан функции)
    
    Узлы для которых метрика не может быть вычислена (например слабо связанные 
    с основным графом) не учитываются. 

    ## Область применения
    Определение размещения одного узла с наилучшими показателями. 
    Дает достаточно точное решение, однако плохо подходит для больших графов.
    Требует длительного времени на проведение расчетов.
    Вместе с тем позволяет определить поле пригодных узлов в тех случаях когда предполагается наличие 
    большого количества приемлемых мест размещения. Например, точек, размещение в которых пожарных депо позволяет обеспечить
    ИП-10 = 100%.

    ## Важно
    Следует помнить, что некорректные узлы в случае их учета посредством снижения appr_val могут 
    давать искаженное представление о метриках графа. При этом в реальности граф дорожной сети 
    как правило изобилует слабосвязанными узлами, поэтому учет только узлов обеспечивающих 100%
    достижимость всего графа может приводить к принципиальной невозможности расчета.
    """
    def __init__(self,
                 state_function: StateBase,
                 metric_function: MetricBase,
                 appr_val: float = 0.95,
                 return_list: bool = False,
                 node_calc_end_function: callable = None,
                 **kwargs) -> None:
        '''
        `state_function` : StateBase
            Функция расчета состояния среды
        `metric_function` : MetricBase
            Функция оценки.
        `appr_val`: = 0.95
            Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики. 
            Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
            то такой узел не рассматривается.
        `return_list`: = False
            Если True - вернет список всех лучших узлов. False - вернет только первый из списка.
            Свойство необходимо для соблюдения правил возврата данных `BestPointsBase`.
            Однако в ряде случаев может потребоваться получить список всех лучших узлов.
        `node_calc_end_function`:callable=None
            Функция вызываемая в конце расчета каждого узла.
            Сигнатура функции:
            ```
                node_calc_end_function()
            ```
        '''
        self.appr_val = appr_val
        self.node_calc_end_function = node_calc_end_function
        self.return_list = return_list
        super().__init__(state_function, metric_function, **kwargs)

    def __call__(self, env:   nx.Graph,
                 area:        pd.Series = None,
                 start_point: int = None,
                 points_list:  set = None,
                 **kwargs):
        """
        Перебирает все узлы графа для поиска оптимального.

        Args:
            env (nx.Graph): Граф улично-дорожной сети
            area (pd.Series, optional): Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
            start_point (int, optional): Не используется.
            points_list (set, optional): Множество узлов-кандидатов.
            **kwargs: Дополнительные параметры.

        Returns:
            tuple: (best_nodes_list, best_metric) — список лучших узлов и значение метрики.
        """

        if not isinstance(env, nx.Graph):
            raise TypeError('Тип данных аргумента `env` должен быть nx.Graph')
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `pd.Series`! Имеет {type(area)}')

        # Если списка узлов изначально не передано, рассматриваются все узлы графа
        nodes_list = points_list
        if nodes_list is None:
            nodes_list = env.nodes()

        # # Если маска приемлемых узлов графа не передана,
        # # то приемлемое количество узлов считается от количества узлов в графе
        # # Иначе - от количества True в маске
        # if area is None:
        #     appr_nodes_count = int(env.number_of_nodes() * self.appr_val)
        # else:
        #     appr_nodes_count = int(sum(area) * self.appr_val)

        best_metric = None
        # best_node = None
        best_nodes_list = []
        # best_nodes_list = set()

        for node in nodes_list:

            # # расчет среды
            # times, _ = self.state_function(env=env, points=[node], area=area, **kwargs)

            # # проверка на корректность охвата графа
            # if len(times)>=appr_nodes_count:

            #     # расчет метрики среды для текущего узла `node`
            #     node_metric = self.metric_function(times, **kwargs)
            
            node_metric_func = NodeMetric(self.state_function, self.metric_function, self.appr_val, err_val=None)
            node_metric = node_metric_func(env=env, node=node, area=area)

            # проверка на корректность охвата графа
            if node_metric:

                # если лучшая метрика еще не указана, присваиваем текущее значение
                if best_metric is None:
                    best_metric = node_metric
                    best_nodes_list = [node]
                    # best_nodes_list = set([node])
                    # best_node = node
                # проверяем лучше ли метрика среды для текущего узла, чем лучшая до этого
                else:
                    # if best_metric > node_metric:
                    if self.metric_function.compare(best_metric, node_metric) == node_metric:
                        if best_metric == node_metric:
                            best_nodes_list.append(node)
                            # best_nodes_list.add(node)
                        else:
                            best_nodes_list = [node]
                            # best_nodes_list = set([node])
                        # best_node = node
                        best_metric = node_metric

            # выполняем функцию завершения расчета для узла
            if self.node_calc_end_function:
                self.node_calc_end_function()

        if self.return_list:
            return best_nodes_list, best_metric
        return best_nodes_list[0], best_metric



class BestNodeHillClimbing(BestPointsBase):
    """
    Поиск лучшего узла графа с использованием алгоритма скалолаза (hill climbing). 
    Могут быть возвращены только узлы из которых можно попасть в большую часть других узлов графа.
    Если таковых узлов нет, возвращается ошибка некорректности графа. 
    (граф должен быть проверен на корректность прежде чем будет передан функции)
    
    Узлы для которых метрика не может быть вычислена (например слабо связанные 
    с основным графом) не учитываются. 

    ## Область применения
    Определение размещения одного узла с наилучшими показателями. 
    Дает достаточно точное решение. Хорошо подходит для больших графов.
    Может давать оценку только для полного набора узлов графа (в отличие от BNF).
    """
    def __init__(self,
                 state_function: StateBase,
                 metric_function: MetricBase,
                 appr_val: float = 0.95,
                 all_neighbors: bool=True,
                 first_node_random: bool = False,
                 node_calc_end_function:callable=None,
                 **kwargs) -> None:
        """
        Args:
            state_function (StateBase): функция расчета состояния окружения
            metric_function (MetricBase): Функция расчета метрики
            appr_val (float, optional): Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики. По умолчанию 0.95.
            all_neighbors (bool, optional): Если True рассматриваются все узлы смежные с рассчитываемым узлом. Если False - только исходящие.
            node_calc_end_function (callable, optional): Функция выполняемая в конце расчета каждого узла.
            **kwargs: Дополнительные параметры.
        """
        if appr_val > 1 or appr_val<0:
            raise ValueError(f'Аргумент `appr_val` должен находиться в диапазоне (0, 1)!'
                             f'Сейчас {appr_val}')
        self.appr_val = appr_val
        self.all_neighbors=all_neighbors
        self.first_node_random = first_node_random
        self.node_calc_end_function = node_calc_end_function
        super().__init__(state_function, metric_function, **kwargs)

    def __call__(self,
                 env:         nx.Graph,
                 area:        pd.Series=None,
                 start_point: int = None,
                 points_list: set = None,
                #  debug_route: bool=False,
                #  node_calc_end_function:callable=None,
                 **kwargs):
        """
        Поиск лучшего узла методом скалолаза.

        Args:
            env (nx.Graph): Граф улично-дорожной сети
            area (pd.Series, optional): Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
            start_point (int, optional): Стартовый узел.
            points_list (set, optional): Не используется.
            **kwargs: Дополнительные параметры.

        Returns:
            tuple: (best_node, best_metric) — лучший узел и значение метрики.
        """
        
        if not isinstance(env, nx.Graph):
            raise TypeError("Тип переменной `env` должен быть Graph!")
        

        node_metric_func = NodeMetric(self.state_function, self.metric_function, self.appr_val, err_val=None, **kwargs)

        nodes_metric = {}
        route={}

        # Если маска приемлемых узлов графа не передана,
        # то приемлемое количество узлов считается от количества узлов в графе
        # Иначе - от количества True в маске
        # if area is None:
        #     appr_nodes_count = int(env.number_of_nodes() * self.appr_val)
        # else:
        #     appr_nodes_count = int(sum(area) * self.appr_val)

        start_node = start_point
        if start_node is None or self.first_node_random:
            # Поиск первого узла из которого можно попасть во все остальные узлы ГДС !ВАЖНО!
            # Иначе можно оказаться в тупике из которого нет выхода
            node_metric = None
            i=0
            nodes_list = list(env.nodes())
            while node_metric is None:
                if i>=env.number_of_nodes():
                    raise ValueError('Определить наиболее выгодный стартовый узел невозможно, ' \
                    'в связи с неприемлемой несвязностью графа')
                # Если указано, что стартовый узел должен выбираться рандомно
                if self.first_node_random:
                    start_node = random.choice(nodes_list)
                else:
                    start_node = nodes_list[i]

                # Расчет метрики для узла `start_node`
                node_metric = node_metric_func(env=env, node=start_node, area=area, **kwargs)
                # Если указана расчетная область и метрика не была рассчитана
                if (node_metric is None) and (not area is None):
                    node_metric = node_metric_func(env=env, node=start_node, **kwargs)

                nodes_metric[start_node] = node_metric
                i+=1
        else:
            # Использование переданного стартового узла
            if not isinstance(start_node,int):
                raise TypeError('Тип данных start_node должен быть только int!')

            node_metric = node_metric_func(env=env, node=start_node, area=area, **kwargs)

            if node_metric is None:
                if not area is None:
                    # raise ValueError(f'Достичь области `area` из стартового узла {start_node}' \
                    #                  'невозможно,')
                    # ВАЖНО! будет произведена оценка оценка метрики для подграфа зоны обслуживания подразделения
                    node_metric = node_metric_func(env=env, node=start_node, **kwargs)
                else:
                    raise ValueError('Указанный стартовый узел неприемлем, в связи с его слабой ' \
                        'связностью с остальной частью графа')

            nodes_metric[start_node] = node_metric
        
        route[start_node] = node_metric
        # logging.debug('ПЕРВЫЙ УЗЕЛ {}, метрика {}'.format(start_node, node_metric))

        # Пошаговый поиск лучшего узла от start_node
        best_metric = node_metric
        best_node = start_node
        tmp_node=None
        while best_node!=tmp_node:
            tmp_node = best_node

            if self.all_neighbors:
                nnodes = get_all_neighbor_nodes(env, tmp_node)
            else:
                nnodes = env[tmp_node]

            for node in nnodes:
                if node in nodes_metric:
                    node_metric = nodes_metric[node]
                else:
                    node_metric = node_metric_func(env=env, node=node, area=area, **kwargs)
                    nodes_metric[node] = node_metric

                # logging.debug('УЗЕЛ {}, метрика {}'.format(node, node_metric))

                # Если метрики одинаковы, в данном случае - ситуация неприемлемая и должна быть исключена.
                if best_metric!=node_metric and \
                        self.metric_function.compare(best_metric, node_metric) == node_metric:
                    best_node = node
                    best_metric = node_metric

            route[best_node] = best_metric
            # logging.debug('ЛУЧШИЙ УЗЕЛ {}, метрика {}'.format(best_node, best_metric))

            # выполняем функцию завершения расчета для узла
            if self.node_calc_end_function:
                self.node_calc_end_function(best_node=best_node, best_metric=best_metric)

        # if debug_route:
        #     return best_node, best_metric, route
        return best_node, best_metric



# Временно здесь - потом вынести в отдельный модуль для кастомизированных решений
class BestNodeHillClimbing_maxMean(BestNodeHillClimbing):
    def __init__(self, 
                 state_function: StateBase,
                 metric_function: MetricBase = None,
                 appr_val: float = 0.95,
                 all_neighbors: bool = True,
                 **kwargs) -> None:
        super().__init__(state_function, metric_function, appr_val, all_neighbors, **kwargs)

    def __call__(self, env:   nx.Graph,
                 area:        pd.Series = None,
                 start_point: int = None,
                 points_list: set = None,
                 debug_route: bool = False,
                 node_calc_end_function: callable = None,
                 **kwargs):
        """
        Сначала поиск по максимуму, затем по среднему.

        Args:
            env (nx.Graph): Граф улично-дорожной сети
            area (pd.Series, optional): Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
            start_point (int, optional): Стартовый узел.
            points_list (set, optional): Не используется.
            debug_route (bool, optional): Не используется.
            node_calc_end_function (callable, optional): Функция после шага.
            **kwargs: Дополнительные параметры.

        Returns:
            tuple: (best_node, best_metric) — лучший узел и значение метрики.
        """
        bnch_max = BestNodeHillClimbing(state_function=self.state_function,
                                        metric_function=ArrivalTime(np.max), appr_val=self.appr_val, all_nodes=self.all_neighbors)
        bnch_mean = BestNodeHillClimbing(state_function=self.state_function,
                                         metric_function=ArrivalTime(), appr_val=self.appr_val, all_nodes=self.all_neighbors)

        best_node, best_metric = bnch_max(env=env, area=area, start_point=start_point, points_list=points_list)
        best_node, best_metric = bnch_mean(env=env, area=area, start_point=start_point, points_list=points_list)

        # выполняем функцию завершения расчета для узла
        if node_calc_end_function:
            node_calc_end_function()

        return best_node, best_metric

class BestNodesHalfDiameter(BestPointsBase):
    """
    Поиск лучших узлов графа с использованием алгоритма точки в середине диаметра графа.

    Args:
        state_function (StateBase, optional): Не используется.
        metric_function (MetricBase, optional): Не используется.
        weight (str, optional): Имя поля веса ребер. По умолчанию 'travel_time'.
        **kwargs: Дополнительные параметры.
    """
    def __init__(self,
                 state_function: StateBase = None,
                 metric_function: MetricBase = None,
                 weight: str = 'travel_time',
                 **kwargs) -> None:
        '''
        ## Аргументы
        `state_function`: StateBase
            Не используется! функция расчета состояния окружения
        `metric_function`: MetricBase
            Не используется! Функция расчета метрики
        `weight`:str или callable  = "travel_time"
            Имя поля содержащего вес ребер, или функция позволяющая вычислять 
            вес динамически.
        '''
        self.weight = weight
        super().__init__(state_function, metric_function, **kwargs)

    def __call__(self, env:nx.Graph, area=None, **kwargs):
        """
        Поиск узла, находящегося примерно посередине диаметра графа.

        Args:
            env (nx.Graph): Граф улично-дорожной сети
            area: Не используется!

        Returns:
            tuple: (node, None) — найденный узел и None.
        """
        # Проверка корректности пришедших данных
        if not isinstance(env, nx.Graph):
            raise TypeError(f'Неверный тип аргумента `env`! Должен быть `nx.Graph` - имеется `{type(env)}`.')

        # 1 Выбираем произвольную точку. По-умолчанию берем просто первую из списка узлов
        nd = list(env.nodes())[0]

        # 2.1 Находим самую отдаленную от нее (входящую) -- периферия №1
        lngs = nx.shortest_path_length(env, target=nd, weight=self.weight)
        corner_1 = max(lngs, key=lngs.get)

        # 2.2 Находим самую отдаленную от нее (входящую) -- периферия №2
        lngs = nx.shortest_path_length(env, target=corner_1, weight=self.weight)
        corner_2 = max(lngs, key=lngs.get)

        # # 2.3 Находим самую отдаленную от нее (входящую) -- периферия №3 (уточняющая)
        # lngs = nx.shortest_path_length(env, target=corner_2, weight=self.weight)
        # corner_3 = max(lngs, key=lngs.get)

        # Рассчитываем длину диаметра и маршрут следования по нему
        diameter = nx.shortest_path_length(env, corner_2, corner_1, weight=self.weight)
        short_path = nx.shortest_path(env, corner_2, corner_1, weight=self.weight)
        # diameter = nx.shortest_path_length(env, corner_1, corner_2, weight=self.weight)
        # short_path = nx.shortest_path(env, corner_1, corner_2, weight=self.weight)

        # Если в кратчайшем маршруте менее двух точек, возвращаем первую
        if len(short_path) < 2:
            return short_path[0], None

        # Находим точку примерно по середине диаметра
        tot_len = 0
        nd1 = short_path[0]
        for nd1, nd2 in zip(short_path[:-1],short_path[1:]):
            cur_len = env.get_edge_data(nd1, nd2, 0)[self.weight]
            tot_len += cur_len
            if tot_len >= diameter / 2: break

        return nd1, None


class BestNodeMonkey(BestNodeHillClimbing):
    """
    Поиск лучших узлов графа с использованием алгоритма обезьяньего поиска (Monkey Search Algorithm).

    Args:
        `state_function` (StateBase): Функция расчета состояния окружения.
        `metric_function` (MetricBase): Функция расчета метрики.
        `appr_val` (float, optional): Доля узлов графа для приемлемого покрытия. По умолчанию 0.95.
        `all_neighbors` (bool, optional): Если True — рассматривать всех соседей. По умолчанию True.
        `global_jumps_count` (int, optional): Количество глобальных прыжков. По умолчанию 3.
        `local_jumps_count` (int, optional): Количество локальных прыжков. По умолчанию 10.
        `local_jump_max_distance` (int, optional): Максимальное расстояние локального прыжка. По умолчанию 1000.
        `after_global_jump_function` (callable, optional): Функция после глобального прыжка.
        `after_local_jump_function` (callable, optional): Функция после локального прыжка.
        `**kwargs`: Дополнительные параметры.
    """
    def __init__(self,
                 state_function: StateBase,
                 metric_function: MetricBase,
                 appr_val: float = 0.95,
                 all_neighbors: bool=True,
                 global_jumps_count: int = 3,
                 local_jumps_count: int = 10,
                 local_jump_max_distance: int = 1000,
                 after_global_jump_function: callable = None,
                 after_local_jump_function: callable = None,
                 **kwargs) -> None:
        '''
        ## Аргументы
        `state_function`: StateBase

            функция расчета состояния окружения

        `metric_function`: MetricBase

            Функция расчета метрики

        `appr_val`: = 0.95

            Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики. 
            Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
            то такой узел не рассматривается.

        `all_neighbors`: bool=True

            Если True рассматриваются все узлы смежные с рассчитываемым узлом.
            Если False - только исходящие.

        `global_jumps_count`: int=3

            Количество глобальных прыжков (начиная со стартового).
            Глобальный прыжок осуществляется путем случайного выбора произвольного узла графа.

        `local_jumps_count`: int=10

            Количество локальных прыжков.

        `local_jump_max_distance`: int=1000

            Максимальное расстояние локального прыжка.
            Локальный прыжок осуществляется путем случайного выбора 
            узла графа на расстоянии `local_jump_max_distance` метров от текущей вершины.

        `after_global_jump_function`: callable=None

            Функция вызываемая после каждого глобального прыжка.

        `after_local_jump_function`: callable=None
        
            Функция вызывается после каждого локального прыжка
        '''
        self.node_metric_func = NodeMetric(state_function,
                                           metric_function,
                                           appr_val,
                                           err_val=None,
                                           **kwargs)
        self.global_jumps_count = global_jumps_count
        self.local_jumps_count = local_jumps_count
        self.local_jump_max_distance = local_jump_max_distance
        self.after_global_jump_function = after_global_jump_function
        self.after_local_jump_function = after_local_jump_function
        super().__init__(state_function,
                         metric_function,
                         appr_val=appr_val,
                         all_neighbors=all_neighbors,
                         **kwargs)


    def _get_sample_node(self, env, area, points_list, **kwargs):
        """
        Получение случайного узла, пригодного для старта.

        Args:
            env (nx.Graph): Граф.
            area: Маска целевых узлов.
            points_list: Множество узлов-кандидатов.
            **kwargs: Дополнительные параметры.

        Returns:
            tuple: (start_node, node_metric) — узел и его метрика.
        """
        node_metric = None
        i=0
        if points_list is None:
            nodes_list = list(env.nodes())
        else:
            nodes_list = points_list
        while node_metric is None:
            if i>=env.number_of_nodes():
                raise ValueError('Определить наиболее выгодный стартовый узел невозможно, ' \
                'в связи с неприемлемой несвязностью графа')
            start_node = random.choice(nodes_list)

            # Расчет метрики для узла `start_node`
            node_metric = self.node_metric_func(env=env, node=start_node, area=area, **kwargs)
            # Если указана расчетная область и метрика не была рассчитана
            if (node_metric is None) and (not area is None):
                node_metric = self.node_metric_func(env=env, node=start_node, **kwargs)

            i+=1

        return start_node, node_metric


    def __call__(self,
                 env: nx.Graph,
                 area = None,
                 start_point: int = None,
                 points_list:  set = None,
                 **kwargs) -> tuple[int | None, float | None]:
        """
        Основной цикл обезьяньего поиска.

        Args:
            env (nx.Graph): Граф улично-дорожной сети
            area (pd.Series, optional): Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
            start_point (int, optional): Не используется.
            points_list (set, optional): Не используется.

        Returns:
            tuple: (best_node, best_metric) — лучший узел и значение метрики.
        """

        if not isinstance(env, nx.Graph):
            raise TypeError("Тип переменной `env` должен быть Graph!")
        

        # Первый глобальный прыжок - случайный выбор старта
        # ! Здесь потом заменить `_get_sample_node`` на передаваемую функцию
        start_node = start_point
        if start_node is None:
            start_node, node_metric = self._get_sample_node(env, area, points_list, **kwargs)
        else:
            # Расчет метрики для узла `start_node`
            node_metric = self.node_metric_func(env=env, node=start_node, area=area, **kwargs)
            # Если указана расчетная область и метрика не была рассчитана
            if (node_metric is None) and (not area is None):
                node_metric = self.node_metric_func(env=env, node=start_node, **kwargs)

        best_node = start_node
        best_metric  = node_metric


        # Совершаем прыжки, начиная с первого
        for global_jump_number in range(self.global_jumps_count):

            # 1 Подъем на гору
            try:
                best_node_local, best_metric_local = super().__call__(
                    env=env,
                    area=area,
                    start_point=start_node,
                    # points_list=points_list,
                    **kwargs,
                    )
            except Exception as _:
                best_node_local, best_metric_local = None, None

            # Проверка на улучшение метрики и узла
            if best_node != best_node_local and \
                    self.metric_function.compare(best_metric, best_metric_local) == best_metric_local:
                best_node = best_node_local
                best_metric = best_metric_local

            # Событие после подъема на гору
            if self.after_global_jump_function is not None:
                self.after_global_jump_function(
                    global_jump_number = global_jump_number,
                    best_node = best_node,
                    best_metric = best_metric,
                    best_node_current = best_node_local,
                    best_metric_current = best_metric_local
                    )

            # 2 Локальные прыжки
            # 2.1 Определение узлов в округе
            g_nodes = ox.graph_to_gdfs(env, edges=False)
            if not ox.projection.is_projected(g_nodes.crs):
                g_nodes = ox.projection.project_gdf(g_nodes)
            buffer = g_nodes.loc[best_node_local:best_node_local].geometry.buffer(self.local_jump_max_distance)
            nodes_in_buffer = g_nodes[g_nodes.within(buffer.iloc[0])]
            nodes_in_buffer = list(nodes_in_buffer.index)

            # 2.2 Локальные прыжки
            global_number = 0
            jump_number = 0
            # while jump_number <= self.local_jumps_count:
            while jump_number < self.local_jumps_count:
                # global_number+=1
                # jump_number+=1
                jump_node = random.choice(nodes_in_buffer)
                # Вычисление метрики для узла `jump_node`
                try:
                    best_node_after_jump, best_metric_after_jump = super().__call__(env=env,
                                                               area=area,
                                                               start_point=jump_node,
                                                            #    points_list=points_list,
                                                               **kwargs)
                except Exception as _:
                    best_node_after_jump, best_metric_after_jump = None, None

                # if best_metric_after_jump<best_metric:  # Здесь заменить на Compare
                if best_node != best_node_after_jump and \
                        self.metric_function.compare(best_metric, best_metric_after_jump) == best_metric_after_jump:
                    best_node = best_node_after_jump
                    best_metric = best_metric_after_jump
                    # jump_number = -1
                    # Переход к новой вершине
                    buffer = g_nodes.loc[best_node_after_jump:best_node_after_jump].geometry.buffer(self.local_jump_max_distance)
                    nodes_in_buffer = g_nodes[g_nodes.within(buffer.iloc[0])]
                    nodes_in_buffer = list(nodes_in_buffer.index)

                # Событие после подъема на гору
                if self.after_local_jump_function is not None:
                    self.after_local_jump_function(
                        global_jump_number = global_jump_number,
                        local_jump_number = jump_number,
                        best_node = best_node,
                        best_metric = best_metric,
                        best_node_current = best_node_after_jump,
                        best_metric_current = best_metric_after_jump
                        )
                global_number+=1
                jump_number+=1

            # Глобальный прыжок (выбор нового случайного узла)
            start_node, node_metric = self._get_sample_node(env, area, points_list, **kwargs)

        return best_node, best_metric


class BestNodeBee(BestNodeHillClimbing):
    """
    Поиск лучших узлов графа с использованием алгоритма пчелиной колонии 
    (Artificial Bee Colony Optimization, ABC).

    Args:
        `state_function` (StateBase): Функция расчета состояния окружения.
        `metric_function` (MetricBase): Функция расчета метрики.
        `appr_val` (float, optional): Доля узлов графа для приемлемого покрытия. По умолчанию 0.95.
        `all_neighbors` (bool, optional): Если True — рассматривать всех соседей. По умолчанию True.
        `scouts_count` (int, optional): Количество пчел-разведчиков. По умолчанию 100.
        `best_scouts_count` (int, optional): Количество лучших разведчиков. По умолчанию 5.
        `**kwargs`: Дополнительные параметры.
    """
    def __init__(self,
                 state_function: StateBase,
                 metric_function: MetricBase,
                 appr_val: float = 0.95,
                 all_neighbors: bool=True,
                 scouts_count: int = 100,
                 best_scouts_count: int = 5,
                #  after_local_jump_function: callable = None,
                 **kwargs) -> None:
        '''
        ## Аргументы
        `state_function`: StateBase
            функция расчета состояния окружения
        `metric_function`: MetricBase
            Функция расчета метрики
        `appr_val`: = 0.95
            Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики. 
            Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
            то такой узел не рассматривается.
        `all_neighbors`: bool=True
            Если True рассматриваются все узлы смежные с рассчитываемым узлом.
            Если False - только исходящие.
        `scouts_count`: int=100
            Количество пчел-разведчиков.
        `best_scouts_count`: int=5
            Количество лучших пчел-разведчиков.
            Для них производится дальнейшая оптимизация.
        
        '''
        self.node_metric_func = NodeMetric(state_function,
                                           metric_function,
                                           appr_val,
                                           err_val=None,
                                           **kwargs)
        self.scouts_count = scouts_count
        self.best_scouts_count = best_scouts_count
        super().__init__(state_function,
                         metric_function,
                         appr_val=appr_val,
                         all_neighbors=all_neighbors,
                         **kwargs)


    def __call__(self,
                 env:nx.Graph,
                 area=None,
                 start_point: int = None,
                 points_list: set = None,
                 **kwargs) -> tuple[int | None, float | None]:
        """
        Основной цикл алгоритма пчелиной колонии.

        Args:
            env (nx.Graph): Граф улично-дорожной сети
            area (pd.Series, optional): Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
            start_point (int, optional): Стартовая точка
            points_list (set, optional): Множество узлов графа которые будут рассмотрены в качестве кандидатов.
            Если не указан, то будут рассмотрены все узлы графа.

        Returns:
            tuple: (best_node, best_metric) — лучший узел и значение метрики.
        """

        if not isinstance(env, nx.Graph):
            raise TypeError("Тип переменной `env` должен быть Graph!")
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `pd.Series`! Имеет {type(area)}')

        # Если списка узлов изначально не передано, рассматриваются все узлы графа
        # Здесь возможначпроблема, т.к. при дальнейшем поиске в любом случае оптимальными могут быть признаны узлы не из списка
        nodes_list = points_list
        if nodes_list is None:
            nodes_list = env.nodes()

        # 1. Разведка местности
        t_list = []
        t_metric = []
        for start_node in random.choices(list(nodes_list), k=self.scouts_count):
            start_metric = self.node_metric_func(env=env, area=area, node=start_node)
            t_list.append(start_node)
            t_metric.append(start_metric)
        
        # 2. Выбираем результаты лучших пчел-разведчиков
        tdf = pd.DataFrame({'node':t_list, 'metric':t_metric})
        if self.metric_function.compare(1,2)==2:
            tdf = tdf.sort_values('metric', ascending=False)
        else:
            tdf = tdf.sort_values('metric', ascending=True)
        # tdf.sort_values('metric', ascending=self.metric_function.compare(1,2)==1)
        # print(tdf['metric'][:10])

        # 3. Для каждого из лучших результатов пытаемся найти еще лучшие значения метрики в окрестности
        best_node = None
        best_metric  = None
        for node in tdf['node'][:self.best_scouts_count]:
            cur_node, cur_metric = super().__call__(
                    env=env,
                    area=area,
                    start_point=node,
                    # **kwargs,
                    )

            if best_node is None:
                best_node, best_metric = cur_node, cur_metric
            else:
                if best_node != cur_node and \
                        self.metric_function.compare(best_metric, cur_metric) == cur_metric:
                    best_node, best_metric = cur_node, cur_metric

        return best_node, best_metric


class BestNodeSquareZoom(BestPointsBase):
    """
    Поиск лучшего узла графа методом рекурсивного деления области на квадраты.

    1. Проекция графа в локальную систему координат.
    2. Определение главного прямоугольника, в который вписаны узлы графа.
    3. Разбиение прямоугольника на квадраты со стороной min(width, height)/2.
    4. В каждом квадрате случайная выборка до n узлов и вычисление метрики.
    5. Выбор квадрата с узлом наилучшей метрики и рекурсия, пока количество узлов в квадрате не <= n_min_count.

    Args:
        `state_function` (StateBase): Функция расчета состояния окружения.
        `metric_function` (MetricBase): Функция расчета метрики.
        `n_samples` (int, optional): Максимальное число узлов для оценки в каждом квадрате. По умолчанию 10.
        `n_min_count` (int, optional): Минимальное число узлов для останова рекурсии. По умолчанию 5.
        `step_end_function` (callable, optional): Функция, вызываемая после каждого шага рекурсии.
        `**kwargs`: Дополнительные параметры.
    """
    def __init__(self, state_function: StateBase,
                metric_function: MetricBase,
                n_samples: int = 10,
                n_min_count: int = 5,
                step_end_function: callable = None,
                **kwargs):
        """
        `state_function`: StateBase — функция расчета состояния окружения.
        `metric_function`: MetricBase — функция расчета метрики.
        `n_samples`: int — максимальное число узлов для оценки в каждом квадрате.
        `n_min_count`: int — минимальное число узлов для останова рекурсии.
        `step_end_function` - функция, вызываемая после каждого шага рекурсии.
        """
        self.n_samples = n_samples
        self.n_min_count = n_min_count
        self.step_end_function = step_end_function
        super().__init__(state_function, metric_function, **kwargs)

    def __call__(self, env: nx.Graph,
                area=None, 
                start_point: int = None,
                points_list: set = None,
                **kwargs) -> tuple[int | None, float | None]:
        """
        Основной цикл рекурсивного деления на квадраты.

        Args:
            env (nx.Graph): Граф улично-дорожной сети.
            area (pd.Series, optional): Маска узлов графа для расчёта доступности.
            start_point (int, optional): Не используется.
            points_list (set, optional): Начальный набор узлов-кандидатов.
            **kwargs: Дополнительные параметры.

        Returns:
            tuple: (best_node, best_metric) — лучший узел и значение метрики.
        """
        # Подготовка исходных узлов
        if points_list is None:
            current_nodes = list(env.nodes())
        else:
            current_nodes = list(points_list)
        # Проекция графа
        # Проекция графа только если он в географической системе координат
        if env.graph['crs'] == 'epsg:4326':
            G = ox.project_graph(env)
        else:
            G = env
        node_metric_fn = NodeMetric(self.state_function, self.metric_function)
        best_node = None
        best_metric = None

        level = 1
        while True:
            # Координаты узлов
            xs = [G.nodes[u]['x'] for u in current_nodes]
            ys = [G.nodes[u]['y'] for u in current_nodes]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            dx, dy = x_max - x_min, y_max - y_min
            size = min(dx, dy) / 2
            # Создание квадратов
            nx_squares = int(math.ceil(dx / size))
            ny_squares = int(math.ceil(dy / size))
            squares = []
            for i in range(nx_squares):
                for j in range(ny_squares):
                    x0, y0 = x_min + i * size, y_min + j * size
                    squares.append((x0, y0, x0 + size, y0 + size))
            # Оценка квадратов
            best_node_sq = None
            best_metric_sq = None
            best_square_nodes = None
            square = 1
            best_square = None
            for x0, y0, x1, y1 in squares:
                # Узлы в квадрате
                nodes_sq = [u for u in current_nodes if x0 <= G.nodes[u]['x'] <= x1 and y0 <= G.nodes[u]['y'] <= y1]
                if not nodes_sq:
                    continue
                # Выборка узлов
                if len(nodes_sq) > self.n_samples:
                    sample = random.sample(nodes_sq, self.n_samples)
                else:
                    sample = nodes_sq
                # Оценка узлов
                for u in sample:
                    node_metric = node_metric_fn(env=G, node=u, area=area, **kwargs)
                    if node_metric is None:
                        continue
                    if best_metric_sq is None or self.metric_function.compare(node_metric, best_metric_sq) == node_metric:
                        best_metric_sq = node_metric
                        best_node_sq = u
                        best_square_nodes = nodes_sq
                        best_square = (x0, y0, x1, y1)
                square += 1
            # best_squares.append(best_square)
            if best_node_sq is None:
                break
            best_node, best_metric = best_node_sq, best_metric_sq
            # Проверка условия останова
            if len(best_square_nodes) <= self.n_min_count:
                break
            # Переход к узлам лучшего квадрата
            current_nodes = best_square_nodes
            # Печать результата
            if self.step_end_function is not None:
                self.step_end_function(
                    step        = level,
                    best_node   = best_node,
                    best_metric = best_metric,
                    best_square = best_square)
            # Уменьшение масштаба квадратов
            level += 1

        return best_node, best_metric



class BestNodeCircleZoom(BestPointsBase):
    """
    Поиск лучшего узла графа методом рекурсивного деления области на вложенные окружности.

    1. Проекция графа в локальную систему координат.
    2. Определение окружности, в которую вписаны все узлы графа.
    3. В пределах окружности случайная выборка до n узлов и вычисление метрики.
    4. Выбор узла с наилучшей метрикой.
    5. Построение окружности с радиусом вдвое меньше предыдущего и центром в наилучшем узле.
    6. Повторять шаги 3-5, пока количество узлов в окружности не <= n_min_count.

    Args:
        state_function (StateBase): Функция расчета состояния окружения.
        metric_function (MetricBase): Функция расчета метрики.
        n_samples (int, optional): Максимальное число узлов для оценки в каждом шаге. По умолчанию 10.
        n_min_count (int, optional): Минимальное число узлов для останова рекурсии. По умолчанию 5.
        step_end_function (callable, optional): Функция, вызываемая после каждого шага рекурсии.
        **kwargs: Дополнительные параметры.
    """
    def __init__(self, state_function: StateBase,
                 metric_function: MetricBase,
                 n_samples: int = 10,
                 n_min_count: int = 5,
                 step_end_function: callable = None,
                 **kwargs):
        self.n_samples = n_samples
        self.n_min_count = n_min_count
        self.step_end_function = step_end_function
        super().__init__(state_function, metric_function, **kwargs)
    
    def __call__(self, env: nx.Graph,
                 area=None,
                 start_point: int = None,
                 points_list: set = None,
                 **kwargs) -> tuple[int | None, float | None]:
        """
        Основной цикл рекурсивного деления на окружности.

        Args:
            env (nx.Graph): Граф улично-дорожной сети.
            area (pd.Series, optional): Маска узлов графа для расчёта доступности.
            start_point (int, optional): Не используется.
            points_list (set, optional): Начальный набор узлов-кандидатов.
            **kwargs: Дополнительные параметры.

        Returns:
            tuple: (best_node, best_metric) — лучший узел и значение метрики.
        """
        # Подготовка исходных узлов
        if points_list is None:
            current_nodes = list(env.nodes())
        else:
            current_nodes = list(points_list)
        # Проекция графа
        if env.graph.get('crs') == 'epsg:4326':
            G = ox.project_graph(env)
        else:
            G = env
        # Координаты узлов
        xs = [G.nodes[u]['x'] for u in current_nodes]
        ys = [G.nodes[u]['y'] for u in current_nodes]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        # Центр и радиус окружности
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        radius = max(math.hypot(G.nodes[u]['x'] - center_x,
                                G.nodes[u]['y'] - center_y)
                     for u in current_nodes)
        node_metric_fn = NodeMetric(self.state_function, self.metric_function)
        best_node = None
        best_metric = None
        level = 1
        current_center = (center_x, center_y)
        current_radius = radius
        while True:
            # Узлы внутри текущей окружности
            circle_nodes = [u for u in current_nodes
                            if math.hypot(G.nodes[u]['x'] - current_center[0],
                                          G.nodes[u]['y'] - current_center[1])
                            <= current_radius]
            if not circle_nodes:
                break
            # Случайная выборка узлов
            if len(circle_nodes) > self.n_samples:
                sample = random.sample(circle_nodes, self.n_samples)
            else:
                sample = circle_nodes
            # Оценка узлов в выборке
            best_node_iter = None
            best_metric_iter = None
            for u in sample:
                m = node_metric_fn(env=G, node=u, area=area, **kwargs)
                if m is None:
                    continue
                if best_metric_iter is None or \
                   self.metric_function.compare(m, best_metric_iter) == m:
                    best_metric_iter = m
                    best_node_iter = u
            if best_node_iter is None:
                break
            best_node, best_metric = best_node_iter, best_metric_iter
            # Условие останова
            if len(circle_nodes) <= self.n_min_count:
                break
            # Обновление центра и радиуса
            current_center = (G.nodes[best_node]['x'], G.nodes[best_node]['y'])
            current_radius /= 2
            # Вызов функции после шага
            if self.step_end_function:
                self.step_end_function(
                    step=level,
                    best_node=best_node,
                    best_metric=best_metric,
                    center=current_center,
                    radius=current_radius)
            level += 1
        return best_node, best_metric


class BestNodeGWO(BestPointsBase):
    """
    Поиск лучших узлов графа с использованием алгоритма серых волков (Grey Wolf Optimizer, GWO).

    Алгоритм имитирует поведение стаи серых волков при охоте, где три лучших волка (альфа, бета, дельта)
    ведут остальных волков к оптимальному решению.

    Args:
        state_function (StateBase): Функция расчета состояния окружения.
        metric_function (MetricBase): Функция расчета метрики.
        appr_val (float, optional): Доля узлов графа для приемлемого покрытия. По умолчанию 0.95.
        max_iter (int, optional): Максимальное количество итераций. По умолчанию 50.
        pack_size (int, optional): Размер стаи волков. По умолчанию 20.
        **kwargs: Дополнительные параметры.
    """
    def __init__(self,
                 state_function:         StateBase,
                 metric_function:        MetricBase,
                 appr_val:               float    = 0.95,
                 max_iter:               int      = 20,
                 pack_size:              int      = 10,
                 hunt_bound:             float    = 1,
                 weight:                 str      = 'travel_time',
                 consensus_value:        float    = 0.75,
                 iteration_end_function: callable = None,
                 **kwargs) -> None:
        '''
        ## Аргументы
        `state_function`: StateBase
            функция расчета состояния окружения
        `metric_function`: MetricBase
            Функция расчета метрики
        `appr_val`: = 0.95
            Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики.
            Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
            то такой узел не рассматривается.
        `max_iter`: int = 50
            Максимальное количество итераций алгоритма.
        `pack_size`: int = 20
            Размер стаи волков.
        `hunt_bound`: float = 1
            Порог начала фазы охоты.
        `weight`: str = 'travel_time'
            Имя поля содержащего вес ребер, или функция позволяющая вычислять 
            вес динамически.
        `consensus_value`: float = 0.75
            Количество волков, находящихся в одном месте (жертве),
            при котором считается, что алгоритм успешно завершен.
        `iteration_end_function`: callable = None
            Функция, вызываемая в конце каждой итерации.
        '''
        if hunt_bound < 0 or hunt_bound > 2:
            raise ValueError(f"Порог охоты не может выходить за пределы диапазона [0, 2]. Сейчас {hunt_bound}")
        if consensus_value < 0 or consensus_value > 1:
            raise ValueError(f"Порог охоты не может выходить за пределы диапазона [0, 1]. Сейчас {consensus_value}")

        self.appr_val = appr_val
        self.max_iter = max_iter
        self.pack_size = pack_size
        self.hunt_bound = hunt_bound
        self.node_metric_func = NodeMetric(state_function, metric_function, appr_val, err_val=None, **kwargs)
        self.weight = weight
        self.consensus_value = consensus_value
        self.iteration_end_function = iteration_end_function
        super().__init__(state_function, metric_function, **kwargs)


    def _find_closest_node_to_three_weighted(self, env, 
                                            nodes,
                                            weight='travel_time'):
        """
        Находит узел графа env, находящийся ближе всего к трем переданным узлам (для взвешенного графа).
        
        Параметры:
        env : networkx.Graph
            Взвешенный граф, в котором происходит поиск
        nodes : list
            Список из трех узлов графа
        weight : str, optional
            Название атрибута ребра, представляющего вес (по умолчанию 'travel_time')
            
        Возвращает:
        int или str
            Идентификатор узла, сумма расстояний от которого до трех заданных узлов минимальна
        """
        if len(nodes) != 3:
            raise ValueError("Должно быть передано ровно три узла")
        
        # Проверяем, что все узлы существуют в графе
        for node in nodes:
            if node not in env.nodes():
                raise ValueError(f"Узел {node} не существует в графе")
        
        # Вычисляем кратчайшие пути от каждого из трех узлов до всех остальных
        distances = {}
        for node in nodes:
            distances[node] = nx.single_source_dijkstra_path_length(env, node, weight=weight)
        
        # Находим узел с минимальной суммой расстояний до трех заданных узлов
        min_sum = float('inf')
        closest_node = None
        
        for node in env.nodes():
            # Сумма расстояний от текущего узла до трех заданных
            distance_sum = max(distances[source_node].get(node, float('inf')) for source_node in nodes)

            # Если сумма конечная (все пути существуют) и меньше текущего минимума
            if distance_sum < min_sum:
                min_sum = distance_sum
                closest_node = node
        
        return closest_node


    def _shortest_path_within_distance(self, env, source, target, max_distance, weight='travel_time'):
        """
        Находит кратчайший путь между исходной и целевой вершинами и возвращает
        только ту часть пути, которая находится в пределах максимального расстояния от источника.
        
        Параметры:
        env : networkx.Graph
            Граф для поиска
        source : node
            Начальная вершина
        target : node
            Целевая вершина
        max_distance : float
            Максимально допустимое расстояние от начальной вершины
        weight : str, optional
            Название атрибута ребра для весов (по умолчанию 'travel_time')
            
        Возвращает:
        list или None
            Частичный путь как список вершин (от источника до самой дальней вершины в пределах расстояния),
            или None, если путь не существует
        """
        try:
            # Найти кратчайший путь между источником и целью
            full_path = nx.shortest_path(env, source=source, target=target, weight=weight)
            
            # Вычислить совокупные расстояния от источника вдоль пути
            cumulative_distances = [0]  # Distance to source is 0
            for i in range(1, len(full_path)):
                prev_node = full_path[i-1]
                curr_node = full_path[i]
                # Получить данные ребра и извлечь вес
                edge_weight = env.get_edge_data(prev_node, curr_node, 0)[weight]
                cumulative_distances.append(cumulative_distances[-1] + edge_weight)
            
            # Найти часть пути в пределах max_distance
            truncated_path = []
            for i, distance in enumerate(cumulative_distances):
                if distance <= max_distance:
                    truncated_path.append(full_path[i])
                else:
                    break
            
            # Вернуть None, если нет вершин в пределах расстояния
            return truncated_path if truncated_path else None
            
        except nx.NetworkXNoPath:
            # Путь между источником и целью не существует
            return None


    def _search(self, env, cur_node, dist_to_prey, **kwargs):
        """
            Поиск узлов в заданном радиусе от текущего узла.
        """
        g_nodes = ox.graph_to_gdfs(env, edges=False)
        if not ox.projection.is_projected(g_nodes.crs):
            g_nodes = ox.projection.project_gdf(g_nodes)
        search_area = g_nodes.loc[[cur_node]].geometry.buffer(dist_to_prey)
        local_nodes = g_nodes[g_nodes.within(search_area.iloc[0])]
        return list(local_nodes.index)

    def __call__(self, 
                env: nx.Graph,
                area=None,
                start_point: int = None,
                points_list: set = None,
                **kwargs) -> tuple[int | None, float | None]:
        """
        Основной цикл алгоритма GWO для поиска оптимального размещения пожарного подразделения.

        Args:
            env (nx.Graph): Граф улично-дорожной сети
            area (pd.Series, optional): Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
            start_point (int, optional): Не используется.
            points_list (set, optional): Множество узлов-кандидатов. Если не указано, используются все узлы графа.

        Returns:
            tuple: (best_node, best_metric) — лучший узел и значение метрики.
        """
        if not isinstance(env, nx.Graph):
            raise TypeError("Тип переменной `env` должен быть Graph!")

        # Если списка узлов изначально не передано, рассматриваются все узлы графа
        nodes_list = list(points_list) if points_list is not None else list(env.nodes())

        if len(nodes_list) < 3:
            raise ValueError("Для работы алгоритма GWO необходимо минимум 3 узла-кандидата")

        # Инициализация стаи волков (случайные узлы графа)
        pack = []
        pack_metrics = []
        alpha_metric = None
        alpha_node = None

        # Заполняем популяцию случайными узлами
        for _ in range(self.pack_size):
            # Выбираем случайный узел, для которого можно рассчитать метрику
            node_metric = None
            attempts = 0
            max_attempts = min(100, len(nodes_list))  # Ограничение на количество попыток

            while node_metric is None and attempts < max_attempts:
                node = random.choice(nodes_list)
                node_metric = self.node_metric_func(env=env, node=node, area=area, **kwargs)
                attempts += 1

            if node_metric is not None:
                pack.append(node)
                pack_metrics.append(node_metric)

        # Если не удалось инициализировать достаточное количество особей
        if len(pack) < 3:
            raise ValueError("Не удалось инициализировать популяцию с достаточным количеством допустимых узлов")

        # Основной цикл оптимизации
        for iter_num in range(self.max_iter):
            # Сортировка волков по значению метрики (лучшие первые)
            sorted_indices = sorted(range(len(pack_metrics)), 
                                  key=lambda i: pack_metrics[i])

            # Предполагаем, что лучшая метрика - минимальная (для времени прибытия)
            # Если лучшая метрика - максимальная (для индекса прикрытия), то разворачиваем список
            if self.metric_function.compare(1, 2) == 2:
                sorted_indices = sorted_indices[::-1]
            
            # Определение альфа, бета и дельта волков (три лучших решения)
            alpha_idx = sorted_indices[0]
            alpha_pos = pack[alpha_idx]
            
            beta_idx = sorted_indices[1]
            beta_pos = pack[beta_idx]
            
            delta_idx = sorted_indices[2]
            delta_pos = pack[delta_idx]

            # Определяем предполагаемую позицию жертвы
            prey_node = self._find_closest_node_to_three_weighted(
                    env = env,
                    nodes = [alpha_pos, beta_pos, delta_pos],
                    weight      = self.weight
            )

            # Обновление позиций всех волков (кроме трех лидеров)
            a = 2 - iter_num * (2 / self.max_iter)  # Коэффициент уменьшающийся от 2 до 0
            
            for i, cur_node in enumerate(pack):
                # Пропускаем трех лидеров
                if cur_node in [alpha_pos, beta_pos, delta_pos]:
                    continue
                    
                # Расчет коэффициентов A и C для каждого волка
                r1 = random.random()
                r2 = random.random()
                
                A = 2 * a * r1 - a  # Коэффициент A
                dist_to_prey = nx.shortest_path_length(env, cur_node, prey_node, weight='length') * r2

                if abs(A) > self.hunt_bound:
                    # глобальный поиск
                    candidates = self._search(
                        env = env, 
                        cur_node = cur_node, 
                        dist_to_prey = dist_to_prey,
                    )

                else:
                    # движение в сторону жертвы
                    candidates = self._shortest_path_within_distance(
                        env = env,
                        source = cur_node, 
                        target = prey_node, 
                        max_distance = dist_to_prey, 
                        weight='travel_time'
                    )

                    if candidates:
                        # разворачиваем маршрут, так, чтобы последний узел был первым
                        candidates = candidates[::-1]
                    else:
                        # Если путь найти не удалось, то выполняем глобальный поиск
                        candidates = self._search(
                            env = env, 
                            cur_node = cur_node, 
                            dist_to_prey = dist_to_prey,
                        )

                # Оценка кандидатов
                if candidates:
                    new_pos = random.choice(candidates)
                    
                    # Проверяем, что новый узел допустим
                    new_metric = self.node_metric_func(env=env, node=new_pos, area=area, **kwargs)
                    
                    # Если метрика рассчитана успешно и она лучше текущей, обновляем позицию
                    if new_metric is not None:
                        if self.metric_function.compare(pack_metrics[i], new_metric) == new_metric:
                            pack[i] = new_pos
                            pack_metrics[i] = new_metric
                            if alpha_metric is None or self.metric_function.compare(alpha_metric, new_metric) == new_metric:
                                alpha_metric = new_metric
                                alpha_node = new_pos

            # Вызов функции обработки завершения итерации
            if self.iteration_end_function:
                self.iteration_end_function(
                    value      = alpha_metric,                          # Лучшая метрика
                    abw_nodes = [alpha_pos, beta_pos, delta_pos],       # Позиция Альфы, Беты и Дельты
                    iteration  = iter_num,                              # Номер итерации
                    pack       = pack,                                  # Текущее положение волков
                    prey_node  = prey_node,                             # Текущая предположительная позиция жертвы
                    )
            
            # Если заданное количество волков нашло одну и ту же жертву, заканчиваем алгоритм
            if int(len(pack) * self.consensus_value) <= max(Counter(pack).values()):
                break
            
        # После завершения всех итераций находим лучшее решение
        best_idx = 0
        for i in range(1, len(pack_metrics)):
            if self.metric_function.compare(pack_metrics[best_idx], pack_metrics[i]) == pack_metrics[i]:
                best_idx = i
        
        return pack[best_idx], pack_metrics[best_idx]
