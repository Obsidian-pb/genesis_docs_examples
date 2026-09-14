'''
Расчет оптимальных зон обслуживания
'''

from heapq import heappush, heappop
from itertools import count
from typing import Any

import numpy as np
import pandas as pd
import geopandas as gpd
# import geopandas as gpd
import networkx as nx
import osmnx as ox

from .utils import Progressbar
from ._errors import KeyNotInMatrixError

from .core import StateBase
from .swiss_knife import DELAY_TIME, MSF
from .tools import get_duplicates_list


class FirstArrivalUnitState(StateBase):
    def __init__(self,
                 state_algorithm  = MSF,
                 weight           = 'travel_time',
                 delay            = DELAY_TIME,
                 **kwargs):
        '''
        Аргументы инициализации:

            `state_algorithm`: callable
                Функция расчета кратчайших путей от множества источников. 
                В качестве функции могут быть переданы реализации алгоритмов из пакета
                `networkx`. Например, реализация алгоритма Дейкстры: `nx.multi_source_dijkstra`.
                Пользователь может использовать собственные функции с
                интерфейсом 
                    `func(G: Graph, sources: Any, target: Any | None = None, cutoff: Any | None = None, 
                                    weight: str = "weight") -> (dict, dict)`

            `weight`:str или callable  = "travel_time"
                Имя поля содержащего вес ребер, или функция позволяющая вычислять 
                вес динамически.

            `delay`:
                Задержка в расчете. Например на обслуживание вызова на пожар.
                По умолчанию указана в swiss_knife.DELAY_TIME
        '''
        self.weight = weight
        self.delay = delay
        super().__init__(state_algorithm, **kwargs)

    def __call__(self,
                 env,
                 points,
                 area=None,
                 **kwargs):
        '''
        Аргументы вызова:

            `env`: nx.Graph
                Граф улично-дорожной сети.

            `points`: list|dict
                Стартовые узлы. Может быть списком узлов вида list(int), или словарем вида dict(int:str), где ключ - 
                идентификатор узла, значение - его наименование. Может использоваться для указания узлов в которых 
                расположены пожарные подразделения: {1234:'ПСЧ-1'}

            `area`: pd.Series = None
                Маска узлов графа. Значениями True отмечены узлы графа для которых
                требуется вернуть результат расчета. Если не указана,
                расчет производится для всех узлов графа.
        

        Возвращает:

            times, nearest -> tuple[Series[float], Series[str] | Series]. 
            times - Время прибытия первого подразделения в каждый из узлов графа. 
            nearest - Соответствие узлов первому прибывающему подразделению. 
        '''

        if not isinstance(env, nx.Graph):
            raise TypeError('Тип данных аргумента `env` должен быть nx.Graph')
        if not isinstance(points, (list, dict)):
            raise TypeError('Тип данных аргумента `points` должен быть (list или dict)')
        if isinstance(points, (list)):
            if len(points)!=len(set(points)):
                duplicates = get_duplicates_list(points)
                raise ValueError(f'Значения элементов аргумента `points` не могут повторяться! '
                                 f'Список повторяющихся значений: {duplicates}')
        if isinstance(points, (dict)):
            if len(points)!=len(set(points.values())):
                duplicates = get_duplicates_list(points.values())
                raise ValueError(f'Значения values элементов аргумента `points` не могут повторяться! '
                                 f'Список повторяющихся значений: {duplicates}')
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `list`! Имеет {type(area)}')

        # Расчет
        # ЗДЕСЬ G - ПОТЕНЦИАЛЬНАЯ ПРОБЛЕМА т.к. Мы явным образом передаем аргумент в nx.shortest_path,
        # что вызовет ошибку, в случае использования функций с иными аргументами
        # Возможно лучшим вариантом будет просто передавать **kwargs
        # Либо переопределять в каждом отдельном случае именно State, а не state_algorithm.
        # т.е. вместо FirstArrivalUnitState будет FirstArrivalUnitStateForG или FirstArrivalUnitStateForGAndBuildings ...
        times, routes = self.state_algorithm(G=env, sources = points, weight = self.weight, **kwargs)

        # # Дополнение строками узлов в которые нет возможности попасть
        # # Необходимо для корректности расчета
        # tl = set(times.keys())
        # nl = env.nodes()
        # ss = nl ^ tl
        # add_dict = {k:np.nan for k in ss}

        # times = {**times, **add_dict}
        times = pd.Series(times, dtype=float, name='times') + self.delay

        # Определение стартового узла для каждого маршрута
        if isinstance(points, dict):
            nearest = {k:points[route[0]] for k, route in routes.items()}
            # nearest = {**nearest, **add_dict}
            nearest = pd.Series(nearest, dtype=str, name='nearest')
        else:
            nearest = {k:route[0] for k, route in routes.items()}
            # nearest = {**nearest, **add_dict}
            nearest = pd.Series(nearest, dtype='int64', name='nearest')

        # Отбор узлов по маске
        if not area is None:
            times = times[area]
            nearest = nearest[area]

        return times, nearest


class FirstArrivalUnitCacheState(StateBase):
    '''
    Класс-функций расчета состояния среды с предварительно рассчитанным состоянием
    для статических узлов.

    Используются перегруженные функции оригинальных функций 
    из библиотеки networkx: `multi_source_dijkstra`, `_dijkstra_multisource`.

    Позволяет существенно сократить время моделирования при наличии статических узлов
    '''
    def __init__(self,
                 state_algorithm = MSF,
                 weight = 'travel_time',
                 delay = DELAY_TIME,
                 times_cache=None,
                 nearest_cache=None,
                 **kwargs):
        '''
            `state_algorithm`: function
                Функция расчета кратчайших путей от единственного источника. 
                В качестве функции могут быть переданы реализации алгоритмов из пакета
                `networkx`. Например, реализация алгоритма Дейкстры: `nx.multi_source_dijkstra`.
                Пользователь может использовать собственные функции с
                интерфейсом `func(G: Graph, sources: Any, target: Any | None = None, cutoff: Any | None = None, 
                                    weight: str = "weight") -> (dict, dict)`
            `weight`:str или callable  = "travel_time"
                Имя поля содержащего вес ребер, или функция позволяющая вычислять 
                вес динамически.
            `delay`: 
                Задержка в расчете. Например на обслуживание вызова на пожар.
                По умолчанию указана в swiss_knife.DELAY_TIME
            `times_cache`:
                Кэш времен прибытия в узлы
            `nearest_cache`:
                Кэш ближайших узлов
            `**kwargs`: 
                Любые другие параметры функции `state_algorithm`

        '''
        self.weight = weight
        self.delay = delay
        self.times_cache = times_cache
        self.nearest_cache = nearest_cache
        super().__init__(state_algorithm, **kwargs)

    def __call__(self,
                 env,
                 points,
                 area=None,
                 **kwargs):
        '''
        # Аргументы
        `env`: nx.Graph (G)
            Граф улично-дорожной сети.
        `points`: list|dict (source)
            Стартовые узлы. Может быть списком узлов вида list(int), или словарем вида dict(int:str), где ключ - 
            идентификатор узла, значение - его наименование. Может использоваться для указания узлов в которых 
            расположены пожарные подразделения: {1234:'ПСЧ-1'}
        `area`: pd.Series = None
            Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
        

        # Возвращает
            times, nearest -> tuple[Series[float], Series[str] | Series]. 
            times - Время прибытия первого подразделения в каждый из узлов графа. 
            nearest - Соответствие узлов первому подразделению. 
        '''

        state_func = FirstArrivalUnitState(
            self.multi_source_dijkstra_cache(dists_cache=self.times_cache.to_dict()),
            weight = self.weight,
            delay  = self.delay
            )
        times, nearest = state_func(env=env, points=points, area=area, **kwargs)

        times_total = self.times_cache.copy()
        times_total.update(times)

        nearest_total = self.nearest_cache.copy()
        nearest_total.update(nearest)

        return times_total, nearest_total

    def _dijkstra_multisource_cache(
        self, G, sources, weight, pred=None, paths=None, cutoff=None, target=None, seen=None
    ):

        G_succ = G._succ if G.is_directed() else G._adj

        push = heappush
        pop = heappop
        dist = {}  # dictionary of final distances
        if seen is None:
            seen = {}
        # fringe is heapq with 3-tuples (distance,c,node)
        # use the count c to avoid comparing nodes (may not be able to)
        c = count()
        fringe = []
        for source in sources:
            if source not in G:
                raise nx.NodeNotFound(f"Source {source} not in G")
            seen[source] = 0
            push(fringe, (0, next(c), source))
        while fringe:
            (d, _, v) = pop(fringe)
            if v in dist:
                continue  # already searched this node.
            dist[v] = d
            if v == target:
                break
            for u, e in G_succ[v].items():
                cost = weight(v, u, e)
                if cost is None:
                    continue
                vu_dist = dist[v] + cost
                if cutoff is not None:
                    if vu_dist > cutoff:
                        continue
                if u in dist:
                    u_dist = dist[u]
                    if vu_dist < u_dist:
                        raise ValueError("Contradictory paths found:", "negative weights?")
                    elif pred is not None and vu_dist == u_dist:
                        pred[u].append(v)
                elif u not in seen or vu_dist < seen[u]:
                    seen[u] = vu_dist
                    push(fringe, (vu_dist, next(c), u))
                    if paths is not None:
                        paths[u] = paths[v] + [u]
                    if pred is not None:
                        pred[u] = [v]
                elif vu_dist == seen[u]:
                    if pred is not None:
                        pred[u].append(v)

        # The optional predecessor and path dictionaries can be accessed
        # by the caller via the pred and paths objects passed as arguments.
        return dist


    def _weight_function_cache(self, G, weight):
        """Returns a function that returns the weight of an edge.

        The returned function is specifically suitable for input to
        functions :func:`_dijkstra` and :func:`_bellman_ford_relaxation`.

        Parameters
        ----------
        G : NetworkX graph.

        weight : string or function
            If it is callable, `weight` itself is returned. If it is a string,
            it is assumed to be the name of the edge attribute that represents
            the weight of an edge. In that case, a function is returned that
            gets the edge weight according to the specified edge attribute.

        Returns
        -------
        function
            This function returns a callable that accepts exactly three inputs:
            a node, an node adjacent to the first one, and the edge attribute
            dictionary for the eedge joining those nodes. That function returns
            a number representing the weight of an edge.

        If `G` is a multigraph, and `weight` is not callable, the
        minimum edge weight over all parallel edges is returned. If any edge
        does not have an attribute with key `weight`, it is assumed to
        have weight one.

        """
        if callable(weight):
            return weight
        # If the weight keyword argument is not callable, we assume it is a
        # string representing the edge attribute containing the weight of
        # the edge.
        if G.is_multigraph():
            return lambda u, v, d: min(attr.get(weight, 1) for attr in d.values())
        return lambda u, v, data: data.get(weight, 1)


    def multi_source_dijkstra_cache(self, dists_cache):
        def multi_source_dijkstra_cache_(G, sources, target=None, cutoff=None, weight="weight"):
            """Find shortest weighted paths and lengths from a given set of
            source nodes.

            Uses Dijkstra's algorithm to compute the shortest paths and lengths
            between one of the source nodes and the given `target`, or all other
            reachable nodes if not specified, for a weighted graph.

            Parameters
            ----------
            G : NetworkX graph

            sources : non-empty set of nodes
                Starting nodes for paths. If this is just a set containing a
                single node, then all paths computed by this function will start
                from that node. If there are two or more nodes in the set, the
                computed paths may begin from any one of the start nodes.

            target : node label, optional
                Ending node for path

            cutoff : integer or float, optional
                Length (sum of edge weights) at which the search is stopped.
                If cutoff is provided, only return paths with summed weight <= cutoff.

            weight : string or function
                If this is a string, then edge weights will be accessed via the
                edge attribute with this key (that is, the weight of the edge
                joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
                such edge attribute exists, the weight of the edge is assumed to
                be one.

                If this is a function, the weight of an edge is the value
                returned by the function. The function must accept exactly three
                positional arguments: the two endpoints of an edge and the
                dictionary of edge attributes for that edge. The function must
                return a number.

            Returns
            -------
            distance, path : pair of dictionaries, or numeric and list
                If target is None, returns a tuple of two dictionaries keyed by node.
                The first dictionary stores distance from one of the source nodes.
                The second stores the path from one of the sources to that node.
                If target is not None, returns a tuple of (distance, path) where
                distance is the distance from source to target and path is a list
                representing the path from source to target.

            Examples
            --------
            >>> G = nx.path_graph(5)
            >>> length, path = nx.multi_source_dijkstra(G, {0, 4})
            >>> for node in [0, 1, 2, 3, 4]:
            ...     print(f"{node}: {length[node]}")
            0: 0
            1: 1
            2: 2
            3: 1
            4: 0
            >>> path[1]
            [0, 1]
            >>> path[3]
            [4, 3]

            >>> length, path = nx.multi_source_dijkstra(G, {0, 4}, 1)
            >>> length
            1
            >>> path
            [0, 1]

            Notes
            -----
            Edge weight attributes must be numerical.
            Distances are calculated as sums of weighted edges traversed.

            The weight function can be used to hide edges by returning None.
            So ``weight = lambda u, v, d: 1 if d['color']=="red" else None``
            will find the shortest red path.

            Based on the Python cookbook recipe (119466) at
            https://code.activestate.com/recipes/119466/

            This algorithm is not guaranteed to work if edge weights
            are negative or are floating point numbers
            (overflows and roundoff errors can cause problems).

            Raises
            ------
            ValueError
                If `sources` is empty.
            NodeNotFound
                If any of `sources` is not in `G`.

            See Also
            --------
            multi_source_dijkstra_path
            multi_source_dijkstra_path_length

            """
            if not sources:
                raise ValueError("sources must not be empty")
            if target in sources:
                return (0, [target])
            weight = self._weight_function_cache(G, weight)

            paths = {source: [source] for source in sources}  # dictionary of paths
            dists = self._dijkstra_multisource_cache(
                G, sources, weight, paths=paths, cutoff=cutoff, target=target, seen=dists_cache
            )

            if target is None:
                return (dists, paths)
            try:
                return (dists[target], paths[target])
            except KeyError as e:
                raise nx.NetworkXNoPath(f"No path to {target}.") from e

        return multi_source_dijkstra_cache_



# Блок расчета по матрице прибытия
class ArrivalTimeMatrixState(StateBase):
    '''
    Класс-функция моделирования параметров прибытия на основе предварительно 
    рассчитанной матрицы времен прибытия.

    Позволяет существенно сократить время моделирования при расчете
    для известных объектов или ограниченного количества узлов графа.
    '''
    def __init__(self,
                 matrix,
                 state_algorithm = np.min,
                 delay = 0,
                 **kwargs):
        '''
        УСТАРЕВШЕЕ - не использовать! В последующих версиях будет удалено.
        
        Расчет параметров прибытия на основе предварительно 
        рассчитанной матрицы времен прибытия

        # Аргументы

            `matrix`: pd.DataFrame
                Матрица времен прибытия, отражающая время прибытия к каждому из объектов из каждого из узлов.

            `state_algorithm`: function
                Функция оценки времени прибытия из всех возможных стартовых узлов `points`.
                По умолчанию - `np.min`, что отражает время прибытия первого подразделения.

            `delay`: float, optional = None
                Задержка в расчете. Например на обслуживание вызова на пожар.
                По умолчанию указана в swiss_knife.DELAY_TIME
                **Предполагается, что уже учтена в `matrix`, поэтому не рекомендуется использовать!**
        '''
        self.matrix = matrix
        self.delay = delay
        super().__init__(state_algorithm, **kwargs)

    def __call__(self,
                 env    = None,
                 points = None,
                 area   = None,
                 **kwargs):
        '''
        Аргументы:
        
            `env`: None.
                Не используется.

            `points`: list|dict (source)
                Стартовые узлы. Может быть списком узлов вида list(int), или словарем вида dict(int:str), где ключ - 
                идентификатор узла, значение - его наименование. Может использоваться для указания узлов в которых 
                расположены пожарные подразделения: {1234:'ПСЧ-1'}

            `area`: pd.Series = None
                Маска узлов графа. Значениями True отмечены узлы графа - цели расчета.
                Потенциальные места размещения.
                Если не указана, расчет производится для всех узлов графа.

            `**kwargs`:
                Данные которые используются для расчета.

        
        Возвращает:
            times, nearest -> tuple[Series[float], Series[str] | Series]. 
            times - Время прибытия первого подразделения в каждый из узлов графа. 
            nearest - Соответствие узлов первому прибывающему подразделению.
            
        '''

        if not isinstance(points, (list, dict)):
            raise TypeError('Тип данных аргумента `points` должен быть (list или dict)')
        if isinstance(points, (list)):
            if len(points)!=len(set(points)):
                duplicates = get_duplicates_list(points)
                raise ValueError(f'Значения элементов аргумента `points` не могут повторяться! '
                                 f'Список повторяющихся значений: {duplicates}')

        # Обрезка матрицы по узлам графа
        # if not env is None:
        #     print('Обрезка матрицы по узлам графа')

        if isinstance(points, dict):
            missed_nodes = []
            for unit_node, unit_name in points.items():
                if not unit_node in self.matrix.columns:
                    missed_nodes.append(unit_node)
            if len(missed_nodes)>0:
                raise KeyNotInMatrixError(f'Узлы `{missed_nodes}` отсутствуют в матрице.')
            data = self.matrix[points.keys()]
        else:
            missed_nodes = []
            for unit_node in points:
                if not unit_node in self.matrix.columns:
                    missed_nodes.append(unit_node)
            if len(missed_nodes)>0:
                raise KeyNotInMatrixError(f'Узлы `{missed_nodes}` отсутствуют в матрице.')
            data = self.matrix[points]

        # Отброс строк в которых нет значений (недостижимых за t узлов
        data = data[data.sum(axis=1) != 0]

        # Получение времен прибытия
        times =  self.state_algorithm(data, axis=1)
        times = pd.Series(times, dtype=float, name='times') + self.delay

        # Получение названий первых прибывающих подразделений
        nearest = np.nanargmin(data, axis=1)
        if isinstance(points, dict):
            units = list(points.values())
        else:
            units = points
        nearest = {k: units[v] for k, v in zip(times.index, nearest)}
        nearest = pd.Series(nearest, dtype=str, name='nearest')

        # Отбор узлов по маске
        if not area is None:
            times = times[area]
            nearest = nearest[area]

        return times, nearest


def get_atm(G,
            data: gpd.GeoDataFrame  = None,
            data_sample_size: int   = None,
            data_node_field: str    = 'node',
            weight: str             = 'travel_time',
            cutoff: float           = None,
            data_cutoff_field: str  = None,
            delay: float            = DELAY_TIME,
            target_set: set         = None,
            print_calc_states: bool = True,
            calc_function: callable = None,
            batch_size: int         = 1000,
            ):
    '''
    Расчет матрицы времен прибытия.

    Аргументы:

        `G`: nx.MultiDiGraph
            Граф улично-дорожной сети

        `data`: pd.DataFrame = None
            Данные для расчета. Может быть указан набор данных для которых следует рассчитать,
            например, здания, или набор узлов графа.
            По умолчанию используются все узлы графа

        `data_sample_size`: int = None
            Размер выборки данных для расчета. Количество записей
            data ,которые будут учтены при расчете.
        
        `data_node_field`: str   = 'node'
            Поле в котором хранится индекс узла, для которого производится расчет.

        `weight`: str или callable  = "travel_time"
            Имя поля содержащего вес ребер, или функция позволяющая вычислять 
            вес динамически.

        `cutoff`: float = None,
            Размер расчетной области. По умолчанию производится расчет для всего графа.

        `delay`: float, optional = None
            Задержка в расчете. Например на обслуживание вызова на пожар.
            По умолчанию указана в swiss_knife.DELAY_TIME

        `target_set`: set = None
            Целевой сет узлов графа, которые рассматриваются в качестве потенциальных мест размещения.

        `print_calc_states`: bool = True
            Флаг отображения прогресса расчета.
        
        `calc_function`: callable
            Функция запускаемая на каждой из добавленных подразделений

        `batch_size`: int
            Размер батча при расчете. Позволяет избежать проблемы переполнения памяти

    Возвращает:

        `matrix`: pd.DataFrame
            Время прибытия первого подразделения в каждый из узлов графа. 
            Столбцами являются узлы графа, индексами - узлы графа, до которых производился расчет.
    '''

    # Если конкретный набор данных не передан, используем узлы графа
    if data is None:
        data = ox.graph_to_gdfs(G, edges=False)
        data[data_node_field] = data.index

    # Если указана доля набора данных, выбираем его случайным образом из data
    if not data_sample_size is None:
        data = data.sample(data_sample_size)

    # Реверсный граф
    GR = nx.reverse(G, copy=False)

    # Матрица
    matrix = pd.DataFrame()

    # перебираем все записи в наборе данных
    if print_calc_states:
        pb = Progressbar(len(data), bins=40)
    i = 0
    for bi in range(0, len(data), batch_size):
        d = {}
        data_batch = data.iloc[bi:bi+batch_size]

        for di, dt in data_batch.iterrows():
            node = dt[data_node_field]
            if not data_cutoff_field is None:
                ctf = dt[data_cutoff_field]
            else:
                ctf = cutoff - delay if not cutoff is None else None
            length = nx.single_source_dijkstra_path_length(
                GR,
                source = node,
                cutoff = ctf,
                weight = weight,
                )
            # Оставляем только узлы в которых можно разместить [предполагалось удалить позже, но почему, пока не понятно]
            if not target_set is None:
                length = {k:v for k,v in length.items() if k in target_set}
                
            d[di] = pd.Series(length) + delay
            # d[node] = pd.Series(length) + delay
            if print_calc_states:
                pb()
            if calc_function:
                value = i / len(data)
                calc_function(value = value)

            i += 1

        dft = pd.DataFrame.from_dict(d, orient='index')
        if len(matrix) == 0:
            matrix = dft
        else:
            matrix = pd.concat([matrix, dft])


    # Удаление зданий, к которым невозможно прибытия из перечня приемлемых узлов
    if not target_set is None:
        # Оставляем только узлы в которых можно разместить
        # matrix = matrix[list(set(matrix.columns) & target_set)]
        # Здания к которым невозможно прибыть после отброса
        zero_buildings = np.sum(matrix, axis=1)==0
        zero_buildings_count = sum(zero_buildings)
        if zero_buildings_count > 0:
            print(f'!{zero_buildings_count} зданий не доступны из указанных мест размещения!')
            matrix = matrix[zero_buildings == False]

    G = nx.reverse(GR, copy=False)
    return matrix
