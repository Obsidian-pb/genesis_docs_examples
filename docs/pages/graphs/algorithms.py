'''
Алгоритмы работы с графами

'''

from collections import defaultdict

import math
import random

import numpy as np
import osmnx as ox
from osmnx import utils
import networkx as nx
import geopandas as gpd
from shapely.ops import unary_union, transform
from shapely.geometry import Point, LineString, MultiLineString



def fix_highway_list(edge):
    '''
    Функция исправления типа улицы, для случая, когда тип указан как список
    '''
    if isinstance(edge, list):
        return edge[0]
    return edge


def floor_coords(geoms):
    """Округляет координаты геометрии до целых чисел в меньшую сторону"""
    def floor_coord(x, y):
        #return (math.floor(x), math.floor(y))
        return (round(x, 0), round(y, 0))
    
    return transform(floor_coord, geoms)




def graph_rise_from_gpkg(roads: gpd.GeoDataFrame,
                         oneway_field_name: str = 'oneway',
                         # lanes_field_name: str = 'lanes',  # Сейчас не реализовано
                         reversed_field_name: str = 'reversed',
                         ):
    '''
    Алгоритм собирает граф дорожной сети на основе геометрии входного векторного GeoDataFrame

    Аргументы
    ---------
    `roads`: gpd.GeoDataFrame
        Датафрейм дорог
    `oneway_field_name`: str
        Поле в котором хранятся сведения о односторонности дороги
    `reversed_field_name`: str
        Поле в котором хранятся сведения о направлении движения

    Возвращает
    ----------
    `nx.MultiDiGraph`
        Граф улично-дорожной сети
    ``
    '''
    # Проверка наличия колонок
    #missing_cols = [col for col in [oneway_field_name, reversed_field_name] if col not in roads.columns]
    #if missing_cols:
    #    raise ValueError(f"В GeoDataFrame отсутствуют необходимые колонки: {missing_cols}")

    # Запомним исходную СК
    crs = roads.crs
    
    # Спроектируем DataFrame в местную метрическую систему координат
    try:
        roads_p = ox.projection.project_gdf(roads)
    except Exception:
        roads_p = roads

    # Округляем координаты геометрии до целых чисел
    roads_p['geometry'] = roads_p['geometry'].apply(floor_coords)

    # Создаем пустой граф
    metadata = {
            "created_date": utils.ts(),
            "created_with": f"OSMnx {ox.__version__}",
            "crs": roads_p.crs
        }
    G = nx.MultiDiGraph(**metadata)

    nodes_dict = {}
    node_id = 0

    # Словарь имеющихся ребер (для учета повторов)
    existed_edges_dict = {}

    # Перебираем все строки с записями фрагментов дорог
    for _, road in roads_p.iterrows():
        geometry = road['geometry']

        # Обработка MultiLineString
        # Это нужно проверить
        if isinstance(geometry, MultiLineString):
            lines = geometry.geoms
        else:
            lines = [geometry]

        road_data = road  #[columns_list]
        road_data = {k: v[0] if isinstance(v, list) else v for k, v in road_data.items()}
        del road_data['geometry']

        # print('---общее')
        road_length = 0
        for line in lines:
            coords = list(line.coords)
            for coord1, coord2 in zip(coords[:-1], coords[1:]):
                road_length_cur = ox.distance.euclidean(y1=coord1[1], x1=coord1[0],
                                            y2=coord2[1], x2=coord2[0])
                road_length += road_length_cur
                # print(road_length, road_length_cur)
        # Предотвращение ошибки нулевой длины участка (временное решение)
        if road_length == 0:
            road_length = 1

        # Добавить интерполирование travel_time для каждой линии (А на будущее и иных данных)
        # Реализовано ниже
        travel_time = road_data.get('travel_time', 0)
        # print(travel_time)
                
        # Перебираем все линии в геометрии (может быть несколько для MultiLineString)
        for line in lines:
            coords = list(line.coords)

            # Добавляем узлы в граф
            for coord in coords:
                if coord not in nodes_dict:
                    nodes_dict[coord] = node_id
                    node_id += 1
                G.add_node(nodes_dict[coord], x=coord[0], y=coord[1])

            # Добавляем ребра в граф
            for coord1, coord2 in zip(coords[:-1], coords[1:]):
                length = ox.distance.euclidean(y1=coord1[1], x1=coord1[0],
                                              y2=coord2[1], x2=coord2[0])
                if travel_time > 0:
                    travel_time_line = travel_time * length / road_length
                    # создаем массив длинн для каждого фрагмента линии
                    road_data['travel_time'] = travel_time_line
                    # print('- ', travel_time_line, length, road_length)


                # Универсальная обработка oneway
                oneway = road_data.get(oneway_field_name, False)
                # Проверка на NaN
                if isinstance(oneway, float) and np.isnan(oneway):
                    oneway = False
                elif isinstance(oneway, str):
                    oneway = oneway.lower() in ['yes', 'true', '1']
                elif isinstance(oneway, (int, float)):
                    oneway = bool(oneway)
                road_data[oneway_field_name] = oneway

                # Универсальная обработка reversed
                reversed_val = road_data.get(reversed_field_name, False)
                if isinstance(reversed_val, float) and np.isnan(reversed_val):
                    reversed_val = False
                elif isinstance(reversed_val, str):
                    reversed_val = reversed_val.lower() in ['yes', 'true', '1']
                elif isinstance(reversed_val, (int, float)):
                    reversed_val = bool(reversed_val)
                road_data[reversed_field_name] = reversed_val

                if not oneway:
                    # Двусторонняя дорога — ребра в обе стороны
                    key = existed_edges_dict.get((nodes_dict[coord2], nodes_dict[coord1]), 0)
                    G.add_edge(nodes_dict[coord2], nodes_dict[coord1], key,
                               **road_data, length=length)
                    existed_edges_dict[(nodes_dict[coord2], nodes_dict[coord1])] = key + 1

                    key = existed_edges_dict.get((nodes_dict[coord1], nodes_dict[coord2]), 0)
                    G.add_edge(nodes_dict[coord1], nodes_dict[coord2], key,
                               **road_data, length=length)
                    existed_edges_dict[(nodes_dict[coord1], nodes_dict[coord2])] = key + 1
                else:
                    # Односторонняя дорога
                    if reversed_val:
                        # Движение в обратном направлении
                        key = existed_edges_dict.get((nodes_dict[coord2], nodes_dict[coord1]), 0)
                        G.add_edge(nodes_dict[coord2], nodes_dict[coord1], key,
                                   **road_data, length=length)
                        existed_edges_dict[(nodes_dict[coord2], nodes_dict[coord1])] = key + 1
                    else:
                        # Движение в прямом направлении
                        key = existed_edges_dict.get((nodes_dict[coord1], nodes_dict[coord2]), 0)
                        G.add_edge(nodes_dict[coord1], nodes_dict[coord2], key,
                                   **road_data, length=length)
                        existed_edges_dict[(nodes_dict[coord1], nodes_dict[coord2])] = key + 1

    # Также вызываем project_graph для перепроецирования геометрии рёбер (если она есть)
    G = ox.projection.project_graph(G, to_crs=crs)

    return G


def net_overlay(G_master:nx.MultiDiGraph, key_nodes:list, weight='length', cutoff=240):
    '''
    Наложение сети. Экспериментально! Требует проверки и уточнения!

    Выбираются ключевые узлы ГДС между которыми выстраиваются кратчайшие 
    маршруты становящиеся ребрами СГДС-- (Граф Дорожной Сети Субъектового уровня 
    максимальный вариант упрощения). 

    Аргументы
    ---------
    G_master: MultiDiGraph
        СГДС
    key_nodes: list
        список ключевых узлов которые должны стать вершинами нового графа
    weight : str
        Имя поля содержащего сведения о весе ребра (км, или время и что-то иное)

    Возвращает
    ----------
    MultiDiGraph
        Упрощенный граф дорожной сети
    '''

    # 1. Построить зоны достижимости всех подразделений с учетом взаимного влияния
    destination_areas = nx.multi_source_dijkstra_path(G_master, key_nodes, weight=weight)

    # 2. Определить смежность подразделений
    links_from=[]
    links_to=[]
    links_list=[]
    for node in destination_areas.keys():
        if len(G_master[node])>0:
            from_node = destination_areas[node][0]
            for next_node in list(G_master[node]):
                to_node=destination_areas[next_node][0]
                if from_node!=to_node:
                    if not (from_node, to_node) in links_list:
                        links_from.append(from_node)
                        links_to.append(to_node)
                        links_list.append((from_node, to_node))

    # 3. Найти кратчайшие пути между 
    links_paths = ox.shortest_path(G_master, links_from, links_to, weight=weight)

    # 4. Объединить ребра и ключевые точки в СГДС--
    G_new = _paths_to_graph(G_master, links_paths, weight=weight)

    return G_new

def _paths_to_graph(G, paths, weight='length'):
    '''
    Возвращает граф полученный из путей на основе некоторого базового графа.

    Требует доработки

    Аргументы
    ---------
    G : MultiDiGraph
        Граф дорожной сети
    paths : list(list)
        Список списков - список путей, где каждый путь 
        список последовательных узлов
    weight : str
        Имя поля содержащего сведения о весе ребра (км, или вермя и что-то иное)
    
    Возвращает
    ----------
    MultiDiGraph
        Упрощенный граф дорожной сети
    '''
    # G_new = ox.graph.nx.MultiDiGraph()
    # Создание пустого мультиграфа
    metadata = {
            "created_date": utils.ts(),
            "created_with": f"OSMnx {ox.__version__}",
            "crs": G.graph['crs']
        }
    G_new = nx.MultiDiGraph(**metadata)

    g_edges = ox.graph_to_gdfs(G, edges=True, nodes=False)
    for path in paths:
        lines=[]
        length=0
        weight_add=0
        pseudo_osmid = 0
        hws = defaultdict(lambda: 0)    # Словарь длины дорог в зависимости от типа
        try:
            for u, v in zip(path[:-1], path[1:]):
                edge_data = g_edges.loc[u,v,0]
                # if 'geometry' in G.edges[u,v,0].keys():
                if 'geometry' in edge_data.keys():
                    lines.append(edge_data['geometry'])
                    # print(edge_data['geometry'])
                if 'length' in edge_data.keys():
                    hw_val = edge_data['length']
                    length+=hw_val
                    # Добавляем длину по типу дорог
                    hw = edge_data['highway']
                    if isinstance(hw, list):
                        hw=hw[0]
                    hws[hw]+=hw_val
                if weight!='length':
                    if weight in edge_data.keys():
                        hw_val = edge_data[weight]
                        weight_add+=hw_val
                        # Добавляем длину по типу дорог
                        hw = edge_data['highway']
                        if isinstance(hw, list):
                            hw=hw[0]
                        hws[hw]+=hw_val

            G_new.add_nodes_from([ (path[0], G.nodes(data=True)[path[0]]) ])
            G_new.add_nodes_from([ (path[-1], G.nodes(data=True)[path[-1]]) ])
            data={
                'osmid': pseudo_osmid,
                'lanes': '2',
                'highway': max(hws, key=hws.get),     # 'tertiary' # до 11/01/2024: 
                'oneway': False,
                'length': length,
                'geometry': unary_union(lines)  # РАЗОБРАТЬСЯ - при перепроецировании не работает
            }
            # print(unary_union(lines))
            # print(lines)
            if weight!='length':
                data[weight]=weight_add
        
            G_new.add_edges_from([ (path[0],path[-1], data) ])

            pseudo_osmid += 1

        except TypeError:
            print("gds._paths_to_graph: проверить работоспособность путей длиной 0", path)
        
    # G_new.graph["crs"]='WGS 84'
    
    return G_new


def generate_grid_graph(n: int,
                        m: int,
                        step: float = 100.0,
                        speed: float = 50.0,
                        crs = 'EPSG:3857',
                        add_geometry: bool = True,
                        diagonals: bool = False) -> nx.MultiDiGraph:
    '''
    Генерация графа улично-дорожной сети в виде прямоугольной сетки с опцией диагональных связей.
    
    Аргументы:
        `n`: int
            Количество узлов по оси X (ширина сетки)
        
        `m`: int
            Количество узлов по оси Y (высота сетки)
        
        `step`: float = 100.0
            Расстояние между соседними узлами (в метрах)
        
        `speed`: float или tuple = 50.0
            Скорость движения по ребрам (км/ч) для расчета времени в пути.
            Может быть:
            - float: единственное значение скорости для всех ребер
            - tuple: ((speed1, speed2, ...), (prob1, prob2, ...))
              где первый кортеж содержит скорости, второй - вероятности их выбора.
              Сумма вероятностей должна быть строго равна 1.0
              Пример: ((30, 40, 50), (0.2, 0.5, 0.3))
        
        `crs`: str = 'EPSG:3857'
            Система координат
        
        `add_geometry`: bool = True
            Добавлять ли геометрию (LineString) к ребрам
        
        `diagonals`: bool = False
            Добавлять ли диагональные связи между узлами
    
    Возвращает:
        `G`: nx.MultiDiGraph
            Направленный мультиграф с узлами и ребрами в виде сетки
    '''
    
    
    # Валидация параметра speed
    if isinstance(speed, tuple):
        if len(speed) != 2:
            raise ValueError("Параметр speed в формате tuple должен содержать 2 элемента: ((speeds...), (probs...))")
        
        speeds, probs = speed
        
        if len(speeds) != len(probs):
            raise ValueError(f"Количество скоростей ({len(speeds)}) должно совпадать с количеством вероятностей ({len(probs)})")
        
        if not isinstance(speeds, (tuple, list)) or not isinstance(probs, (tuple, list)):
            raise ValueError("Скорости и вероятности должны быть tuple или list")
        
        # Проверка суммы вероятностей
        prob_sum = sum(probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(f"Сумма вероятностей должна быть равна 1.0, текущая сумма: {prob_sum}")
        
        # Проверка, что все вероятности неотрицательны
        if any(p < 0 for p in probs):
            raise ValueError("Все вероятности должны быть неотрицательными")
        
        # Проверка, что все скорости положительны
        if any(s <= 0 for s in speeds):
            raise ValueError("Все значения скорости должны быть положительными")

    # Создание пустого мультиграфа
    metadata = {
            "created_date": utils.ts(),
            "created_with": f"OSMnx {ox.__version__}",
            "crs": crs
        }
    G = nx.MultiDiGraph(**metadata)

    # Генерация узлов
    node_id = 0
    node_positions = {}  # Для хранения соответствия (i, j) -> node_id

    for i in range(n):
        for j in range(m):
            # Координаты узла
            x = i * step
            y = j * step

            # Добавление узла с атрибутами
            G.add_node(node_id, 
                      x=x, 
                      y=y,
                      pos=(x, y),
                      geometry=Point(x, y) if add_geometry else None)

            node_positions[(i, j)] = node_id
            node_id += 1
    
    # Генерация ребер между соседними узлами
    for i in range(n):
        for j in range(m):
            current_node = node_positions[(i, j)]
            
            # Горизонтальные и вертикальные связи
            # Соединяем с правым соседом (i+1, j)
            if i + 1 < n:
                right_node = node_positions[(i + 1, j)]
                add_edge_with_attributes(G, current_node, right_node, step, speed, add_geometry)
                add_edge_with_attributes(G, right_node, current_node, step, speed, add_geometry)
            
            # Соединяем с верхним соседом (i, j+1)
            if j + 1 < m:
                top_node = node_positions[(i, j + 1)]
                add_edge_with_attributes(G, current_node, top_node, step, speed, add_geometry)
                add_edge_with_attributes(G, top_node, current_node, step, speed, add_geometry)
            
            # Диагональные связи
            if diagonals:
                diagonal_length = step * math.sqrt(2)
                
                # Диагональ вправо-вверх (i+1, j+1)
                if i + 1 < n and j + 1 < m:
                    diag_node = node_positions[(i + 1, j + 1)]
                    add_edge_with_attributes(G, current_node, diag_node, diagonal_length, speed, add_geometry)
                    add_edge_with_attributes(G, diag_node, current_node, diagonal_length, speed, add_geometry)
                
                # Диагональ влево-вверх (i-1, j+1)
                if i - 1 >= 0 and j + 1 < m:
                    diag_node = node_positions[(i - 1, j + 1)]
                    add_edge_with_attributes(G, current_node, diag_node, diagonal_length, speed, add_geometry)
                    add_edge_with_attributes(G, diag_node, current_node, diagonal_length, speed, add_geometry)
    
    return G

def add_edge_with_attributes(G, u, v, length, speed, add_geometry=True):
    '''
    Добавляет ребро с необходимыми атрибутами.
    
    Аргументы:
        `G`: nx.MultiDiGraph
            Граф
        `u`, `v`: int
            Узлы начала и конца ребра
        `length`: float
            Длина ребра (в метрах)
        `speed`: float или tuple
            Скорость движения (км/ч). Может быть:
            - float: единственное значение скорости
            - tuple: ((speed1, speed2, ...), (prob1, prob2, ...))
        `add_geometry`: bool
            Добавлять ли геометрию LineString
    '''
    
    # Определение скорости для данного ребра
    if isinstance(speed, tuple):
        # Формат: ((speeds...), (probs...))
        speeds, probs = speed
        actual_speed = np.random.choice(speeds, p=probs)
    else:
        # Единственное значение скорости
        actual_speed = speed
    
    # Расчет времени в пути (в секундах)
    # length в метрах, actual_speed в км/ч
    # travel_time = (length / 1000.0) / actual_speed * 3600.0  # секунды
    travel_time = length / (actual_speed * 1000 / 60)        # минуты
    
    # Атрибуты ребра
    edge_attrs = {
        'length': length,
        'travel_time': travel_time,
        'speed_kph': actual_speed,
    }
    
    # Добавление геометрии если требуется
    # По непонятной причине - это вызывает ошибку при перепроецироании, поэтому не используем
    if add_geometry:
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        edge_attrs['geometry'] = LineString([
            (u_data['x'], u_data['y']),
            (v_data['x'], v_data['y'])
        ])
        
    
    G.add_edge(u, v, **edge_attrs)


def update_edge_geometries_from_nodes(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    '''
    Возможно нет необходимости.

    Обновляет геометрию ребер графа на основе текущих координат узлов.
    
    Эта функция полезна после перепроецирования графа (ox.project_graph()),
    когда координаты узлов обновляются, но геометрия ребер остается в старой системе координат.
    
    Аргументы:
        `G`: nx.MultiDiGraph
            Граф с узлами, имеющими атрибуты x, y
    
    Возвращает:
        `G`: nx.MultiDiGraph
            Граф с обновленной геометрией ребер
    '''
    # Проходим по всем ребрам графа
    for u, v, key, data in G.edges(keys=True, data=True):
        # Если у ребра есть геометрия, обновляем её
        if 'geometry' in data:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            
            # Получаем координаты узлов
            u_x = u_data.get('x')
            u_y = u_data.get('y')
            v_x = v_data.get('x')
            v_y = v_data.get('y')
            
            # Если координаты узлов доступны, пересоздаем геометрию
            if u_x is not None and u_y is not None and v_x is not None and v_y is not None:
                data['geometry'] = LineString([
                    (u_x, u_y),
                    (v_x, v_y)
                ])
    
    return G


def remove_random_elements(G, 
                          remove_nodes: int = 0, 
                          remove_edges: int = 0,
                          seed: int = None,
                          copy: bool = True) -> nx.MultiDiGraph:
    '''
    Случайное удаление узлов или ребер из графа.
    
    Аргументы:
        `G`: nx.MultiDiGraph
            Исходный граф
        
        `remove_nodes`: int = 0
            Количество узлов для удаления
        
        `remove_edges`: int = 0
            Количество ребер для удаления

        `seed`: int = None
            Зерно случайного выбора
        
        `copy`: bool = True
            Создавать копию графа (True) или изменять исходный (False)
    
    Возвращает:
        `G_modified`: nx.MultiDiGraph
            Граф с удаленными элементами
    '''
    
    if copy:
        G = G.copy()

    if seed is not None:
        random.seed(seed)
    
    # Удаление случайных узлов
    if remove_nodes > 0:
        nodes_list = list(G.nodes())
        nodes_to_remove = random.sample(nodes_list, min(remove_nodes, len(nodes_list)))
        G.remove_nodes_from(nodes_to_remove)
    
    # Удаление случайных ребер
    if remove_edges > 0:
        edges_list = list(G.edges(keys=True))
        edges_to_remove = random.sample(edges_list, min(remove_edges, len(edges_list)))
        for u, v, k in edges_to_remove:
            G.remove_edge(u, v, k)
    
    return G

def remove_edges_intersecting_lines(G: nx.MultiDiGraph,
                                    lines: list,
                                    copy: bool = True) -> nx.MultiDiGraph:
    '''
    Удаляет из графа все ребра, которые пересекаются с линиями (LineString) из переданного списка.
    
    Аргументы:
        `G`: nx.MultiDiGraph
            Исходный граф
        
        `lines`: list
            Список объектов LineString (из shapely.geometry) или других геометрических объектов,
            с которыми проверяется пересечение ребер графа
        
        `copy`: bool = True
            Создавать копию графа (True) или изменять исходный (False)
    
    Возвращает:
        `G_modified`: nx.MultiDiGraph
            Граф с удаленными ребрами, которые пересекаются с линиями
    '''

    
    if copy:
        G = G.copy()
    
    # Если список линий пуст, возвращаем граф без изменений
    if not lines:
        return G
    
    # Список ребер для удаления
    edges_to_remove = []
    
    # Проверяем каждое ребро графа
    for u, v, key, data in G.edges(keys=True, data=True):
        # Получаем геометрию ребра
        edge_geometry = data.get('geometry')
        
        # Если у ребра нет геометрии, пропускаем его
        if edge_geometry is None:
            continue
        
        # Проверяем пересечение с каждой линией из списка
        for line in lines:
            # Проверяем пересечение
            if edge_geometry.intersects(line):
                edges_to_remove.append((u, v, key))
                break  # Если пересечение найдено, не проверяем другие линии
    
    # Удаляем найденные ребра
    for u, v, key in edges_to_remove:
        G.remove_edge(u, v, key)
    
    return G


def remove_nodes_in_polygons(G: nx.MultiDiGraph,
                             polygons: list,
                             copy: bool = True) -> nx.MultiDiGraph:
    '''
    Удаляет из графа все узлы, которые попадают в полигоны из переданного списка.
    
    Аргументы:
        `G`: nx.MultiDiGraph
            Исходный граф
        
        `polygons`: list
            Список объектов Polygon (из shapely.geometry) или других геометрических объектов,
            в которые проверяется попадание узлов графа
        
        `copy`: bool = True
            Создавать копию графа (True) или изменять исходный (False)
    
    Возвращает:
        `G_modified`: nx.MultiDiGraph
            Граф с удаленными узлами, которые попадают в полигоны
    '''

    
    if copy:
        G = G.copy()
    
    # Если список полигонов пуст, возвращаем граф без изменений
    if not polygons:
        return G
    
    # Список узлов для удаления
    nodes_to_remove = []
    
    # Проверяем каждый узел графа
    for node, data in G.nodes(data=True):
        # Получаем геометрию узла
        node_geometry = data.get('geometry')
        
        # Если у узла нет геометрии, пытаемся создать Point из координат x, y
        if node_geometry is None:
            x = data.get('x')
            y = data.get('y')
            if x is not None and y is not None:
                node_geometry = Point(x, y)
            else:
                # Если нет ни геометрии, ни координат, пропускаем узел
                continue
        
        # Проверяем попадание в каждый полигон из списка
        for polygon in polygons:
            # Проверяем, попадает ли узел в полигон
            if polygon.contains(node_geometry) or polygon.intersects(node_geometry):
                nodes_to_remove.append(node)
                break  # Если попадание найдено, не проверяем другие полигоны
    
    # Удаляем найденные узлы (при удалении узла автоматически удаляются все связанные ребра)
    G.remove_nodes_from(nodes_to_remove)
    
    return G


def update_speeds_on_random_routes(G: nx.MultiDiGraph,
                                    num_nodes: int,
                                    max_route_length: float,
                                    weight: str = 'length',
                                    max_routes_per_node: int = 5,
                                    speed: float = 50.0,
                                    seed: int = None,
                                    copy: bool = True,
                                    selected_nodes: list = None) -> nx.MultiDiGraph:
    '''
    Находит кратчайшие маршруты между случайными узлами графа и устанавливает
    для всех ребер в каждом из маршрутов заданную скорость.
    
    Функция случайным образом выбирает узлы из графа, находит кратчайшие пути
    между ними (с учетом максимальной протяженности) и обновляет скорость для
    всех ребер, входящих в найденные маршруты.
    
    Аргументы:
        `G`: nx.MultiDiGraph
            Граф дорожной сети
        `num_nodes`: int
            Общее количество случайным образом взятых узлов графа
        `max_route_length`: float
            Максимальная протяженность маршрута (в единицах веса ребер)
        `weight`: str = 'length'
            Атрибут ребер, который следует рассматривать как вес для поиска
            кратчайшего пути (например, 'length', 'travel_time')
        `max_routes_per_node`: int = 10
            Максимально возможное количество маршрутов из одного узла
        `speed`: float = 50.0
            Скорость движения (км/ч) для установки на ребрах маршрутов
        `seed`: int = None
            Зерно для генератора случайных чисел (для воспроизводимости)
        `copy`: bool = True
            Создавать копию графа (True) или изменять исходный (False)
    
    Возвращает:
        `G`: nx.MultiDiGraph
            Граф с обновленными скоростями на ребрах найденных маршрутов
    '''
    
    if copy:
        G = G.copy()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Получаем список всех узлов графа
    all_nodes = list(G.nodes())
    
    if len(all_nodes) < num_nodes:
        num_nodes = len(all_nodes)
    
    # Случайным образом выбираем узлы
    if selected_nodes is None:
        selected_nodes = random.sample(all_nodes, num_nodes)
    
    # Словарь для подсчета количества маршрутов из каждого узла
    routes_from_node = defaultdict(int)
    
    # Множество всех ребер, которые нужно обновить
    edges_to_update = set()
    
    # Находим кратчайшие пути между случайными парами узлов
    #for i in range(len(selected_nodes)):
    #    source = selected_nodes[i]
    for source in selected_nodes:
        
        # Проверяем, не превышен ли лимит маршрутов из этого узла
        if routes_from_node[source] >= max_routes_per_node:
            continue
        
        # Выбираем случайные целевые узлы (исключая сам источник)
        possible_targets = [n for n in selected_nodes if n != source]
        
        if not possible_targets:
            continue
        
        # Случайным образом выбираем целевой узел
        # target = random.choice(possible_targets)
        
        for target in possible_targets:
            try:
                # Проверяем, не превышен ли лимит маршрутов из этого узла
                if routes_from_node[source] >= max_routes_per_node:
                    break
                    
                # Пытаемся найти кратчайший путь
                # Используем cutoff для ограничения максимальной длины пути
                path = nx.shortest_path(
                    G,
                    source=source,
                    target=target,
                    weight=weight,
                    method='dijkstra'
                )
                
                # Вычисляем длину пути
                path_length = 0
                for u, v in zip(path[:-1], path[1:]):
                    # Берем первое ребро между узлами (key=0)
                    edge_data = G[u][v][0]
                    edge_weight = edge_data.get(weight, 1.0)
                    path_length += edge_weight
                
                # Проверяем, не превышает ли путь максимальную длину
                if path_length <= max_route_length:
                    # Добавляем все ребра пути в множество для обновления
                    for u, v in zip(path[:-1], path[1:]):
                        # В MultiDiGraph может быть несколько ребер между узлами
                        # Добавляем все ключи
                        for key in G[u][v].keys():
                            edges_to_update.add((u, v, key))
                    
                    # Увеличиваем счетчик маршрутов из исходного узла
                    routes_from_node[source] += 1
            
            
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Если путь не найден, пропускаем эту пару узлов
                continue
    
    # Обновляем скорость для всех найденных ребер
    for u, v, key in edges_to_update:
        data = G[u][v][key]
        
        # Получаем длину ребра
        length = data.get('length')
        if length is None:
            # Если длины нет, вычисляем из геометрии или координат узлов
            edge_geometry = data.get('geometry')
            if edge_geometry is not None:
                length = edge_geometry.length
            else:
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
                    length = ox.distance.euclidean(
                        y1=u_data['y'], x1=u_data['x'],
                        y2=v_data['y'], x2=v_data['x']
                    )
                else:
                    # Если не можем вычислить длину, пропускаем это ребро
                    continue
        
        # Пересчитываем travel_time
        # length в метрах, speed в км/ч
        travel_time = length / (speed * 1000 / 60)  # минуты
        
        # Обновляем атрибуты ребра
        data['speed_kph'] = speed
        data['travel_time'] = travel_time
    
    return G

