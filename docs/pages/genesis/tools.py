'''
Инструменты обработки используемые в расетах.
'''

from typing import Any

import time
import pandas as pd
import geopandas as gpd
import logging as lg
import networkx as nx




def get_all_neighbor_nodes(G: nx.Graph, node: int) -> list:
    '''
    Возвращает полный список всех соседних узлов.
    '''
    nnodes = []
    for edge in G.out_edges(node):
        nnodes.append(edge[1])
    for edge in G.in_edges(node):
        nnodes.append(edge[0])
    return nnodes


def multi_source_dijkstra_reversed(G, sources, **kwargs) -> Any:
    '''
    Алгоритм расчета кратчайших маршрутов в точки `source`
    с использованием алгоритма Дейкстры
    в реализации `nx.multi_source_dijkstra`.
    '''
    G = nx.reverse(G.copy())
    return nx.multi_source_dijkstra(G=G, sources=sources, **kwargs)


def list_dict_concat(a:list|dict,b:list|dict) -> list|dict:
    '''
    Корректно склеивает между собой списки и словари.
    При условии, что аргумент `a` и аргумент `b` одного типа.

    #Аргументы
    `a`, `b`: list|dict
        Аргументы которые следует склеить между собой
    '''
    if isinstance(a,list) and isinstance(b,list):
        return a+b
    elif isinstance(a,dict) and isinstance(b,dict):
        return {**a, **b}
    else:
        raise TypeError(f'Аргументы имеют различный тип данных: {a:type(a)}, {b:type(b)}')

def k_v_dict(d, f=min):
    '''
    Получаем номер ключа в словаре которому соответствует значение с
    минимальным/максимальным/средним и т.д. значением, в зависимости
    от функции f.

    Самый быстрый способ.
    '''
    v=list(d.values())
    k=list(d.keys())
    return k[v.index(f(v))]

def get_dict_key(dictionary: dict, value: Any):
    '''
    Получение ключа в словаре по значению

    Аргументы
    ---------
    `dictionary`: dict
        Словарь в котором ищем значение
    `value`: Any
    '''
    for key, value_in_dict in dictionary.items():
        if value_in_dict == value:
            return key
    return None

def get_duplicates_list(seq: list):
    '''
    Получение списка дублирующихся во входящем списке значений

    Аргументы
    ---------
    `seq` (list)  
        Список значений
    '''
    duplicates = []
    unique = []
    for s in seq:
        if s not in unique:
            unique.append(s)
        else:
            duplicates.append(s)
    return duplicates
