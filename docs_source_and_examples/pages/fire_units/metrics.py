'''
Метрики параметров прибытия пожарных подразделений к зданиям
'''

import numpy as np
import pandas as pd
import geopandas as gpd

# Обработка использование как модуля QGIS
try:
    from genesis.core import MetricBase
    from genesis.metrics import ArrivalTime
except:
    from ..genesis.core import MetricBase
    from ..genesis.metrics import ArrivalTime



class CoverIndexBuilding(MetricBase):
    '''
    Класс-функция расчета индекса прибытия к зданиям
    '''
    def __init__(self,
                 buildings,
                #  f:callable = np.mean,
                 comp_func:callable = max,
                 zero_val=0,
                 ip_val=10,
                 buildings_node_id_field: str = 'node',
                #  buildings_demand_field: str = 'demand',
                 ) -> None:
        '''
        `buildings`: pd.GeoDataFrame
            Геодатафрейм со сведениями о зданиях.
        `f`: callable
            Функция расчета показателя. По-умолчанию = np.mean,
            т.е. вычисляется среднее время следования.
        `comp_func`: callable
            Функция сравнения значений метрики.
            По умолчанию - лучшей считается большая.
        `zero_val`: float = 0
            Значение которое будет возвращено в случае передачи набора данных `route_times` без элементов.
        `buildings_node_id_field`: str = 'node'
            Имя поля в `buildings` содержащего идентификаторы узлов.
        buildings_demand_field: str = 'demand'
            Имя поля в `buildings` содержащего значения спроса
        '''
        if not buildings_node_id_field in buildings.columns:
            raise KeyError(f"Поле `{buildings_node_id_field}` отсутствует в 'buildings'")

        self.buildings = buildings
        # self.f = f
        self.zero_val = zero_val
        self.buildings_node_id_field = buildings_node_id_field
        # self.buildings_demand_field = buildings_demand_field
        self.ip_val = ip_val
        super().__init__(comp_func)

    def __call__(self, state, area=None):
        '''
        `state` (`состояние`): pd.Series
            Серия времен прибытия в разные точки окружения.

        `area`: pd.Series
            Серия данных содержащих маску точек которые должны быть учтены при расчете метрики.
        '''

        if not isinstance(state, (list, pd.Series)):
            raise TypeError(
                f"Аргумент state может быть только типа list или pd.Series"
                f" Имеется {type(state)}"
                )

        # Отбор узлов по area (списку узлов которые следует учесть в расчете)
        if not area is None:
            if isinstance(state, pd.Series):
                state_c = state.loc[area]
            else:
                state_c = [x for x in state if x in area]
        else:
            state_c = state

        if len(state_c)==0:
            return self.zero_val

        # Расчет метрики по спросу
        buildings_s = self.buildings.query(f'{self.buildings_node_id_field} in @state_c.keys()')
        merged_df = pd.merge(buildings_s,
                             state_c,
                             how='left',
                             left_on=self.buildings_node_id_field,
                             right_index=True)
        # Расчет
        state_c = merged_df['times']
        if len(state_c)==0:
            return self.zero_val
        # ip_len = sum([1 for t in state_c if t<=self.ip_val])
        ip_len = sum(state_c <= self.ip_val)
        tot_len = len(state_c)

        return 100*ip_len/tot_len


        # # Расчет собственно состояния удовлетворенности спроса
        # return merged_df['times']


class ArrivalTimeBuilding(MetricBase):
    '''
    Класс-функция расчета времени прибытия к зданиям
    '''
    def __init__(self,
                 buildings,
                 f:callable = np.mean,
                 comp_func:callable = min,
                 zero_val=1000,
                 buildings_node_id_field: str = 'node',
                 ) -> None:
        '''
        `buildings`: pd.GeoDataFrame
            Геодатафрейм со сведениями о зданиях.
        `f`: callable
            Функция расчета показателя. По-умолчанию = np.mean,
            т.е. вычисляется среднее время следования.
        `comp_func`: callable
            Функция сравнения значений метрики.
            По умолчанию - лучшей считается большая.
        `zero_val`: float = 0
            Значение которое будет возвращено в случае передачи набора данных `route_times` без элементов.
        `buildings_node_id_field`: str = 'node'
            Имя поля в `buildings` содержащего идентификаторы узлов.
        `buildings_demand_field`: str = 'demand'
            Имя поля в `buildings` содержащего значения спроса
        '''
        if not buildings_node_id_field in buildings.columns:
            raise KeyError(f"Поле `{buildings_node_id_field}` отсутствует в 'buildings'")

        self.buildings = buildings
        self.f = f
        self.zero_val = zero_val
        self.buildings_node_id_field = buildings_node_id_field
        super().__init__(comp_func)

    def __call__(self, state, area=None):
        '''
        `state` (`состояние`): pd.Series
            Серия времен прибытия в разные точки окружения.

        `area`: pd.Series
            Серия данных содержащих маску точек которые должны быть учтены при расчете метрики.
        '''

        if not isinstance(state, (list, pd.Series)):
            raise TypeError(
                f"Аргумент state может быть только типа list или pd.Series"
                f" Имеется {type(state)}"
                )

        # Отбор узлов по area (списку узлов которые следует учесть в расчете)
        if not area is None:
            if isinstance(state, pd.Series):
                state_c = state.loc[area]
            else:
                state_c = [x for x in state if x in area]
        else:
            state_c = state

        if len(state_c)==0:
            return self.zero_val

        # Расчет метрики по спросу
        buildings_s = self.buildings.query(f'{self.buildings_node_id_field} in @state_c.keys()')
        merged_df = pd.merge(buildings_s,
                             state_c,
                             how='left',
                             left_on=self.buildings_node_id_field,
                             right_index=True)
        # Расчет
        state_c = merged_df['times']
        if len(state_c)==0:
            return self.zero_val

        return self.f(state_c)


# Метрики для ADD
class CoverIndexBuildingADD(MetricBase):
    '''
    !Только для верификации ADD

    Класс-функция расчета индекса прибытия к зданиям
    '''
    def __init__(self,
                 buildings,
                #  f:callable = np.mean,
                 comp_func:callable = max,
                 zero_val=0,
                 ip_val=10,
                 buildings_node_id_field: str = 'node',
                #  buildings_demand_field: str = 'demand',
                 ) -> None:
        '''
        `buildings`: pd.GeoDataFrame
            Геодатафрейм со сведениями о зданиях.
        `f`: callable
            Функция расчета показателя. По-умолчанию = np.mean,
            т.е. вычисляется среднее время следования.
        `comp_func`: callable
            Функция сравнения значений метрики.
            По умолчанию - лучшей считается большая.
        `zero_val`: float = 0
            Значение которое будет возвращено в случае передачи набора данных `route_times` без элементов.
        `buildings_node_id_field`: str = 'node'
            Имя поля в `buildings` содержащего идентификаторы узлов.
        buildings_demand_field: str = 'demand'
            Имя поля в `buildings` содержащего значения спроса
        '''
        if not buildings_node_id_field in buildings.columns:
            raise KeyError(f"Поле `{buildings_node_id_field}` отсутствует в 'buildings'")

        self.buildings = buildings
        self.zero_val = zero_val
        self.buildings_node_id_field = buildings_node_id_field
        self.ip_val = ip_val
        super().__init__(comp_func)

    def __call__(self, state, area=None):
        '''
        `state` (`состояние`): pd.Series
            Серия времен прибытия в разные точки окружения.

        `area`: pd.Series
            Серия данных содержащих маску точек которые должны быть учтены при расчете метрики.
        '''

        if not isinstance(state, (list, pd.Series)):
            raise TypeError(
                f"Аргумент state может быть только типа list или pd.Series"
                f" Имеется {type(state)}"
                )

        
        # Отбор узлов по area (списку узлов которые следует учесть в расчете)
        if not area is None:
            if isinstance(state, pd.Series):
                state_c = state.loc[area]
            else:
                state_c = [x for x in state if x in area]
        else:
            state_c = state

        if len(state_c)==0:
            return self.zero_val

        # Расчет метрики по спросу
        buildings_s = self.buildings.query(f'{self.buildings_node_id_field} in @state_c.keys()') # Вот здесь!!!
        merged_df = pd.merge(buildings_s,
                             state_c,
                             how='left',
                             left_on=self.buildings_node_id_field,
                             right_index=True)
        # Расчет
        state_c = merged_df['times']
        if len(state_c)==0:
            return self.zero_val
        ip_len = sum(state_c <= self.ip_val)

        # только для проверки ADD:
        tot_len = len(self.buildings)

        return 100*ip_len/tot_len
