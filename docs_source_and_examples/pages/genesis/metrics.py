'''
Estimated Arrival Parameters Problem - Задача определения ожидаемых параметров реагирования пожарных подразделений

Метрики прибытия пожарных подразделений
'''

import numpy as np
import pandas as pd
import geopandas as gpd

from .core import MetricBase







class ArrivalTime(MetricBase):
    '''
    Класс-функция расчета метрик времени первого подразделения.
    '''
    def __init__(self, f=np.mean, zero_val=0) -> None:
        '''
        `f`: function
            Функция расчета показателя. По-умолчанию = np.mean,
            т.е. вычисляется среднее время следования.
        `zero_val`: float = 0
            Значение которое будет возвращено в случае передачи набора данных `route_times` без элементов.
        '''
        self.f = f
        self.zero_val = zero_val
        super().__init__()

    def __call__(self, state, area=None):
        '''
        `state` (`состояние`): pd.Series
            Серия времен прибытия в разные точки окружения.

        `area`: pd.Series
            Серия данных содержащих маску точек которые должны быть учтены при расчете метрики.
        '''

        if not isinstance(state, (list, pd.Series)):
            raise TypeError(
                f"Аргумент times может быть только типа list или pd.Series"
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
        return self.f(state_c)


class CoverIndex(MetricBase):
    '''
    Класс-функция расчета индекса прикрытия территорий

    ВНИМАНИЕ: Данная метрика рассчитывается только для тех узлов в которые можно попасть.
    Для расчета исходя из количества всех узлов графа следует использовать fire_units.CoverIndexBuilding
    А позже metrics.CoverIndexPoints
    '''
    def __init__(self, ip_val:int=10, zero_val:float=0, comp_func:callable=max) -> None:
        '''
        `ip_val`:int
            Пороговое значение для определения индекса прикрытия.
            Рекомендуется использовать 10 для городских населенных пунктов и 
            20 для сельских.
        `zero_val`: float = 0
            Значение которое будет возвращено в случае передачи набора данных 
            `route_times` без элементов.
        `comp_func`: callable
            Функция сравнения значений метрики
        '''
        self.ip_val = ip_val
        self.zero_val = zero_val
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
                f"Аргумент times может быть только типа list или pd.Series"
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

        # Расчет
        if len(state_c)==0:
            return self.zero_val
        ip_len = sum([1 for t in state_c if t <= self.ip_val])
        tot_len = len(state_c)


        return 100 * ip_len / tot_len





class CoverIndexValue(MetricBase):
    '''
    Класс-функция расчета взвешенного индекса прикрытия.
    Подходит для задач, где необходимо учитывать влияние некоторых
    дополнительных факторов. Например площади зданий, 
    численности населения, потребности в защите и т.д.
    '''
    def __init__(self,
                 data: gpd.GeoDataFrame,
                #  f:callable = np.mean,
                 value_field: str,
                 comp_func:callable = max,
                 zero_val = 0,
                 ip_val = 10,
                 data_node_id_field: str = 'node',
                 ) -> None:
        '''
        `data`: pd.GeoDataFrame
            Геодатафрейм со сведениями о зданиях.
        `f`: callable
            Функция расчета показателя. По-умолчанию = np.mean,
            т.е. вычисляется среднее время следования.
        `comp_func`: callable
            Функция сравнения значений метрики.
            По умолчанию - лучшей считается большая.
        `zero_val`: float = 0
            Значение которое будет возвращено в случае передачи набора данных `route_times` без элементов.
        `value_field`: str = None
            Имя поля в `data` содержащего вес (например численность населения).
        `data_node_id_field`: str = 'node'
            Имя поля в `data` содержащего идентификаторы узлов.
        '''
        if not value_field in data.columns:
            raise KeyError(f"Поле `{value_field}` отсутствует в 'buildings'")

        self.data = data
        self.value_field = value_field
        self.zero_val = zero_val
        self.ip_val = ip_val
        self.data_node_id_field = data_node_id_field
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

        if len(state_c) == 0:
            return self.zero_val

        # Объединение данных об узлах и временах прибытия в каждый из них
        data_s = self.data.query(f'{self.data_node_id_field} in @state_c.keys()')
        merged_df = pd.merge(data_s,
        # merged_df = pd.merge(self.data,
                             state_c,
                             how='left',
                             left_on = self.data_node_id_field,
                             right_index = True)

        # Расчет полного веса по всему набору данных
        total_value = merged_df[self.value_field].sum()

        # Оставляем только строки для узлов в которые время прибытия меньше или равно 10 минут
        merged_df = merged_df[merged_df['times'] <= self.ip_val]


        # Если таких узлов нет - возвращаем значение для ноля
        if len(merged_df) == 0:
            return self.zero_val


        return 100 * merged_df[self.value_field].sum() / total_value



class _CoverIndexComplex(MetricBase):
    '''
    Составная метрика. 
    Не использовать! В настоящий момент не реализована.
    '''
    def __init__(self, metrics:list, **kwargs):
        self.metrics = metrics
        super().__init__(**kwargs)

    def __call__(self, state, area=None, **kwargs):
        result = []
        for m in self.metrics:
            result.append(m(state, area, **kwargs))
        return result
    
    def compare(self, a, b):
        if a==b:
            return b        # Требуется доп. проверка: Не понятно как это будет себя вести с другими алгоритмами.
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        
        # result = []
        for m, ma, mb in zip(self.metrics, a, b):
            if ma == mb:
                continue
            if m.compare(ma, mb) == ma:
                return a
            else:
                return b
        # print('all equal')
        return b