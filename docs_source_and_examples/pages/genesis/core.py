'''
Классы ядра
'''

from typing import Any
import numpy as np


class MetricBase:
    '''
    Базовый класс для Класс-функций расчета метрики по набору данных.
    '''
    # self.comp_func=min


    def __init__(self, comp_func=min, **kwargs):
        '''

        `comp_func`: function
            Функция по которой происходит сравнение двух метрик методом compare.
            По умолчанию выбирается минимальная.

        `**kwargs`:
            Основные настройки объекта.

            В наиболее общем случае - функция np.mean,
            которая будет отвечать за расчет значения

            Пользователь может указать собственную функцию.
        '''
        self.comp_func=comp_func

    def __call__(self, state, area=None, **kwargs):
        '''
        `state` (`состояние`): abstract
            Абстрактный объект состояния среды.
            Например, словарь времен прибытия в разные точки.

        `area` (`область`): abstract
            Абстрактный объект конкретизирующий часть среды,
            состояние которой должно быть учтено при расчете метрики. 

        `**kwargs`:
            Прочие данные которые используются для вычислений.
            Например, частные настройки алгоритма, дополнительные данные и т.д.

        # Возвращает
        `metric` (`значение метрики`): float
            значение метрики
        '''
        return None

    def compare(self, a, b):
        '''
        Сравнение.       
        Возвращает значение в соответствии с логикой расчетов.
        '''
        if a==b:
            return b        # Требуется доп. проверка: Не понятно как это будет себя вести с другими алгоритмами.
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return self.comp_func(a, b)


class StateBase():
    '''
    Базовый класс для класс-функций расчета состояния среды.
    '''
    def __init__(self, state_algorithm, **kwargs):
        '''
            `state_algorithm` (`алгоритм`): abstract
                алгоритм (или алгоритмы) используемые для проведения расчетов состояния окружения.
                Например функции nx.multi_source_dijkstra
        '''
        self.state_algorithm = state_algorithm

    def __call__(self, env, points, area=None, **kwargs):
        '''
        `**kwargs`:
            Данные которые используются для расчета.
            
            Согласно правилам genesis, должны быть переданы следующие данные:
            
            `env` (`среда`): abstract
                данные среды используемые для проведения расчетов.
                Граф, растр или иное представление пространства, в том числе композитное.

            `points`: list|dict (source)
                Стартовые точки. Может быть списком узлов вида list(int), или словарем вида dict(int:str), где ключ - 
                идентификатор узла, значение - его наименование. 
                Точки должны быть соотносимы с окружением `env`.

            `area` (`область оценки`): abstract = None
                область для которой происходит оценивание метрики.
                ни какие точки являющиеся частью среды `env`, но расположенные
                вне `area` учтены при оценке метрики не будут.

            Конкретный тип данных и их именование в аргументах осуществляются 
            непосредственно при реализации
        
        # Возвращает
        `state` : abstract
            Состояние среды. Зависит от конкретной реализации
        '''
        return None


class BestPointsBase:
    '''
    Базовый класс алгоритма расчета оптимальной точки
    '''
    def __init__(self, 
                 state_function: StateBase, 
                 metric_function: MetricBase, 
                 **kwargs) -> None:
        '''
        `state_function`: StateBase
            функция расчета состояния окружения
        `metric_function`: MetricBase
            Функция расчета метрики
        '''
        self.state_function = state_function
        self.metric_function = metric_function

    def __call__(self,
                 env,
                 area=None,
                 start_point: int = None,
                 points_list:  set = None,
                 **kwargs) -> tuple[None, None]:
        '''
        # Аргументы
        `env` (`среда`): abstract
            Данные среды используемые для проведения расчетов.
            Граф, растр или иное представление пространства, в том числе композитное.

        `area` (`область оценки`): abstract = None
            Область для которой происходит оценивание метрики.
            ни какие точки являющиеся частью среды `env`, но расположенные
            вне `area` учтены при оценке метрики не будут.
        `start_point`: int
            Идентификатор стартовой точки
        `points_list`: set = None
            Множество точек среды которые будут рассмотрены в качестве кандидатов.
            Если не указан, то будут рассмотрены все узлы графа.
        
        # Возвращает
        `point` : int
            идентификатор оптимальной точки
        `metric` (`значение метрики`): float
            значение метрики
        '''


class MCLPBase:
    '''
    Базовый класс алгоритма расчета лучшего размещения точек
    '''
    def __init__(self, state_function: StateBase, metric_function: MetricBase,  **kwargs) -> None:
        '''
        `state_function`: StateBase
            функция расчета состояния окружения
        `metric_function`: MetricBase
            Функция расчета метрики
        '''
        self.state_function = state_function
        self.metric_function = metric_function

    def __call__(self, env, area=None, **kwargs):
        '''
        `env` (`среда`): abstract
            данные среды используемые для проведения расчетов.
            Граф, растр или иное представление пространства, в том числе композитное.

        `area` (`область оценки`): abstract = None
            область для которой происходит оценивание метрики.
            ни какие точки являющиеся частью среды `env`, но расположенные
            вне `area` учтены при оценке метрики не будут.
        '''


class StopCaseBase:
    '''
    Базовый класс для Класс-функций оценки достижения критерия остановки.
    '''
    def __init__(self, **kwargs):
        pass

    def __call__(self, value, **kwargs):
        '''
        Возвращает True если критерий достигнут, иначе False.

        `value`
            Передаваемое функции значение, по которому будет оцениваться
            достигнут критерий остановки или нет.
        `**kwargs`
            Прочие (дополнительные или альтернативные) значения которые
            будут использовать для оценивания
        '''
        return False


class PointSelectorBase:
    '''
    Класс-функция выбора новой точки (кандидата).

    В соответствии с правилами реализованными в логике класса осуществляется выбор
    точки в среде.
    '''
    def __init__(self, **kwargs):
        pass

    def __call__(self, **kwargs):
        '''
       `**kwargs`:
            Данные которые используются для расчета.
            
            Согласно правилам genesis, должны быть переданы следующие данные:
            
            `env` (`среда`): abstract
                данные среды используемые для проведения расчетов.
                Граф, растр или иное представление пространства, в том числе композитное.

            `area` (`область оценки`): abstract = None
                область для которой происходит оценивание метрики.
                ни какие точки являющиеся частью среды `env`, но расположенные
                вне `area` учтены при оценке метрики не будут.
            `points`: dict
                список точек, которые будут рассматриваться в качестве имеющихся размещений.
        '''
        return None






class LSCPBase:
    '''
    Базовый класс алгоритма расчета лучшей размещения точек
    '''
    def __init__(self,
                 mclp_function: MCLPBase,
                 point_selector: PointSelectorBase,
                 stop_case_function: StopCaseBase,
                 metric_function: MetricBase,
                 names_pattern: str = '{}',
                 start_names_index: int = 1,
                 **kwargs) -> None:
        '''
        `mclp_function`: StateBase
            функция расчета лучшего размещения точек
        `point_selector`: PointSelectorBase
            Функция выбора новой точки (кандидата).
        `stop_case_function`: MetricBase
            Функция оценки достижения критерия остановки
        `metric_function`: MetricBase
            Функция расчет метрики для оптимизации LSCP
        `names_pattern`: str
            Шаблон имен новых точек. По умолчанию names_pattern: `str = '{}'`.
            Применяется как `names_pattern.format(iteration)`, где 
            `iteration` - номер итерации.
        `start_names_index`: int
            Начальный индекс имени новых точек.
        '''
        self.mclp_function = mclp_function
        self.point_selector = point_selector
        self.stop_case_function = stop_case_function
        self.metric_function = metric_function
        self.names_pattern = names_pattern
        self.start_names_index = start_names_index

    def __call__(self, env, area=None, **kwargs):
        '''
        `env` (`среда`): abstract
            данные среды используемые для проведения расчетов.
            Граф, растр или иное представление пространства, в том числе композитное.

        `area` (`область оценки`): abstract = None
            область для которой происходит оценивание метрики.
            ни какие точки являющиеся частью среды `env`, но расположенные
            вне `area` учтены при оценке метрики не будут.
        '''
