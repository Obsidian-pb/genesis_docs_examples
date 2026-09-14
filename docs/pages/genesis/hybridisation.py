'''
Реализация алгоритмов гибридизации
'''






import warnings

import networkx as nx
import pandas as pd

from .core import BestPointsBase, LSCPBase, MCLPBase, MetricBase, StateBase



class Hybridization(MCLPBase):
    '''
    Реализация базового алгоритма MCLP как гибрида

    Является функцией гибридизации алгоритмов BestPoints, MCLP и LSCP,
    методами бустинга и беггинга, т.е. последовательного выполнения функций и передачи полученного 
    результата далее.
    '''
    def __init__(self,
                 state_function:        StateBase,
                 metric_function:       MetricBase,
                 functions:             list,
                 type_of_hybrid:        str        = 'Boosting',
                 function_end_function: callable   = None,
                 **kwargs):
        '''
        ## Аргументы

        `functions`: list
            Список функций для гибридизации.

        `type_of_hybrid`: str = 'Boosting'  
            Тип гибридизации. Возможные варианты: 'Boosting' или 'Stacking'.
            * 'Boosting' - метод бустинга, т.е. последовательного выполнения функций и передачи полученного 
            результата далее.
            * 'Stacking' - метод стекинга, т.е. параллельного выполнения функций и выбора лучшего результата.  

        `state_function`: StateBase = None
            Функция для расчета состояния.

        `metric_function`: MetricBase = None
            Функция для расчета основной метрики.

        `function_end_function`: callable = None
            Функция для выполнения после выполнения каждой функции.

        `**kwargs`
            Дополнительные аргументы для функций.
        '''

        # 0. Проверка корректности пришедших данных
        if not isinstance(functions, list):
            raise TypeError('Аргумент `functions` должен быть типа list!')
        if not callable(function_end_function) and function_end_function is not None:
            raise TypeError('Аргумент `function_end_function` должен быть типа callable или иметь значение None!')
        if not isinstance(state_function, StateBase) and not state_function is None:
            raise TypeError('Аргумент `state_function` должен быть типа StateBase или иметь значение None!')
        if not isinstance(metric_function, MetricBase) and not metric_function is None:
            raise TypeError('Аргумент `metric_function` должен быть типа MetricBase или иметь значение None!')
        if type_of_hybrid not in ['Boosting', 'Stacking']:
            raise ValueError('Аргумент `type_of_hybrid` должен быть типа str и иметь значение "Boosting" или "Stacking"!')

        # 1. Сохранение данных в свойствах функции
        self.functions             = functions
        self.type_of_hybrid        = type_of_hybrid
        self.function_end_function = function_end_function
        super().__init__(state_function, metric_function, **kwargs)


    def append(self, func):
        '''
        Добавить функцию расчета
        '''
        if isinstance(func, (BestPointsBase, MCLPBase, LSCPBase)):
            self.functions.append(func)
        else:
            raise TypeError('Аргумент `func` должен быть типом класса BestPoints, MCLP или LSCP!')


    def __call__(self,
                 env:           nx.Graph,
                 dynamic_nodes: dict,
                 static_nodes:  dict      = None,
                 area:          pd.Series = None,
                 **kwargs):
        '''
        ## Аргументы

        `env`:nx.Graph

            Граф улично-дорожной сети

        `dynamic_nodes`: dict

            Список стартовых узлов графа в которых размещены
            подразделения оптимальные места которых следует определить.

        `static_nodes`: dict = None

            Список стартовых узлов графа в которых размещены
            подразделения изменять размещение которых не следует.

        `area`: pd.Series = None

            Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.

        ## Возвращает

        `best_dynamic_nodes, best_metric`: tuple[list | dict, float]

            Список или словарь лучших мест размещения, значение метрики metric_function
            для лучшего размещения.
        '''
        # 0. Проверка корректности пришедших данных
        if not isinstance(env, nx.Graph):
            raise TypeError("Тип аргумента `env` должен быть Graph!")
        if not isinstance(dynamic_nodes, dict) and not dynamic_nodes is None:
            raise TypeError(f'Аргумент `dynamic_nodes` должен иметь тип `dict`! Имеет {type(area)}')
        if not isinstance(static_nodes, dict) and not static_nodes is None:
            raise TypeError(f'Аргумент `static_nodes` должен иметь тип `dict`! Имеет {type(area)}')
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `pd.Series`! Имеет {type(area)}')


        main_best_metric = None
        main_best_nodes  = None
        best_nodes = dynamic_nodes.copy()

        # 2. Последовательный перебор всех функций
        i = 0
        for m in self.functions:
            if self.type_of_hybrid == 'Boosting':
                best_nodes, cur_metric = m(env       = env,
                                       area          = area,
                                       dynamic_nodes = best_nodes,
                                       static_nodes  = static_nodes,
                                       **kwargs)
            elif self.type_of_hybrid == 'Stacking':
                best_nodes, cur_metric = m(env       = env,
                                       area          = area,
                                       dynamic_nodes = dynamic_nodes.copy(),
                                       static_nodes  = static_nodes,
                                       **kwargs)
            
            # Оценка основной метрики
            # if self.metric_function and self.state_function:
            if not static_nodes is None:
                all_nodes = {**best_nodes, **static_nodes}
            else:
                all_nodes = best_nodes
            # all_nodes  = list_dict_concat(best_nodes, static_nodes)
            times, nearest = self.state_function(env=env, points=all_nodes, **kwargs)
            cur_metric = self.metric_function(times, area=area, **kwargs)

            # Проверяем улучшилась ли основная метрика, если да, то сохраняем результат
            # print('!', main_best_metric, cur_metric)
            if main_best_metric is None:
                main_best_nodes  = best_nodes
                main_best_metric = cur_metric
            elif self.metric_function.compare(main_best_metric, cur_metric) == cur_metric:
                main_best_nodes  = best_nodes
                main_best_metric = cur_metric
            

            # Печать результатов
            if self.function_end_function:
                self.function_end_function(turn = i,
                            best_metric = main_best_metric,
                            cur_metric  = cur_metric,
                            best_nodes  = best_nodes)
            i += 1

        # 3. Возвращаем результат
        return main_best_nodes, main_best_metric






class AdapterBLP2MCLP(MCLPBase):
    '''
    Адаптер алгоритмов BLP к MCLP.

    Позволяет использовать алгоритмы BestPointsBase как MCLPBase.
    '''

    def __init__(self,
                 blp_function:    BestPointsBase,
                 state_function:  StateBase      = None,
                 metric_function: MetricBase     = None,
                 **kwargs):
        '''
        Перечень аргументов соответствует алгоритму BLP переданному через аргумент `blp_function`.

        ## Аргументы
        
        `blp_function`: BestPointsBase
            
            функция реализующая алгоритм BLP.

        `state_function`: StateBase

            функция расчета состояния окружения

        `metric_function`: MetricBase

            Функция расчета метрики

        `**kwargs`
            
            прочие аргументы определяющиеся реализацией blp_function.
        '''
        if not isinstance(blp_function, BestPointsBase):
            raise TypeError("Аргумент `blp_function` должен иметь тип `BestPointsBase`!")
        if not isinstance(state_function, StateBase) and not state_function is None:
            raise TypeError("Аргумент `state_function` должен иметь тип `StateBase`!")
        if not isinstance(metric_function, MetricBase) and not metric_function is None:
            raise TypeError("Аргумент `metric_function` должен иметь тип `MetricBase`!")

        self.blp_function = blp_function
        super().__init__(state_function, metric_function, **kwargs)


    def __call__(self,
                 env,
                 area                = None,
                 dynamic_nodes: dict = None,
                 static_nodes:  dict = None,
                 **kwargs) -> tuple[dict, int | float]:
        '''
        # Аргументы

        `env`:nx.Graph

            Граф улично-дорожной сети

        `area`: pd.Series = None

            Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.

        `dynamic_nodes`: dict

            Список стартовых узлов графа в которых размещены
            подразделения оптимальные места которых следует определить.
        
        `static_nodes`: dict = None

            Список стартовых узлов графа в которых размещены
            подразделения изменять размещение которых не следует.
            Для BLP не используется.

        `**kwargs`
            
            прочие аргументы определяющиеся реализацией blp_function.
        
        '''

        # Проверка входящих данных
        # Выполняется в оборачиваемой функции, поэтому здесь смысле не имеет - позже удалить
        if dynamic_nodes is None:
            raise ValueError("Аргумент `dynamic_nodes` должен быть словарем и не может быть равен None!")
        if len(dynamic_nodes) < 1:
            raise ValueError("Аргумент `dynamic_nodes` должен содержать не менее 1 узла!")
        if len(dynamic_nodes) > 1:
            warnings.warn(f"Аргумент `dynamic_nodes` содержит больше одного узла!"
                    f"Это приведет к отбрасыванию всех узлов кроме первого")

        # Получение первого узла из словаря
        start_point, start_key = list(dynamic_nodes.items())[0]

        # Расчет лучшего узла с использованием BLP
        best_node, best_metric = self.blp_function(
            env=env,
            area=area,
            start_point = start_point,
            points_list = dynamic_nodes,
            **kwargs)

        # Конвертация результата в результат метода MCLP
        return {best_node: start_key}, best_metric