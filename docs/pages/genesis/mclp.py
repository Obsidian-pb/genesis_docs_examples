'''
Реализация алгоритмов поиска оптимального размещения нескольких наилучшим образом расположенных узлов (MCLP)
'''


import random
import warnings
import math

import networkx as nx
import numpy as np
import pandas as pd

from .core import BestPointsBase, MetricBase, PointSelectorBase, StateBase, MCLPBase
from .best_points import NodeMetric
from .tools import list_dict_concat



class NodesMetric(BestPointsBase):
    '''
    Расчет метрики для набора узлов графа.
    
    ## Важно
    Применяется строго к графам!

    ОЧЕНЬ ВНИМАТЕЛЬНО ПОДУМАТЬ НАД ЭТОЙ ФУНКЦИЕЙ!
    

    '''
    def __init__(self, state_function: StateBase,
                 metric_function: MetricBase,
                #  appr_val = 0.95,
                #  err_val=None,
                 **kwargs) -> None:
        '''
            `state_function`: StateBase
                функция расчета состояния окружения
            `metric_function`: MetricBase
                Функция расчета метрики
            
            ### Устарело:
            `appr_val`: = 0.95
                Доля узлов графа, покрытие которой считается приемлемой для принятия расчетной метрики. 
                Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
                то такой узел не рассматривается.
            `err_val`: any = None
                Значение которое будет возвращено в случае если из точки `node`
                невозможно будет достичь требуемой доли узлов графа.
        '''
        # self.appr_val = appr_val
        # self.err_val = err_val
        super().__init__(state_function, metric_function, **kwargs)

    def __call__(self, env:nx.Graph, nodes:list, area=None, **kwargs):
        '''
            ## Параметры
            `env` : Graph (G)
                Граф дорожной сети
            `node`: list
                Список идентификаторов узлов графа для которого происходит расчет
            `area`: pd.Series = None
                Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
                Если не указана, расчет производится для всех узлов графа.
        '''

        # Если маска приемлемых узлов графа не передана,
        # то приемлемое количество узлов считается от количества узлов в графе
        # Иначе - от количества True в маске
        # if area is None:
        #     appr_nodes_count = int(env.number_of_nodes() * self.appr_val)
        # else:
        #     appr_nodes_count = int(sum(area) * self.appr_val)
        # # Приемлемое количество узлов считается от количества узлов в графе

        # Собственно расчет
        times, _ = self.state_function(env=env, points=nodes, area=area, **kwargs)
        cur_val = self.metric_function(times, **kwargs)
        # if len(times)>=appr_nodes_count:
        #     cur_val = self.metric_function(times, **kwargs)
        # else:
        #     cur_val = self.err_val
        return cur_val


class BestNodesKoptG(MCLPBase):
    '''
    Поиск лучших узлов для размещения n объектов.

    Расчет производится алгоритмом k-mean адаптированным для расчета
    в графовом пространстве и абстрагированным от применения метрики
    mean (среднее).

    ## Область применения
    Определение размещения множества узлов в графе, при котором 
    обеспечиваются наилучшие некоторые целевые метрики. 
    Дает достаточно точное решение. Хорошо подходит для больших графов.
    '''

    def __init__(self,
                 state_function: StateBase,
                 metric_function: MetricBase,
                 best_point_function: BestPointsBase,
                 iterations:int=5,
                 appr_val_in_area=0,
                 before_iters_start_function: callable = None,
                 iter_calc_end_function: callable = None,
                 stop_case_function: callable = None,
                 **kwargs) -> None:
        '''
        ## Аргументы

        `state_function`: StateBase

            функция расчета состояния окружения

        `metric_function`: MetricBase

            Функция расчета метрики

        `best_point_function`: BestPointsBase

            Функция расчета лучшего размещения узла

        `iterations`:int=5

            Количество итераций расчета

        `appr_val_in_area`: int=0

            Доля узлов графа в пределах area, покрытие которой считается приемлемой для принятия расчетной метрики. 
            Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
            то такой узел не рассматривается.
            При расчет размещения нескольких узлов неминуемо возникает ситуация при которой
            часть территории area будет недостижима. Поэтому по умолчанию считается

        `before_iters_start_function`: callable=None

            Функция выполняемая перед началом итеративного расчета.
            Сигнатура функции:
            ```
            before_iters_start_function(
                                    best_metric: float,
                                    dynamic_nodes: list|dict,
                                    static_nodes: list|dict
                                    )
            ```
            Если не указана, ничего не происходит.

        `iter_calc_end_function`: callable=None
        
            Функция выполняемая в конце каждой итерации.
            Сигнатура функции:
            ```
            before_iters_start_function(
                                    best_metric: float,
                                    dynamic_nodes: list|dict,
                                    static_nodes: list|dict
                                    )
            ```
            Если не указана, ничего не происходит.

        `stop_case_function`: callable = None
        
            Функция проверки достигнута ли цель расчета
        '''
        
        self.best_point_function = best_point_function
        self.iterations = iterations
        self.appr_val_in_area = appr_val_in_area
        self.before_iters_start_function = before_iters_start_function
        self.iter_calc_end_function = iter_calc_end_function
        self.stop_case_function = stop_case_function
        super().__init__(state_function, metric_function, **kwargs)


    def __call__(self,
                 env:nx.Graph,
                 dynamic_nodes: dict,
                 static_nodes: dict = None,
                 area: pd.Series = None,
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
        if not static_nodes is None:
            if not (isinstance(dynamic_nodes,dict) and isinstance(static_nodes,dict)):
                raise TypeError(f'Аргументы `dynamic_nodes` и `static_nodes` должны быть одинакового типа: dict'
                                f'Имеют: {type(dynamic_nodes)}, {type(static_nodes)}')
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `pd.Series`! Имеет {type(area)}')
        # if (len(dynamic_nodes) + len(static_nodes))<1:
        if len(dynamic_nodes) <1:
            raise ValueError(f'Количество элементов `dynamic_nodes` не может быть равно 0! Сейчас {(len(dynamic_nodes) + len(static_nodes))}')

        # Если передана область которую следует учитывать в расчете
        # установить для нее приемлемый охват равный `self.appr_val_in_area`
        # (0 по умолчанию)
        if not area is None:
            self.best_point_function.__setattr__('appr_val',
                                                 self.appr_val_in_area)

        # 1.1. Если статические узлы не указаны - заменяем значение переменной с None
        # на {}
        if static_nodes is None:
            static_nodes = {}

        # 1.2. Получение суммарного списка (или словаря) узлов для расчета состояния и метрик
        start_nodes = list_dict_concat(dynamic_nodes, static_nodes)

        # 2.1. Расчет состояния
        times, nearest = self.state_function(env=env, points=start_nodes, **kwargs)

        # 2.2. Расчет стартовой метрики состояния и определение стартового размещения
        best_metric = self.metric_function(times, area=area, **kwargs)
        best_dynamic_nodes = dynamic_nodes

        # 3. Выполняем функцию перед началом итераций
        if self.before_iters_start_function:
            self.before_iters_start_function(
                                    best_metric=best_metric,
                                    dynamic_nodes=dynamic_nodes,
                                    static_nodes=static_nodes
                                    )

        # 4. Итерации расчета лучшего размещения
        for iteration in range(self.iterations):

            # 1. Поиск для каждого из dinamic_nodes наилучшего размещения в пределах своей зоны обслуживания
            # 1.1. Для списка:
            if isinstance(dynamic_nodes, list):
                new_dynamic_nodes = []
                for dynamic_node in dynamic_nodes:

                    # Определяем список узлов в которые из узла dynamic_node можно прибыть первым
                    node_area_nodes = [n for n,v in nearest.items() if v==dynamic_node]

                    # Определяем подграф зоны обслуживания для узла dynamic_node
                    node_area_G = nx.subgraph(env, node_area_nodes).copy()

                    # Определяем лучший узел
                    best_nodes, _ = self.best_point_function(env=node_area_G, 
                                                                    start_point=dynamic_node,  # start_point
                                                                    area=area,
                                                                    **kwargs)[:2] # [:2] Это для ограничения вывода дебаг-данных в некоторых функциях
                    del node_area_G

                    if isinstance(best_nodes, list):
                        best_nodes = best_nodes[0]
                    # Добавляем полученный узел в новый список
                    new_dynamic_nodes.append(best_nodes)
            # 1.2. Для словаря:
            else:
                new_dynamic_nodes = {}
                for dynamic_node_id, dynamic_node_key in dynamic_nodes.items():

                    # Определяем список узлов в которые из узла dynamic_node можно прибыть первым
                    node_area_nodes = [k for k,v in nearest.items() if v==dynamic_node_key]

                    # Определяем подграф зоны обслуживания для узла dynamic_node
                    node_area_G = nx.subgraph(env, node_area_nodes).copy()

                    # Определяем лучший узел
                    best_nodes, _ = self.best_point_function(env=node_area_G,
                                                                    start_point=dynamic_node_id,
                                                                    area=area,
                                                                    **kwargs)[:2] # [:2] Это для ограничения вывода дебаг-данных в некоторых функциях
                    del node_area_G

                    if isinstance(best_nodes, list):
                        best_nodes = best_nodes[0]
                    # Добавляем полученный узел в новый список
                    new_dynamic_nodes[best_nodes] = dynamic_node_key

            # 2. Заменяем список dynamic_nodes списком с новыми, лучшими узлами
            dynamic_nodes = new_dynamic_nodes.copy()

            # 3. Приведение к типу переменной в соответствии с типом `dynamic_nodes`
            start_nodes = list_dict_concat(dynamic_nodes, static_nodes)

            # 4. Расчет состояния
            times, nearest = self.state_function(env=env, points=start_nodes, **kwargs)

            # 5. Расчет метрики состояния
            state_metric = self.metric_function(times, area=area, **kwargs)

            # 6. Оценка метрики состояния
            if self.metric_function.compare(best_metric, state_metric) == state_metric:
                best_metric = state_metric
                best_dynamic_nodes = dynamic_nodes

            # 7. Выполняем функцию завершения расчета для итерации
            if self.iter_calc_end_function:
                self.iter_calc_end_function(i=iteration,
                                       best_metric=best_metric,
                                       dynamic_nodes=best_dynamic_nodes,
                                       static_nodes=static_nodes)

            # 8 Если достигнута цель расчета, выходим из цикла
            if self.stop_case_function:
                if self.stop_case_function(value=best_metric,
                            iteration=iteration,
                            best_metric=best_metric,
                            dynamic_nodes=best_dynamic_nodes,
                            static_nodes=static_nodes):
                    break

        return best_dynamic_nodes, best_metric

        
class BestNodesGA(MCLPBase):
    '''
    Поиск лучших узлов для размещения n объектов.

    Расчет производится генетическим алгоритмом.

    ## Область применения
    Определение размещения множества узлов в графе, при котором 
    обеспечиваются наилучшие некоторые целевые метрики. 
    
    Дает решение высокой точности. Хорошо подходит для больших графов.
    '''

    def __init__(self,
                 state_function     :StateBase,
                 metric_function    :MetricBase,
                 node_selector      :PointSelectorBase,
                 population_size    :int = 25,
                 epochs             :int = 50,
                 mutation_rate      :float = 0.5,
                 elite_size         :int = 0,
                 appr_val_in_area   :float = 0,
                 mutation_max_count :int | float = 1,
                 bad_val_in_area    :int = 1000,
                 epoch_end_function :callable = None,
                 stop_case_function :callable = None,
                 **kwargs) -> None:
        '''
        ## Аргументы
        `state_function`: StateBase
            функция расчета состояния окружения
        `metric_function`: MetricBase
            Функция расчета метрики
        `node_selector`: PointSelectorBase
            Функция выбора узла.
        `population_size`:int = 25
            Размер популяции
        `epochs`: int = 50
            Количество эпох расчета
        `mutation_rate`: float = 0.1
            Вероятность единичной мутации
        `elite_size`: int =  0
            Количество особей в элитной группе.
            Если указан 0, то механизм элитизма не задействуется.
        `appr_val_in_area`: int=0
            Доля узлов графа в пределах area, покрытие которой считается приемлемой для принятия расчетной метрики. 
            Если при расчете метрик, из стартового узла (узлов) достижимо меньшее количество узлов,
            то такой узел не рассматривается.
            При расчет размещения нескольких узлов неминуемо возникает ситуация при которой
            часть территории area будет недостижима. Поэтому по умолчанию считается
        `mutation_max_count`: int = 1
            Максимальное количество единиц мутации.
            Если указано десятичное число от 0 до 1, то используется доля 
            от общего количества хромосом (подразделений)
        `bad_val_in_area`: int=1000
            Значение указываемое для узла, в случае если
            из него нельзя попасть в `appr_val_in_area` долю узлов в пределах
            `area`.
        `epoch_end_function`: callable = None
            Функция вызываемая после каждой эпохи расчета.
        `stop_case_function`: callable = None
            Функция проверки достигнута ли цель расчета
        '''
        if population_size <= 5:
            raise ValueError('Размер популяции должен быть > 0, так же не рекомендуется использовать размер популяции < 5')
        if epochs  <=  0:
            raise ValueError('Количество эпох должно быть > 0')
        if mutation_max_count  <=  0:
            raise ValueError('Максимальное количество единиц мутации должно быть > 0')
        if mutation_rate <0 or mutation_rate > 1:
            raise ValueError('Вероятность мутации должна лежать в диапазоне (0,1)')
        if appr_val_in_area <0 or appr_val_in_area > 1:
            raise ValueError('Доля узлов графа в пределах area, должна лежать в диапазоне (0,1)')
        if elite_size < 0:
            raise ValueError('Количество особей в элитной группе должно быть >= 0')
        
        self.population_size = population_size
        self.epochs = epochs
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.appr_val_in_area = appr_val_in_area
        self.mutation_max_count = mutation_max_count
        self.bad_val_in_area = bad_val_in_area
        self.epoch_end_function = epoch_end_function
        self.stop_case_function = stop_case_function
        self.node_selector = node_selector
        super().__init__(state_function, metric_function, **kwargs)

    def _fit_function(self, env, nodes, area=None, **kwargs):
        '''
        Расчет функции приспособленности
        '''
        # 2.1. Расчет состояния
        times, _ = self.state_function(env=env, points=nodes, area=area, **kwargs)

        if not area is None:
            appr_nodes_count = area.sum() * self.appr_val_in_area
            if len(times) < appr_nodes_count:
                print('Расстановка не обеспечивает требуемую степень прикрытия территории area')
                return self.bad_val_in_area

        # 2.2. Расчет стартовой метрики состояния и определение стартового размещения
        best_metric = self.metric_function(times, **kwargs)

        return best_metric

    def _mutate(self, env, new_dynamic_nodes, area):
        '''
        Мутация особи
        '''
        if self.mutation_max_count >= 1:
            mutation_count = self.mutation_max_count
        elif self.mutation_max_count > 0 and self.mutation_max_count < 1:
            mutation_count = int(len(new_dynamic_nodes) * self.mutation_max_count)
        else:
            mutation_count = len(new_dynamic_nodes)
        if mutation_count == 0:
            mutation_count = 1
        for _ in range(mutation_count):
            if random.random() < self.mutation_rate:
                # Выбор случайного элемента в словаре и удаление его из new_dynamic_nodes
                node, unit = random.choice(list(new_dynamic_nodes.items()))
                del new_dynamic_nodes[node]

                # Поиск нового узла, которого при этом нет в new_dynamic_nodes
                node = self.node_selector(env=env, points=new_dynamic_nodes, area=area)
                new_dynamic_nodes[node] = unit

        return  new_dynamic_nodes

    def __call__(self,
                 env:nx.Graph,
                 dynamic_nodes: dict,
                 static_nodes: dict = None,
                 area: pd.Series = None,
                 **kwargs):
        '''
        Запуск работы генетического алгоритма

        ## Аргументы
        `env`:nx.Graph
            Граф улично-дорожной сети
        `dynamic_nodes`: list|dict
            Список стартовых узлов графа в которых размещены
            подразделения оптимальные места которых следует определить.
        `static_nodes`: list|dict = None
            Список стартовых узлов графа в которых размещены
            подразделения изменять размещение которых не следует.
        `area`: pd.Series = None
            Маска узлов графа. Значениями True отмечены узлы графа - цели расчета леса Вороного.
            Если не указана, расчет производится для всех узлов графа.
        '''

        # 0. Проверка корректности пришедших данных
        if not isinstance(env, nx.Graph):
            raise TypeError("Тип аргумента `env` должен быть `Graph`!")
        if not isinstance(dynamic_nodes, dict):
            raise TypeError("Тип аргумента `dynamic_nodes` должен быть `dict`!")
        if not static_nodes is None and not isinstance(static_nodes, dict):
            raise TypeError("Тип аргумента `static_nodes` должен быть `dict`!")
        if len(dynamic_nodes) < 2:
            raise ValueError(f'Количество элементов `dynamic_nodes` не может быть меньше 2. Сейчас {len(dynamic_nodes)}')
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `pd.Series`! Имеет {type(area)}')


        # 1.1. Если статические узлы не указаны - заменяем значение переменной с None
        # на {}
        if static_nodes is None:
            static_nodes = {}

        # Последовательность узлов графа (для последующего обращения к нему)
        # g_nodes = list(env.nodes())

        # ===================================== Генетический алгоритм ==============
        # 1. Создание стартовой популяции
        population = [dynamic_nodes for _ in range(self.population_size)]
        # # Обязательно сохраняется один экземпляр исходного бота!
        # population = [self._mutate(env, dynamic_nodes, area) for _ in range(self.population_size - 1)] + [dynamic_nodes]

        # Оценка приспособленности всех особей
        bot_fit = [self._fit_function(env=env,
                                      nodes=list_dict_concat(dn, static_nodes),
                                      area=area,
                                      **kwargs) for dn in population]
        if self.metric_function.compare(1,2) == 1:        # Для минимизации:
            max_val  = max(bot_fit)
            weights = [1.1*max_val-x for x in bot_fit]
            best_metric = min(bot_fit)
        else:                                           # Для максимизации:
            weights = bot_fit
            best_metric = max(bot_fit)
        best_bot = population[bot_fit.index(best_metric)]

        # 2. На каждой эпохе
        for epoch in range(self.epochs):

            # 2.0 Элитарность
            if self.elite_size>0:
                # Определение весов
                pop_weight = pd.DataFrame({'w': weights, 'p': population})
                pop_weight = pop_weight.sort_values('w', ascending=False)
                pop_weight = pop_weight.iloc[:self.elite_size]
                population = pop_weight['p'].to_list()
                weights = pop_weight['w'].to_list()

                new_population = population.copy()
            else:
                new_population = []

            # 2.1 Генерация новой популяции
            for _ in range(self.population_size - self.elite_size):

                # Отбор по правилу рулетки
                parent_bot_1 = random.choices(population, weights=weights)[0]
                parent_bot_2 = random.choices(population, weights=weights)[0]

                # Скрещивание (одноточечное)
                split_point = int(len(parent_bot_1)/2)
                left_gen_vals = list(parent_bot_1.values())[:split_point]
                right_gen_vals  = list(parent_bot_1.values())[split_point:]
                left_genome_part = {k:v for k, v in parent_bot_1.items() if v in left_gen_vals}
                right_genome_part = {k:v for k, v in parent_bot_2.items() if v in right_gen_vals}
                new_dynamic_nodes = {**left_genome_part, **right_genome_part}

                # Мутация (выбор произвольного узла)
                new_dynamic_nodes = self._mutate(env=env,
                                                 new_dynamic_nodes=new_dynamic_nodes,
                                                 area=area)

                # Добавляем его в новую популяцию
                new_population.append(new_dynamic_nodes)

            # 2.2 Заменяем предыдущую популяцию новой
            population = new_population

            # 2.3 Оценка приспособленности всех особей
            bot_fit = [self._fit_function(env = env,
                                      nodes   = list_dict_concat(dn, static_nodes),
                                      area    = area,
                                      **kwargs) for dn in population]

            # 2.4 Определение лучшего на эпохе значения метрики
            # и весов в зависимости от приспособленности        
            if self.metric_function.compare(1,2)==1:    # Для минимизации:
                max_val  = max(bot_fit)
                weights = [1.1*max_val-x for x in bot_fit]
                cur_metric = min(bot_fit)
            else:                                       # Для максимизации
                weights = bot_fit
                cur_metric = max(bot_fit)
            
            # 2.5 Нормализация весов (для более выраженной точности расчета)
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

            # 2.6 Определяем лучше ли лучшее на эпохе решение чем имеющееся
            if self.metric_function.compare(best_metric, cur_metric) == cur_metric:
                best_bot = population[bot_fit.index(cur_metric)]
                best_metric  = cur_metric

            # 2.7 Выполнение функции окончания расчета на эпохе
            if self.epoch_end_function:
                self.epoch_end_function(epoch=epoch,
                                        best_metric=best_metric,
                                        cur_metric=cur_metric,
                                        best_bot=best_bot)

            # 2.8 Если достигнута цель расчета, выходим из цикла
            if self.stop_case_function:
                if self.stop_case_function(value=best_metric,
                            iteration=epoch,
                            best_metric=best_metric,
                            dynamic_nodes=best_bot,
                            static_nodes=static_nodes):
                    break

        return best_bot, best_metric



class BestNodesSA(MCLPBase):
    '''
    Поиск лучших узлов для размещения n объектов.

    Расчет производится алгоритмом имитации отжига.

    ## Область применения
    Определение размещения множества узлов в графе, при котором 
    обеспечиваются наилучшие некоторые целевые метрики. 
    
    ### Достоинства
    Дает решение высокой точности. Подходит для средних графов.
    
    ### Недостатки
    Долго считает. 
    '''
    def __init__(self, state_function: StateBase,
                 metric_function: MetricBase,
                 node_selector:PointSelectorBase,
                 initial_temperature: float = 1,
                 end_temperature: float = 0.0001,
                 turns: int = 1000000,
                 mutation_max_count: int = 1,
                 appr_val_in_area: float = 0,
                 bad_val_in_area: int|None = 1000,
                 turn_end_function: callable = None,
                 best_state_find_function: callable = None,
                 stop_case_function: callable = None,
                 **kwargs) -> None:
        '''
        ## Аргументы
        `state_function`: StateBase
            функция расчета состояния окружения
        `metric_function`: MetricBase
            Функция расчета метрики
        `initial_temperature`: float  = 1
            Начальная температура
        `end_temperature`: float = 0.0001
            Конечная температура
        `mutation_max_count`: int = 1
            Максимальное количество единиц мутации.
        `appr_val_in_area`: int=0
            Доля узлов графа в пределах area, покрытие которой 
            считается приемлемой для принятия расчетной метрики. 
            Если при расчете метрик, из стартового узла (узлов) достижимо меньшее 
            количество узлов, то такой узел не рассматривается.
            При расчет размещения нескольких узлов неминуемо возникает ситуация при которой
            часть территории area будет недостижима. Поэтому по умолчанию считается
        `bad_val_in_area`: int=1000
            Значение указываемое для узла, в случае если
            из него нельзя попасть в `appr_val_in_area` долю узлов в пределах
            `area`.
        `turn_end_function`: callable  = None
            Функция вызываемая на каждой итерации расчета.
        `best_state_find_function`: callable = None
            Функция вызываемая после каждого улучшения абсолютного оптимума.
        '''

        if initial_temperature  <  0:
            raise ValueError('Начальная температура должна быть >= 0')
        if end_temperature  <  0:
            raise ValueError('Конечная температура должна быть >= 0')
        if end_temperature  >  initial_temperature:
            raise ValueError('Конечная температура должна быть <= начальной')
        if mutation_max_count  <=  0:
            raise ValueError('Максимальное количество единиц мутации должно быть > 0')
        if appr_val_in_area < 0 or appr_val_in_area > 1:
            raise ValueError('Доля узлов графа в пределах area, должна лежать в диапазоне (0,1)')

        if turn_end_function:
            try:
                turn_end_function(turn = None,
                                        T = None,
                                        best_metric = None,
                                        cur_metric = None,
                                        best_bot = None)
            except TypeError:
                print('Сигнатура функции `turn_end_function` должна быть:')
                print('`turn_end_function(turn:int, T:float, best_metric:float, cur_metric:float, best_bot:dict)`')
        
        if best_state_find_function:
            try:
                best_state_find_function(turn = None,
                                    T = None,
                                    best_metric = None,
                                    cur_metric = None,
                                    best_bot = None)
            except TypeError:
                print('Сигнатура функции `best_state_find_function` должна быть:')
                print('`best_state_find_function(turn:int, T:float, best_metric:float, cur_metric:float, best_bot:dict)`')

        self.initial_temperature = initial_temperature
        self.end_temperature = end_temperature
        self.node_selector = node_selector
        self.turns = turns
        self.appr_val_in_area = appr_val_in_area
        self.mutation_max_count = mutation_max_count
        self.bad_val_in_area = bad_val_in_area
        self.turn_end_function = turn_end_function
        self.best_state_find_function = best_state_find_function
        self.stop_case_function = stop_case_function
        super().__init__(state_function, metric_function, **kwargs)


    def _decrease_temperature(self, T0, t):
        '''
        Функция изменения температуры

        ## Аргументы
        `T0` - начальная температура

        `t` - текущая итерация

        ## Возвращает
        Температура на итерации t
        '''
        return T0/(1+t)

    def _get_transition_probability(self, dE, T):
        '''
        Функция расчета вероятности перехода из состояния i в состояние j

        ## Аргументы
        `dE` - энергия перехода

        `T` - температура

        ## Возвращает
        Вероятность перехода
        '''
        try:
            return math.exp(-dE/T)
        except OverflowError as _:
            # print(dE, T)
            return 0
            # raise OverflowError(dE, T)

    def _calculate_energy(self, env: nx.Graph, nodes: list, area=None, **kwargs):
        '''
        Расчет энергии состояния.
        Здесь это значение целевой метрики.

        ## Аргументы
        `env`: nx.Graph
            окружение
        `nodes`: list
            список стартовых узлов. Например мест размещения пожарных депо
        `area`: pd.Series
            маска доступности узлов
        `**kwargs` - дополнительные аргументы
        '''
        # 1. Расчет состояния
        times, _ = self.state_function(env=env, points=nodes, area=area, **kwargs)

        if not area is None:
            appr_nodes_count = area.sum() * self.appr_val_in_area
            if len(times) < appr_nodes_count:
                print('Расстановка не обеспечивает требуемую степень прикрытия территории area')
                return self.bad_val_in_area

        # 2. Расчет стартовой метрики состояния и определение стартового размещения
        best_metric = self.metric_function(times, **kwargs)

        return best_metric

    def _generate_state_candidate(self, env, state, area):
                              #  (self, env, new_dynamic_nodes, g_nodes, area):
        '''
        Функция порождает новое состояние
        '''
        new_state = state.copy()

        # Здесь вся магия
        for _ in range(self.mutation_max_count):
            # Выбор случайного элемента в словаре и удаление его из new_state
            node, unit = random.choice(list(new_state.items()))
            del new_state[node]

            # Поиск нового узла, которого при этом нет в new_state
            node = self.node_selector(env=env, points=new_state, area=area)
            # node = random.choice(g_nodes)
            # while node in new_state.keys():
            #     node = random.choice(g_nodes)

            new_state[node] = unit

        # Возвращаем новое состояние
        return new_state

    def _is_transition(self, probability):
        value = random.random()
        return value <= probability

    def __call__(self,
                 env,
                 dynamic_nodes: dict,
                 static_nodes: dict = None,
                 area=None,
                 **kwargs) -> tuple[dict, int | None]:
        '''
        Запуск работы алгоритма имитации отжига

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
        '''

        # 0. Проверяем пришедшие данные
        if not isinstance(env, nx.Graph):
            raise TypeError("Тип аргумента `env` должен быть `Graph`!")
        if not isinstance(dynamic_nodes, dict):
            raise TypeError("Тип аргумента `dynamic_nodes` должен быть `dict`!")
        if not static_nodes is None and not isinstance(static_nodes, dict):
            raise TypeError("Тип аргумента `static_nodes` должен быть `dict`!")
        if len(dynamic_nodes) < 1:
            raise ValueError(f'Количество элементов `dynamic_nodes` не может быть меньше 1. Сейчас {len(dynamic_nodes)}')
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `pd.Series`! Имеет {type(area)}')

        if static_nodes is None:
            static_nodes = {}
        g_nodes = list(env.nodes())

        # 1. Рассчитываем исходное состояние и энергию, устанавливаем начальную температуру
        best_state = dynamic_nodes
        current_state = dynamic_nodes

        current_energy = self._calculate_energy(env=env,
                                      nodes=list_dict_concat(current_state, static_nodes),
                                      area=area,
                                      **kwargs)
        best_energy = current_energy
        T = self.initial_temperature

        # 2. Выполняем расчет
        for i in range(self.turns):
            state_candidate  = self._generate_state_candidate(env=env,
                                                             state=current_state,
                                                             area=area)
            candidate_energy = self._calculate_energy(env=env,
                                nodes=list_dict_concat(state_candidate, static_nodes),
                                area=area,
                                **kwargs)

            if current_energy != candidate_energy and \
                    self.metric_function.compare(current_energy, candidate_energy) == candidate_energy:
                current_energy = candidate_energy
                current_state = state_candidate

                if best_energy != current_energy and \
                    self.metric_function.compare(best_energy, current_energy) == current_energy: # Здесь проверку на тип оптимизации (мин/макс)
                    best_energy = current_energy
                    best_state = state_candidate

                    # Выполнение функции окончания расчета на эпохе
                    if self.best_state_find_function:
                        self.best_state_find_function(turn=i,
                                                T = T,
                                                best_metric=best_energy,
                                                cur_metric=candidate_energy,
                                                best_bot=best_state)
            else:
                p = self._get_transition_probability(candidate_energy-current_energy, T)
                if self._is_transition(p):
                    current_energy = candidate_energy
                    current_state = state_candidate

            T = self._decrease_temperature(self.initial_temperature, i)
            if T < self.end_temperature:
                return best_state, best_energy

            # Выполнение функции окончания расчета на эпохе
            if self.turn_end_function:
                self.turn_end_function(turn = i,
                                        T = T,
                                        best_metric = best_energy,
                                        cur_metric = candidate_energy,
                                        best_bot = best_state)
                
            # 2.8 Если достигнута цель расчета, выходим из цикла
            if self.stop_case_function:
                if self.stop_case_function(value=best_energy,
                            iteration=i,
                            best_metric=best_energy,
                            dynamic_nodes=best_state,
                            static_nodes=static_nodes):
                    break

        return best_state, best_energy



