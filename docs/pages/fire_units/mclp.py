'''
Решение MCLP применительно к пожарным подразделениям
'''




import random
import warnings

import numpy as np
import networkx as nx
import pandas as pd

try:
    from ..genesis.core import MCLPBase, PointSelectorBase, StateBase, MetricBase
    from ..genesis.mclp import BestNodesKoptG
    from ..genesis.tools import list_dict_concat
except:
    from genesis.core import MCLPBase, PointSelectorBase, StateBase, MetricBase
    from genesis.mclp import BestNodesKoptG
    from genesis.tools import list_dict_concat



class BestNodesGAKopt(MCLPBase):
    '''
    Поиск лучших узлов для размещения n объектов.

    Расчет производится генетическим алгоритмом.
    При этом приспособленность каждой особи дополнительно
    улучшается за счет поиска наилучшего размещения динамических узлов
    при помощи алгоритма Kopt.

    ## Область применения
    Определение размещения множества узлов в графе, при котором 
    обеспечиваются наилучшие некоторые целевые метрики. 
    
    Дает решение высокой точности. Хорошо подходит для больших графов.
    '''

    def __init__(self,
                 state_function: StateBase,
                 metric_function: MetricBase,
                 kopt_function: BestNodesKoptG,
                 node_selector:PointSelectorBase,
                 population_size:int = 25,
                 epochs:int = 50,
                 mutation_rate:float = 0.5,
                 elite_size:int = 0,
                 appr_val_in_area:float = 0,
                 mutation_max_count: int | float = 1,
                 bad_val_in_area:int = 1000, # &! Возможна ошибка при возрастающих метриках...
                 epoch_end_function:callable = None,
                 stop_case_function:callable = None,
                 **kwargs) -> None:
        '''
        ## Аргументы
        `state_function`: StateBase
            функция расчета состояния окружения
        `metric_function`: MetricBase
            Функция расчета метрики
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
        `mutation_max_count`: int | float = 1
            Максимальное количество единиц мутации. Если указано в диапазоне от 0 до 1,
            то она будет рассчитываться относительно размера популяции.
        `bad_val_in_area`: int=1000
            Значение указываемое для узла, в случае если
            из него нельзя попасть в `appr_val_in_area` долю узлов в пределах
            `area`.
        `epoch_end_function`: callable = None
            Функция вызываемая после каждой эпохи расчета.
        '''
        if population_size <= 5:
            raise ValueError('Размер популяции должен быть > 0, так же не рекомендуется использовать размер популяции < 5')
        if epochs  <=  0:
            raise ValueError('Количество эпох должно быть > 0')
        # if mutation_max_count  <=  0:
        #     raise ValueError('Максимальное количество единиц мутации должно быть > 0')
        if mutation_rate <0 or mutation_rate > 1:
            raise ValueError('Вероятность мутации должна лежать в диапазоне (0,1)')
        if appr_val_in_area <0 or appr_val_in_area > 1:
            raise ValueError('Доля узлов графа в пределах area, должна лежать в диапазоне (0,1)')
        if elite_size < 0:
            raise ValueError('Количество особей в элитной группе должно быть >= 0')

        self.kopt_function = kopt_function
        self.population_size = population_size
        self.node_selector = node_selector
        self.epochs = epochs
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.appr_val_in_area = appr_val_in_area
        self.mutation_max_count = mutation_max_count
        self.bad_val_in_area = bad_val_in_area
        self.epoch_end_function = epoch_end_function
        self.stop_case_function = stop_case_function
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
        g_nodes = list(env.nodes())

        # ===================================== Генетический алгоритм ==============
        # 1. Создание стартовой популяции
        population = [dynamic_nodes for _ in range(self.population_size)]
        # Обязательно сохраняется один экземпляр исходного бота!
        # population = [self._mutate(env, dynamic_nodes, area) for _ in range(self.population_size - 1)] + [dynamic_nodes]

        # Оценка приспособленности всех особей
        bot_fit = [self._fit_function(env=env,
                                      nodes=list_dict_concat(dn, static_nodes),
                                      area=area,
                                      **kwargs) for dn in population]
        if self.metric_function.compare(1,2)==1:        # Для минимизации:
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
                # print(len(population), len(weights))

                # Отбор по правилу рулетки
                try:
                    parent_bot_1 = random.choices(population, weights=weights)[0]
                    parent_bot_2 = random.choices(population, weights=weights)[0]
                except:
                    print(weights)
                    print(sum(weights))
                    raise Warning('Здесь заменить на лакончиный код')
                    raise Exception('***')

                # Скрещивание (одноточечное)
                split_point = int(len(parent_bot_1)/2)
                left_gen_vals = list(parent_bot_1.values())[:split_point]
                right_gen_vals  = list(parent_bot_1.values())[split_point:]
                left_genome_part = {k:v for k, v in parent_bot_1.items() if v in left_gen_vals}
                right_genome_part = {k:v for k, v in parent_bot_2.items() if v in right_gen_vals}
                new_dynamic_nodes = {**left_genome_part, **right_genome_part}

                # Мутация (выбор произвольного узла)
                # new_dynamic_nodes = self._mutate(new_dynamic_nodes, g_nodes)
                new_dynamic_nodes = self._mutate(env=env,
                                                 new_dynamic_nodes=new_dynamic_nodes,
                                                 area=area)

                # Добавляем его в новую популяцию
                new_population.append(new_dynamic_nodes)

            # 2.2 Заменяем предыдущую популяцию новой
            population = new_population

            # 2.3 Оценка приспособленности всех особей
            bot_fit = [self._fit_function(env=env,
                                      nodes=list_dict_concat(dn, static_nodes),
                                      area=area,
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
                    # return best_bot, best_metric
                    break
            
        # Применение чистовой обработки с использованием локального поиска
        # warnings.warn('Здесь нужно пересмотреть расчет метрики - сейчас рассчитывается по метрике вложенной функции, что не правильно!')
        best_bot, best_metric = self.kopt_function(
            env = env,
            dynamic_nodes = best_bot,
            static_nodes = static_nodes,
            area = area,
        )

        return best_bot, best_metric
    
