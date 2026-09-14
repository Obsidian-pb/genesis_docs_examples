'''
Реализация алгоритмов поиска оптимального размещения
заранее неизвестного количества
наилучшим образом расположенных узлов (LSCP)
'''

import warnings

import networkx as nx
import numpy as np
import pandas as pd

from .core import LSCPBase, BestPointsBase, MCLPBase, MetricBase, PointSelectorBase, StateBase, StopCaseBase
from .mclp import NodesMetric
from .tools import list_dict_concat


class LSCPCommon(LSCPBase):
    '''
    Наиболее общий модульный алгоритм расчета количества и оптимального 
    размещения узлов для достижения целевой метрики.

    ## Область применения
        Определение размещения требуемого количества подразделений пожарной охраны
        исходя из цели расчета. 
    '''
    def __init__(self,
                 mclp_function: MCLPBase,
                 point_selector: PointSelectorBase,
                 stop_case_function: StopCaseBase,
                 metric_function: MetricBase,
                 names_pattern: str = '{}',
                 start_names_index: int = 1,
                 after_mclp_function: callable = None,
                 **kwargs):
        '''
        ## Аргументы

        `mclp_function`: MCLPBase
            функция решения задачи MCLP
        `point_selector`: PointSelectorBase
            функция выбора узла
        `stop_case_function`: StopCaseBase
            функция проверки достигнутой цели расчета
        `metric_function`: MetricBase
            Функция расчета метрики
        `names_pattern`: str = '{}'
            Шаблон имен подразделений.
        `start_names_index`: int = 1
            Стартовый номер подразделений.
        `after_mclp_function`: callable = None
            функция выполняемая после выполнения каждой итерации алгоритма.
            Сигнатура функции:

                after_mclp_function(iteration,
                            best_metric,
                            current_metric,
                            dynamic_nodes,
                            static_nodes)
        '''
        if not hasattr(mclp_function, 'state_function'):
            raise TypeError('Аргумент `mclp_function` не является MCLPBase! Возможно передаваемая функция обернута или декорирована.')
        self.after_mclp_function = after_mclp_function
        super().__init__(mclp_function, point_selector, stop_case_function, metric_function, names_pattern, start_names_index, **kwargs)

    def __call__(self,
                 env:nx.Graph,
                 dynamic_nodes: dict,
                 static_nodes: dict = None,
                 area: pd.Series = None,
                 nodes_list:set=None,
                 **kwargs):
        '''
        Запуск работы алгоритма

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
        `nodes_list`:set=None
            Множество узлов графа которые будут рассмотрены в качестве кандидатов.
            Если не указан, то будут рассмотрены все узлы графа.
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
      
        # 0.1. Если статические узлы не указаны - заменяем значение переменной с None на {}
        if static_nodes is None:
            static_nodes = {}
        if len(static_nodes) + len(dynamic_nodes) == 0:
            raise ValueError('Суммарное количество элементов `static_nodes` и `dynamic_nodes` не может быть равно 0! ' + \
                            'В случае если в `env` отсутствуют известные размещения `points`, ' + \
                            'используйте методы класса `BestPointsBase`')

        # Итерации
        best_dynamic_nodes = dynamic_nodes
        iteration = 0
        name_index = self.start_names_index
        while True:
            # 1. Расчет оптимального размещения подразделений
            best_dynamic_nodes, best_metric = self.mclp_function(env=env,
                                            dynamic_nodes=best_dynamic_nodes,
                                            static_nodes=static_nodes,
                                            area=area,
                                            **kwargs)
            # 2. Расчет текущей метрики
            if static_nodes is None:
                all_nodes = best_dynamic_nodes
            else:
                all_nodes = list_dict_concat(best_dynamic_nodes, static_nodes)
            current_metric = NodesMetric(self.mclp_function.state_function,
                                         self.metric_function
                                         )(env,
                                           list(all_nodes.keys()),
                                           area=area)
            # 3. Печать отчета расчета
            if not self.after_mclp_function is None:
                self.after_mclp_function(iteration=iteration,
                            best_metric=best_metric,
                            current_metric=current_metric,
                            dynamic_nodes=best_dynamic_nodes,
                            static_nodes=static_nodes)

            # 4. Если достигнута цель расчета, выходим из цикла
            if self.stop_case_function(value=current_metric,
                            iteration=iteration,
                            best_metric=best_metric,
                            current_metric=current_metric,
                            dynamic_nodes=best_dynamic_nodes,
                            static_nodes=static_nodes):
                return best_dynamic_nodes, current_metric

            # ============================================================================================
            # 2. Если нет - создаем новое подразделение
            # 2.1. Получение суммарного списка (или словаря) узлов для расчета состояния и метрик
            start_nodes = list_dict_concat(best_dynamic_nodes, static_nodes)

            # 2.2. Выбор нового узла в соответствии с переданной логикой
            new_node = self.point_selector(env=env, points=start_nodes, area=area)
            # print(new_node)

            # 2.3. Добавление нового узла в словарь динамических узлов:
            while self.names_pattern.format(name_index) in all_nodes.values():
                name_index += 1
            best_dynamic_nodes[new_node] = self.names_pattern.format(name_index)
            # print(best_dynamic_nodes)

            # ============================================================================================
            iteration += 1
            name_index += 1

        return best_dynamic_nodes, best_metric


class LSCP_ADD(LSCPBase):
    '''
    Решение задачи LSCP алгоритмом жадного добавления.
    '''
    def __init__(self,
                 state_function:      StateBase,
                 matrix:              pd.DataFrame,
                 metric_function:     MetricBase,
                 ip_val:              int = 10,
                 mclp_function:       MCLPBase = None,
                 point_selector:      PointSelectorBase = None,
                 stop_case_function:  StopCaseBase = None,
                 names_pattern:       str = '{}',
                 start_names_index:   int = 1,
                 target_weights:      pd.Series = None,
                 after_mclp_function: callable = None,
                 check_for_best_time: bool = True,
                 **kwargs):
        '''
        
        # Аргументы

        `state_function`: StateBase

            Функция расчета состояния парметров реагирования.

        `matrix`: pd.DataFrame

            Матрица времен прибытия, отражающая время прибытия к каждому из объектов из каждого из узлов.

        `metric_function`: MetricBase

            Функция расчета метрики оптимальности размещения подразделений.
        
        `ip_val`:int

            Пороговое значение для определения индекса прикрытия.
            Рекомендуется использовать 10 для городских населенных пунктов и 
            20 для сельских.

        `mclp_function`: MCLPBase = None,

            Реализация алгоритма решения задачи MCLP.

        `point_selector`: PointSelectorBase = None

            Реализация алгоритма выбора узлов.

        `stop_case_function``: callable = None,

            Функция условия остановки. Если не указана, расчет производится 
            до тех пор пока не будут удалены все строки в matrix.

        `names_pattern`: str = '{}'

            Шаблон имен подразделений.
        
        `start_names_index`: int = 1

            Стартовый номер подразделений.
        
        `after_mclp_function`: callable = None

            Функция выполняемая после выполнения каждой итерации алгоритма.
            
            Сигнатура функции:

                after_mclp_function(iteration,
                            best_metric,
                            current_metric,
                            dynamic_nodes,
                            static_nodes)
        
        `check_for_best_time`: bool = True

            Нужно ли проверять выбор оптимального узла по минимальному времени.
            Если True, то при выборе оптимального узла будет проверяться 
            Нужно для отладки и исследования. Потом будет удалено.
        
        '''

        # Сохраняем матрицу прибытия в пределах максимального времени прибытия
        self.matrix = matrix
        self.ip_val = ip_val

        if not target_weights is None:
            warnings.warn("Учет веса целей в настоящее время не протестирован!", UserWarning)
        self.target_weights = target_weights

        self.state_function      = state_function
        self.after_mclp_function = after_mclp_function
        self.check_for_best_time = check_for_best_time
        super().__init__(mclp_function, point_selector, stop_case_function, metric_function, names_pattern, start_names_index, **kwargs)

    def __call__(self,
                 env:nx.Graph = None,
                 dynamic_nodes: dict = None,
                 static_nodes: dict = None,
                 area: pd.Series = None,
                 nodes_list:set=None,
                 **kwargs):
        '''
        Запуск работы алгоритма

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

        `nodes_list`:set=None
        
            Множество узлов графа которые будут рассмотрены в качестве кандидатов.
            Если не указан, то будут рассмотрены все узлы графа.

        ## Возвращает

        `best_dynamic_nodes, best_metric`

            Словарь лучших размещений ({узел: имя}), лучшая метрика.
        '''

        # 0. Проверка корректности пришедших данных
        if not static_nodes is None:
            if not isinstance(static_nodes,dict):
                raise TypeError(f'Аргумент `static_nodes` должен быть типа dict'
                                f'Имеет: {type(static_nodes)}')
        if not dynamic_nodes is None:
            if not isinstance(dynamic_nodes,dict):
                raise TypeError(f'Аргумент `dynamic_nodes` должен быть типа dict'
                                f'Имеет: {type(dynamic_nodes)}')
        if not area is None and not isinstance(area, pd.Series):
            raise TypeError(f'Аргумент `area` должен иметь тип `pd.Series`! Имеет {type(area)}')

        # Если передан список возможных для размещения узлов
        if not nodes_list is None:
            self.matrix = self.matrix[nodes_list]

        # Создание копии матрицы для использования в алгоритме
        matrix_temp = self.matrix.copy()

        # Установка для матрицы признака прикрытой исходя из
        # максимально допустимого времени прибытия
        matrix_temp = matrix_temp <= self.ip_val #- self.delay

        # Если следует учесть веса объектов, умножаем матрицу на них
        if not self.target_weights is None:
            warnings.warn("Учет веса целей в настоящее время не протестирован! Возможно следует использовать matrix_temp", UserWarning)
            weights = self.target_weights.loc[matrix_temp.index]
            matrix_temp = matrix_temp.mul(weights, axis=0)


        # 0.1. Если статические узлы не указаны - заменяем значение переменной с None на {}
        if static_nodes is None:
            static_nodes = {}
        else:
            # Отброс узлов прикрытых имеющимися подразделениями
            # Отключил т.к. подразумевается, что учет влияния существующих подразделений 
            # будет выполнен ранее - на этапе подготовки данных 
            # (отброс прикрытых объектов до расчета матрицы прибытия)
            try:
                # node_column = matrix_temp[static_nodes.keys()]
                # matrix_temp = matrix_temp[np.any(node_column, axis=1) == False]
                node_cols = list(static_nodes.keys())
                node_column = matrix_temp[node_cols]
                mask = ~node_column.any(axis=1)
                matrix_temp = matrix_temp[mask]
            except KeyError:
                warnings.warn('Влияние подразделений уже было учтено на этапе подготовки данных')

            # 0.2. Если статические узлы были переданы, проверяем, следует ли проводить расчет
            # возможно условие расчета уже было достигнуто
            # Расчет метрики
            current_metric = NodesMetric(self.state_function,
                                        self.metric_function
                                        )(env,
                                        list(static_nodes.keys()),
                                        area=area)
            if self.stop_case_function(value = current_metric,
                        iteration      = 0,
                        best_metric    = current_metric,
                        current_metric = current_metric,
                        dynamic_nodes  = {},
                        static_nodes   = static_nodes,
                        matrix_        = matrix_temp,
                        ):
                if not self.after_mclp_function is None:
                    self.after_mclp_function(
                            iteration      = 0,
                            best_metric    = current_metric,
                            current_metric = current_metric,
                            dynamic_nodes  = {},
                            static_nodes   = static_nodes,
                            matrix_        = matrix_temp,
                             )
                return {}, current_metric
        
        # 0.3. Если динамические узлы не указаны - заменяем значение переменной с None на {}
        if dynamic_nodes is None:
            best_dynamic_nodes = {}
        else:
            best_dynamic_nodes = dynamic_nodes


        # Итерации
        iteration = 0
        name_index = self.start_names_index
        while len(matrix_temp) > 0:
           
            # ! Проверить !
            # Если статические узлы не были переданы, выполняем расчет
            if static_nodes is None:
                all_nodes = best_dynamic_nodes
            else:
                all_nodes = list_dict_concat(best_dynamic_nodes, static_nodes)
                
            # if len(all_nodes) == 0:
            #     # 1. Расчет оптимального размещения подразделений
            #     node_id = matrix_temp.columns[np.argmax(np.sum(matrix_temp, axis=0))]
            #     while self.names_pattern.format(name_index) in best_dynamic_nodes.values():
            #         name_index += 1
            #     best_dynamic_nodes[node_id] = self.names_pattern.format(name_index)
            #     node_column = matrix_temp[node_id]
            #     # name_index += 1

            #     # 2. Отброс прикрытых узлов
            #     matrix_temp = matrix_temp[node_column == False]

            # =============== Здесь проверить корректно ли удаляются ===================
            # 1. Расчет оптимального размещения подразделений
            if self.check_for_best_time:
                max_value = np.max(np.sum(matrix_temp, axis=0))
                max_indices = [i for i, val in enumerate(np.sum(matrix_temp, axis=0)) if val == max_value]
                # Для случая если есть только один узел с максимальным значением
                if len(max_indices) == 1:
                    node_id = matrix_temp.columns[max_indices[0]]
                # Для случая если есть несколько узлов с максимальным значением (имеет место неопределенность размещения)
                else:
                    # Ситуация когда узлы с максимальным значениям покрывают различные здания 
                    # (если брать по всем в таком случае, то оптимальное размещение будет находиться посередине зон,
                    # что некорректно)
                    # 1.1. Выбираем первый узел
                    node_id_single = matrix_temp.columns[max_indices[0]]
                    # 1.2. выбираем какие здания прикрыты из этого узла
                    covered_buildings = matrix_temp[node_id_single]
                    # 1.3. Оставляем в max_indices только те узлы которые обеспечивают прикрытие зданий из covered_buildings
                    # Здесь мы обращаемся к столбцу по его порядковому номеру (индексу), а не по имени
                    max_indices = [mi for mi in max_indices if matrix_temp.iloc[:, mi].equals(covered_buildings)]

                    # Непосредственно выбираем узел с минимальным максимальным временем прибытия
                    tmp_matrix_cols = matrix_temp.columns[max_indices]
                    tmp_matrix = self.matrix[tmp_matrix_cols]
                    # Здесь неопределенность выбранной метрики: np.max(tmp_matrix, axis=0) или np.mean(tmp_matrix, axis=0)
                    # node_id = tmp_matrix.columns[np.argmin(np.max(tmp_matrix, axis=0), axis=0)]
                    node_id = tmp_matrix.columns[np.argmin(np.mean(tmp_matrix, axis=0), axis=0)]
                    # print(node_id, tmp_matrix.shape)
                    # Здесь корректный код! (потом удалить):
                    # tmp_matrix_cols = matrix_temp.columns[max_indices]
                    # tmp_matrix = matrix[tmp_matrix_cols]
                    # npm = np.max(tmp_matrix, axis=0)
                    # node_id = tmp_matrix.columns[np.argmin(npm)]
                    # node_id
            else:
                node_id = matrix_temp.columns[np.argmax(np.sum(matrix_temp, axis=0))]


            while self.names_pattern.format(name_index) in all_nodes.values():
                    name_index += 1
            best_dynamic_nodes[node_id] = self.names_pattern.format(name_index)
            node_column = matrix_temp[node_id]

            # 2. Отброс прикрытых узлов
            matrix_temp = matrix_temp[node_column == False]
            # ==========================================================================
            
            # 3. Расчет текущей метрики
            # Объединяем статические и динамические узлы
            if static_nodes is None:
                all_nodes = best_dynamic_nodes
            else:
                all_nodes = list_dict_concat(best_dynamic_nodes, static_nodes)
            # Расчет метрики
            current_metric = NodesMetric(self.state_function,
                                         self.metric_function
                                         )(env,
                                           list(all_nodes.keys()),
                                           area=area)
            # times, _ = self.state_function(env = env, points = all_nodes)
            # current_metric = self.metric_function(times)
            # 4. Печать отчета расчета
            if not self.after_mclp_function is None:
                self.after_mclp_function(
                            iteration      = iteration,
                            best_metric    = current_metric,
                            current_metric = current_metric,
                            dynamic_nodes  = best_dynamic_nodes,
                            static_nodes   = static_nodes,
                            matrix_        = matrix_temp,
                            )

            # 5. Если достигнута цель расчета, выходим из цикла
            if not self.stop_case_function is None:
                if self.stop_case_function(
                                value          = current_metric,
                                iteration      = iteration,
                                best_metric    = current_metric,
                                current_metric = current_metric,
                                dynamic_nodes  = best_dynamic_nodes,
                                static_nodes   = static_nodes,
                                matrix_        = matrix_temp,):
                    return best_dynamic_nodes, current_metric

            # # ========================================================
            # # 1. Расчет оптимального размещения подразделений
            # node_id = matrix_temp.columns[np.argmax(np.sum(matrix_temp, axis=0))]
            # while self.names_pattern.format(name_index) in best_dynamic_nodes.values():
            #         name_index += 1
            # best_dynamic_nodes[node_id] = self.names_pattern.format(name_index)
            # node_column = matrix_temp[node_id]

            # # 2. Отброс прикрытых узлов
            # matrix_temp = matrix_temp[node_column == False]

            # # ========================================================
            iteration += 1
            # name_index += 1

        return best_dynamic_nodes, current_metric




def drop_trash_points(env,
                    state_function: StateBase,
                    metric_function: MetricBase,
                    stop_case_function: StopCaseBase,
                    dynamic_nodes,
                    static_nodes=None,
                    area=None,
                    ):
    '''
    Функция отбора «мусорных» (лишних) точек размещения методом жадного удаления.

    На каждом шаге поочередно пробует удалить каждую точку из dynamic_nodes и вычисляет метрику для нового набора.
    Если после удаления метрика удовлетворяет критерию останова (stop_case_function), точка окончательно удаляется.
    Процесс повторяется, пока не останется одна точка или не будет достигнут критерий останова.

    Аргументы
    ---------
    `env` : object
        Окружение или данные, необходимые для вычисления метрики.
    `state_function` : StateBase
        Функция или объект для вычисления состояния.
    `metric_function` : MetricBase
        Функция или объект для вычисления метрики.
    `stop_case_function` : StopCaseBase
        Функция или объект, определяющий критерий останова.
    `dynamic_nodes` : dict
        Словарь динамических (удаляемых) точек размещения.
    `static_nodes` : dict, optional
        Словарь статических (неудаляемых) точек размещения (по умолчанию None).
    `area` : object, optional
        Дополнительная область или параметры для метрики (по умолчанию None).

    Возвращает
    ----------
    `dynamic_nodes` : dict
        Оставшиеся динамические точки размещения после отбора.
    `metric` : float
        Значение метрики для итогового набора точек.
    '''

    # TODO: В будущем переписать как MCLP

    # Жадное удаление точек: поочередно пробуем удалить каждую точку
    count = 0
    for node in dynamic_nodes:
        if len(dynamic_nodes) == 1:
            break  # Оставляем хотя бы одну точку
        tmp = dynamic_nodes.copy()
        tmp.pop(node)  # Пробуем удалить текущую точку

        # Формируем новый набор точек для оценки
        if static_nodes is None:
            all_nodes = tmp
        else:
            all_nodes = list_dict_concat(tmp, static_nodes)

        # Вычисляем метрику для нового набора точек
        metric = NodesMetric(state_function, metric_function)(env, list(all_nodes.keys()), area=area)
        # Проверяем критерий останова
        if stop_case_function(value        = metric,
                            iteration      = count,
                            best_metric    = metric,
                            dynamic_nodes  = tmp,
                            current_metric = metric,
                            static_nodes   = static_nodes):
            dynamic_nodes = tmp  # Если критерий выполнен — удаляем точку

    # Итоговая метрика для оставшихся точек
    if static_nodes is None:
        all_nodes = dynamic_nodes
    else:
        all_nodes = list_dict_concat(dynamic_nodes, static_nodes)
    metric = NodesMetric(state_function, metric_function)(env, list(all_nodes.keys()), area=area)

    return dynamic_nodes, metric