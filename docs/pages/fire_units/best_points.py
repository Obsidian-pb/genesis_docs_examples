'''
Дополнительные функции расчета лучшего узла,
учитывающие специфику пожарных подразделений
'''


import networkx as nx
import pandas as pd
import numpy as np


try:
    from ..genesis.best_points import BestNodeHillClimbing, BestNodesHalfDiameter
    from ..genesis.core import MetricBase, StateBase
    from ..genesis.metrics import ArrivalTime
    from ..genesis.states import FirstArrivalUnitState
except:
    from genesis.best_points import BestNodeHillClimbing, BestNodesHalfDiameter
    from genesis.core import MetricBase, StateBase
    from genesis.metrics import ArrivalTime
    from genesis.states import FirstArrivalUnitState    



class BestNodeHillClimbing_HD_Max_Mean_Metric(BestNodeHillClimbing):
    def __init__(self, 
                 state_function: StateBase,
                 metric_function: MetricBase = None,
                 appr_val: float = 0.95,
                 all_neighbors: bool = True,
                 node_calc_end_function: callable = None,
                 **kwargs) -> None:
        self.node_calc_end_function = node_calc_end_function
        super().__init__(state_function, metric_function, appr_val, all_neighbors, **kwargs)

    def __call__(self, env: nx.Graph,
                 area: pd.Series = None,
                 start_node: int = None,
                 debug_route: bool = False,
                 **kwargs):

        bnch_d = BestNodesHalfDiameter()
        bnch_max = BestNodeHillClimbing(state_function=self.state_function,
                                        metric_function=ArrivalTime(np.max), appr_val=self.appr_val, all_nodes=self.all_neighbors)
        bnch_mean = BestNodeHillClimbing(state_function=self.state_function,
                                         metric_function=ArrivalTime(), appr_val=self.appr_val, all_nodes=self.all_neighbors)
        bnch_meetric = BestNodeHillClimbing(state_function=self.state_function,
                                         metric_function=self.metric_function, appr_val=self.appr_val, all_nodes=self.all_neighbors)

        best_node, _               =    bnch_d(env=env, area=area)
        try:
            best_node, best_metric =    bnch_max(env=env, area=area, start_point=best_node)
        except:
            best_node, best_metric =    bnch_max(env=env, area=area)
        best_node, best_metric     =    bnch_mean(env=env, area=area, start_point=best_node)
        best_node, best_metric     =    bnch_meetric(env=env, area=area, start_point=best_node)

        # выполняем функцию завершения расчета для узла
        if self.node_calc_end_function:
            self.node_calc_end_function(best_node=best_node, best_metric=best_metric)

        return best_node, best_metric


class BestNodeHillClimbing_maxMean_Metric(BestNodeHillClimbing):
    def __init__(self, 
                 state_function: StateBase,
                 metric_function: MetricBase = None,
                 appr_val: float = 0.95,
                 all_neighbors: bool = True,
                 node_calc_end_function: callable = None,
                 **kwargs) -> None:
        self.node_calc_end_function = node_calc_end_function
        super().__init__(state_function, metric_function, appr_val, all_neighbors, **kwargs)

    def __call__(self, env: nx.Graph,
                 area: pd.Series = None,
                 start_point: int = None,
                 debug_route: bool = False,
                 **kwargs):

        bnch_max = BestNodeHillClimbing(state_function=self.state_function,
                                        metric_function=ArrivalTime(np.max), appr_val=self.appr_val, all_nodes=self.all_neighbors)
        bnch_mean = BestNodeHillClimbing(state_function=self.state_function,
                                         metric_function=ArrivalTime(), appr_val=self.appr_val, all_nodes=self.all_neighbors)
        bnch_meetric = BestNodeHillClimbing(state_function=self.state_function,
                                         metric_function=self.metric_function, appr_val=self.appr_val, all_nodes=self.all_neighbors)

        best_node, best_metric =     bnch_max(env=env, area=area, start_point=start_point)
        best_node, best_metric =    bnch_mean(env=env, area=area, start_point=best_node)
        best_node, best_metric = bnch_meetric(env=env, area=area, start_point=best_node)

        # выполняем функцию завершения расчета для узла
        if self.node_calc_end_function:
            self.node_calc_end_function(best_node=best_node, best_metric=best_metric)

        return best_node, best_metric
    

class BestNodeHillClimbing_HD_Max_Mean(BestNodeHillClimbing):
    def __init__(self, 
                 state_function: StateBase,
                 metric_function: MetricBase = None,
                 appr_val: float = 0.95,
                 all_neighbors: bool = True,
                 node_calc_end_function: callable = None,
                 **kwargs) -> None:
        self.node_calc_end_function = node_calc_end_function
        super().__init__(state_function, metric_function, appr_val, all_neighbors, **kwargs)

    def __call__(self, env: nx.Graph,
                 area: pd.Series = None,
                 start_node: int = None,
                 debug_route: bool = False,
                 **kwargs):

        bnch_d = BestNodesHalfDiameter()
        bnch_max = BestNodeHillClimbing(state_function=self.state_function,
                                        metric_function=ArrivalTime(np.max), appr_val=self.appr_val, all_nodes=self.all_neighbors)
        bnch_mean = BestNodeHillClimbing(state_function=self.state_function,
                                         metric_function=ArrivalTime(), appr_val=self.appr_val, all_nodes=self.all_neighbors)

        best_node, _               =    bnch_d(env=env, area=area)
        try:
            best_node, best_metric =    bnch_max(env=env, area=area, start_point=best_node)
        except:
            best_node, best_metric =    bnch_max(env=env, area=area)
        best_node, best_metric     =    bnch_mean(env=env, area=area, start_point=best_node)


        # выполняем функцию завершения расчета для узла
        if self.node_calc_end_function:
            self.node_calc_end_function(best_node=best_node, best_metric=best_metric)

        return best_node, best_metric




class BestNodeHillClimbingHD(BestNodeHillClimbing):
    '''
    Расчет лучшего узла с использованием алгоритма
    Hill Climbing по метрике и
    предварительным определением центра диаметра графа (Half Diameter).
    '''
    def __init__(self,
                 state_function: StateBase,
                 metric_function: MetricBase = None,
                 appr_val: float = 0.95,
                 all_neighbors: bool = True,
                 node_calc_end_function: callable = None,
                 **kwargs) -> None:
        self.node_calc_end_function = node_calc_end_function
        super().__init__(state_function, metric_function, appr_val, all_neighbors, **kwargs)

    def __call__(self, env: nx.Graph,
                 area: pd.Series = None,
                 start_point: int = None,
                 debug_route: bool = False,
                 **kwargs):

        bnhd = BestNodesHalfDiameter()
        bnhc = BestNodeHillClimbing(state_function=self.state_function,
                                         metric_function=self.metric_function,
                                         appr_val=self.appr_val,
                                         all_nodes=self.all_neighbors)

        start_node, _ = bnhd(env=env, area=area)
        best_node, best_metric = bnhc(env=env, area=area, start_point=start_node)

        # выполняем функцию завершения расчета для узла
        if self.node_calc_end_function:
            self.node_calc_end_function(best_node=best_node, best_metric=best_metric)

        return best_node, best_metric