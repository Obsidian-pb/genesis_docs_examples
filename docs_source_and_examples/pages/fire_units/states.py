'''
Расчет параметров прибытия пожарных подразделений к зданиям
'''

import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

from genesis.core import StateBase
from genesis.states import FirstArrivalUnitState
from genesis.swiss_knife import DELAY_TIME, MSF
from genesis.tools import get_duplicates_list



class FireDemandSatisfyState(FirstArrivalUnitState):
    def __init__(self, state_algorithm=MSF, weight='travel_time', delay=DELAY_TIME, **kwargs):
        super().__init__(state_algorithm, weight, delay, **kwargs)

    def __call__(self,
                 env: nx.Graph,
                 points: dict,
                 buildings: gpd.GeoDataFrame,
                 buildings_node_id_field: str = 'node',
                 buildings_demand_field: str = 'demand',
                 area=None,
                 **kwargs):

        # print(1, points)

        # Вот это будет вызывать базовый алгоритм:
        times, nearest = super().__call__(env, points, area, **kwargs)
        arrival_state = pd.concat([times, nearest], axis=1)

        # print(2, len(arrival_state))

        # здесь сопоставляем building и times
        merged_df = pd.merge(buildings, arrival_state, left_on=buildings_node_id_field, right_index=True)  # здесь подумать! #, how='left')

        # Расчет собственно состояния удовлетворенности спроса
        return merged_df[buildings_demand_field] / merged_df['times'], merged_df['nearest']   #, merged_df['times']


