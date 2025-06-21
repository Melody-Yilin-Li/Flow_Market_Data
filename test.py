import numpy as np 
import pandas as pd
import itertools 
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter
import statistics
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
import seaborn as sns
import faulthandler; faulthandler.enable()
from functools import reduce           
from sys import exit

# Replace NaN by empty dict
def replace_nans_with_dict(series):
    for idx in series[series.isnull()].index:
        series.at[idx] = {}
    return series

# Explodes list and dicts
def df_explosion(df, col_name:str):
    if df[col_name].isna().any():
        df[col_name] = replace_nans_with_dict(df[col_name])
    df.reset_index(drop=True, inplace=True)
    df1 = pd.DataFrame(df.loc[:,col_name].values.tolist())
    df = pd.concat([df,df1], axis=1)
    df.drop([col_name], axis=1, inplace=True)
    return df

num_groups = 4
players_per_group = 8
prac_rounds = 2 
num_rounds = 22
round_length = 120
leave_out_seconds = 10
penalty_cost = 20
price = [14, 6, 9, 6, 14]
ce_price = [p for p in price for _ in range(players_per_group // 2)]
quantity = [1200, 1300, 1500, 1300, 1200]
ce_quantity = [q for q in quantity for _ in range(players_per_group // 2)]
profits = [11700, 10700, 13200, 10700, 11700]
ce_profit = [p for p in profits for _ in range(players_per_group // 2)]
moving_average_size = 5

participant = {}
for g in range(1, num_groups + 1): 
    for r in range(1, num_rounds - prac_rounds + 1): 
        path = '/Users/YilinLi/Downloads/flow production/cda{}/{}/1_market.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(
            path,
        )
        # print(rnd[50:100])
        rnd['clearing_price'].fillna(method='bfill', inplace=True)
        rnd.fillna(0, inplace=True)
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)
        rnd['cumulative_quantity'] = rnd['clearing_rate'].cumsum()
        # print(rnd.loc[round_length - 1, 'cumulative_quantity'] * 2)
        # # break

        name = 'par{}'.format(r)
        path = '/Users/YilinLi/Downloads/flow production/cda{}/{}/1_participant.json'.format(g, r + prac_rounds)
        participant[name] = pd.read_json(
            path,
            )
        participant[name] = participant[name].explode('executed_contracts')
        participant[name].reset_index(drop=True, inplace=True)
        participant[name] = df_explosion(participant[name], 'executed_contracts')
        participant[name] = participant[name][(participant[name]['before_transaction'] == False)].reset_index()

        participant[name]['change_in_inventory'] = participant[name][participant[name]['timestamp'] < round_length - 1].groupby('id_in_group')['inventory'].diff().abs()
        participant[name]['change_in_inventory'].fillna(0, inplace=True)
        participant[name]['transacted_volume'] = participant[name].groupby(['id_in_group'])['change_in_inventory'].cumsum()

        def calculate_final_volume(row):
            return row['transacted_volume'] + max(0, row['fill_quantity'] - row['transacted_volume'])
        participant[name]['transacted_volume'] = participant[name].apply(calculate_final_volume, axis=1)
        participant[name]['round'] = r
        print(participant[name][['change_in_inventory', 'transacted_volume', 'fill_quantity', 'id_in_group', 'timestamp']][-8:])
        exit(0)


        # volume_change = participant[name][participant[name]['timestamp'] < round_length - 1].groupby('id_in_group', as_index=False)['inventory'].diff().abs().groupby(participant[name]['id_in_group'], as_index=False).sum()            
        # volume_change['id_in_group'] = [i for i in range(1, players_per_group + 1)]
        # volume = pd.merge(volume_change, participant[name][~participant[name]['fill_quantity'].isna()][['id_in_group', 'fill_quantity']], on='id_in_group', how='left')
        # def calculate_volume(row):
        #     return row['inventory'] + max(0, row['fill_quantity'] - row['inventory'])
        # volume['total_transacted_volume'] = volume.apply(calculate_volume, axis = 1)
        # volume.drop(columns=['inventory', 'fill_quantity'], inplace=True)


        tmp_df = participant[name][(participant[name]['timestamp'] == round_length - 1)]
        df = tmp_df.groupby('id_in_subsession').aggregate({'cash': 'sum', 'fill_quantity': 'sum', 'quantity': 'sum', 'transacted_volume': 'sum'}).reset_index()
        df['ce_profit'] = ce_profit[r - 1]
        df['ce_quantity'] = ce_quantity[r - 1] 
        df['payoff_percent'] = round(df['cash'] / df['ce_profit'], 4)
        df['contract_percent'] = round(df['fill_quantity'] / df['ce_quantity'] / 2, 4)
        df['round'] = r

        # print(df.loc[0, 'cash'])
        
        # print(rnd.loc[round_length - 1, 'cumulative_quantity'] * 2, )
        if df.loc[0, 'transacted_volume'] < df.loc[0, 'fill_quantity']:    
            print(g, r, df.loc[0, 'fill_quantity'], df.loc[0, 'transacted_volume'] >= df.loc[0, 'fill_quantity'])
        print(g, r, df, df.loc[0, 'fill_quantity'], df.loc[0, 'transacted_volume'] >= df.loc[0, 'fill_quantity'])

        # exit(0)