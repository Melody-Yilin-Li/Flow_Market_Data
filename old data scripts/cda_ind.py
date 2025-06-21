import numpy as np 
import pandas as pd
import itertools 
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
import seaborn as sns
import faulthandler; faulthandler.enable()
from functools import reduce                # Import reduce function
from sys import exit

# input session constants 
from config import *

directory = '/Users/YilinLi/Documents/UCSC/Flow Data/'


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

plt.close()


def main():
    print("params imported")

if __name__ == "__main__":
     main()


regress_cda_ind = pd.DataFrame()

# read in data 

# df for market clearing_prices and clearing_rates/quantities for all groups 
# create a list of dfs to be merged 
cda_ind = []


clearing_prices = []
clearing_rates = []
cum_quantities = []
rates = []
inventories = []
contract_quantities = []
contract_prices = []
contract_percents = []
fill_quantities = []
in_market_quantities = []
in_market_percents = []
order_directions = []
order_fill_quantities = []
order_ids = []
order_prices = []
order_quantities = []
order_timestamps = []
cashes = []
benchmark_paces = []
projected_profits = []
ind_ce_profits = []
realized_surpluses = []
transacted_quantities = []
excess_profits = []

colors = ['lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'lightcoral', 'lightskyblue', 'pink'] # add more colors with more than 6 groups



for g in range(1, num_groups_cda + 1):
    name = 'group' + str(g)
    group_mkt = []
    for r in range(1, num_rounds - prac_rounds + 1): 
        path = directory + 'flow production/cda{}/{}/1_market.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(
            path,
        )
        # rnd['clearing_price'].fillna(method='bfill', inplace=True)
        # rnd.fillna(0, inplace=True)
        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['timestamp'] < round_length - leave_out_seconds_end) & (rnd['timestamp'] < round_length) & (rnd['before_transaction'] == False)]
        rnd = rnd.drop(columns=['id_in_subsession', 'before_transaction'])
        # rnd['cumulative_quantity'] = rnd['clearing_rate'].cumsum()
        rnd['ce_quantity'] = ce_quantity[r - 1]
        rnd['ce_price'] = ce_price[r - 1]
        rnd['timestamp'] = np.arange(leave_out_seconds, round_length - leave_out_seconds_end, 1)
        group_mkt.append(rnd) 


    group_par = []

    for r in range(1, num_rounds - prac_rounds + 1):
        order_price_buy = []
        order_num_buy = 0
        order_quantity_buy = []

        order_price_sell = []
        order_num_sell = 0
        order_quantity_sell = []

        visited = set()
        
        path = directory + 'flow production/cda{}/{}/1_participant.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(
            path,
        )
        # rnd.fillna(0, inplace=True)
        # rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['timestamp'] < round_length) & (rnd['before_transaction'] == False)]
        rnd = pd.merge(rnd, group_mkt[r - 1], how='left', on='timestamp') # attache clearing price and clearing rate 
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)
        
        max_quantity_orders_buy = max_quantity_orders_sell = 0
        executed_percent_buy, executed_percent_sell = {}, {}
        for ind, row in rnd.iterrows():
            for order in row['active_orders']:
                if order['direction'] == 'sell':
                    if order['order_id'] not in visited:
                        order_price_sell.append(order['price'])
                        order_num_sell += 1
                        order_quantity_sell.append(order['quantity'])
                        if order['quantity'] == max_order_quantity:
                            max_quantity_orders_sell += 1
                    executed_percent_sell[order['order_id']] = order['fill_quantity'] / order['quantity']
                elif order['direction'] == 'buy':
                    if order['order_id'] not in visited:
                        order_price_buy.append(order['price'])
                        order_num_buy += 1
                        order_quantity_buy.append(order['quantity'])
                        if order['quantity'] == max_order_quantity:
                            max_quantity_orders_buy += 1
                    executed_percent_buy[order['order_id']] = order['fill_quantity'] / order['quantity']
                visited.add(order['order_id'])
        
        for ind, row in rnd.iterrows(): # determine which order is in the market given multiple 
            if np.isnan(row['clearing_price']):
                rnd.at[ind, 'active_orders'] = []
                continue
            if len(row['active_orders']) >= 1: 
                for order in row['active_orders']:
                    if (order['direction'] == 'sell' and order['price'] > row['clearing_price']):
                        rnd.at[ind, 'active_orders'].remove(order)
                    elif (order['direction'] == 'buy' and order['price'] < row['clearing_price']):
                        rnd.at[ind, 'active_orders'].remove(order)
        if 'executed_orders' in rnd.columns:
            rnd.drop('executed_orders', axis=1, inplace=True)
        rnd = rnd.explode('active_orders')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'active_orders')
        rnd.columns = ['timestamp', 'id_in_subsession', 'id_in_group', 'participant_id', 'before_transaction', 
            'active_contracts', 'executed_contracts', 'cash', 'inventory', 'rate', 
            'clearing_price', 'clearing_rate',
            'ce_quantity', 'ce_price', 'order_id', 'order_direction', 'order_quantity', 'order_fill_quantity', 
            'order_timestamp', 'order_price']
        rnd = rnd.explode('active_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'active_contracts')
        rnd = rnd.explode('executed_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'executed_contracts')
        rnd = rnd.groupby(level=0, axis=1).first()

        rnd['change_in_inventory'] = rnd[rnd['timestamp'] < round_length - 1].groupby('id_in_group')['inventory'].diff().abs()
        rnd['change_in_inventory'].fillna(0, inplace=True)
        rnd['transacted_quantity'] = rnd.groupby(['id_in_group'])['change_in_inventory'].cumsum()
        def calculate_final_volume(row):
            return row['transacted_quantity'] + max(0, row['fill_quantity'] - row['transacted_quantity'])
        rnd['transacted_quantity'] = rnd.apply(calculate_final_volume, axis=1)

        def calculate_final_inventory(row):
            if row['direction'] == 'sell':
                return row['inventory'] - max(0, row['fill_quantity'] - abs(row['inventory']))
            elif row['direction'] == 'buy':
                return row['inventory'] + max(0, row['fill_quantity'] - abs(row['inventory']))
        rnd['inventory'] = rnd.apply(calculate_final_inventory, axis=1)

        rnd['cumulative_quantity'] = rnd.groupby('timestamp')['transacted_quantity'].transform('sum')
        rnd['clearing_price'].fillna(method='bfill', inplace=True)
        # rnd.fillna(0, inplace=True)
        rational_shares = 0
        for ind, row in rnd.iterrows():
            if row['change_in_inventory'] != 0:
                if row['direction'] == 'sell':
                    if (row['direction'] == row['order_direction'] and row['clearing_price'] >= row['price']) or (row['direction'] != row['order_direction'] and row['clearing_price'] <= row['price']):
                        rational_shares += row['change_in_inventory']
                else:
                    if (row['direction'] == row['order_direction'] and row['clearing_price'] <= row['price']) or (row['direction'] != row['order_direction'] and row['clearing_price'] >= row['price']):
                        rational_shares += row['change_in_inventory']

        rnd = rnd[['timestamp', 'id_in_subsession', 'id_in_group', 'inventory', 'rate', 'direction', 'clearing_price', 'clearing_rate', 
        'cumulative_quantity', 'fill_quantity', 'quantity', 'price', 'ce_quantity', 'ce_price', 'order_direction', 
        'order_fill_quantity', 'order_id', 'order_price', 'order_quantity', 'order_timestamp', 'cash', 'transacted_quantity']]
        # rnd = rnd.drop(columns=['timestamp'])
    
        rnd['contract_percent'] = rnd['inventory'] / rnd['quantity']
        # timestamp = [i for i in np.arange(leave_out_seconds, round_length, 1) for _ in range(players_per_group)] # create correct timestamps 
        # rnd['timestamp'] = timestamp
        rnd['in_market_quantity'] = 0
        rnd['projected_profit'] = 0
        for ind, row in rnd.iterrows():
            contract_set = r // (players_per_group // 2) + int(r % (players_per_group // 2) != 0)
            if row['direction'] == 'sell': 
                rnd.loc[ind, 'in_market_quantity'] = contract_sell[contract_set][int(row['price'])] 
                if row['timestamp'] == round_length - 1: 
                    rnd.loc[ind, 'projected_profit'] = row['cash']
                else:
                    rnd.loc[ind, 'projected_profit'] = row['cash'] - min(abs(row['inventory']), row['quantity']) * row['price']
            else:
                rnd.loc[ind, 'in_market_quantity'] = contract_buy[contract_set][int(row['price'])]
                if row['timestamp'] == round_length - 1: 
                    rnd.loc[ind, 'projected_profit'] = row['cash']
                else:
                    rnd.loc[ind, 'projected_profit'] = row['cash'] + min(abs(row['inventory']), row['quantity']) * row['price']
        rnd['in_market_percent'] = rnd['inventory'] / rnd['in_market_quantity']
        rnd.loc[(rnd['timestamp'] == round_length - 1) & (rnd['direction'] == 'sell'), 'in_market_percent'] = - rnd['fill_quantity'] / rnd['in_market_quantity']
        rnd.loc[(rnd['timestamp'] == round_length - 1) & (rnd['direction'] == 'buy'), 'in_market_percent'] = rnd['fill_quantity'] / rnd['in_market_quantity']
        # benchmark_pace -- if a trader accumulates contract quantity at a uniform pace, what she has as a % of what she should have 
        rnd['benchmark_pace'] = rnd['inventory'] / (rnd['in_market_quantity'] / round_length * rnd['timestamp']) 
        rnd['ind_ce_profit'] = rnd['in_market_quantity'] * abs(rnd['price'] - rnd['ce_price'])        
        rnd['realized_surplus'] = rnd['projected_profit'] / rnd['ind_ce_profit']
        rnd['excess_profit'] = rnd['projected_profit'] - rnd['ind_ce_profit']
        rnd.loc[(rnd['ind_ce_profit'] == 0) & (rnd['projected_profit'] >= 0), 'realized_surplus'] = 1
        rnd.loc[(rnd['ind_ce_profit'] == 0) & (rnd['projected_profit'] < 0), 'realized_surplus'] = -1
        
        # buy low sell high behavior & suboptimal behavior 
        opposite_direction = 0
        same_direction = 0
        opposite_sell = set()
        opposite_buy = set()
        same_sell = set()
        same_buy = set()
        
        for ind, row in rnd.iterrows():
            if row['direction'] != row['order_direction']: 
                if row['order_direction'] == 'sell' and row['order_price'] < row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']: 
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    opposite_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    opposite_buy.add(row['id_in_group'])
                if row['order_direction'] == 'buy' and row['order_price'] > row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']: 
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    opposite_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    opposite_sell.add(row['id_in_group'])
            else:
                if row['order_direction'] == 'sell' and row['order_price'] < row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']:
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    same_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    same_sell.add(row['id_in_group'])
                elif row['order_direction'] == 'buy' and row['order_price'] > row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']:
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    same_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    same_buy.add(row['id_in_group'])
        # print(g, r, opposite_direction, same_direction)
        # print(g, r, opposite_buy, opposite_sell, same_buy, same_sell)

        regress_df = rnd[['direction', 'cumulative_quantity', 'fill_quantity', 'ce_quantity', 'ce_price', 'cash', 'in_market_quantity', 'ind_ce_profit', 'excess_profit']][-players_per_group:].copy()
        regress_df = regress_df.groupby('direction', as_index=False).aggregate({'cumulative_quantity': 'mean', 'fill_quantity': 'sum', 'ce_quantity': 'mean', 'ce_price': 'mean', 'cash': 'sum', 'in_market_quantity': 'sum', 'ind_ce_profit': 'sum', 'excess_profit': 'sum'}).reset_index(drop=True)
        regress_df['round'] = r
        regress_df['group'] = g
        regress_df['block'] = regress_df['block'] = regress_df['round'] // ((num_rounds - prac_rounds) // blocks) + (regress_df['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        regress_df['format'] = 'CDA'
        regress_df.rename(columns={'ind_ce_profit': 'ce_profit'}, inplace=True)
        regress_df['gross_profits_norm'] = regress_df['cash'] / regress_df['ce_profit']
        regress_df['order_num'] = 0
        regress_df['order_price_low'] = 0
        regress_df['order_price_high'] = 0
        regress_df['order_quantity'] = 0
        regress_df['order_rate'] = 0
        regress_df['max_quantity/rate_orders_sell'] = max_quantity_orders_sell
        regress_df['max_quantity/rate_orders_buy'] = max_quantity_orders_buy

        for ind, row in regress_df.iterrows():
            if row['direction'] == 'sell':
                regress_df.loc[ind, 'order_num'] = order_num_sell
                regress_df.loc[ind, 'order_price_low'] = np.mean(order_price_sell)
                regress_df.loc[ind, 'order_price_high'] = np.mean(order_price_sell)
                regress_df.loc[ind, 'order_quantity'] = np.mean(order_quantity_sell)
            else:
                regress_df.loc[ind, 'order_num'] = order_num_buy
                regress_df.loc[ind, 'order_price_low'] = np.mean(order_price_buy)
                regress_df.loc[ind, 'order_price_high'] = np.mean(order_price_buy)
                regress_df.loc[ind, 'order_quantity'] = np.mean(order_quantity_buy)
        regress_df['rational_shares'] = rational_shares

        regress_cda_ind = pd.concat([regress_cda_ind, regress_df], ignore_index=True)

        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['timestamp'] < round_length - leave_out_seconds_end)]

        group_par.append(rnd)
    


    df = pd.concat(group_par, ignore_index=True, sort=False)

    # print(df, df.columns)

    id_in_subsession = 'id_in_subsession_{}'.format(g)
    # id_in_group = 'id_in_group_{}'.format(g)
    
    inventory = 'inventory_{}'.format(g)
    inventories.append(inventory)
    
    rate = 'rate_{}'.format(g)
    rates.append(rate)
    
    # direction = 'direction_{}'.format(g)
    
    clearing_price = 'clearing_price_{}'.format(g)
    clearing_prices.append(clearing_price)
    
    clearing_rate = 'clearing_rate_{}'.format(g)
    clearing_rates.append(clearing_rate)
    
    cumsum = 'cumulative_quantity_{}'.format(g)
    cum_quantities.append(cumsum)
    
    fill_quantity = 'fill_quantity_{}'.format(g)
    fill_quantities.append(fill_quantity)
    
    contract_quantity = 'contract_quantity_{}'.format(g)
    contract_quantities.append(contract_quantity) 
    
    contract_price = 'contract_price_{}'.format(g)
    contract_prices.append(contract_price) 

    order_direction = 'order_direction_{}'.format(g)
    order_directions = []

    order_fill_quantity = 'order_fill_quantity_{}'.format(g)
    order_fill_quantities.append(order_fill_quantity)

    order_id = 'order_id_{}'.format(g)
    order_ids.append(order_id)

    order_price = 'order_price_{}'.format(g)
    order_prices.append(order_price)

    order_quantity = 'order_quantity_{}'.format(g)
    order_quantities.append(order_quantity)

    order_timestamp = 'order_timestamp_{}'.format(g)
    order_timestamps.append(order_timestamp)
 
    cash = 'cash_{}'.format(g) 
    cashes.append(cash)

    transacted_quantity = 'transacted_quantities_{}'.format(g)
    transacted_quantities.append(transacted_quantity)

    contract_percent = 'contract_percent_{}'.format(g)
    contract_percents.append(contract_percent)
    
    in_market_quantity = 'in_market_quantity_{}'.format(g)
    in_market_quantities.append(in_market_quantity)
    
    projected_profit = 'projected_profit_{}'.format(g)
    projected_profits.append(projected_profit)

    in_market_percent = 'in_market_percent_{}'.format(g)
    in_market_percents.append(in_market_percent)
    
    benchmark_pace = 'benchmark_pace_{}'.format(g)
    benchmark_paces.append(benchmark_pace)

    ind_ce_profit = 'ind_ce_profit_{}'.format(g)
    ind_ce_profits.append(ind_ce_profit)

    realized_surplus = 'realized_surplus_{}'.format(g)
    realized_surpluses.append(realized_surplus)

    excess_profit = 'excess_profit_{}'.format(g)
    excess_profits.append(excess_profit)


    df.columns = ['timestamp', id_in_subsession, 'id_in_group', inventory, rate, 'direction', clearing_price, 
        clearing_rate, cumsum, fill_quantity, contract_quantity, contract_price, 'ce_quantity', 'ce_price', 
        order_direction, order_fill_quantity, order_id, order_price, order_quantity, order_timestamp, cash,
        transacted_quantity, contract_percent, in_market_quantity, projected_profit, in_market_percent, 
        benchmark_pace, ind_ce_profit, realized_surplus, excess_profit, ]
    df['timestamp'] = df.groupby([id_in_subsession, 'id_in_group'])['id_in_group'].cumcount() + 1


    # print('LOSSES')
    # print('group {}'.format(g), 'COUNT\n', 
    #     (df[(df['timestamp'] % (round_length - leave_out_seconds) == 0)]['projected_profit_{}'.format(g)] < 0).sum(), 
    #     (df[(df['timestamp'] % (round_length - leave_out_seconds) == 0) & (df['timestamp'] // (round_length - leave_out_seconds) > ((num_rounds - prac_rounds) // 2))]['projected_profit_{}'.format(g)] < 0).sum(), )
    # print('group {}'.format(g), 'VALUE\n', 
    #     df[(df['timestamp'] % (round_length - leave_out_seconds) == 0) & (df['projected_profit_{}'.format(g)] < 0)]['projected_profit_{}'.format(g)].sum(), 
    #     df[(df['timestamp'] % (round_length - leave_out_seconds) == 0) & (df['timestamp'] // (round_length - leave_out_seconds) > ((num_rounds - prac_rounds) // 2)) & (df['projected_profit_{}'.format(g)] < 0)]['projected_profit_{}'.format(g)].sum(), 
    #     )

    cda_ind.append(df)
    
data_cda_ind = reduce(lambda left, right:    # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['timestamp', 'id_in_group', 'direction', 'ce_price', 'ce_quantity',],
                            #   suffixes=tuple(['_{}'.format(i) for i in range(1, num_groups_cda + 1)]),
                              ),
                     cda_ind) 

# aggregate the individual data by direction
aggregate = {q: 'sum' for q in projected_profits + fill_quantities}
data_cda_contract_by_direction = data_cda_ind[data_cda_ind['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0].groupby(['timestamp', 'direction'], as_index=False).agg(aggregate)
data_cda_contract_by_direction['ce_profits'] = 0
data_cda_contract_by_direction['round'] = data_cda_contract_by_direction['timestamp'] // (round_length - leave_out_seconds - leave_out_seconds_end)
for ind, row in data_cda_contract_by_direction.iterrows():
    if row['direction'] == 'buy': 
        data_cda_contract_by_direction.loc[ind, 'ce_profits'] = ce_profit_buy[row['round'] - 1]
    else:
        data_cda_contract_by_direction.loc[ind, 'ce_profits'] = ce_profit_sell[row['round'] - 1]
for g in range(1, len(projected_profits) + 1): 
    data_cda_contract_by_direction['realized_surplus_{}'.format(g)] = data_cda_contract_by_direction[projected_profits[g - 1]] / data_cda_contract_by_direction['ce_profits']
data_cda_contract_by_direction = data_cda_contract_by_direction.drop('timestamp', axis = 1)
# print(data_cda_contract_by_direction[fill_quantities], data_cda_contract_by_direction.columns, data_cda_ind.columns)
# print(cda_ind, data_cda_ind, data_cda_ind.columns)
# exit(0)
mean_cda = []
for g in range(1, num_groups_cda + 1):
    mean_df = data_cda_ind.groupby(['timestamp'], as_index=False)[['benchmark_pace_{}'.format(g), 'projected_profit_{}'.format(g)]].apply(lambda x: x.abs().mean())
    mean_cda.append(mean_df)

data_mean_cda = reduce(
    lambda left, right:
    pd.merge(left, right, on = ['timestamp']),
    mean_cda
)

mean_cda_by_direction = []
for g in range(1, num_groups_cda + 1):
    mean_df_by_direction = data_cda_ind.groupby(['timestamp', 'direction'], as_index=False)[['benchmark_pace_{}'.format(g), 'projected_profit_{}'.format(g)]].apply(lambda x: x.abs().mean())
    mean_cda_by_direction.append(mean_df_by_direction)

data_mean_cda_by_direction = reduce(
    lambda left, right:
    pd.merge(left, right, on = ['timestamp',  'direction']),
    mean_cda_by_direction
)

sum_cda_by_direction = []
for g in range(1, num_groups_cda + 1):
    sum_df_by_direction = data_cda_ind.groupby(['timestamp', 'direction'], as_index=False)['projected_profit_{}'.format(g)].apply(lambda x: x.abs().sum())
    sum_cda_by_direction.append(sum_df_by_direction)

data_sum_cda_by_direction = reduce(
    lambda left, right:
    pd.merge(left, right, on = ['timestamp', 'direction']),
    sum_cda_by_direction
)

# print(data_mean_cda, data_mean_cda_by_direction, data_sum_cda_by_direction)
# exit(0)

# average benchmark_pace at group level 
plt.figure(figsize=(15, 5))
for l in range(len(benchmark_paces)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_mean_cda[data_mean_cda[benchmark_paces[l]] > 0]['timestamp'], data_mean_cda[data_mean_cda[benchmark_paces[l]] > 0][benchmark_paces[l]], linestyle='solid', c=colors[l], label=lab)
plt.hlines(y=1, xmin=0, xmax=(num_rounds-prac_rounds) * (round_length - leave_out_seconds - leave_out_seconds_end), colors='plum', linestyles='--')
for x in [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]:
    plt.vlines(x, ymin=0, ymax=5, colors='lightgrey', linestyles='dotted')
# plt.legend(bbox_to_anchor=(1, 1),
#         loc='upper left',
#         borderaxespad=.5)
plt.ylim(0, 5)
plt.xlabel('Time')
plt.ylabel('Benchmark Pace')
plt.title('Benchmark Pace vs Time')
plt.savefig('groups_cda_benchmark_pace.png')
plt.close()

# average projected profit at group level 
plt.figure(figsize=(15, 5))
for l in range(len(projected_profits)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_mean_cda['timestamp'], data_mean_cda[projected_profits[l]], linestyle='solid', c=colors[l], label=lab)
# plt.legend(bbox_to_anchor=(1, 1),
#         loc='upper left',
#         borderaxespad=.5)
plt.ylim(0, 1800)
plt.xlabel('Time')
plt.ylabel('Projected Profit')
plt.title('Mean Projected Profit vs Time')
plt.savefig('groups_cda_projected_profit.png')
plt.close()

# average projected profit at group level (by direction)
plt.figure(figsize=(15, 5))
data_mean_cda_by_direction.loc[data_mean_cda_by_direction['direction'] == 'sell', projected_profits] *= -1
df_long = data_mean_cda_by_direction.melt(id_vars=['timestamp', 'direction'], value_vars=projected_profits, var_name='group_id', value_name='projected_profit')
df_long['group_id'] = df_long['group_id'].str.replace('projected_profit', 'group')

sns.lineplot(data=df_long, x='timestamp', y='projected_profit', hue='group_id', style='direction', palette=colors[:num_groups_cda], legend='full')
plt.hlines(y=0, xmin=0, xmax=(num_rounds-prac_rounds) * (round_length - leave_out_seconds - leave_out_seconds_end), colors='plum', linestyles='dotted')
plt.legend(bbox_to_anchor=(1, 1),
        loc='upper left',
        borderaxespad=.5)
plt.ylim(-2550, 2550)
plt.xlabel('Time')
plt.ylabel('Projected Profit')
plt.title('Mean Projected Profit vs Time')
plt.savefig('groups_cda_projected_profit_by_direction.png')
plt.close()


end_of_period_profits_cda = data_sum_cda_by_direction[data_sum_cda_by_direction['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0].copy()
end_of_period_profits_cda['period'] = data_sum_cda_by_direction['timestamp'] // (round_length - leave_out_seconds - leave_out_seconds_end)

mean_end_of_period_profits_cda = end_of_period_profits_cda.groupby(['direction'], as_index=False)[projected_profits].mean()
mean_end_of_period_profits_cda['mean'] = mean_end_of_period_profits_cda[projected_profits].mean(axis=1)
mean_end_of_period_profits_cda_half = end_of_period_profits_cda[end_of_period_profits_cda['period'] > (num_rounds - prac_rounds) // 2].groupby(['direction'], as_index=False)[projected_profits].mean()
mean_end_of_period_profits_cda_half['mean'] = mean_end_of_period_profits_cda_half[projected_profits].mean(axis=1)
# print('All 20 periods', mean_end_of_period_profits_cda, '\nLast 10 periods\n', mean_end_of_period_profits_cda_half)

data_cda_ind[in_market_percents] = data_cda_ind[in_market_percents].abs()
individual_agg_data_cda = data_cda_ind[data_cda_ind['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0].groupby('timestamp', as_index=False)[projected_profits + in_market_percents + realized_surpluses].mean()


# average projected profit at group level 
plt.figure(figsize=(15, 5))
for l in range(len(realized_surpluses)): 
    lab = '_group' + str(l + 1)
    plt.plot(individual_agg_data_cda['timestamp'], individual_agg_data_cda[realized_surpluses[l]], linestyle='solid', c=colors[l], label=lab)
# plt.legend(bbox_to_anchor=(1, 1),
#         loc='upper left',
#         borderaxespad=.5)
plt.ylim(-0.1, 1.5)
plt.xlabel('Time')
plt.ylabel('Realized Surplus')
plt.savefig('groups_cda_realized_surplus_ind_truncated.png')
plt.close()


summary_cda_by_direction = data_cda_contract_by_direction
summary_cda_ind_by_direction = data_cda_ind[data_cda_ind['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0][projected_profits + ['ind_ce_profit_1'] + excess_profits + ['direction']].reset_index(drop=True)
summary_cda_ind_by_direction['round'] = [r for r in range(1, num_rounds - prac_rounds + 1) for _ in range(players_per_group)]
# print(summary_cda_ind_by_direction)

realized_surplus_buy_cda_full = []
realized_surplus_buy_cda_half = []
realized_surplus_buy_cda_first = []
realized_surplus_buy_cda_test = []

realized_surplus_sell_cda_full = []
realized_surplus_sell_cda_half = []
realized_surplus_sell_cda_first = []
realized_surplus_sell_cda_test = []

profits_buy_cda_ind_full = []
profits_buy_cda_ind_half = []
profits_buy_cda_ind_first = []

profits_sell_cda_ind_full = []
profits_sell_cda_ind_half = []
profits_sell_cda_ind_first = []

excess_profits_buy_cda_ind_full = []
excess_profits_buy_cda_ind_half = []
excess_profits_buy_cda_ind_first = []

excess_profits_sell_cda_ind_full = []
excess_profits_sell_cda_ind_half = []
excess_profits_sell_cda_ind_first = []

for g in range(1, num_groups_cda + 1):
    realized_surplus_buy_cda_full.append(summary_cda_by_direction[summary_cda_by_direction['direction'] == 'buy']['realized_surplus_{}'.format(g)].mean())
    realized_surplus_buy_cda_half.append(summary_cda_by_direction[(summary_cda_by_direction['direction'] == 'buy') & (summary_cda_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_buy_cda_first.append(summary_cda_by_direction[(summary_cda_by_direction['direction'] == 'buy') & (summary_cda_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_buy_cda_test.extend(summary_cda_by_direction[(summary_cda_by_direction['direction'] == 'buy') & (summary_cda_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].tolist())

    realized_surplus_sell_cda_full.append(summary_cda_by_direction[summary_cda_by_direction['direction'] == 'sell']['realized_surplus_{}'.format(g)].mean())
    realized_surplus_sell_cda_half.append(summary_cda_by_direction[(summary_cda_by_direction['direction'] == 'sell') & (summary_cda_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_sell_cda_first.append(summary_cda_by_direction[(summary_cda_by_direction['direction'] == 'sell') & (summary_cda_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_sell_cda_test.extend(summary_cda_by_direction[(summary_cda_by_direction['direction'] == 'sell') & (summary_cda_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].tolist())

    profits_buy_cda_ind_full.extend(summary_cda_ind_by_direction[summary_cda_ind_by_direction['direction'] == 'buy']['projected_profit_{}'.format(g)].tolist())
    profits_buy_cda_ind_half.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'buy') & (summary_cda_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())
    profits_buy_cda_ind_first.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'buy') & (summary_cda_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())

    profits_sell_cda_ind_full.extend(summary_cda_ind_by_direction[summary_cda_ind_by_direction['direction'] == 'sell']['projected_profit_{}'.format(g)].tolist())
    profits_sell_cda_ind_half.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'sell') & (summary_cda_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())
    profits_sell_cda_ind_first.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'sell') & (summary_cda_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())

    excess_profits_buy_cda_ind_full.extend(summary_cda_ind_by_direction[summary_cda_ind_by_direction['direction'] == 'buy']['excess_profit_{}'.format(g)].tolist())
    excess_profits_buy_cda_ind_half.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'buy') & (summary_cda_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())
    excess_profits_buy_cda_ind_first.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'buy') & (summary_cda_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())

    excess_profits_sell_cda_ind_full.extend(summary_cda_ind_by_direction[summary_cda_ind_by_direction['direction'] == 'sell']['excess_profit_{}'.format(g)].tolist())
    excess_profits_sell_cda_ind_half.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'sell') & (summary_cda_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())
    excess_profits_sell_cda_ind_first.extend(summary_cda_ind_by_direction[(summary_cda_ind_by_direction['direction'] == 'sell') & (summary_cda_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())

    
# print(
    # realized_surplus_buy_cda_full,
#     realized_surplus_buy_cda_half,
#     realized_surplus_buy_cda_test,
    # realized_surplus_sell_cda_full,
#     realized_surplus_sell_cda_half,
#     realized_surplus_sell_cda_test,
# )

# print(data_cda_contract_by_direction)
# print(profits_buy_cda_ind_full, profits_sell_cda_ind_full, profits_buy_cda_ind_half, profits_sell_cda_ind_half)
# print(len(profits_buy_cda_ind_full), len(profits_sell_cda_ind_full), len(profits_buy_cda_ind_half), len(profits_sell_cda_ind_half))


# cdf of excess profits at ind level
sorted_excess_profit_cda_ind_full = np.sort(excess_profits_buy_cda_ind_full + excess_profits_sell_cda_ind_full)
sorted_excess_profit_buy_cda_ind_full = np.sort(excess_profits_buy_cda_ind_full)
sorted_excess_profit_sell_cda_ind_full = np.sort(excess_profits_sell_cda_ind_full)
cumulative_prob_excess_profit_cda_ind_full = np.arange(1, len(sorted_excess_profit_cda_ind_full) + 1) / len(sorted_excess_profit_cda_ind_full)
cumulative_prob_excess_profit_buy_cda_ind_full = np.arange(1, len(sorted_excess_profit_buy_cda_ind_full) + 1) / len(sorted_excess_profit_buy_cda_ind_full)
cumulative_prob_excess_profit_sell_cda_ind_full = np.arange(1, len(sorted_excess_profit_sell_cda_ind_full) + 1) / len(sorted_excess_profit_sell_cda_ind_full)
plt.figure(figsize=(15, 10))
plt.step(sorted_excess_profit_cda_ind_full, cumulative_prob_excess_profit_cda_ind_full, label='CDF', where='post')
plt.step(sorted_excess_profit_buy_cda_ind_full, cumulative_prob_excess_profit_buy_cda_ind_full, label='buyers', where='post')
plt.step(sorted_excess_profit_sell_cda_ind_full, cumulative_prob_excess_profit_sell_cda_ind_full, label='sellers', where='post')
plt.title('CDF of the Excess Profits (CDA)')
plt.xlabel('Excess Profits')
plt.ylabel('Probability')
plt.legend()
plt.savefig('groups_cda_excess_profits_cdf.png')
plt.close()