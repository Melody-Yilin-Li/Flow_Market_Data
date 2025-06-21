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

if high_flow_max_rate:
    directory = '/Users/YilinLi/Documents/UCSC/Flow Data/flow production/flow high max rate/'
else:
    directory = '/Users/YilinLi/Documents/UCSC/Flow Data/flow production/flow low max rate/'


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

     
regress_flow_ind = pd.DataFrame()

# read in data 

# df for market clearing_prices and clearing_rates/quantities for all groups 
# create a list of dfs to be merged 
flow_ind = []


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
order_max_prices = []
order_max_rates = []
order_min_prices = []
order_quantities = []
order_timestamps = []
cashes = []
benchmark_paces = []
projected_profits = []
ind_ce_profits = []
realized_surpluses = []
excess_profits = []


colors = ['lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'lightcoral', 'lightskyblue', 'pink'] # add more colors with more than 6 groups



for g in range(1, num_groups_flow + 1):
    name = 'group' + str(g)
    group_mkt = []
    for r in range(1, num_rounds - prac_rounds + 1): 
        path = directory + 'flow{}/{}/1_market.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(path)
        rnd = rnd[(rnd['before_transaction'] == False)]
        rnd['no_trans'] = rnd['clearing_price'].isna().sum()
        rnd = rnd.drop(columns=['id_in_subsession', 'before_transaction'])
        rnd['cumulative_quantity'] = rnd['clearing_rate'].cumsum()
        rnd['ce_rate'] =  ce_rate[r - 1]
        rnd['ce_quantity'] = ce_quantity[r - 1]
        rnd['ce_price'] = ce_price[r - 1]
        group_mkt.append(rnd) 

    group_par = []
    for r in range(1, num_rounds - prac_rounds + 1):
        order_price_low_buy = []
        order_price_high_buy = []
        order_num_buy = 0
        order_quantity_buy = []
        order_rate_buy = []

        order_price_low_sell = []
        order_price_high_sell = []
        order_num_sell = 0
        order_quantity_sell = []
        order_rate_sell = []

        visited = set()
        
        path = directory + 'flow{}/{}/1_participant.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(
            path,
        )
        rnd = pd.merge(rnd, group_mkt[r - 1], how='left', on='timestamp') # attach clearing price and clearing rate 
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)

        max_rate_orders_buy = max_rate_orders_sell = 0
        executed_percent_buy, executed_percent_sell = {}, {}
        for ind, row in rnd.iterrows():
            for order in row['active_orders']:
                
                if order['direction'] == 'sell':
                    if order['order_id'] not in visited:
                        order_price_low_sell.append(order['min_price'])
                        order_price_high_sell.append(order['max_price'])
                        order_num_sell += 1
                        order_quantity_sell.append(order['quantity'])
                        order_rate_sell.append(order['max_rate'])
                        if order['max_rate'] == max_order_rate:
                            max_rate_orders_sell += 1
                    executed_percent_sell[order['order_id']] = order['fill_quantity'] / order['quantity']
                elif order['direction'] == 'buy':
                    if order['order_id'] not in visited:
                        order_price_low_buy.append(order['min_price'])
                        order_price_high_buy.append(order['max_price'])
                        order_num_buy += 1
                        order_quantity_buy.append(order['quantity'])
                        order_rate_buy.append(order['max_rate'])
                        if order['max_rate'] == max_order_rate:
                            max_rate_orders_buy += 1 
                    executed_percent_buy[order['order_id']] = order['fill_quantity'] / order['quantity']
                visited.add(order['order_id'])
        
        for ind, row in rnd.iterrows(): # determine which order is in the market given multiple 
            if np.isnan(row['clearing_price']):
                rnd.at[ind, 'active_orders'] = []
                continue
            if len(row['active_orders']) >= 1: 
                for order in row['active_orders']:
                    if order['direction'] == 'sell' and order['min_price'] > row['clearing_price']:
                        rnd.at[ind, 'active_orders'].remove(order)
                    if order['direction'] == 'buy' and order['max_price'] < row['clearing_price']:
                        rnd.at[ind, 'active_orders'].remove(order)
        if 'executed_orders' in rnd.columns:
            rnd.drop('executed_orders', axis=1, inplace=True)
        rnd = rnd.explode('active_orders')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'active_orders')
        rnd.columns = ['timestamp', 'id_in_subsession', 'id_in_group', 'participant_id', 'before_transaction', 'active_contracts', 
        'executed_contracts', 'cash', 'inventory', 'rate', 'clearing_price', 'clearing_rate', 'no_trans', 'cumulative_quantity', 'ce_rate', 
        'ce_quantity', 'ce_price', 'order_id', 'order_direction', 'order_quantity', 'order_fill_quantity', 'order_timestamp', 
        'order_max_price', 'order_min_price', 'order_max_rate']
        rnd = rnd.explode('active_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'active_contracts')
        rnd = rnd.explode('executed_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'executed_contracts')
        rnd = rnd.groupby(level=0, axis=1).first()

        rnd['change_in_inventory'] = rnd[rnd['timestamp'] < round_length - 1].groupby('id_in_group')['inventory'].diff().abs()
        rnd['change_in_inventory'].fillna(0, inplace=True)
        def calculate_final_inventory(row):
            if row['direction'] == 'sell':
                return row['inventory'] - max(0, row['fill_quantity'] - abs(row['inventory']))
            elif row['direction'] == 'buy':
                return row['inventory'] + max(0, row['fill_quantity'] - abs(row['inventory']))
        rnd['inventory'] = rnd.apply(calculate_final_inventory, axis=1)
        
        rnd['clearing_price'].fillna(method='bfill', inplace=True)
        # rnd.fillna(0, inplace=True)
        rational_shares = 0
        for ind, row in rnd.iterrows():
            if row['change_in_inventory'] != 0:
                if row['direction'] == 'sell':
                    if (row['direction'] == row['order_direction'] and row['clearing_price'] >= row['price']) \
                        or (row['direction'] != row['order_direction'] and row['clearing_price'] <= row['price']):
                        rational_shares += row['change_in_inventory']
                else:
                    if (row['direction'] == row['order_direction'] and row['clearing_price'] <= row['price']) \
                        or (row['direction'] != row['order_direction'] and row['clearing_price'] >= row['price']):
                        rational_shares += row['change_in_inventory']

       
        rnd = rnd[['timestamp', 'id_in_subsession', 'id_in_group', 'inventory', 'rate', 'direction', 'clearing_price', 'clearing_rate', 
        'cumulative_quantity', 'fill_quantity', 'quantity', 'price', 'ce_rate', 'ce_quantity', 'ce_price', 'order_direction', 
        'order_fill_quantity', 'order_id', 'order_max_price', 'order_max_rate', 'order_min_price', 'order_quantity', 'order_timestamp', 
        'cash', 'no_trans']]
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
        
        opposite_direction = 0
        same_direction = 0
        opposite_sell = set()
        opposite_buy = set()
        same_sell = set()
        same_buy = set()

        for ind, row in rnd.iterrows():
            if row['direction'] != row['order_direction']: 
                if row['order_direction'] == 'sell' and row['order_max_price'] < row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']: 
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    opposite_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    opposite_buy.add(row['id_in_group'])
                if row['order_direction'] == 'buy' and row['order_min_price'] > row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']: 
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    opposite_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    opposite_sell.add(row['id_in_group'])
            else:
                if row['order_direction'] == 'sell' and row['order_max_price'] < row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']:
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    same_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    same_sell.add(row['id_in_group'])
                elif row['order_direction'] == 'buy' and row['order_min_price'] > row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']:
                    # print(g, r, row['timestamp'], row['id_in_group'], rnd.loc[ind - 1, 'projected_profit'], row['projected_profit'])
                    same_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    same_buy.add(row['id_in_group'])
        # print(g, r, opposite_direction, same_direction)
        # print(g, r, opposite_buy, opposite_sell, same_buy, same_sell)

        regress_df = rnd[['direction', 'cumulative_quantity', 'fill_quantity', 'ce_quantity', 'ce_price', 'cash', 'in_market_quantity', 'ind_ce_profit', 'excess_profit', 'no_trans']][-players_per_group:].copy()
        regress_df = regress_df.groupby('direction', as_index=False) \
                .aggregate({'cumulative_quantity': 'mean', 'fill_quantity': 'sum', 'ce_quantity': 'mean', 'ce_price': 'mean', 'cash': 'sum', 'in_market_quantity': 'sum', 'ind_ce_profit': 'sum', 'excess_profit': 'sum', 'no_trans': 'mean'})\
                .reset_index(drop=True)
        regress_df['round'] = r
        regress_df['group'] = g
        regress_df['block'] = regress_df['round'] // ((num_rounds - prac_rounds) // blocks) + (regress_df['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        regress_df['format'] = 'FLOW'
        regress_df.rename(columns={'ind_ce_profit': 'ce_profit'}, inplace=True)
        regress_df['gross_profits_norm'] = regress_df['cash'] / regress_df['ce_profit']
        regress_df['order_num'] = 0
        regress_df['order_price_low'] = 0
        regress_df['order_price_high'] = 0
        regress_df['order_quantity'] = 0
        regress_df['order_rate'] = 0
        regress_df['max_quantity/rate_orders_buy'] = max_rate_orders_buy
        regress_df['max_quantity/rate_orders_sell'] = max_rate_orders_sell
        for ind, row in regress_df.iterrows():
            if row['direction'] == 'sell':
                regress_df.loc[ind, 'order_num'] = order_num_sell
                regress_df.loc[ind, 'order_price_low'] = np.mean(order_price_low_sell)
                regress_df.loc[ind, 'order_price_high'] = np.mean(order_price_high_sell)
                regress_df.loc[ind, 'order_quantity'] = np.mean(order_quantity_sell)
                regress_df.loc[ind, 'order_rate'] = np.mean(order_rate_sell)
            else:
                regress_df.loc[ind, 'order_num'] = order_num_buy
                regress_df.loc[ind, 'order_price_low'] = np.mean(order_price_low_buy)
                regress_df.loc[ind, 'order_price_high'] = np.mean(order_price_high_buy)
                regress_df.loc[ind, 'order_quantity'] = np.mean(order_quantity_buy)
                regress_df.loc[ind, 'order_rate'] = np.mean(order_rate_buy)
        regress_df['rational_shares'] = rational_shares
        regress_flow_ind = pd.concat([regress_flow_ind, regress_df], ignore_index=True)

        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['timestamp'] < round_length - leave_out_seconds_end)]
        del rnd['no_trans']
        # rnd["group"] = g
        # rnd["round"] = r
        # rnd["block"] = r // ((num_rounds - prac_rounds) // blocks) + (r % ((num_rounds - prac_rounds) // blocks) != 0)
        group_par.append(rnd)

    df = pd.concat(group_par, ignore_index=True, sort=False)
    df['ind_ce_rate'] = df['ce_rate'].div(players_per_group / 2)

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

    order_max_price = 'order_max_price_{}'.format(g)
    order_max_prices.append(order_max_price)

    order_max_rate = 'order_max_rate_{}'.format(g)
    order_max_rates.append(order_max_rate)

    order_min_price = 'order_min_price_{}'.format(g)
    order_min_prices.append(order_min_price)

    order_quantity = 'order_quantity_{}'.format(g)
    order_quantities.append(order_quantity)

    order_timestamp = 'order_timestamp_{}'.format(g)
    order_timestamps.append(order_timestamp)
 
    cash = 'cash_{}'.format(g) 
    cashes.append(cash)

    contract_percent = 'contract_percent_{}'.format(g)
    contract_percents.append(contract_percent)
    
    in_market_quantity = 'in_market_quantity_{}'.format(g)
    in_market_quantities.append(in_market_quantity)

    projected_profit = 'projected_profit_{}'.format(g)
    projected_profits.append(projected_profit)
    
    in_market_percent = 'in_market_percent_{}'.format(g)
    in_market_percents.append(in_market_percent)

    ind_ce_rate = 'ind_ce_rate_{}'.format(g)
    
    benchmark_pace = 'benchmark_pace_{}'.format(g)
    benchmark_paces.append(benchmark_pace)
    
    ind_ce_profit = 'ind_ce_profit_{}'.format(g)
    ind_ce_profits.append(ind_ce_profit)

    realized_surplus = 'realized_surplus_{}'.format(g)
    realized_surpluses.append(realized_surplus)

    excess_profit = 'excess_profit_{}'.format(g)
    excess_profits.append(excess_profit)

    df.columns = ['timestamp', id_in_subsession, 'id_in_group', inventory, rate, 'direction', clearing_price, 
        clearing_rate, cumsum, fill_quantity, contract_quantity, contract_price, 'ce_rate', 'ce_quantity', 'ce_price', 
        order_direction, order_fill_quantity, order_id, order_max_price, order_max_rate, order_min_price, order_quantity, 
        order_timestamp, cash, contract_percent, in_market_quantity, projected_profit, in_market_percent, benchmark_pace, 
        ind_ce_profit, realized_surplus, excess_profit, ind_ce_rate, ]
    df['timestamp'] = df.groupby([id_in_subsession, 'id_in_group'])['id_in_group'].cumcount() + 1

    # print('LOSSES')
    # print('group {}'.format(g), 'COUNT\n', 
    #     (df[(df['timestamp'] % (round_length - leave_out_seconds) == 0)]['projected_profit_{}'.format(g)] < 0).sum(), 
    #     (df[(df['timestamp'] % (round_length - leave_out_seconds) == 0) & (df['timestamp'] // (round_length - leave_out_seconds) > ((num_rounds - prac_rounds) // 2))]['projected_profit_{}'.format(g)] < 0).sum(), )
    # print('group {}'.format(g), 'VALUE\n', 
    #     df[(df['timestamp'] % (round_length - leave_out_seconds) == 0) & (df['projected_profit_{}'.format(g)] < 0)]['projected_profit_{}'.format(g)].sum(), 
    #     df[(df['timestamp'] % (round_length - leave_out_seconds) == 0) & (df['timestamp'] // (round_length - leave_out_seconds) > ((num_rounds - prac_rounds) // 2)) & (df['projected_profit_{}'.format(g)] < 0)]['projected_profit_{}'.format(g)].sum(), 
    #     )

    flow_ind.append(df)

data_flow_ind = reduce(lambda left, right:    # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['timestamp', 'id_in_group', 'direction', 'ce_price', 'ce_quantity', 'ce_rate'],
                              ),
                     flow_ind) 

# data_flow_ind = pd.concat(flow_ind, ignore_index=True, sort=False)


# aggregate the individual data by direction
aggregate = {q: 'sum' for q in projected_profits}
data_flow_contract_by_direction = data_flow_ind[data_flow_ind['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0].groupby(['timestamp', 'direction'], as_index=False).agg(aggregate)
data_flow_contract_by_direction['ce_profits'] = 0
data_flow_contract_by_direction['round'] = data_flow_contract_by_direction['timestamp'] // (round_length - leave_out_seconds - leave_out_seconds_end)
for ind, row in data_flow_contract_by_direction.iterrows():
    if row['direction'] == 'buy': 
        data_flow_contract_by_direction.loc[ind, 'ce_profits'] = ce_profit_buy[row['round'] - 1]
    else:
        data_flow_contract_by_direction.loc[ind, 'ce_profits'] = ce_profit_sell[row['round'] - 1]
for g in range(1, len(projected_profits) + 1): 
    data_flow_contract_by_direction['realized_surplus_{}'.format(g)] = data_flow_contract_by_direction[projected_profits[g - 1]] / data_flow_contract_by_direction['ce_profits']
data_flow_contract_by_direction = data_flow_contract_by_direction.drop('timestamp', axis = 1)


mean_flow = []
for g in range(1, num_groups_flow + 1):
    mean_df = data_flow_ind.groupby(['timestamp'], as_index=False)[['benchmark_pace_{}'.format(g), 'projected_profit_{}'.format(g), 'in_market_percent_{}'.format(g)]].apply(lambda x: x.abs().mean())
    mean_flow.append(mean_df)

data_mean_flow = reduce(
    lambda left, right:
    pd.merge(left, right, on = ['timestamp']),
    mean_flow
)

mean_flow_by_direction = []
for g in range(1, num_groups_flow + 1):
    mean_df_by_direction = data_flow_ind.groupby(['timestamp', 'direction'], as_index=False)[['benchmark_pace_{}'.format(g), 'projected_profit_{}'.format(g), 'in_market_percent_{}'.format(g)]].apply(lambda x: x.abs().mean())
    mean_flow_by_direction.append(mean_df_by_direction)

data_mean_flow_by_direction = reduce(
    lambda left, right:
    pd.merge(left, right, on = ['timestamp', 'direction']),
    mean_flow_by_direction
)

sum_flow_by_direction = []
for g in range(1, num_groups_flow + 1):
    sum_df_by_direction = data_flow_ind.groupby(['timestamp', 'direction'], as_index=False)['projected_profit_{}'.format(g)].apply(lambda x: x.abs().sum())
    sum_flow_by_direction.append(sum_df_by_direction)

data_sum_flow_by_direction = reduce(
    lambda left, right:
    pd.merge(left, right, on = ['timestamp', 'direction']),
    sum_flow_by_direction
)

# average benchmark_pace at group level 
plt.figure(figsize=(15, 5))
for l in range(len(benchmark_paces)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_mean_flow[data_mean_flow[benchmark_paces[l]] > 0]['timestamp'], data_mean_flow[data_mean_flow[benchmark_paces[l]] > 0][benchmark_paces[l]], linestyle='solid', c=colors[l], label=lab)
plt.hlines(y=1, xmin=0, xmax=(num_rounds-prac_rounds) * (round_length - leave_out_seconds - leave_out_seconds_end), colors='plum', linestyles='--')
for x in [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]:
    plt.vlines(x, ymin=0, ymax=5, colors='lightgrey', linestyles='dotted')
# plt.legend(bbox_to_anchor=(1, 1),
#         loc='upper left',
#         borderaxespad=.5)
plt.ylim(0, 5)
plt.xlabel('Time')
plt.ylabel('Benchmark Pace')
plt.title('Mean Benchmark Pace vs Time')
plt.savefig('groups_flow_benchmark_pace.png')
plt.close()

# average projected profit at group level 
plt.figure(figsize=(15, 5))
for l in range(len(projected_profits)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_mean_flow['timestamp'], data_mean_flow[projected_profits[l]], linestyle='solid', c=colors[l], label=lab)
# plt.legend(bbox_to_anchor=(1, 1),
#         loc='upper left',
#         borderaxespad=.5)
plt.ylim(0, 1800)
plt.xlabel('Time')
plt.ylabel('Projected Profit')
plt.title('Mean Projected Profit vs Time')
plt.savefig('groups_flow_projected_profit.png')
plt.close()


# average projected profit at group level (by direction)
plt.figure(figsize=(15, 5))
data_mean_flow_by_direction.loc[data_mean_flow_by_direction['direction'] == 'sell', projected_profits] *= -1
df_long = data_mean_flow_by_direction.melt(id_vars=['timestamp', 'direction'], value_vars=projected_profits, var_name='group_id', value_name='projected_profit')
df_long['group_id'] = df_long['group_id'].str.replace('projected_profit', 'group')

sns.lineplot(data=df_long, x='timestamp', y='projected_profit', hue='group_id', style='direction', palette=colors[:num_groups_flow], legend='full')
plt.hlines(y=0, xmin=0, xmax=(num_rounds-prac_rounds) * (round_length - leave_out_seconds - leave_out_seconds_end), colors='plum', linestyles='dotted')
plt.legend(bbox_to_anchor=(1, 1),
        loc='upper left',
        borderaxespad=.5)
plt.ylim(-2550, 2550)
plt.xlabel('Time')
plt.ylabel('Projected Profit (+ for buyers/- for sellers)')
plt.title('Mean Projected Profit vs Time')
plt.savefig('groups_flow_projected_profit_by_direction.png')
plt.close()


end_of_period_profits_flow = data_sum_flow_by_direction[data_sum_flow_by_direction['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0].copy()
end_of_period_profits_flow['period'] = data_sum_flow_by_direction['timestamp'] // (round_length - leave_out_seconds - leave_out_seconds_end)


mean_end_of_period_profits_flow = end_of_period_profits_flow.groupby(['direction'], as_index=False)[projected_profits].mean()
mean_end_of_period_profits_flow['mean'] = mean_end_of_period_profits_flow[projected_profits].mean(axis=1)
mean_end_of_period_profits_flow_half = end_of_period_profits_flow[end_of_period_profits_flow['period'] > (num_rounds - prac_rounds) // 2].groupby(['direction'], as_index=False)[projected_profits].mean()
mean_end_of_period_profits_flow_half['mean'] = mean_end_of_period_profits_flow_half[projected_profits].mean(axis=1)
# print('All 20 periods', mean_end_of_period_profits_flow, '\nLast 10 periods\n', mean_end_of_period_profits_flow_half)
# print(data_mean_flow)

data_flow_ind[in_market_percents] = data_flow_ind[in_market_percents].abs()
individual_agg_data_flow = data_flow_ind[data_flow_ind['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0].groupby('timestamp', as_index=False)[projected_profits + in_market_percents + realized_surpluses].mean()

# average projected profit at group level 
plt.figure(figsize=(15, 5))
for l in range(len(realized_surpluses)): 
    lab = '_group' + str(l + 1)
    plt.plot(individual_agg_data_flow['timestamp'], individual_agg_data_flow[realized_surpluses[l]], linestyle='solid', c=colors[l], label=lab)
# plt.legend(bbox_to_anchor=(1, 1),
#         loc='upper left',
#         borderaxespad=.5)
plt.ylim(-0.1, 1.5)
plt.xlabel('Time')
plt.ylabel('Realized Surplus')
plt.title('Mean Realized Surplus vs Time')
plt.savefig('groups_flow_realized_surplus_ind_truncated.png')
plt.close()


summary_flow_by_direction = data_flow_contract_by_direction
summary_flow_ind_by_direction = data_flow_ind[data_flow_ind['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0][projected_profits + ['ind_ce_profit_1'] + excess_profits + ['direction']].reset_index(drop=True)
summary_flow_ind_by_direction['round'] = [r for r in range(1, num_rounds - prac_rounds + 1) for _ in range(players_per_group)]

print(summary_flow_ind_by_direction)
exit(0)

realized_surplus_buy_flow_full = []
realized_surplus_buy_flow_half = []
realized_surplus_buy_flow_first = []
realized_surplus_buy_flow_test = []

realized_surplus_sell_flow_full = []
realized_surplus_sell_flow_half = []
realized_surplus_sell_flow_first = []
realized_surplus_sell_flow_test = []

profits_buy_flow_ind_full = []
profits_buy_flow_ind_half = []
profits_buy_flow_ind_first = []

profits_sell_flow_ind_full = []
profits_sell_flow_ind_half = []
profits_sell_flow_ind_first = []

excess_profits_buy_flow_ind_full = []
excess_profits_buy_flow_ind_half = []
excess_profits_buy_flow_ind_first = []

excess_profits_sell_flow_ind_full = []
excess_profits_sell_flow_ind_half = []
excess_profits_sell_flow_ind_first = []

for g in range(1, num_groups_flow + 1):
    realized_surplus_buy_flow_full.append(summary_flow_by_direction[summary_flow_by_direction['direction'] == 'buy']['realized_surplus_{}'.format(g)].mean())
    realized_surplus_buy_flow_half.append(summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'buy') & (summary_flow_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_buy_flow_first.append(summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'buy') & (summary_flow_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_buy_flow_test.extend(summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'buy') & (summary_flow_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].tolist())

    realized_surplus_sell_flow_full.append(summary_flow_by_direction[summary_flow_by_direction['direction'] == 'sell']['realized_surplus_{}'.format(g)].mean())
    realized_surplus_sell_flow_half.append(summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'sell') & (summary_flow_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_sell_flow_first.append(summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'sell') & (summary_flow_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].mean())
    realized_surplus_sell_flow_test.extend(summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'sell') & (summary_flow_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus_{}'.format(g)].tolist())

    profits_buy_flow_ind_full.extend(summary_flow_ind_by_direction[summary_flow_ind_by_direction['direction'] == 'buy']['projected_profit_{}'.format(g)].tolist())
    profits_buy_flow_ind_half.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') & (summary_flow_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())
    profits_buy_flow_ind_first.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') & (summary_flow_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())

    profits_sell_flow_ind_full.extend(summary_flow_ind_by_direction[summary_flow_ind_by_direction['direction'] == 'sell']['projected_profit_{}'.format(g)].tolist())
    profits_sell_flow_ind_half.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') & (summary_flow_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())
    profits_sell_flow_ind_first.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') & (summary_flow_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['projected_profit_{}'.format(g)].tolist())

    excess_profits_buy_flow_ind_full.extend(summary_flow_ind_by_direction[summary_flow_ind_by_direction['direction'] == 'buy']['excess_profit_{}'.format(g)].tolist())
    excess_profits_buy_flow_ind_half.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') & (summary_flow_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())
    excess_profits_buy_flow_ind_first.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') & (summary_flow_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())
    
    excess_profits_sell_flow_ind_full.extend(summary_flow_ind_by_direction[summary_flow_ind_by_direction['direction'] == 'sell']['excess_profit_{}'.format(g)].tolist())
    excess_profits_sell_flow_ind_half.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') & (summary_flow_ind_by_direction['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())
    excess_profits_sell_flow_ind_first.extend(summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') & (summary_flow_ind_by_direction['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit_{}'.format(g)].tolist())

# print(
    # realized_surplus_buy_flow_full,
#     realized_surplus_buy_flow_half,
#     realized_surplus_buy_flow_test,
    # realized_surplus_sell_flow_full,
#     realized_surplus_sell_flow_half,
#     realized_surplus_sell_flow_test,
# )
# print(data_flow_contract_by_direction)

# print(profits_buy_flow_ind_full, profits_sell_flow_ind_full, profits_buy_flow_ind_half, profits_sell_flow_ind_half)
# print(len(profits_buy_flow_ind_full), len(profits_sell_flow_ind_full), len(profits_buy_flow_ind_half), len(profits_sell_flow_ind_half))

# cdf of excess profits at ind level
sorted_excess_profit_flow_ind_full = np.sort(excess_profits_buy_flow_ind_full + excess_profits_sell_flow_ind_full)
sorted_excess_profit_buy_flow_ind_full = np.sort(excess_profits_buy_flow_ind_full)
sorted_excess_profit_sell_flow_ind_full = np.sort(excess_profits_sell_flow_ind_full)
cumulative_prob_excess_profit_flow_ind_full = np.arange(1, len(sorted_excess_profit_flow_ind_full) + 1) / len(sorted_excess_profit_flow_ind_full)
cumulative_prob_excess_profit_buy_flow_ind_full = np.arange(1, len(sorted_excess_profit_buy_flow_ind_full) + 1) / len(sorted_excess_profit_buy_flow_ind_full)
cumulative_prob_excess_profit_sell_flow_ind_full = np.arange(1, len(sorted_excess_profit_sell_flow_ind_full) + 1) / len(sorted_excess_profit_sell_flow_ind_full)
plt.figure(figsize=(15, 10))
plt.step(sorted_excess_profit_flow_ind_full, cumulative_prob_excess_profit_flow_ind_full, label='CDF', where='post')
plt.step(sorted_excess_profit_buy_flow_ind_full, cumulative_prob_excess_profit_buy_flow_ind_full, label='buyers', where='post')
plt.step(sorted_excess_profit_sell_flow_ind_full, cumulative_prob_excess_profit_sell_flow_ind_full, label='sellers', where='post')
plt.title('CDF of the Excess Profits (FLOW)')
plt.xlabel('Excess Profits')
plt.ylabel('Probability')
plt.legend()
plt.savefig('groups_flow_excess_profits_cdf.png')
plt.close()