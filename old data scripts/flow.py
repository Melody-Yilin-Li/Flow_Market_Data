import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
import seaborn as sns
import faulthandler; faulthandler.enable()
from functools import reduce           
from sys import exit


# input session constants 
from config import *

directory = '/Users/YilinLi/Documents/UCSC/Flow Data/flow production/'

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


# read in data 

# df for market prices and rates/quantities for all groups 
# create a list of dfs to be merged 
groups_mkt_flow = []
prices = []
rates = []
cum_quantities = []
moving_averages = []
delta_prices_flow_full = []
delta_prices_flow_half = []
delta_prices_flow_first = []
delta_prices_flow_full_interval = []
delta_prices_flow_half_interval = []
delta_prices_flow_first_interval = []
regress_flow = pd.DataFrame()
regress_flow_period = pd.DataFrame()
regress_flow_second = pd.DataFrame()

colors = ['lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'peachpuff', 'lightskyblue', 'lavender', 'moccasin', 'lightsteelblue', 'peachpuff'] # add more colors with more than 6 groups

for g in range(1, num_groups_flow + 1):
    name = 'group' + str(g)
    group = []
    delta_price_full = []
    delta_price_half = []
    delta_price_first = []
    for r in range(1, num_rounds - prac_rounds + 1): 
        path = directory + 'flow{}/{}/1_market.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(
            path,
        )
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)
        rnd['clearing_price'].fillna(method='bfill', inplace=True)
        rnd['clearing_price'].fillna(method='ffill', inplace=True)
        delta_price_full.extend(rnd['clearing_price'].diff())
        if r > (num_rounds - prac_rounds) // 2:
            delta_price_half.extend(rnd['clearing_price'].diff())
        else:
            delta_price_first.extend(rnd['clearing_price'].diff())
        rnd.fillna(0, inplace=True)
        rnd = rnd.drop(columns=['id_in_subsession', 'before_transaction'])
        # rnd['cumulative_quantity'] = rnd['clearing_rate'].cumsum()
        rnd['moving_average'] = rnd['clearing_price'].rolling(window=moving_average_size).mean()

        path_par = directory + 'flow{}/{}/1_participant.json'.format(g, r + prac_rounds)
        par = pd.read_json(
            path_par,
            )
        par = par.explode('executed_contracts')
        par.reset_index(drop=True, inplace=True)
        par = df_explosion(par, 'executed_contracts')
        par = par[(par['before_transaction'] == False)].reset_index(drop=True)
        par['change_in_inventory'] = par[par['timestamp'] < round_length - 1].groupby('id_in_group')['inventory'].diff().abs()
        par['change_in_inventory'].fillna(0, inplace=True)
        par['cumulative_quantity'] = par.groupby(['id_in_group'])['change_in_inventory'].cumsum()
        def calculate_final_inv_change(row):
            return row['change_in_inventory'] + max(0, row['fill_quantity'] - row['cumulative_quantity'])
        par['change_in_inventory'] = par.apply(calculate_final_inv_change, axis=1)
        def calculate_final_volume(row):
            return row['cumulative_quantity'] + max(0, row['fill_quantity'] - row['cumulative_quantity'])
        par['cumulative_quantity'] = par.apply(calculate_final_volume, axis=1)
        par_agg = par.groupby('timestamp', as_index=False).aggregate({'cumulative_quantity': 'sum', 'change_in_inventory': 'sum'}).reset_index(drop=True)
        par_agg['cumulative_quantity'] = par_agg['cumulative_quantity'] / 2
        rnd = pd.merge(rnd, par_agg, on='timestamp', how='left')
        rnd.drop(columns=['clearing_rate'], inplace=True)
        rnd.rename(columns={'change_in_inventory': 'clearing_rate'}, inplace=True)

        # get df for each second
        reg_df_sec = rnd.copy()
        reg_df_sec['round'] = r
        reg_df_sec['group'] = g
        reg_df_sec['block'] = reg_df_sec['round'] // ((num_rounds - prac_rounds) // blocks) + (reg_df_sec['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        reg_df_sec['format'] = 'FLOW'
        reg_df_sec['ce_price'] = ce_price[r - 1]
        reg_df_sec['ce_quantity'] = ce_quantity[r - 1]
        regress_flow_second = pd.concat([regress_flow_second, reg_df_sec], ignore_index=True)


        # compute prices for each 5-second intervals
        reg_df = rnd.copy()
        reg_df['interval'] = (reg_df['timestamp'] // price_interval_size) + 1

        def calculate_difference(group):
            return group.iloc[-1] - group.iloc[0]

        result_reg_df = reg_df.groupby('interval').apply(lambda x: pd.Series({
            'quantity': x['clearing_rate'].sum(),
            'weighted_price': (x['clearing_price'] * x['clearing_rate']).sum() / x['clearing_rate'].sum() if x['clearing_rate'].sum() != 0 else np.nan,
            'price_change_int': calculate_difference(x['clearing_price']),
        })).reset_index()
        result_reg_df['weighted_price'].fillna(method='ffill', inplace=True)
        result_reg_df['weighted_price'].fillna(method='bfill', inplace=True)
        result_reg_df['round'] = r
        result_reg_df['group'] = g
        result_reg_df['block'] = result_reg_df['round'] // ((num_rounds - prac_rounds) // blocks) + (result_reg_df['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        result_reg_df['format'] = 'FLOW'
        result_reg_df['price_change'] = result_reg_df['weighted_price'].diff()
        result_reg_df['price_change_int'] = result_reg_df['price_change_int'] * result_reg_df['quantity'] / result_reg_df['quantity'].sum()
        result_reg_df['ce_price'] = ce_price[r - 1]
        result_reg_df['ce_quantity'] = ce_quantity[r - 1]
        result_reg_df['cum_volume'] = result_reg_df['quantity'].cumsum()
        result_reg_df['%cumsum'] = result_reg_df['cum_volume'] / result_reg_df['ce_quantity']
        
        # print(result_reg_df)
        regress_flow = pd.concat([regress_flow, result_reg_df], ignore_index=True)

        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['timestamp'] < round_length - leave_out_seconds_end)]
        # print(rnd, rnd.columns)
        # exit(0)
        group.append(rnd) 
    delta_prices_flow_full.append(delta_price_full)
    delta_prices_flow_half.append(delta_price_half)
    delta_prices_flow_first.append(delta_price_first)
    df = pd.concat(group, ignore_index=True, sort=False)
    price = 'clearing_price_' + str(g)
    prices.append(price)
    rate = 'clearing_rate_' + str(g)
    rates.append(rate)
    cumsum = 'cumulative_quantity_' + str(g)
    cum_quantities.append(cumsum)
    moving_average = 'moving_average_' + str(g)
    moving_averages.append(moving_average)
    df.columns = ['timestamp', price, rate, cumsum, moving_average]

    df['timestamp'] = np.arange(1, len(df) + 1)
    groups_mkt_flow.append(df)
    
# merge the list of df's
data_groups_mkt_flow = reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['timestamp']),
                     groups_mkt_flow)
data_groups_mkt_flow = data_groups_mkt_flow.replace(0, np.NaN)
data_groups_mkt_flow['mean_clearing_price'] = data_groups_mkt_flow[prices].mean(skipna=True, axis=1)
data_groups_mkt_flow['mean_moving_average_price'] = data_groups_mkt_flow[moving_averages].mean(skipna=True, axis=1)
data_groups_mkt_flow['mean_clearing_rate'] = data_groups_mkt_flow[rates].mean(skipna=True, axis=1)
data_groups_mkt_flow['mean_cumulative_quantity'] = data_groups_mkt_flow[cum_quantities].mean(skipna=True, axis=1)
data_groups_mkt_flow = data_groups_mkt_flow.replace(np.NaN, 0)
data_groups_mkt_flow['ce_price'] = [p for p in ce_price for i in range(round_length - leave_out_seconds - leave_out_seconds_end)]
data_groups_mkt_flow['ce_quantity'] = [q for q in ce_quantity for i in range(round_length - leave_out_seconds - leave_out_seconds_end)]
data_groups_mkt_flow['ce_rate'] = data_groups_mkt_flow['ce_quantity'] / (round_length - leave_out_seconds - leave_out_seconds_end) 


# plot clearing prices in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(len(prices)): 
    lab = '_group' + str(l + 1)
    plt.step(data=data_groups_mkt_flow[data_groups_mkt_flow[prices[l]] > 0], x='timestamp', y=prices[l], where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow[data_groups_mkt_flow['mean_clearing_price'] > 0], x='timestamp', y='mean_clearing_price', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow, x='timestamp', y='ce_price', where='pre', c='plum', label='CE Price')
for x in [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]:
    plt.vlines(x, ymin=0, ymax=20, colors='lightgrey', linestyles='dotted')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.ylim(0, 20)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Flow Transaction Prices vs Time')
plt.savefig('groups_flow_price.png')
plt.close()


# plot clearing rates in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(len(rates)): 
    lab = '_group' + str(l + 1)
    plt.step(data=data_groups_mkt_flow[data_groups_mkt_flow[prices[l]] > 0], x='timestamp', y=rates[l], where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow[data_groups_mkt_flow['mean_clearing_price'] > 0], x='timestamp', y='mean_clearing_rate', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow, x='timestamp', y='ce_rate', where='pre', c='plum', label='CE Rate')
for x in [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]:
    plt.vlines(x, ymin=0, ymax=35, colors='lightgrey', linestyles='dotted')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.ylabel('Shares/second')
plt.ylim(0, 35)
plt.title('Flow Transaction Rates vs Time')
plt.savefig('groups_flow_rate.png')
plt.close()

# plot cumulative quantities in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(len(rates)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_mkt_flow['timestamp'], data_groups_mkt_flow[cum_quantities[l]], c=colors[l], label=lab)
plt.plot(data_groups_mkt_flow['timestamp'], data_groups_mkt_flow['mean_cumulative_quantity'], c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow, x='timestamp', y='ce_quantity', where='pre', c='plum', label='CE Quantity')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.ylabel('Shares')
plt.ylim(0, 2000)
plt.title('Flow Cumulative Quantity vs Time')
plt.savefig('groups_flow_cumsum.png')
plt.close()

# participant-level data 
groups_par_flow = []

for g in range(1, num_groups_flow + 1): 
    # dictionary for market prices and rates/quantities 
    # create a list of dataframes to be concatenated after groupby 
    data_mkt = []

    # each round X is denoted as 'mktX'
    market = {}
    for r in range(1, num_rounds - prac_rounds + 1):
        name = 'mkt' + str(r)
        path = directory + 'flow{}/{}/1_market.json'.format(g, r + prac_rounds)
        market[name] = pd.read_json(
            path,
            )
        market[name].fillna(0, inplace=True)
        # print("market", market[name])
        market[name]['unit_weighted_price'] = market[name]['clearing_price'] * market[name]['clearing_rate']
        df = market[name][market[name]['before_transaction'] == False].groupby('id_in_subsession').aggregate({'clearing_price': 'mean', 'clearing_rate': 'sum', 'unit_weighted_price': 'sum'}).reset_index()
        df['unit_weighted_price'] = df['unit_weighted_price'] / df['clearing_rate']
        df['ce_price'] = ce_price[r - 1]
        df['round'] = r
        df['id_in_subsession'] = g
        df.columns = ['group_id', 'time_weighted_price_' + str(g), 'quantity_' + str(g), 'unit_weighted_price_' + str(g), 'ce_price_' + str(g), 'round']
        df.fillna(0, inplace=True)
        data_mkt.append(df)

    # dictionary for participant cash, inventories, and transaction rates if any
    # create a list of dataframes to be concatenated after groupby 
    data_par = []

    # each round X is denoted as 'parX'
    participant = {}
    for r in range(1, num_rounds - prac_rounds + 1):
        name = 'par' + str(r)
        path = directory + 'flow{}/{}/1_participant.json'.format(g, r + prac_rounds)
        participant[name] = pd.read_json(
            path,
            )
        all_orders = set()
        for idx, row in participant[name].iterrows():
            for o in row['active_orders']:
                if o['order_id'] not in all_orders:
                    all_orders.add(o['order_id'])
        number_of_orders = max(all_orders) - min(all_orders) + 1
        # participant[name]['orders'] = participant[name]['executed_orders'].apply(lambda x: len(x))
        participant[name].fillna(0, inplace=True)
        participant[name] = participant[name].explode('executed_contracts')
        participant[name].reset_index(drop=True, inplace=True)
        participant[name] = df_explosion(participant[name], 'executed_contracts')

        participant[name]['change_in_inventory'] = participant[name][participant[name]['timestamp'] < round_length - 1].groupby('id_in_group')['inventory'].diff().abs()
        participant[name]['change_in_inventory'].fillna(0, inplace=True)
        participant[name]['transacted_quantity'] = participant[name].groupby(['id_in_group'])['change_in_inventory'].cumsum()
        def calculate_final_volume(row):
            return row['transacted_quantity'] + max(0, row['fill_quantity'] - row['transacted_quantity'])
        participant[name]['transacted_quantity'] = participant[name].apply(calculate_final_volume, axis=1)
        
        def calculate_final_inventory(row):
            if row['direction'] == 'sell':
                return row['inventory'] - max(0, row['fill_quantity'] - abs(row['inventory']))
            elif row['direction'] == 'buy':
                return row['inventory'] + max(0, row['fill_quantity'] - abs(row['inventory']))
        participant[name]['inventory'] = participant[name].apply(calculate_final_inventory, axis=1)

        tmp_df = participant[name][(participant[name]['before_transaction'] == False) & (participant[name]['timestamp'] == round_length - 1)]
        df = tmp_df.groupby('id_in_subsession').aggregate({'cash': 'sum', 'fill_quantity': 'sum', 'quantity': 'sum', 'transacted_quantity': 'sum',}).reset_index()
        df['ce_profit'] = ce_profit[r - 1]
        df['ce_quantity'] = ce_quantity[r - 1] 
        df['payoff_percent'] = round(df['cash'] / df['ce_profit'], 4)
        df['contract_percent'] = round(df['fill_quantity'] / df['ce_quantity'] / 2, 4)
        df['round'] = r
        df['orders'] = number_of_orders
        df['id_in_subsession'] = g
        df['transacted_quantity'] = df['transacted_quantity'] / 2
        df['extra_traded_quantity'] = df['transacted_quantity'] - df['fill_quantity'] / 2
        df.columns = ['group_id', 'payoff_{}'.format(g), 'fill_quantity_{}'.format(g), 'contract_quantity_{}'.format(g), 'transacted_quantity_{}'.format(g),
            'ce_profit_{}'.format(g), 'ce_quantity_{}'.format(g), 'payoff_percent_{}'.format(g), 'contract_percent_{}'.format(g), 
            'round', 'orders_{}'.format(g), 'extra_traded_quantity_{}'.format(g)]
        df.fillna(0, inplace=True)
        # print(df.columns)
        # exit(0)
        # print("DF", df)
        data_par.append(df)

    ########## Between-period ##########
    between_df1 = pd.concat(data_mkt, ignore_index=True, axis=0)
    between_df2 = pd.concat(data_par, ignore_index=True, axis=0)
    between_df = pd.merge(between_df1, between_df2, on=['group_id', 'round'])
    between_df['order_size_{}'.format(g)] = 2 *  between_df['quantity_{}'.format(g)] / between_df['orders_{}'.format(g)] 
    between_df = between_df.drop(columns=['group_id'])
    groups_par_flow.append(between_df)


# merge the list of df's
data_groups_par_flow = reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['round']),
                     groups_par_flow)         

data_groups_par_flow = data_groups_par_flow.replace(0, np.NaN)
payoffs = ['payoff_percent_' + str(g) for g in range(1, num_groups_flow + 1)]
contracts = ['contract_percent_' + str(g) for g in range(1, num_groups_flow + 1)]
time_weighted = ['time_weighted_price_' + str(g) for g in range(1, num_groups_flow + 1)]
unit_weighted = ['unit_weighted_price_' + str(g) for g in range(1, num_groups_flow + 1)]
quantities = ['quantity_' + str(g) for g in range(1, num_groups_flow + 1)]
orders = ['orders_' + str(g) for g in range(1, num_groups_flow + 1)]
transacted_quantities = ['transacted_quantity_{}'.format(g) for g in range(1, num_groups_flow + 1)]
extra_traded_quantities = ['extra_traded_quantity_{}'.format(g) for g in range(1, num_groups_flow + 1)]
order_sizes = ['order_size_' + str(g) for g in range(1, num_groups_flow + 1)]
data_groups_par_flow['mean_realized_surplus'] = data_groups_par_flow[payoffs].mean(skipna=True, axis=1)
data_groups_par_flow['mean_contract_execution'] = data_groups_par_flow[contracts].mean(skipna=True, axis=1)
data_groups_par_flow['mean_time_weighted_price'] = data_groups_par_flow[time_weighted].mean(skipna=True, axis=1)
data_groups_par_flow['mean_unit_weighted_price'] = data_groups_par_flow[unit_weighted].mean(skipna=True, axis=1)
data_groups_par_flow['mean_quantity'] = data_groups_par_flow[transacted_quantities].mean(skipna=True, axis=1)
data_groups_par_flow = data_groups_par_flow.replace(np.NaN, 0)


# realized surplus for all groups
plt.figure(figsize=(8, 5))
for l in range(len(payoffs)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[payoffs[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[payoffs[l]] > 0][payoffs[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_realized_surplus'], linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Realized Surplus vs Period')
plt.savefig('groups_flow_surplus.png')
plt.close()

# contract execution for all groups
plt.figure(figsize=(8, 5))
for l in range(len(contracts)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[contracts[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[contracts[l]] > 0][contracts[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_contract_execution'], linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Filled Contract vs Period')
plt.savefig('groups_flow_contract.png')
plt.close()

# traded volume for all groups
plt.figure(figsize=(8, 5))
for l in range(len(transacted_quantities)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[transacted_quantities[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[transacted_quantities[l]] > 0][transacted_quantities[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_quantity'], linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow, x='round', y='ce_quantity_1', where='mid', c='plum', label='CE Quantity')
plt.legend(loc='lower right')
plt.ylim(0, 2000)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Shares')
plt.title('Traded Volume vs Period')
plt.savefig('groups_flow_quantity.png')
plt.close()

# time weighted price for all groups
plt.figure(figsize=(8, 5))
for l in range(len(time_weighted)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[time_weighted[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[time_weighted[l]] > 0][time_weighted[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_time_weighted_price'], linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow, x='round', y='ce_price_1', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Time-Weighted Price vs Period')
plt.savefig('groups_flow_time_weighted_price.png')
plt.close()

# unit weighted price for all groups
plt.figure(figsize=(8, 5))
for l in range(len(unit_weighted)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[data_groups_par_flow[unit_weighted[l]] > 0]['round'], data_groups_par_flow[data_groups_par_flow[unit_weighted[l]] > 0][unit_weighted[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow['round'], data_groups_par_flow['mean_unit_weighted_price'], linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow, x='round', y='ce_price_1', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Unit-Weighted Price vs Period')
plt.savefig('groups_flow_unit_weighted_price.png')
plt.close()


########## ---------- summary_flow statistics ---------- ##########
summary_flow = data_groups_par_flow[['round', 'ce_price_1'] + unit_weighted + payoffs + contracts + transacted_quantities + orders + order_sizes + extra_traded_quantities]
summary_flow = summary_flow.rename(columns={'ce_price_1': 'ce_price'})

for g in range(1, num_groups_flow + 1):
    summary_flow['price_dev_{}'.format(g)] = 0
    for ind, row in summary_flow.iterrows():
        if row['ce_price'] == 9:
            if 8 <= row['unit_weighted_price_{}'.format(g)] <= 10: 
                summary_flow.at[ind, 'price_dev_{}'.format(g)] = 0  
            elif row['unit_weighted_price_{}'.format(g)] > 10:
                summary_flow.at[ind, 'price_dev_{}'.format(g)] =  row['unit_weighted_price_{}'.format(g)] - 10
            else:
                summary_flow.at[ind, 'price_dev_{}'.format(g)] =  8 - row['unit_weighted_price_{}'.format(g)]
        else: 
            summary_flow.at[ind, 'price_dev_{}'.format(g)] = abs(row['unit_weighted_price_{}'.format(g)] - row['ce_price'])

price_deviation_flow_full = []
price_deviation_flow_half = []
price_deviation_flow_first = []
price_deviation_flow_test = []

percent_contract_flow_full = []
percent_contract_flow_half = []
percent_contract_flow_first = []
percent_contract_flow_test = []

realized_surplus_flow_full = []
realized_surplus_flow_half = []
realized_surplus_flow_first = []
realized_surplus_flow_test = []

total_quantity_flow_full = []
total_quantity_flow_half = []
total_quantity_flow_first = []
total_quantity_flow_test = []

price_volatility_flow_full = []
price_volatility_flow_half = []
price_volatility_flow_first = []

clearing_rate_flow_full = []
clearing_rate_flow_half = []
clearing_rate_flow_first = []

order_number_flow_full = []
order_number_flow_half = []
order_number_flow_first = []
order_number_flow_test = []

order_size_flow_full = []
order_size_flow_half = []
order_size_flow_first = []
order_size_flow_test = []

extra_traded_quantities_flow_full = []
extra_traded_quantities_flow_half = []
extra_traded_quantities_flow_first = []
extra_traded_quantities_flow_test = []

for g in range(1, num_groups_flow + 1):
    price_deviation_flow_full.extend(summary_flow['price_dev_{}'.format(g)].tolist())
    price_deviation_flow_half.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['price_dev_{}'.format(g)].tolist())
    price_deviation_flow_first.extend(summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['price_dev_{}'.format(g)].tolist())
    price_deviation_flow_test.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['price_dev_{}'.format(g)].tolist())
    
    realized_surplus_flow_full.extend(summary_flow['payoff_percent_{}'.format(g)].tolist())
    realized_surplus_flow_half.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['payoff_percent_{}'.format(g)].tolist())
    realized_surplus_flow_first.extend(summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['payoff_percent_{}'.format(g)].tolist())
    realized_surplus_flow_test.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['payoff_percent_{}'.format(g)].tolist())
    
    percent_contract_flow_full.extend(summary_flow['contract_percent_{}'.format(g)].tolist())
    percent_contract_flow_half.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['contract_percent_{}'.format(g)].tolist())
    percent_contract_flow_first.extend(summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['contract_percent_{}'.format(g)].tolist())
    percent_contract_flow_test.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['contract_percent_{}'.format(g)].tolist())
    
    total_quantity_flow_full.extend(summary_flow['transacted_quantity_{}'.format(g)].tolist())
    total_quantity_flow_half.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity_{}'.format(g)].tolist())
    total_quantity_flow_first.extend(summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['transacted_quantity_{}'.format(g)].tolist())
    total_quantity_flow_test.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity_{}'.format(g)].tolist())
    
    price_volatility_flow_full.extend(data_groups_mkt_flow[data_groups_mkt_flow['clearing_price_{}'.format(g)] > 0]['clearing_price_{}'.format(g)].tolist())
    price_volatility_flow_half.extend(data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price_{}'.format(g)] > 0) & (data_groups_mkt_flow['timestamp'] > (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price_{}'.format(g)].tolist())
    price_volatility_flow_first.extend(data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price_{}'.format(g)] > 0) & (data_groups_mkt_flow['timestamp'] <= (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price_{}'.format(g)].tolist())
    
    clearing_rate_flow_full.extend(data_groups_mkt_flow[data_groups_mkt_flow['clearing_rate_{}'.format(g)] > 0]['clearing_rate_{}'.format(g)].tolist())
    clearing_rate_flow_half.extend(data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate_{}'.format(g)] > 0) & (data_groups_mkt_flow['timestamp'] > (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_rate_{}'.format(g)].tolist())
    clearing_rate_flow_first.extend(data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate_{}'.format(g)] > 0) & (data_groups_mkt_flow['timestamp'] <= (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_rate_{}'.format(g)].tolist())

    order_number_flow_full.extend(summary_flow['orders_{}'.format(g)].tolist())
    order_number_flow_half.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['orders_{}'.format(g)].tolist())
    order_number_flow_first.extend(summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['orders_{}'.format(g)].tolist())
    order_number_flow_test.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['orders_{}'.format(g)])

    order_size_flow_full.extend(summary_flow['order_size_{}'.format(g)].tolist())
    order_size_flow_half.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].tolist())
    order_size_flow_first.extend(summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].tolist())
    order_size_flow_test.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)])

    extra_traded_quantities_flow_full.extend(summary_flow['order_size_{}'.format(g)].tolist())
    extra_traded_quantities_flow_half.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].tolist())
    extra_traded_quantities_flow_first.extend(summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].tolist())
    extra_traded_quantities_flow_test.extend(summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)])

    regress_df = pd.DataFrame(
        {
            'round': [i for i in range(1, (num_rounds - prac_rounds) + 1)],
            'block': [i for i in range(1, blocks + 1) for _ in range((num_rounds - prac_rounds) // blocks) ], 
            'price_deviation': summary_flow['price_dev_{}'.format(g)].tolist(),
            'realized_surplus': summary_flow['payoff_percent_{}'.format(g)].tolist(),
            'traded_volume': summary_flow['transacted_quantity_{}'.format(g)].tolist(),
            'filled_contract': summary_flow['contract_percent_{}'.format(g)].tolist(), 
            'format' : ['FLOW' for _ in range(num_rounds - prac_rounds)],
            'group': [g for _ in range(num_rounds - prac_rounds)],
            'ce_quantity': ce_quantity, 
        }
    )
    regress_flow_period = pd.concat([regress_flow_period, regress_df], ignore_index=True)


regress_flow_period['filled_ce_quantity'] = regress_flow_period['traded_volume'] / regress_flow_period['ce_quantity']
# regress_flow_period['order_number'] = order_number_flow_full


regress_flow['price_deviation'] = 0
for ind, row in regress_flow.iterrows():
        if row['block'] == 3:
            if 8 <= row['weighted_price'] <= 10: 
                regress_flow.at[ind, 'price_deviation'] = 0 
            elif row['weighted_price'] > 10: 
                regress_flow.at[ind, 'price_deviation'] =  row['weighted_price'] - 10
            else:
                regress_flow.at[ind, 'price_deviation'] =  8 - row['weighted_price']   
        else:
            regress_flow.at[ind, 'price_deviation'] = abs(row['weighted_price'] - ce_price[row['round'] - 1])

print(regress_flow)
print(regress_flow_period)