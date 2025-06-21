import numpy as np 
import pandas as pd
import itertools 
import statistics
import matplotlib.pyplot as plt 
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
import seaborn as sns
import faulthandler; faulthandler.enable()
from functools import reduce                # Import reduce function
from sys import exit

directory = '/Users/YilinLi/Documents/UCSC/Flow Data/flow production/data/'

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

# input session constants 
from config import *

def main():
    print("params imported")

if __name__ == "__main__":
     main()

# read in data 

# df for market prices and rates/quantities for all groups 
# create a list of dfs to be merged 
groups_mkt_cda = []
prices = []
quantities = []
cum_quantities = []
moving_averages = []
delta_prices_cda_full = []
delta_prices_cda_half = []
delta_prices_cda_first = []
regress_cda = pd.DataFrame()
regress_cda_period = pd.DataFrame()
regress_cda_second = pd.DataFrame()

colors = ['lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'peachpuff', 'lightskyblue'] # add more colors with more than 6 groups
volume_volatility_cda_full = []
volume_volatility_cda_half = []
volume_volatility_cda_first = []
transaction_numbers = ['transactions_{}'.format(g) for g in range(1, num_groups_cda + 1)]


for g in range(1, num_groups_cda + 1):
    name = 'group' + str(g)
    group = []
    delta_price_full = []
    delta_price_half = []
    delta_price_first = []

    for r in range(1, num_rounds - prac_rounds + 1): 
        path_mkt = directory + 'cda{}/{}/1_market.json'.format(g, r + prac_rounds)
        mkt = pd.read_json(
            path_mkt,
        )
        # mkt.fillna(0, inplace=True)
        mkt = mkt[(mkt['before_transaction'] == False)].reset_index(drop=True)
        mkt['clearing_price'].fillna(method='bfill', inplace=True)
        mkt['clearing_price'].fillna(method='ffill', inplace=True)
        delta_price_full.extend(mkt['clearing_price'].diff())
        if r > (num_rounds - prac_rounds) // 2:
            delta_price_half.extend(mkt['clearing_price'].diff())
        else:
            delta_price_first.extend(mkt['clearing_price'].diff())
        mkt.fillna(0, inplace=True)
        mkt = mkt.drop(columns=['id_in_subsession', 'before_transaction'])
        mkt['moving_average'] = mkt['clearing_price'].rolling(window=moving_average_size).mean()
        path_par = directory + 'cda{}/{}/1_participant.json'.format(g, r + prac_rounds)
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
        # print(g, r, sum(par_agg['change_in_inventory']) == par_agg['cumulative_quantity'].iloc[-1])
        # if not sum(par_agg['change_in_inventory']) == par_agg['cumulative_quantity'].iloc[-1]:
            # print(sum(par_agg['change_in_inventory']), par_agg['cumulative_quantity'].iloc[-1])
        # print(par_agg)
        par_agg['cumulative_quantity'] = par_agg['cumulative_quantity'] / 2
        par_agg['change_in_inventory'] = par_agg['change_in_inventory'] / 2
        mkt = pd.merge(mkt, par_agg, on='timestamp', how='left')

        mkt.drop(columns=['clearing_rate'], inplace=True)
        mkt.rename(columns={'change_in_inventory': 'clearing_rate'}, inplace=True)
        mkt['transactions'] = (mkt['clearing_rate'] > 0).sum()


        volume_volatility_cda_full.extend(mkt[mkt['clearing_rate'] > 0]['clearing_rate'].tolist())
        if r > (num_rounds - prac_rounds) // 2:
            volume_volatility_cda_half.extend(mkt[(mkt['clearing_rate'] > 0)]['clearing_rate'].tolist())
        else:
            volume_volatility_cda_first.extend(mkt[(mkt['clearing_rate'] > 0)]['clearing_rate'].tolist())


        # get df for each second
        reg_df_sec = mkt.copy()
        reg_df_sec['round'] = r
        reg_df_sec['group'] = g
        reg_df_sec['block'] = reg_df_sec['round'] // ((num_rounds - prac_rounds) // blocks) + (reg_df_sec['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        reg_df_sec['format'] = 'CDA'
        reg_df_sec['ce_price'] = ce_price[r - 1]
        reg_df_sec['ce_quantity'] = ce_quantity[r - 1]
        regress_cda_second = pd.concat([regress_cda_second, reg_df_sec], ignore_index=True)

        # compute prices for each 5-second intervals
        reg_df = mkt.copy()
        reg_df['interval'] = (reg_df['timestamp'] // price_interval_size) + 1
        # print(reg_df)

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
        result_reg_df['format'] = 'CDA'
        result_reg_df['price_change'] = result_reg_df['weighted_price'].diff()
        result_reg_df['price_change_int'] = result_reg_df['price_change_int'] * result_reg_df['quantity'] / result_reg_df['quantity'].sum()
        result_reg_df['ce_price'] = ce_price[r - 1]
        result_reg_df['ce_quantity'] = ce_quantity[r - 1]
        result_reg_df['cum_volume'] = result_reg_df['quantity'].cumsum()
        result_reg_df['%cumsum'] = result_reg_df['cum_volume'] / result_reg_df['ce_quantity']

        # print(result_reg_df)
        regress_cda = pd.concat([regress_cda, result_reg_df], ignore_index=True)
        
        mkt = mkt[(mkt['timestamp'] >= leave_out_seconds) & (mkt['timestamp'] < round_length - leave_out_seconds_end)].reset_index(drop=True)
        group.append(mkt)


    delta_prices_cda_full.append(delta_price_full)
    delta_prices_cda_half.append(delta_price_half)
    delta_prices_cda_first.append(delta_price_first)
    df = pd.concat(group, ignore_index=True, sort=False)

    price = 'clearing_price_' + str(g)
    prices.append(price)
    quantity = 'clearing_quantity_' + str(g)
    quantities.append(quantity)
    cumsum = 'cumulative_quantity' + str(g)
    cum_quantities.append(cumsum)
    moving_average = 'moving_average_' + str(g)
    moving_averages.append(moving_average)

    transaction = 'transactions_' + str(g)
    
    df.columns = ['timestamp', price, moving_average, cumsum, quantity, transaction]
    df['timestamp'] = np.arange(1, len(df) + 1)

    # plot price vs quantity for each transaction
    # plt.plot(df[quantity][np.isfinite(df[quantity].astype(np.double))], df[price][np.isfinite(df[price].astype(np.double))], linestyle='dashed', marker='o')
    # plt.show()
    
    groups_mkt_cda.append(df)
    
# merge the list of df's
data_groups_mkt_cda = reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['timestamp']),
                     groups_mkt_cda)

data_groups_mkt_cda = data_groups_mkt_cda.replace(0, np.NaN)
data_groups_mkt_cda['mean_clearing_price'] = data_groups_mkt_cda[prices].mean(skipna=True, axis=1)
data_groups_mkt_cda['mean_moving_average_price'] = data_groups_mkt_cda[moving_averages].mean(skipna=True, axis=1)
data_groups_mkt_cda['mean_clearing_quantity'] = data_groups_mkt_cda[quantities].mean(skipna=True, axis=1)
data_groups_mkt_cda['mean_cumulative_quantity'] = data_groups_mkt_cda[cum_quantities].mean(skipna=True, axis=1)
data_groups_mkt_cda = data_groups_mkt_cda.replace(np.NaN, 0)
data_groups_mkt_cda['ce_price'] = [p for p in ce_price for i in range(round_length - leave_out_seconds - leave_out_seconds_end)]
data_groups_mkt_cda['ce_quantity'] = [q for q in ce_quantity for i in range(round_length - leave_out_seconds - leave_out_seconds_end)]

# plot clearing prices in all rounds for all groups 

## with scatter points
plt.figure(figsize=(20, 5))
for l in range(len(prices)): 
    lab = '_group' + str(l + 1)
    plt.scatter(data=data_groups_mkt_cda[data_groups_mkt_cda[prices[l]] > 0], x='timestamp', y=prices[l], c=colors[l], s=5,
        label=lab
        )
plt.scatter(data=data_groups_mkt_cda[data_groups_mkt_cda['mean_clearing_price'] > 0], x='timestamp', y='mean_clearing_price', c='green', label='Mean Price', s=10,)
plt.step(data=data_groups_mkt_cda, x='timestamp', y='ce_price', where='pre', c='plum', label='CE Price')
for x in [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]:
    plt.vlines(x, ymin=0, ymax=20, colors='lightgrey', linestyles='dotted')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.ylim(0, 20)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('CDA Transaction Prices vs Time')
plt.savefig('groups_cda_price.png')
plt.close()


# plot cumulative quantities in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(len(cum_quantities)): 
    lab = '_group' + str(l + 1)
    plt.step(data=data_groups_mkt_cda, x='timestamp', y=cum_quantities[l], where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_cda, x='timestamp', y='mean_cumulative_quantity', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_cda, x='timestamp', y='ce_quantity', where='pre', c='plum', label='CE Quantity')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.ylabel('Shares')
plt.ylim(0, 2000)
plt.title('CDA Cumulative Quantity vs Time')
plt.savefig('groups_cda_cumsum.png')
plt.close()

# participant-level data 
groups_par_cda = []

for g in range(1, num_groups_cda + 1): 
    # dictionary for market prices and rates/quantities 
    # create a list of dataframes to be concatenated after groupby 
    data_mkt = []

    # each round X is denoted as 'mktX'
    market = {}
    for r in range(1, num_rounds - prac_rounds + 1):
        name = 'mkt' + str(r)
        # path = '/Users/YilinLi/Downloads/flow production/cda{}/{}/1_market.json'.format(g, r + prac_rounds)
        # market[name] = pd.read_json(
        #     path,
        #     )
        # market[name] = market[name][market[name]['before_transaction'] == False]
        market[name] = groups_mkt_cda[g - 1].iloc[(round_length - leave_out_seconds - leave_out_seconds_end) * (r - 1): (round_length - leave_out_seconds - leave_out_seconds_end) * r].copy()
        market[name].columns = ['timestamp', 'clearing_price', 'moving_average', 'cumulative_quantity', 'clearing_rate', 'transactions']
        # market[name].fillna(0, inplace=True)
        market[name]['unit_weighted_price'] = market[name]['clearing_price'] * market[name]['clearing_rate']
        df = pd.DataFrame({
                'clearing_rate': [market[name]['clearing_rate'].sum()], 
                'unit_weighted_price': [market[name]['unit_weighted_price'].sum()], 
                'transactions': [market[name]['transactions'].mean()],
            })
        # df = market[name].groupby('id_in_subsession').aggregate({'clearing_rate': 'sum', 'unit_weighted_price': 'sum'}).reset_index()
        df['unit_weighted_price'] = df['unit_weighted_price'] / df['clearing_rate']
        df['ce_price'] = ce_price[r - 1]
        df['round'] = r
        df['group_id'] = g
        # df['transactions'] = (market[name]['clearing_rate'] > 0).sum()
        df.columns = ['quantity_{}'.format(g), 'unit_weighted_price_{}'.format(g), 'transactions_{}'.format(g),
            'ce_price_{}'.format(g), 'round', 'group_id',]
        data_mkt.append(df)
        # print(name, '\n', market[name])

    # print('MARKET', data_mkt)

    # dictionary for participant cash, inventories, and transaction rates if any
    # create a list of dataframes to be concatenated after groupby 
    data_par = []

    # each round X is denoted as 'parX'
    participant = {}
    for r in range(1, num_rounds - prac_rounds + 1):
        name = 'par' + str(r)
        path = directory + 'cda{}/{}/1_participant.json'.format(g, r + prac_rounds)
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

        
        # transactions = 0
        # for t in range(round_length):
        #     cur = participant[name][(participant[name]['timestamp'] == t) & (participant[name]['before_transaction'] == False)]
        #     cur.reset_index(drop=True, inplace=True)
        #     buy = (cur.head(players_per_group // 2)['change_in_inventory'] != 0).sum()
        #     sell = (cur.tail(players_per_group // 2)['change_in_inventory'] != 0).sum()
        #     transactions += min(buy, sell) 

        tmp_df = participant[name][(participant[name]['before_transaction'] == False) & (participant[name]['timestamp'] == round_length - 1)]
        # print("TMP", tmp_df[['id_in_group', 'transacted_quantity', 'cash','inventory','quantity', 'fill_quantity']], tmp_df.columns)
        # exit(0)
        # print('round', r)
        df = tmp_df.groupby('id_in_subsession').aggregate({'cash': 'sum', 'fill_quantity': 'sum', 'quantity': 'sum', 'transacted_quantity': 'sum',}).reset_index()
        df['ce_profit'] = ce_profit[r - 1]
        df['ce_quantity'] = ce_quantity[r - 1] 
        df['payoff_percent'] = round(df['cash'] / df['ce_profit'], 4)
        df['contract_percent'] = round(df['fill_quantity'] / df['ce_quantity'] / 2, 4)
        df['round'] = r
        df['orders'] = number_of_orders
        # df['transactions'] = transactions
        df['id_in_subsession'] = g
        df['transacted_quantity'] = df['transacted_quantity'] / 2
        df['extra_traded_quantity'] = df['transacted_quantity'] - df['fill_quantity'] / 2
        df.columns = ['group_id', 'payoff_{}'.format(g), 'fill_quantity_{}'.format(g), 'contract_quantity_{}'.format(g), 'transacted_quantity_{}'.format(g), 'ce_profit_{}'.format(g), 
            'ce_quantity_{}'.format(g), 'payoff_percent_{}'.format(g), 'contract_percent_{}'.format(g), 'round', 'orders_{}'.format(g), 'extra_traded_quantity_{}'.format(g)]
        df.fillna(0, inplace=True)
        data_par.append(df)

    ########## Between-period ##########
    between_df1 = pd.concat(data_mkt, ignore_index=True, axis=0)
    between_df2 = pd.concat(data_par, ignore_index=True, axis=0)
    between_df = pd.merge(between_df1, between_df2, on=['group_id', 'round'])
    between_df['order_size_{}'.format(g)] = 2 * between_df['quantity_{}'.format(g)] / between_df['orders_{}'.format(g)] 
    between_df = between_df.drop(columns=['group_id'])
    # print('here\n', between_df)

    groups_par_cda.append(between_df)


# merge the list of df's
data_groups_par_cda = reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ['round']),
                     groups_par_cda)
data_groups_par_cda = data_groups_par_cda.replace(0, np.NaN)
payoffs = ['payoff_percent_{}'.format(g) for g in range(1, num_groups_cda + 1)]
contracts = ['contract_percent_{}'.format(g) for g in range(1, num_groups_cda + 1)]
unit_weighted = ['unit_weighted_price_{}'.format(g) for g in range(1, num_groups_cda + 1)]
quantities = ['quantity_{}'.format(g) for g in range(1, num_groups_cda + 1)]
orders = ['orders_{}'.format(g) for g in range(1, num_groups_cda + 1)]
# transaction_numbers = ['transactions_{}'.format(g) for g in range(1, num_groups_cda + 1)]
transacted_quantities = ['transacted_quantity_{}'.format(g) for g in range(1, num_groups_cda + 1)]
extra_traded_quantities = ['extra_traded_quantity_{}'.format(g) for g in range(1, num_groups_cda + 1)]
order_sizes = ['order_size_{}'.format(g) for g in range(1, num_groups_cda + 1)]
data_groups_par_cda['mean_realized_surplus'] = data_groups_par_cda[payoffs].mean(skipna=True, axis=1)
data_groups_par_cda['mean_contract_execution'] = data_groups_par_cda[contracts].mean(skipna=True, axis=1)
data_groups_par_cda['mean_unit_weighted_price'] = data_groups_par_cda[unit_weighted].mean(skipna=True, axis=1)
data_groups_par_cda['mean_quantity'] = data_groups_par_cda[quantities].mean(skipna=True, axis=1)
data_groups_par_cda = data_groups_par_cda.replace(np.NaN, 0)

# realized surplus for all groups
plt.figure(figsize=(8, 5))
for l in range(len(payoffs)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda['round'], data_groups_par_cda[payoffs[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda['round'], data_groups_par_cda['mean_realized_surplus'], linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Realized Surplus vs Period')
plt.savefig('groups_cda_surplus.png')
plt.close()

# contract execution for all groups
plt.figure(figsize=(8, 5))
for l in range(len(contracts)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda['round'], data_groups_par_cda[contracts[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda['round'], data_groups_par_cda['mean_contract_execution'], linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Filled Contract vs Period')
plt.savefig('groups_cda_contract.png')
plt.close()

# traded volume for all groups
plt.figure(figsize=(8, 5))
for l in range(len(quantities)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda['round'], data_groups_par_cda[quantities[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda['round'], data_groups_par_cda['mean_quantity'], linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_cda, x='round', y='ce_quantity_1', where='mid', c='plum', label='CE Quantity')
plt.legend(loc='lower right')
plt.ylim(0, 2000)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Shares')
plt.title('Traded Volume vs Period')
plt.savefig('groups_cda_quantity.png')
plt.close()

# unit weighted price for all groups
plt.figure(figsize=(8, 5))
for l in range(len(unit_weighted)): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda['round'], data_groups_par_cda[unit_weighted[l]], linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda['round'], data_groups_par_cda['mean_unit_weighted_price'], linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_cda, x='round', y='ce_price_1', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Unit-Weighted Price vs Period')
plt.savefig('groups_cda_unit_weighted_price.png')
plt.close()



########## ---------- summary_cda statistics ---------- ##########
summary_cda = data_groups_par_cda[['round', 'ce_price_1'] + unit_weighted + payoffs + contracts + transacted_quantities + orders + order_sizes + transaction_numbers + extra_traded_quantities]
summary_cda = summary_cda.rename(columns={'ce_price_1': 'ce_price'})

for g in range(1, num_groups_cda + 1):
    summary_cda['price_dev_{}'.format(g)] = 0
    for ind, row in summary_cda.iterrows():
        if row['ce_price'] == 9:
            if 8 <= row['unit_weighted_price_{}'.format(g)] <= 10: 
                summary_cda.at[ind, 'price_dev_{}'.format(g)] = 0  
            elif row['unit_weighted_price_{}'.format(g)] > 10:
                summary_cda.at[ind, 'price_dev_{}'.format(g)] =  row['unit_weighted_price_{}'.format(g)] - 10
            else:
                summary_cda.at[ind, 'price_dev_{}'.format(g)] =  8 - row['unit_weighted_price_{}'.format(g)]
        else: 
            summary_cda.at[ind, 'price_dev_{}'.format(g)] = abs(row['unit_weighted_price_{}'.format(g)] - row['ce_price'])


price_deviation_cda_full = []
price_deviation_cda_half = []
price_deviation_cda_first = []
price_deviation_cda_test = []

realized_surplus_cda_full = []
realized_surplus_cda_half = []
realized_surplus_cda_first = []
realized_surplus_cda_test = []

percent_contract_cda_full = []
percent_contract_cda_half = []
percent_contract_cda_first = []
percent_contract_cda_test = []

total_quantity_cda_full = []
total_quantity_cda_half = []
total_quantity_cda_first = []
total_quantity_cda_test = []

price_volatility_cda_full = []
price_volatility_cda_half = []
price_volatility_cda_first = []

# volume_volatility_cda_full = []
# volume_volatility_cda_half = []
# volume_volatility_cda_first = []

order_number_cda_full = []
order_number_cda_half = []
order_number_cda_first = []
order_number_cda_test = []

order_size_cda_full = []
order_size_cda_half = []
order_size_cda_first = []
order_size_cda_test = [] 

extra_traded_quantities_cda_full = []
extra_traded_quantities_cda_half = []
extra_traded_quantities_cda_first = []
extra_traded_quantities_cda_test = []

transaction_number_cda_full = []
transaction_number_cda_half = []
transaction_number_cda_first = []

for g in range(1, num_groups_cda + 1):
    price_deviation_cda_full.extend(summary_cda['price_dev_{}'.format(g)].tolist())
    price_deviation_cda_half.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['price_dev_{}'.format(g)].tolist())
    price_deviation_cda_first.extend(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['price_dev_{}'.format(g)].tolist())
    price_deviation_cda_test.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['price_dev_{}'.format(g)].tolist())
    
    realized_surplus_cda_full.extend(summary_cda['payoff_percent_{}'.format(g)].tolist())
    realized_surplus_cda_half.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['payoff_percent_{}'.format(g)].tolist())
    realized_surplus_cda_first.extend(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['payoff_percent_{}'.format(g)].tolist())
    realized_surplus_cda_test.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2 ]['payoff_percent_{}'.format(g)].tolist())
    
    percent_contract_cda_full.extend(summary_cda['contract_percent_{}'.format(g)].tolist())
    percent_contract_cda_half.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['contract_percent_{}'.format(g)].tolist())
    percent_contract_cda_first.extend(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['contract_percent_{}'.format(g)].tolist())
    percent_contract_cda_test.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['contract_percent_{}'.format(g)].tolist())
    
    total_quantity_cda_full.extend(summary_cda['transacted_quantity_{}'.format(g)].tolist())
    total_quantity_cda_half.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity_{}'.format(g)].tolist())
    total_quantity_cda_first.extend(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['transacted_quantity_{}'.format(g)].tolist())
    total_quantity_cda_test.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity_{}'.format(g)].tolist())
    
    price_volatility_cda_full.extend(data_groups_mkt_cda[data_groups_mkt_cda['clearing_price_{}'.format(g)] > 0]['clearing_price_{}'.format(g)].tolist())
    price_volatility_cda_half.extend(data_groups_mkt_cda[(data_groups_mkt_cda['clearing_price_{}'.format(g)] > 0) & (data_groups_mkt_cda['timestamp'] > (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price_{}'.format(g)].tolist())
    price_volatility_cda_first.extend(data_groups_mkt_cda[(data_groups_mkt_cda['clearing_price_{}'.format(g)] > 0) & (data_groups_mkt_cda['timestamp'] <= (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price_{}'.format(g)].tolist())
    
    # volume_volatility_cda_full.extend(data_groups_mkt_cda[data_groups_mkt_cda['clearing_quantity_{}'.format(g)] > 0]['clearing_quantity_{}'.format(g)].tolist())
    # volume_volatility_cda_half.extend(data_groups_mkt_cda[(data_groups_mkt_cda['clearing_quantity_{}'.format(g)] > 0) & (data_groups_mkt_cda['timestamp'] > (round_length - leave_out_seconds) * (num_rounds - prac_rounds) // 2)]['clearing_quantity_{}'.format(g)].tolist())
    # volume_volatility_cda_first.extend(data_groups_mkt_cda[(data_groups_mkt_cda['clearing_quantity_{}'.format(g)] > 0) & (data_groups_mkt_cda['timestamp'] <= (round_length - leave_out_seconds) * (num_rounds - prac_rounds) // 2)]['clearing_quantity_{}'.format(g)].tolist())

    order_number_cda_full.extend(summary_cda['orders_{}'.format(g)].tolist())
    order_number_cda_half.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['orders_{}'.format(g)].tolist())
    order_number_cda_first.extend(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['orders_{}'.format(g)].tolist())
    order_number_cda_test.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['orders_{}'.format(g)])

    order_size_cda_full.extend(summary_cda['order_size_{}'.format(g)].tolist())
    order_size_cda_half.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].tolist())
    order_size_cda_first.extend(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].tolist())
    order_size_cda_test.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)])

    transaction_number_cda_full.extend(summary_cda['transactions_{}'.format(g)].tolist())
    transaction_number_cda_half.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['transactions_{}'.format(g)].tolist())
    transaction_number_cda_first.extend(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['transactions_{}'.format(g)].tolist())

    extra_traded_quantities_cda_full.append(summary_cda['order_size_{}'.format(g)].mean())
    extra_traded_quantities_cda_half.append(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].mean())
    extra_traded_quantities_cda_first.append(summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)].mean())
    extra_traded_quantities_cda_test.extend(summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size_{}'.format(g)])


    regress_df = pd.DataFrame(
        {
            'round': [i for i in range(1, (num_rounds - prac_rounds) + 1)],
            'block': [i for i in range(1, blocks + 1) for _ in range((num_rounds - prac_rounds) // blocks) ], 
            'price_deviation': summary_cda['price_dev_{}'.format(g)].tolist(),
            'realized_surplus': summary_cda['payoff_percent_{}'.format(g)].tolist(),
            'traded_volume': summary_cda['transacted_quantity_{}'.format(g)].tolist(),
            'filled_contract': summary_cda['contract_percent_{}'.format(g)].tolist(), 
            'format' : ['CDA' for _ in range(num_rounds - prac_rounds)],
            'group': [g for _ in range(num_rounds - prac_rounds)],
            'ce_quantity': ce_quantity, 
        }
    )
    regress_cda_period = pd.concat([regress_cda_period, regress_df], ignore_index=True)


regress_cda_period['filled_ce_quantity'] = regress_cda_period['traded_volume'] / regress_cda_period['ce_quantity']
# regress_cda_period['order_number'] = order_number_cda_full

regress_cda['price_deviation'] = 0
for ind, row in regress_cda.iterrows():
        if row['block'] == 3:
            if 8 <= row['weighted_price'] <= 10: 
                regress_cda.at[ind, 'price_deviation'] = 0 
            elif row['weighted_price'] > 10: 
                regress_cda.at[ind, 'price_deviation'] =  row['weighted_price'] - 10
            else:
                regress_cda.at[ind, 'price_deviation'] =  8 - row['weighted_price']
        else: 
            regress_cda.at[ind, 'price_deviation'] = abs(row['weighted_price'] - ce_price[row['round'] - 1])

print(regress_cda)
print(regress_cda_period)
