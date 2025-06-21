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

# input session constants 
from config import *

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


def main():
    print("params imported")

if __name__ == "__main__":
     main()

# read in data 

# df for market prices and rates/quantities for all groups 
# create a list of dfs to be merged 
groups_mkt_cda = []

delta_prices_cda_full = []
delta_prices_cda_half = []
delta_prices_cda_first = []
regress_cda = pd.DataFrame()
regress_cda_period = pd.DataFrame()
regress_cda_second = pd.DataFrame()

colors = [
    'lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'lightcoral', 'lightskyblue', 'pink',
    'peachpuff', 'thistle', 'honeydew', 'powderblue', 'mistyrose', 'palegreen', 'paleturquoise', 'lightyellow',
    'cornsilk', 'lemonchiffon', 'azure', 'aliceblue', 'seashell', 'beige', 'oldlace', 'floralwhite'
]

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

    df.columns = ['timestamp', 'clearing_price', 'moving_average', 'cumulative_quantity', 'clearing_quantity', 'transactions']
    df['timestamp'] = np.arange(1, len(df) + 1)
    df['group_id'] = g

    # plot price vs quantity for each transaction
    # plt.plot(df[quantity][np.isfinite(df[quantity].astype(np.double))], df[price][np.isfinite(df[price].astype(np.double))], linestyle='dashed', marker='o')
    # plt.show()
    
    groups_mkt_cda.append(df)

    
# merge the list of df's
data_groups_mkt_cda = pd.concat(groups_mkt_cda, ignore_index=True, sort=False)
data_groups_mkt_cda = data_groups_mkt_cda.replace(0, np.nan)

means = data_groups_mkt_cda\
    .groupby('timestamp')[['clearing_price', 'moving_average', 'cumulative_quantity', 'clearing_quantity', 'transactions']]\
    .transform('mean')

data_groups_mkt_cda['mean_clearing_price'] = means['clearing_price']
data_groups_mkt_cda['mean_moving_average_price'] = means['moving_average']
data_groups_mkt_cda['mean_clearing_quantity'] = means['clearing_quantity']
data_groups_mkt_cda['mean_cumulative_quantity'] = means['cumulative_quantity']

data_groups_mkt_cda = data_groups_mkt_cda.replace(np.nan, 0)
data_groups_mkt_cda['round'] = data_groups_mkt_cda['timestamp'] // round_length + (data_groups_mkt_cda['timestamp'] % round_length != 0)
data_groups_mkt_cda['block'] = data_groups_mkt_cda['round'] // ((num_rounds - prac_rounds) // blocks) + (data_groups_mkt_cda['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
data_groups_mkt_cda['ce_price'] = data_groups_mkt_cda['block'].apply(lambda x: price[x - 1])
data_groups_mkt_cda['ce_quantity'] = data_groups_mkt_cda['block'].apply(lambda x: quantity[x - 1])


# plot clearing prices in all rounds for all groups 

## with scatter points
plt.figure(figsize=(20, 5))
for l in range(num_groups_cda): 
    lab = '_group ' + str(l + 1)
    plt.scatter(data=data_groups_mkt_cda[(data_groups_mkt_cda['clearing_price'] > 0) \
                                         & (data_groups_mkt_cda['group_id'] == l + 1)], \
                x='timestamp', y='clearing_price', c=colors[l], s=5, label=lab
        )
plt.scatter(data=data_groups_mkt_cda[(data_groups_mkt_cda['clearing_price'] > 0) \
                                         & (data_groups_mkt_cda['group_id'] == 1)], \
            x='timestamp', y='mean_clearing_price', c='green', label='Mean Price', s=10,)
plt.step(data=data_groups_mkt_cda[(data_groups_mkt_cda['group_id'] == 1)], \
         x='timestamp', y='ce_price', where='pre', c='plum', label='CE Price')

vline_xs = [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]
for i, x in enumerate(vline_xs, 1):
    color = 'slategray' if i in [4, 8, 12, 16] else 'lightgray'
    plt.vlines(x, ymin=0, ymax=20, colors=color, linestyles='dotted')
round_label_xs = list(range(60, 2401, 120))
for i, x in enumerate(round_label_xs, 1):
    plt.text(x, 2, str(i), color='slategray', ha='center', fontsize=9)
block_label_xs = list(range(240, 2161, 480))  # [240, 720, 1200, 1680, 2160]
for i, x in enumerate(block_label_xs, 1):
    plt.text(x, 18, f'Block {i}', color='slategray', ha='center', fontsize=10, fontweight='bold')

plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.ylim(0, 20)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_rounds - prac_rounds) + 2, round_length), np.arange(0, round_length * (num_rounds - prac_rounds) + 1, round_length))
plt.ylabel('Price')
# plt.title('CDA Transaction Prices vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_cda_price.png'))
plt.close()


# plot clearing quantity
rate_comparable = data_groups_mkt_cda.copy()
rate_comparable['time_bin'] = ((rate_comparable['timestamp'] - leave_out_seconds - 1) // price_interval_size) * price_interval_size + 1
grouped_rate_comparable = rate_comparable.groupby(['group_id', 'time_bin']).agg({'clearing_quantity': 'sum'}).reset_index()
pivot = grouped_rate_comparable.pivot(index='time_bin', columns='group_id', values='clearing_quantity')
pivot['mean_clearing_quantity'] = pivot.mean(axis=1)

plt.figure(figsize=(20, 5))
for l in range(num_groups_cda):
    lab = '_group ' + str(l + 1)
    plt.step(pivot.index, pivot[l + 1], linestyle='solid', c=colors[l], label=lab)
plt.step(pivot.index, pivot['mean_clearing_quantity'], linestyle='solid', c='green', label='Mean Quantity')
for x in [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]:
    plt.vlines(x, ymin=0, ymax=500, colors='lightgrey', linestyles='dotted')
plt.step(data_groups_mkt_cda[data_groups_mkt_cda['group_id'] == 1]['timestamp'], \
        data_groups_mkt_cda[data_groups_mkt_cda['group_id'] == 1]['ce_quantity'] / (round_length - leave_out_seconds - leave_out_seconds_end) * price_interval_size, \
        where='pre', c='plum', label='CE Quantity')

vline_xs = [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]
for i, x in enumerate(vline_xs, 1):
    color = 'slategray' if i in [4, 8, 12, 16] else 'lightgray'
    plt.vlines(x, ymin=0, ymax=20, colors=color, linestyles='dotted')
round_label_xs = list(range(60, 2401, 120))
for i, x in enumerate(round_label_xs, 1):
    plt.text(x, 20, str(i), color='slategray', ha='center', fontsize=9)
block_label_xs = list(range(240, 2161, 480))  # [240, 720, 1200, 1680, 2160]
for i, x in enumerate(block_label_xs, 1):
    plt.text(x, 450, f'Block {i}', color='slategray', ha='center', fontsize=10, fontweight='bold')

plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_rounds - prac_rounds) + 2, round_length), np.arange(0, round_length * (num_rounds - prac_rounds) + 1, round_length))
plt.ylabel('Clearing Quantity')
plt.ylim(0, 500)
# plt.title('CDA Clearing Quantity vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_cda_rate.png'))
plt.close()

# plot cumulative quantities in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(num_groups_cda): 
    lab = '_group' + str(l + 1)
    plt.step(data=data_groups_mkt_cda[(data_groups_mkt_cda['group_id'] == l + 1)], \
             x='timestamp', y='cumulative_quantity', where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_cda[(data_groups_mkt_cda['group_id'] == 1)], \
         x='timestamp', y='mean_cumulative_quantity', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_cda[(data_groups_mkt_cda['group_id'] == 1)], \
         x='timestamp', y='ce_quantity', where='pre', c='plum', label='CE Quantity')

vline_xs = [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]
for i, x in enumerate(vline_xs, 1):
    color = 'slategray' if i in [4, 8, 12, 16] else 'lightgray'
    plt.vlines(x, ymin=0, ymax=20, colors=color, linestyles='dotted')
round_label_xs = list(range(60, 2401, 120))
for i, x in enumerate(round_label_xs, 1):
    plt.text(x, 200, str(i), color='slategray', ha='center', fontsize=9)
block_label_xs = list(range(240, 2161, 480))  # [240, 720, 1200, 1680, 2160]
for i, x in enumerate(block_label_xs, 1):
    plt.text(x, 1800, f'Block {i}', color='slategray', ha='center', fontsize=10, fontweight='bold')

plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_rounds - prac_rounds) + 2, round_length), np.arange(0, round_length * (num_rounds - prac_rounds) + 1, round_length))
plt.ylabel('Shares')
plt.ylim(0, 2000)
# plt.title('CDA Cumulative Quantity vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_cda_cumsum.png'))
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
        market[name] = groups_mkt_cda[g - 1]\
            .iloc[(round_length - leave_out_seconds - leave_out_seconds_end) * (r - 1): (round_length - leave_out_seconds - leave_out_seconds_end) * r]\
            .copy()
        market[name]['unit_weighted_price'] = market[name]['clearing_price'] * market[name]['clearing_quantity']
        df = pd.DataFrame({
                'quantity': [market[name]['clearing_quantity'].sum()], 
                'unit_weighted_price': [market[name]['unit_weighted_price'].sum()], 
                'transactions': [market[name]['transactions'].mean()],
            })
        df['unit_weighted_price'] = df['unit_weighted_price'] / df['quantity']
        df['ce_price'] = ce_price[r - 1]
        df['round'] = r
        df['group_id'] = g
        data_mkt.append(df)


    # dictionary for participant cash, inventories, and transaction rates if any
    # create a list of dataframes to be concatenated after groupby 
    data_par = []

    # each round X is denoted as 'parX'
    participant = {}
    for r in range(1, num_rounds - prac_rounds + 1):
        name = 'par' + str(r)
        path = directory + 'cda{}/{}/1_participant.json'.format(g, r + prac_rounds)
        participant[name] = pd.read_json(path)
        all_orders = set()
        for idx, row in participant[name].iterrows():
            for o in row['active_orders']:
                if o['order_id'] not in all_orders:
                    all_orders.add(o['order_id'])
        number_of_orders = max(all_orders) - min(all_orders) + 1
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
        df.rename(columns={'id_in_subsession': 'group_id', 'cash': 'payoff', 'quantity': 'contract_quantity'}, inplace=True)
        df.fillna(0, inplace=True)
        data_par.append(df)

    ########## Between-period ##########
    between_df1 = pd.concat(data_mkt, ignore_index=True, axis=0)
    between_df2 = pd.concat(data_par, ignore_index=True, axis=0)
    between_df = pd.merge(between_df1, between_df2, on=['group_id', 'round'])
    between_df['order_size'] = 2 * between_df['quantity'] / between_df['orders'] 
    groups_par_cda.append(between_df)


# merge the list of df's
data_groups_par_cda = pd.concat(groups_par_cda, ignore_index=True, sort=False)
data_groups_par_cda = data_groups_par_cda.replace(0, np.nan)
payoffs = ['payoff_percent_{}'.format(g) for g in range(1, num_groups_cda + 1)]
contracts = ['contract_percent_{}'.format(g) for g in range(1, num_groups_cda + 1)]
unit_weighted = ['unit_weighted_price_{}'.format(g) for g in range(1, num_groups_cda + 1)]
quantities = ['quantity_{}'.format(g) for g in range(1, num_groups_cda + 1)]
orders = ['orders_{}'.format(g) for g in range(1, num_groups_cda + 1)]
# transaction_numbers = ['transactions_{}'.format(g) for g in range(1, num_groups_cda + 1)]
transacted_quantities = ['transacted_quantity_{}'.format(g) for g in range(1, num_groups_cda + 1)]
extra_traded_quantities = ['extra_traded_quantity_{}'.format(g) for g in range(1, num_groups_cda + 1)]
order_sizes = ['order_size_{}'.format(g) for g in range(1, num_groups_cda + 1)]

means = data_groups_par_cda\
    .groupby('round')[['payoff_percent', 'contract_percent', 'unit_weighted_price', 'quantity']]\
    .transform('mean')

data_groups_par_cda['mean_realized_surplus'] = means['payoff_percent']
data_groups_par_cda['mean_contract_execution'] = means['contract_percent']
data_groups_par_cda['mean_unit_weighted_price'] = means['unit_weighted_price']
data_groups_par_cda['mean_quantity'] = means['quantity']
data_groups_par_cda = data_groups_par_cda.replace(np.nan, 0)


# realized surplus for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_cda): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['round'], \
            data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['payoff_percent'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['round'], \
        data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['mean_realized_surplus'], \
        linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Realized Surplus vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_cda_surplus.png'))
plt.close()

# contract execution for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_cda): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['round'], \
            data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['contract_percent'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['round'], \
        data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['mean_contract_execution'], \
        linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Filled Contract vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_cda_contract.png'))
plt.close()

# traded volume for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_cda): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['round'], \
            data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['quantity'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['round'], \
        data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['mean_quantity'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)], \
         x='round', y='ce_quantity', where='mid', c='plum', label='CE Quantity')
plt.legend(loc='lower right')
plt.ylim(0, 2000)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Shares')
plt.title('Traded Volume vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_cda_quantity.png'))
plt.close()

# unit weighted price for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_cda): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['round'], \
            data_groups_par_cda[(data_groups_par_cda['group_id'] == l + 1)]['unit_weighted_price'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['round'], \
        data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)]['mean_unit_weighted_price'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_cda[(data_groups_par_cda['group_id'] == 1)], \
         x='round', y='ce_price', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Unit-Weighted Price vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_cda_unit_weighted_price.png'))
plt.close()



########## ---------- summary_cda statistics ---------- ##########
summary_cda = data_groups_par_cda[['group_id', 'round', 'ce_price', 'unit_weighted_price', 'payoff_percent', \
                                'contract_percent', 'transacted_quantity', 'orders', 'order_size', 'transactions', \
                                'extra_traded_quantity']].copy()
summary_cda['price_dev'] = abs(summary_cda['unit_weighted_price'] - summary_cda['ce_price'])

# Handle ce_price == 9 case separately
mask = summary_cda['ce_price'] == 9

# Vectorized sub-cases
within_range = mask & summary_cda['unit_weighted_price'].between(8, 10)
above_10 = mask & (summary_cda['unit_weighted_price'] > 10)
below_8 = mask & (summary_cda['unit_weighted_price'] < 8)

# Assign accordingly
summary_cda.loc[within_range, 'price_dev'] = 0
summary_cda.loc[above_10, 'price_dev'] = summary_cda['unit_weighted_price'] - 10
summary_cda.loc[below_8, 'price_dev'] = 8 - summary_cda['unit_weighted_price']

price_deviation_cda_full = summary_cda['price_dev'].tolist()
price_deviation_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['price_dev'].tolist()
price_deviation_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['price_dev'].tolist()
price_deviation_cda_test = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['price_dev'].tolist()

realized_surplus_cda_full = summary_cda['payoff_percent'].tolist()
realized_surplus_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['payoff_percent'].tolist()
realized_surplus_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['payoff_percent'].tolist()
realized_surplus_cda_test = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2 ]['payoff_percent'].tolist()

percent_contract_cda_full = summary_cda['contract_percent'].tolist()
percent_contract_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['contract_percent'].tolist()
percent_contract_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['contract_percent'].tolist()
percent_contract_cda_test = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['contract_percent'].tolist()

total_quantity_cda_full = summary_cda['transacted_quantity'].tolist()
total_quantity_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity'].tolist()
total_quantity_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['transacted_quantity'].tolist()
total_quantity_cda_test = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity'].tolist()

price_volatility_cda_full = data_groups_mkt_cda[data_groups_mkt_cda['clearing_price'] > 0]['clearing_price'].tolist()
price_volatility_cda_half = data_groups_mkt_cda[(data_groups_mkt_cda['clearing_price'.format(g)] > 0) & (data_groups_mkt_cda['timestamp'] > (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price'].tolist()
price_volatility_cda_first = data_groups_mkt_cda[(data_groups_mkt_cda['clearing_price'.format(g)] > 0) & (data_groups_mkt_cda['timestamp'] <= (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price'].tolist()

order_number_cda_full = summary_cda['orders'].tolist()
order_number_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['orders'].tolist()
order_number_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['orders'].tolist()
order_number_cda_test = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['orders']

order_size_cda_full = summary_cda['order_size'].tolist()
order_size_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size'].tolist()
order_size_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['order_size'].tolist()
order_size_cda_test = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size']

transaction_number_cda_full = summary_cda['transactions'].tolist()
transaction_number_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['transactions'].tolist()
transaction_number_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['transactions'].tolist()

extra_traded_quantities_cda_full = summary_cda['order_size'].mean()
extra_traded_quantities_cda_half = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size'].mean()
extra_traded_quantities_cda_first = summary_cda[summary_cda['round'] <= (num_rounds - prac_rounds) // 2]['order_size'].mean()
extra_traded_quantities_cda_test = summary_cda[summary_cda['round'] > (num_rounds - prac_rounds) // 2]['order_size']

regress_cda_period = summary_cda[['group_id', 'round', 'price_dev', 'payoff_percent', 'transacted_quantity', 'contract_percent']].copy()

regress_cda_period['block'] = regress_cda_period['round'] // ((num_rounds - prac_rounds) // blocks) + (regress_cda_period['round'] % ((num_rounds - prac_rounds) // blocks) != 0).astype(int)
regress_cda_period['format'] = 'CDA'
regress_cda_period['ce_quantity'] = regress_cda_period['block'].apply(lambda x: quantity[x - 1])

regress_cda_period.rename(columns={'group_id': 'group', 'payoff_percent': 'realized_surplus', 'transacted_quantity': 'traded_volume', 'contract_percent': 'filled_contract', 'price_dev': 'price_deviation'}, inplace=True)
regress_cda_period['filled_ce_quantity'] = regress_cda_period['traded_volume'] / regress_cda_period['ce_quantity']


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
