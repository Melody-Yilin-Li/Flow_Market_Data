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
groups_mkt_flow = []
delta_prices_flow_r_full = []
delta_prices_flow_s_full = []
delta_prices_flow_s_half = []
delta_prices_flow_r_half = []
delta_prices_flow_s_first = []
delta_prices_flow_r_first = []
delta_prices_flow_full_interval = []
delta_prices_flow_half_interval = []
delta_prices_flow_first_interval = []
regress_flow = pd.DataFrame()
regress_flow_period = pd.DataFrame()
regress_flow_second = pd.DataFrame()

colors = [
    'lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'lightcoral', 'lightskyblue', 'pink',
    'peachpuff', 'thistle', 'honeydew', 'powderblue', 'mistyrose', 'palegreen', 'paleturquoise', 'lightyellow',
    'cornsilk', 'lemonchiffon', 'azure', 'aliceblue', 'seashell', 'beige', 'oldlace', 'floralwhite'
]
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
        reg_df_sec['format'] = 'FlowR' if g <= num_groups_flow_low else 'FlowS'
        reg_df_sec['ce_price'] = ce_price[r - 1]
        reg_df_sec['ce_quantity'] = ce_quantity[r - 1]
        reg_df_sec['treat'] = 'L' if g <= 5 else 'H'
        
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
        result_reg_df['format'] = 'FlowR' if g <= num_groups_flow_low else 'FlowS'
        result_reg_df['price_change'] = result_reg_df['weighted_price'].diff()
        result_reg_df['price_change_int'] = result_reg_df['price_change_int'] * result_reg_df['quantity'] / result_reg_df['quantity'].sum()
        result_reg_df['ce_price'] = ce_price[r - 1]
        result_reg_df['ce_quantity'] = ce_quantity[r - 1]
        result_reg_df['cum_volume'] = result_reg_df['quantity'].cumsum()
        result_reg_df['%cumsum'] = result_reg_df['cum_volume'] / result_reg_df['ce_quantity']
        
        regress_flow = pd.concat([regress_flow, result_reg_df], ignore_index=True)

        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['timestamp'] < round_length - leave_out_seconds_end)]
        group.append(rnd) 
    if g <= num_groups_flow_low:
        delta_prices_flow_r_full.append(delta_price_full)
        delta_prices_flow_r_half.append(delta_price_half)
        delta_prices_flow_r_first.append(delta_price_first)
    else:
        delta_prices_flow_s_full.append(delta_price_full)
        delta_prices_flow_s_half.append(delta_price_half)
        delta_prices_flow_s_first.append(delta_price_first)
    df = pd.concat(group, ignore_index=True, sort=False)
    df.columns = ['timestamp', 'clearing_price', 'clearing_rate', 'cumulative_quantity', 'moving_average']
    df['timestamp'] = np.arange(1, len(df) + 1)
    df['group_id'] = g
    groups_mkt_flow.append(df)
    
# merge the list of df's
data_groups_mkt_flow = pd.concat(groups_mkt_flow, ignore_index=True, sort=False)
data_groups_mkt_flow = data_groups_mkt_flow.replace(0, np.nan)

mask_l = data_groups_mkt_flow['group_id'] <= num_groups_flow_low
mask_h = data_groups_mkt_flow['group_id'] > num_groups_flow_low

means_l = data_groups_mkt_flow[mask_l]\
    .groupby('timestamp')[['clearing_price', 'clearing_rate', 'cumulative_quantity']]\
    .transform('mean')
means_h = data_groups_mkt_flow[mask_h]\
    .groupby('timestamp')[['clearing_price', 'clearing_rate', 'cumulative_quantity']]\
    .transform('mean')

data_groups_mkt_flow.loc[mask_l, 'mean_clearing_price'] = means_l['clearing_price']
data_groups_mkt_flow.loc[mask_l, 'mean_clearing_rate'] = means_l['clearing_rate']
data_groups_mkt_flow.loc[mask_l, 'mean_cumulative_quantity'] = means_l['cumulative_quantity']

data_groups_mkt_flow.loc[mask_h, 'mean_clearing_price'] = means_h['clearing_price']
data_groups_mkt_flow.loc[mask_h, 'mean_clearing_rate'] = means_h['clearing_rate']
data_groups_mkt_flow.loc[mask_h, 'mean_cumulative_quantity'] = means_h['cumulative_quantity']

data_groups_mkt_flow = data_groups_mkt_flow.replace(np.nan, 0)
data_groups_mkt_flow['round'] = data_groups_mkt_flow['timestamp'] // round_length + (data_groups_mkt_flow['timestamp'] % round_length != 0)
data_groups_mkt_flow['block'] = data_groups_mkt_flow['round'] // ((num_rounds - prac_rounds) // blocks) + (data_groups_mkt_flow['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
data_groups_mkt_flow['ce_price'] = data_groups_mkt_flow['block'].apply(lambda x: price[x - 1])
data_groups_mkt_flow['ce_quantity'] = data_groups_mkt_flow['block'].apply(lambda x: quantity[x - 1])
data_groups_mkt_flow['ce_rate'] = data_groups_mkt_flow['ce_quantity'] / (round_length - leave_out_seconds - leave_out_seconds_end) 


# plot clearing prices in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(num_groups_flow_low):
    lab = '_group ' + str(l + 1)
    plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price'] > 0) \
                                       & (data_groups_mkt_flow['group_id'] == l + 1)], \
            x='timestamp', y='clearing_price', where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['mean_clearing_price'] > 0) \
                                   & (data_groups_mkt_flow['group_id'] == 1)], \
            x='timestamp', y='mean_clearing_price', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 1)], \
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
# plt.title('Flow Transaction Prices vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_flow_price_r.png'))
plt.close()

plt.figure(figsize=(20, 5))
for l in range(num_groups_flow_high):
    lab = '_group ' + str(l + 1)
    plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price'] > 0) \
                                       & (data_groups_mkt_flow['group_id'] == l + 1 + num_groups_flow_low)], \
            x='timestamp', y='clearing_price', where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['mean_clearing_price'] > 0) \
                                   & (data_groups_mkt_flow['group_id'] == 6)], \
            x='timestamp', y='mean_clearing_price', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 6)], \
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
# plt.title('Flow Transaction Prices vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_flow_price_s.png'))
plt.close()

# plot clearing rates in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(num_groups_flow_low): 
    lab = '_group ' + str(l + 1)
    plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price'] > 0) \
                                       & (data_groups_mkt_flow['group_id'] == l + 1)], \
            x='timestamp', y='clearing_rate', where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['mean_clearing_price'] > 0) \
                                   & (data_groups_mkt_flow['group_id'] == 1)], \
        x='timestamp', y='mean_clearing_rate', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 1)], \
         x='timestamp', y='ce_rate', where='pre', c='plum', label='CE Rate')
vline_xs = [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]
for i, x in enumerate(vline_xs, 1):
    color = 'slategray' if i in [4, 8, 12, 16] else 'lightgray'
    plt.vlines(x, ymin=0, ymax=35, colors=color, linestyles='dotted')
round_label_xs = list(range(60, 2401, 120))
for i, x in enumerate(round_label_xs, 1):
    plt.text(x, 2, str(i), color='slategray', ha='center', fontsize=9)
block_label_xs = list(range(240, 2161, 480))  # [240, 720, 1200, 1680, 2160]
for i, x in enumerate(block_label_xs, 1):
    plt.text(x, 30, f'Block {i}', color='slategray', ha='center', fontsize=10, fontweight='bold')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_rounds - prac_rounds) + 2, round_length), np.arange(0, round_length * (num_rounds - prac_rounds) + 1, round_length))
plt.ylabel('Shares/second')
plt.ylim(0, 35)
# plt.title('Flow Transaction Rates vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_flow_rate_r.png'))
plt.close()

plt.figure(figsize=(20, 5))
for l in range(num_groups_flow_high): 
    lab = '_group ' + str(l + 1)
    plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price'] > 0) \
                                       & (data_groups_mkt_flow['group_id'] == l + 1 + num_groups_flow_low)], \
            x='timestamp', y='clearing_rate', where='pre', c=colors[l], label=lab)
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['mean_clearing_price'] > 0) \
                                   & (data_groups_mkt_flow['group_id'] == 6)], \
        x='timestamp', y='mean_clearing_rate', where='pre', c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 6)], \
         x='timestamp', y='ce_rate', where='pre', c='plum', label='CE Rate')
vline_xs = [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_rounds - prac_rounds)]
for i, x in enumerate(vline_xs, 1):
    color = 'slategray' if i in [4, 8, 12, 16] else 'lightgray'
    plt.vlines(x, ymin=0, ymax=35, colors=color, linestyles='dotted')
round_label_xs = list(range(60, 2401, 120))
for i, x in enumerate(round_label_xs, 1):
    plt.text(x, 2, str(i), color='slategray', ha='center', fontsize=9)
block_label_xs = list(range(240, 2161, 480))  # [240, 720, 1200, 1680, 2160]
for i, x in enumerate(block_label_xs, 1):
    plt.text(x, 30, f'Block {i}', color='slategray', ha='center', fontsize=10, fontweight='bold')
plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_rounds - prac_rounds) + 2, round_length), np.arange(0, round_length * (num_rounds - prac_rounds) + 1, round_length))
plt.ylabel('Shares/second')
plt.ylim(0, 35)
# plt.title('Flow Transaction Rates vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_flow_rate_s.png'))
plt.close()

# plot cumulative quantities in all rounds for all groups 
plt.figure(figsize=(20, 5))
for l in range(num_groups_flow_low): 
    lab = '_group ' + str(l + 1)
    plt.plot(data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == l + 1)]['timestamp'], \
            data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == l + 1)]['cumulative_quantity'], \
            c=colors[l], label=lab)
plt.plot(data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 1)]['timestamp'], \
        data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 1)]['mean_cumulative_quantity'], \
        c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 1)], \
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
    plt.text(x, 1900, f'Block {i}', color='slategray', ha='center', fontsize=10, fontweight='bold')

plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_rounds - prac_rounds) + 2, round_length), np.arange(0, round_length * (num_rounds - prac_rounds) + 1, round_length))
plt.ylabel('Shares')
plt.ylim(0, 2000)
# plt.title('Flow Cumulative Quantity vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_flow_cumsum_r.png'))
plt.close()



plt.figure(figsize=(20, 5))
for l in range(num_groups_flow_high): 
    lab = '_group ' + str(l + 1)
    plt.plot(data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == l + 1 + num_groups_flow_low)]['timestamp'], \
            data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == l + 1 + num_groups_flow_low)]['cumulative_quantity'], \
            c=colors[l], label=lab)
plt.plot(data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 6)]['timestamp'], \
        data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 6)]['mean_cumulative_quantity'], \
        c='green', label='Mean', linestyle='solid')
plt.step(data=data_groups_mkt_flow[(data_groups_mkt_flow['group_id'] == 6)], \
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
    plt.text(x, 1900, f'Block {i}', color='slategray', ha='center', fontsize=10, fontweight='bold')

plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_rounds - prac_rounds) + 2, round_length), np.arange(0, round_length * (num_rounds - prac_rounds) + 1, round_length))
plt.ylabel('Shares')
plt.ylim(0, 2000)
# plt.title('Flow Cumulative Quantity vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_flow_cumsum_s.png'))
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
        market[name] = pd.read_json(path)
        market[name].fillna(0, inplace=True)
        market[name]['unit_weighted_price'] = market[name]['clearing_price'] * market[name]['clearing_rate']
        df = market[name][market[name]['before_transaction'] == False].groupby('id_in_subsession').aggregate({'clearing_price': 'mean', 'clearing_rate': 'sum', 'unit_weighted_price': 'sum'}).reset_index()
        df['unit_weighted_price'] = df['unit_weighted_price'] / df['clearing_rate']
        df['ce_price'] = ce_price[r - 1]
        df['round'] = r
        df.rename(columns={'id_in_subsession': 'group_id', 'clearing_price': 'time_weighted_price', 'clearing_rate': 'quantity'}, inplace=True)
        df['group_id'] = g
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
        participant[name] = pd.read_json(path)
        all_orders = set()
        for idx, row in participant[name].iterrows():
            for o in row['active_orders']:
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
        df.rename(columns={'id_in_subsession': 'group_id', 'cash': 'payoff', 'quantity': 'contract_quantity'}, inplace=True)
        df.fillna(0, inplace=True)
        data_par.append(df)

    ########## Between-period ##########
    between_df1 = pd.concat(data_mkt, ignore_index=True, axis=0)
    between_df2 = pd.concat(data_par, ignore_index=True, axis=0)

    between_df = pd.merge(between_df1, between_df2, on=['group_id', 'round'])
    between_df['order_size'] = 2 *  between_df['quantity'] / between_df['orders'] 
    groups_par_flow.append(between_df)


# merge the list of df's
data_groups_par_flow = pd.concat(groups_par_flow, ignore_index=True, sort=False)
data_groups_par_flow = data_groups_par_flow.replace(0, np.nan)

mask_l = data_groups_par_flow['group_id'] <= num_groups_flow_low
mask_h = data_groups_par_flow['group_id'] > num_groups_flow_low

means_l = data_groups_par_flow[mask_l].groupby('round')[['payoff_percent', 'contract_percent', 'time_weighted_price', 'unit_weighted_price', 'transacted_quantity']].transform('mean')
means_h = data_groups_par_flow[mask_h].groupby('round')[['payoff_percent', 'contract_percent', 'time_weighted_price', 'unit_weighted_price', 'transacted_quantity']].transform('mean')

data_groups_par_flow.loc[mask_l, 'mean_realized_surplus'] = means_l['payoff_percent']
data_groups_par_flow.loc[mask_l, 'mean_contract_execution'] = means_l['contract_percent']
data_groups_par_flow.loc[mask_l, 'mean_time_weighted_price'] = means_l['time_weighted_price']
data_groups_par_flow.loc[mask_l, 'mean_unit_weighted_price'] = means_l['unit_weighted_price']
data_groups_par_flow.loc[mask_l, 'mean_quantity'] = means_l['transacted_quantity']

data_groups_par_flow.loc[mask_h, 'mean_realized_surplus'] = means_h['payoff_percent']
data_groups_par_flow.loc[mask_h, 'mean_contract_execution'] = means_h['contract_percent']
data_groups_par_flow.loc[mask_h, 'mean_time_weighted_price'] = means_h['time_weighted_price']
data_groups_par_flow.loc[mask_h, 'mean_unit_weighted_price'] = means_h['unit_weighted_price']
data_groups_par_flow.loc[mask_h, 'mean_quantity'] = means_h['transacted_quantity']
data_groups_par_flow = data_groups_par_flow.replace(np.nan, 0)


# realized surplus for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_low): 
    lab = '_group ' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['payoff_percent'] > 0) \
                                  & (data_groups_par_flow['group_id'] == l + 1)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['payoff_percent'] > 0) \
                                 & (data_groups_par_flow['group_id'] == l + 1)]['payoff_percent'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['mean_realized_surplus'], \
        linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Realized Surplus vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_surplus_r.png'))
plt.close()

plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_high): 
    lab = '_group ' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['payoff_percent'] > 0) \
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['payoff_percent'] > 0) \
                                 & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['payoff_percent'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['mean_realized_surplus'], \
        linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Realized Surplus vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_surplus_s.png'))
plt.close()

# contract execution for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_low): 
    lab = '_group ' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['contract_percent'] > 0) \
                                  & (data_groups_par_flow['group_id'] == l + 1)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['contract_percent'] > 0) \
                                 & (data_groups_par_flow['group_id'] == l + 1)]['contract_percent'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['mean_contract_execution'], \
        linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Filled Contract vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_contract_r.png'))
plt.close()

plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_high): 
    lab = '_group ' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['contract_percent'] > 0) \
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['contract_percent'] > 0) \
                                 & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['contract_percent'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['mean_contract_execution'], \
        linestyle='solid', c='green', label='Mean')
plt.hlines(y=1, xmin=1, xmax=num_rounds-prac_rounds, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Percent')
plt.title('Filled Contract vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_contract_s.png'))
plt.close()

# traded volume for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_low): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['transacted_quantity'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['transacted_quantity'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1)]['transacted_quantity'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['mean_quantity'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)], x='round', y='ce_quantity', where='mid', c='plum', label='CE Quantity')
plt.legend(loc='lower right')
plt.ylim(0, 2000)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Shares')
plt.title('Traded Volume vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_quantity_r.png'))
plt.close()


plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_high): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['transacted_quantity'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['transacted_quantity'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['transacted_quantity'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['mean_quantity'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)], x='round', y='ce_quantity', where='mid', c='plum', label='CE Quantity')
plt.legend(loc='lower right')
plt.ylim(0, 2000)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Shares')
plt.title('Traded Volume vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_quantity_s.png'))
plt.close()


# time weighted price for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_low): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['time_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['time_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1)]['time_weighted_price'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['mean_time_weighted_price'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)], x='round', y='ce_price', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Time-Weighted Price vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_time_weighted_price_r.png'))
plt.close()

plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_high): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['time_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['time_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['time_weighted_price'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['mean_time_weighted_price'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)], x='round', y='ce_price', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Time-Weighted Price vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_time_weighted_price_s.png'))
plt.close()

# unit weighted price for all groups
plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_low): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['unit_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['unit_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1)]['unit_weighted_price'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)]['mean_unit_weighted_price'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow[(data_groups_par_flow['group_id'] == 1)], x='round', y='ce_price', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Unit-Weighted Price vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_unit_weighted_price_r.png'))
plt.close()

plt.figure(figsize=(8, 5))
for l in range(num_groups_flow_high): 
    lab = '_group' + str(l + 1)
    plt.plot(data_groups_par_flow[(data_groups_par_flow['unit_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['round'], \
            data_groups_par_flow[(data_groups_par_flow['unit_weighted_price'] > 0)\
                                  & (data_groups_par_flow['group_id'] == l + 1 + num_groups_flow_low)]['unit_weighted_price'], \
            linestyle='solid', c=colors[l], label=lab)
plt.plot(data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['round'], \
        data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)]['mean_unit_weighted_price'], \
        linestyle='solid', c='green', label='Mean')
plt.step(data=data_groups_par_flow[(data_groups_par_flow['group_id'] == 6)], x='round', y='ce_price', where='mid', c='plum', label='CE Price')
plt.legend(loc='lower right')
plt.ylim(0, 20)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_rounds - prac_rounds + 1), np.arange(1, num_rounds - prac_rounds + 1))
plt.ylabel('Price')
plt.title('Unit-Weighted Price vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_flow_unit_weighted_price_s.png'))
plt.close()

########## ---------- summary_flow statistics ---------- ##########
summary_flow = data_groups_par_flow[['group_id', 'round', 'ce_price', 'unit_weighted_price', \
                                    'payoff_percent', 'contract_percent', 'transacted_quantity', \
                                    'orders', 'order_size', 'extra_traded_quantity']]\
                                        .copy()

summary_flow['price_dev'] = abs(summary_flow['unit_weighted_price'] - summary_flow['ce_price'])

# Handle ce_price == 9 case separately
mask = summary_flow['ce_price'] == 9

# Vectorized sub-cases
within_range = mask & summary_flow['unit_weighted_price'].between(8, 10)
above_10 = mask & (summary_flow['unit_weighted_price'] > 10)
below_8 = mask & (summary_flow['unit_weighted_price'] < 8)

# Assign accordingly
summary_flow.loc[within_range, 'price_dev'] = 0
summary_flow.loc[above_10, 'price_dev'] = summary_flow['unit_weighted_price'] - 10
summary_flow.loc[below_8, 'price_dev'] = 8 - summary_flow['unit_weighted_price']

price_deviation_flow_full = summary_flow['price_dev'].tolist()
price_deviation_flow_half = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['price_dev'].tolist()
price_deviation_flow_first = summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['price_dev'].tolist()
price_deviation_flow_test = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['price_dev'].tolist()

realized_surplus_flow_full = summary_flow['payoff_percent'].tolist()
realized_surplus_flow_half = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['payoff_percent'].tolist()
realized_surplus_flow_first = summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['payoff_percent'].tolist()
realized_surplus_flow_test = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['payoff_percent'].tolist()

percent_contract_flow_full = summary_flow['contract_percent'].tolist()
percent_contract_flow_half = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['contract_percent'].tolist()
percent_contract_flow_first = summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['contract_percent'].tolist()
percent_contract_flow_test = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['contract_percent'].tolist()

total_quantity_flow_full = summary_flow['transacted_quantity'].tolist()
total_quantity_flow_half = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity'].tolist()
total_quantity_flow_first = summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['transacted_quantity'].tolist()
total_quantity_flow_test = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['transacted_quantity'].tolist()

price_volatility_flow_full = data_groups_mkt_flow[data_groups_mkt_flow['clearing_price'] > 0]['clearing_price'].tolist()
price_volatility_flow_half = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price'] > 0) & (data_groups_mkt_flow['timestamp'] > (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price'].tolist()
price_volatility_flow_first = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_price'] > 0) & (data_groups_mkt_flow['timestamp'] <= (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2)]['clearing_price'].tolist()

clearing_rate_flow_r_full = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate'] > 0) & (data_groups_mkt_flow['group_id'] <= num_groups_flow_low)]['clearing_rate'].tolist()
clearing_rate_flow_r_half = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate'] > 0) & (data_groups_mkt_flow['timestamp'] > (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2) & (data_groups_mkt_flow['group_id'] <= num_groups_flow_low)]['clearing_rate'].tolist()
clearing_rate_flow_r_first = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate'] > 0) & (data_groups_mkt_flow['timestamp'] <= (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2) & (data_groups_mkt_flow['group_id'] <= num_groups_flow_low)]['clearing_rate'].tolist()
clearing_rate_flow_s_full = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate'] > 0) & (data_groups_mkt_flow['group_id'] > num_groups_flow_low)]['clearing_rate'].tolist()
clearing_rate_flow_s_half = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate'] > 0) & (data_groups_mkt_flow['timestamp'] > (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2) & (data_groups_mkt_flow['group_id'] > num_groups_flow_low)]['clearing_rate'].tolist()
clearing_rate_flow_s_first = data_groups_mkt_flow[(data_groups_mkt_flow['clearing_rate'] > 0) & (data_groups_mkt_flow['timestamp'] <= (round_length - leave_out_seconds - leave_out_seconds_end) * (num_rounds - prac_rounds) // 2) & (data_groups_mkt_flow['group_id'] > num_groups_flow_low)]['clearing_rate'].tolist()

order_number_flow_full = summary_flow['orders'].tolist()
order_number_flow_half = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['orders'].tolist()
order_number_flow_first = summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['orders'].tolist()
order_number_flow_test = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['orders']

order_size_flow_full = summary_flow['order_size'].tolist()
order_size_flow_half = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size'].tolist()
order_size_flow_first = summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['order_size'].tolist()
order_size_flow_test = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size']

extra_traded_quantities_flow_full = summary_flow['order_size'].tolist()
extra_traded_quantities_flow_half = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size'].tolist()
extra_traded_quantities_flow_first = summary_flow[summary_flow['round'] <= (num_rounds - prac_rounds) // 2]['order_size'].tolist()
extra_traded_quantities_flow_test = summary_flow[summary_flow['round'] > (num_rounds - prac_rounds) // 2]['order_size']

regress_flow_period = summary_flow[['group_id', 'round', 'price_dev', 'payoff_percent', 'transacted_quantity', 'contract_percent']].copy()

regress_flow_period['block'] = regress_flow_period['round'] // ((num_rounds - prac_rounds) // blocks) + (regress_flow_period['round'] % ((num_rounds - prac_rounds) // blocks) != 0).astype(int)
regress_flow_period['format'] = regress_flow_period['group_id'].apply(
    lambda x: 'FlowR' if x <= num_groups_flow_low else 'FlowS'
)
regress_flow_period['ce_quantity'] = regress_flow_period['block'].apply(lambda x: quantity[x - 1])

regress_flow_period.rename(columns={'group_id': 'group', 'payoff_percent': 'realized_surplus', 'transacted_quantity': 'traded_volume', 'contract_percent': 'filled_contract', 'price_dev': 'price_deviation'}, inplace=True)
regress_flow_period['filled_ce_quantity'] = regress_flow_period['traded_volume'] / regress_flow_period['ce_quantity']

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
