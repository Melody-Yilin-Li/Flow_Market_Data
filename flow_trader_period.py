import numpy as np 
import pandas as pd
from collections import defaultdict 
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

      
flow_trader_period = pd.DataFrame()
cda_trader_period = pd.DataFrame()


# read in data 
colors = [
    'lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'lightcoral', 'lightskyblue', 'pink',
    'peachpuff', 'thistle', 'honeydew', 'powderblue', 'mistyrose', 'palegreen', 'paleturquoise', 'lightyellow',
    'cornsilk', 'lemonchiffon', 'azure', 'aliceblue', 'seashell', 'beige', 'oldlace', 'floralwhite'
]

for g in range(1, num_groups_cda + 1):
    name = 'group' + str(g)
    group_par = []
    for r in range(1, num_rounds - prac_rounds + 1):
        
        path = directory + 'cda{}/{}/1_participant.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(path)
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)


        # Step 1: Flatten all active orders into one big DataFrame
        records = []

        for _, row in rnd.iterrows():   
            for order in row['active_orders']:
                records.append({
                    'group_id': g,
                    'round': r, 
                    'id_in_group': row['id_in_group'], 
                    'order_id': order.get('order_id'),
                    'order_price': order.get('price'),
                })

        # Step 2: Create DataFrame from the flattened list
        orders_df = pd.DataFrame(records)

        orders_df.drop_duplicates() 

        summary = orders_df.groupby(['group_id', 'id_in_group'])[['order_price']].mean().reset_index()
        summary['round'] = r
        summary['block'] = summary['round'] // ((num_rounds - prac_rounds) // blocks) + (summary['round'] % ((num_rounds - prac_rounds) // blocks) != 0)

        rnd = rnd.explode('active_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'active_contracts')
        rnd = rnd.explode('executed_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'executed_contracts')
        rnd = rnd.groupby(level=0, axis=1).first()


        rnd = rnd[-players_per_group:][['id_in_group', 'direction', 'fill_quantity', 'cash', 'price', 'quantity']]
        rnd.rename(columns={'cash': 'profit', 'price': 'contract_price', 'quantity': 'contract_quantity'}, inplace=True)
        rnd = pd.merge(rnd, summary, on='id_in_group', how='left')
        rnd['group_id'] = g
        rnd['round'] = r
        rnd['block'] = rnd['round'] // ((num_rounds - prac_rounds) // blocks) + (rnd['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        for ind, row in rnd.iterrows():
            if pd.isna(row['order_price']):
                if row['direction'] == 'buy':
                    rnd.loc[ind, 'order_price'] = row['contract_price'] - row['profit'] / row['fill_quantity']
                else:
                    rnd.loc[ind, 'order_price'] = row['contract_price'] + row['profit'] / row['fill_quantity']
        
        rnd['price_dev_from_contract'] = 0
        rnd['price_dev_from_contract'] = rnd['price_dev_from_contract'].astype(float)
        for ind, row in rnd.iterrows():
            contract_set = r // (players_per_group // 2) + int(r % (players_per_group // 2) != 0)
            if row['direction'] == 'buy':
                rnd.loc[ind, 'in_market_quantity'] = contract_buy[contract_set][int(row['contract_price'])]
                rnd.loc[ind, 'price_dev_from_contract'] = row['contract_price'] - row['order_price'] 
            else:
                rnd.loc[ind, 'in_market_quantity'] = contract_sell[contract_set][int(row['contract_price'])]
                rnd.loc[ind, 'price_dev_from_contract'] = row['order_price'] - row['contract_price']

        rnd['ce_price'] = rnd['block'].apply(lambda x: price[x - 1])
        rnd['format'] = 'CDA'
        rnd['time']  = 'T1-T10' if r <= (num_rounds - prac_rounds) // 2 else 'T11-T20'
    
        cda_trader_period = pd.concat([cda_trader_period, rnd], ignore_index=True, sort=False)


cda_trader_period.to_csv(os.path.join(tables_dir, 'cda_trader_period.csv'), index=False)


for g in range(1, num_groups_flow + 1):
    name = 'group' + str(g)
    group_par = []
    for r in range(1, num_rounds - prac_rounds + 1):
        
        path = directory + 'flow{}/{}/1_participant.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(path)
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)


        # Step 1: Flatten all active orders into one big DataFrame
        records = []

        for _, row in rnd.iterrows():            
            for order in row['active_orders']:
                records.append({
                    'group_id': g,
                    'round': r, 
                    'id_in_group': row['id_in_group'], 
                    'order_id': order.get('order_id'),
                    'min_price': order.get('min_price'),
                    'max_price': order.get('max_price'),
                    'max_rate': order.get('max_rate')
                })

        # Step 2: Create DataFrame from the flattened list
        orders_df = pd.DataFrame(records)

        orders_df.drop_duplicates() 
        orders_df['order_price_diff'] = orders_df['max_price'] - orders_df['min_price']

        summary = orders_df.groupby(['group_id', 'id_in_group'])[['max_price', 'min_price', 'order_price_diff', 'max_rate']].mean().reset_index()
        summary['round'] = r
        summary['block'] = summary['round'] // ((num_rounds - prac_rounds) // blocks) + (summary['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        summary['max_rate_percent'] = summary['max_rate'] / max_order_rate_low if g <= num_groups_flow_low else summary['max_rate'] / max_order_rate_high


        rnd = rnd.explode('active_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'active_contracts')
        rnd = rnd.explode('executed_contracts')
        rnd.reset_index(drop=True, inplace=True)
        rnd = df_explosion(rnd, 'executed_contracts')
        rnd = rnd.groupby(level=0, axis=1).first()


        rnd = rnd[-players_per_group:][['id_in_group', 'direction', 'fill_quantity', 'cash', 'price', 'quantity']]
        rnd.rename(columns={'cash': 'profit', 'price': 'contract_price', 'quantity': 'contract_quantity'}, inplace=True)

        rnd = pd.merge(rnd, summary, on='id_in_group', how='left')

        rnd['contract_percent'] = rnd['fill_quantity'] / rnd['contract_quantity']
        rnd['in_market_quantity'] = 0
        rnd['price_dev_from_contract'] = 0
        for ind, row in rnd.iterrows():
            contract_set = r // (players_per_group // 2) + int(r % (players_per_group // 2) != 0)
            if row['direction'] == 'buy':
                rnd.loc[ind, 'in_market_quantity'] = contract_buy[contract_set][int(row['contract_price'])]
                rnd.loc[ind, 'price_dev_from_contract'] = row['contract_price'] - row['max_price'] 
            else:
                rnd.loc[ind, 'in_market_quantity'] = contract_sell[contract_set][int(row['contract_price'])]
                rnd.loc[ind, 'price_dev_from_contract'] = row['min_price'] - row['contract_price']

        rnd['in_market_percent'] = rnd['fill_quantity'] / rnd['in_market_quantity']
        rnd['ce_price'] = rnd['block'].apply(lambda x: price[x - 1])
        rnd['ind_ce_profit'] = rnd['in_market_quantity'] * abs(rnd['contract_price'] - rnd['ce_price'])        
        rnd['realized_surplus'] = rnd['profit'] / rnd['ind_ce_profit']
        rnd['excess_profit'] = rnd['profit'] - rnd['ind_ce_profit']
        rnd.loc[(rnd['ind_ce_profit'] == 0) & (rnd['profit'] >= 0), 'realized_surplus'] = np.nan
        rnd.loc[(rnd['ind_ce_profit'] == 0) & (rnd['profit'] < 0), 'realized_surplus'] = np.nan
        rnd['format'] = 'FlowR' if g <= num_groups_flow_low else 'FlowS'
        rnd['time']  = 'T1-T10' if r <= (num_rounds - prac_rounds) // 2 else 'T11-T20'
    
        flow_trader_period = pd.concat([flow_trader_period, rnd], ignore_index=True, sort=False)


flow_trader_period.to_csv(os.path.join(tables_dir, 'flow_trader_period.csv'), index=False)

########## CDF ##########
# pH - pL
sorted_order_price_diff_buy_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['order_price_diff'].tolist())
sorted_order_price_diff_buy_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['order_price_diff'].tolist())
sorted_order_price_diff_sell_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['order_price_diff'].tolist())
sorted_order_price_diff_sell_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['order_price_diff'].tolist())

sorted_order_price_diff_buy_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())
sorted_order_price_diff_buy_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())
sorted_order_price_diff_sell_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())
sorted_order_price_diff_sell_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())

sorted_order_price_diff_buy_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())
sorted_order_price_diff_buy_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())
sorted_order_price_diff_sell_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())
sorted_order_price_diff_sell_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['order_price_diff'].tolist())

cumulative_prob_order_price_diff_buy_flow_r = np.arange(1, len(sorted_order_price_diff_buy_flow_r) + 1) / len(sorted_order_price_diff_buy_flow_r)
cumulative_prob_order_price_diff_buy_flow_s = np.arange(1, len(sorted_order_price_diff_buy_flow_s) + 1) / len(sorted_order_price_diff_buy_flow_s)
cumulative_prob_order_price_diff_sell_flow_r = np.arange(1, len(sorted_order_price_diff_sell_flow_r) + 1) / len(sorted_order_price_diff_sell_flow_r)
cumulative_prob_order_price_diff_sell_flow_s = np.arange(1, len(sorted_order_price_diff_sell_flow_s) + 1) / len(sorted_order_price_diff_sell_flow_s)

cumulative_prob_order_price_diff_buy_flow_r_first = np.arange(1, len(sorted_order_price_diff_buy_flow_r_first) + 1) / len(sorted_order_price_diff_buy_flow_r_first)
cumulative_prob_order_price_diff_buy_flow_s_first = np.arange(1, len(sorted_order_price_diff_buy_flow_s_first) + 1) / len(sorted_order_price_diff_buy_flow_s_first)
cumulative_prob_order_price_diff_sell_flow_r_first = np.arange(1, len(sorted_order_price_diff_sell_flow_r_first) + 1) / len(sorted_order_price_diff_sell_flow_r_first)
cumulative_prob_order_price_diff_sell_flow_s_first = np.arange(1, len(sorted_order_price_diff_sell_flow_s_first) + 1) / len(sorted_order_price_diff_sell_flow_s_first)

cumulative_prob_order_price_diff_buy_flow_r_last = np.arange(1, len(sorted_order_price_diff_buy_flow_r_last) + 1) / len(sorted_order_price_diff_buy_flow_r_last)
cumulative_prob_order_price_diff_buy_flow_s_last = np.arange(1, len(sorted_order_price_diff_buy_flow_s_last) + 1) / len(sorted_order_price_diff_buy_flow_s_last)
cumulative_prob_order_price_diff_sell_flow_r_last = np.arange(1, len(sorted_order_price_diff_sell_flow_r_last) + 1) / len(sorted_order_price_diff_sell_flow_r_last)
cumulative_prob_order_price_diff_sell_flow_s_last = np.arange(1, len(sorted_order_price_diff_sell_flow_s_last) + 1) / len(sorted_order_price_diff_sell_flow_s_last)

plt.figure(figsize=(15, 10))
plt.plot(sorted_order_price_diff_buy_flow_r, cumulative_prob_order_price_diff_buy_flow_r, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer')
plt.plot(sorted_order_price_diff_sell_flow_r, cumulative_prob_order_price_diff_sell_flow_r, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller')
plt.plot(sorted_order_price_diff_buy_flow_s, cumulative_prob_order_price_diff_buy_flow_s, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer')
plt.plot(sorted_order_price_diff_sell_flow_s, cumulative_prob_order_price_diff_sell_flow_s, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller')
plt.title('CDF of the Order Price Difference (T1-T20)')
plt.xlabel('Order Width')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_order_price_diff_cdf_full.png'))
plt.close()


plt.figure(figsize=(15, 10))
plt.plot(sorted_order_price_diff_buy_flow_r_first, cumulative_prob_order_price_diff_buy_flow_r_first, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T1-T10')
plt.plot(sorted_order_price_diff_sell_flow_r_first, cumulative_prob_order_price_diff_sell_flow_r_first, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T1-T10')
plt.plot(sorted_order_price_diff_buy_flow_s_first, cumulative_prob_order_price_diff_buy_flow_s_first, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T1-T10')
plt.plot(sorted_order_price_diff_sell_flow_s_first, cumulative_prob_order_price_diff_sell_flow_s_first, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T1-T10')
plt.plot(sorted_order_price_diff_buy_flow_r_last, cumulative_prob_order_price_diff_buy_flow_r_last, marker=',', linestyle='dashdot', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T11-T20')
plt.plot(sorted_order_price_diff_sell_flow_r_last, cumulative_prob_order_price_diff_sell_flow_r_last, marker=',', linestyle='dashdot', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T11-T20')
plt.plot(sorted_order_price_diff_buy_flow_s_last, cumulative_prob_order_price_diff_buy_flow_s_last, linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T11-T20')
plt.plot(sorted_order_price_diff_sell_flow_s_last, cumulative_prob_order_price_diff_sell_flow_s_last, linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T11-T20')
plt.title('CDF of the Order Price Difference')
plt.xlabel('Order Width')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_order_price_diff_cdf_all.png'))
plt.close()

# realized surplus
sorted_realized_surplus_buy_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['realized_surplus'].tolist())
sorted_realized_surplus_buy_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['realized_surplus'].tolist())

sorted_realized_surplus_buy_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())
sorted_realized_surplus_buy_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())

sorted_realized_surplus_buy_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())
sorted_realized_surplus_buy_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['realized_surplus'].tolist())

cumulative_prob_realized_surplus_buy_flow_r = np.arange(1, len(sorted_realized_surplus_buy_flow_r) + 1) / len(sorted_realized_surplus_buy_flow_r)
cumulative_prob_realized_surplus_buy_flow_s = np.arange(1, len(sorted_realized_surplus_buy_flow_s) + 1) / len(sorted_realized_surplus_buy_flow_s)
cumulative_prob_realized_surplus_sell_flow_r = np.arange(1, len(sorted_realized_surplus_sell_flow_r) + 1) / len(sorted_realized_surplus_sell_flow_r)
cumulative_prob_realized_surplus_sell_flow_s = np.arange(1, len(sorted_realized_surplus_sell_flow_s) + 1) / len(sorted_realized_surplus_sell_flow_s)

cumulative_prob_realized_surplus_buy_flow_r_first = np.arange(1, len(sorted_realized_surplus_buy_flow_r_first) + 1) / len(sorted_realized_surplus_buy_flow_r_first)
cumulative_prob_realized_surplus_buy_flow_s_first = np.arange(1, len(sorted_realized_surplus_buy_flow_s_first) + 1) / len(sorted_realized_surplus_buy_flow_s_first)
cumulative_prob_realized_surplus_sell_flow_r_first = np.arange(1, len(sorted_realized_surplus_sell_flow_r_first) + 1) / len(sorted_realized_surplus_sell_flow_r_first)
cumulative_prob_realized_surplus_sell_flow_s_first = np.arange(1, len(sorted_realized_surplus_sell_flow_s_first) + 1) / len(sorted_realized_surplus_sell_flow_s_first)

cumulative_prob_realized_surplus_buy_flow_r_last = np.arange(1, len(sorted_realized_surplus_buy_flow_r_last) + 1) / len(sorted_realized_surplus_buy_flow_r_last)
cumulative_prob_realized_surplus_buy_flow_s_last = np.arange(1, len(sorted_realized_surplus_buy_flow_s_last) + 1) / len(sorted_realized_surplus_buy_flow_s_last)
cumulative_prob_realized_surplus_sell_flow_r_last = np.arange(1, len(sorted_realized_surplus_sell_flow_r_last) + 1) / len(sorted_realized_surplus_sell_flow_r_last)
cumulative_prob_realized_surplus_sell_flow_s_last = np.arange(1, len(sorted_realized_surplus_sell_flow_s_last) + 1) / len(sorted_realized_surplus_sell_flow_s_last)

plt.figure(figsize=(20, 10))
plt.plot(sorted_realized_surplus_buy_flow_r, cumulative_prob_realized_surplus_buy_flow_r, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer')
plt.plot(sorted_realized_surplus_sell_flow_r, cumulative_prob_realized_surplus_sell_flow_r, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller')
plt.plot(sorted_realized_surplus_buy_flow_s, cumulative_prob_realized_surplus_buy_flow_s, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer')
plt.plot(sorted_realized_surplus_sell_flow_s, cumulative_prob_realized_surplus_sell_flow_s, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller')
plt.title('CDF of the Realized Surplus (T1-T20)')
plt.xlabel('Realized Surplus')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_realized_surplus_cdf_full.png'))
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(sorted_realized_surplus_buy_flow_r_first, cumulative_prob_realized_surplus_buy_flow_r_first, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T1-T10')
plt.plot(sorted_realized_surplus_sell_flow_r_first, cumulative_prob_realized_surplus_sell_flow_r_first, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T1-T10')
plt.plot(sorted_realized_surplus_buy_flow_s_first, cumulative_prob_realized_surplus_buy_flow_s_first, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T1-T10')
plt.plot(sorted_realized_surplus_sell_flow_s_first, cumulative_prob_realized_surplus_sell_flow_s_first, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T1-T10')
plt.plot(sorted_realized_surplus_buy_flow_r_last, cumulative_prob_realized_surplus_buy_flow_r_last, marker=',', linestyle='dashdot', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T11-T20')
plt.plot(sorted_realized_surplus_sell_flow_r_last, cumulative_prob_realized_surplus_sell_flow_r_last, marker=',', linestyle='dashdot', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T11-T20')
plt.plot(sorted_realized_surplus_buy_flow_s_last, cumulative_prob_realized_surplus_buy_flow_s_last, linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T11-T20')
plt.plot(sorted_realized_surplus_sell_flow_s_last, cumulative_prob_realized_surplus_sell_flow_s_last, linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T11-T20')
plt.title('CDF of the Realized Surplus')
plt.xlabel('Realized Surplus')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_realized_surplus_cdf_all.png'))
plt.close()


# price deviation from contract price
sorted_price_dev_from_contract_buy_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_buy_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_buy_cda = np.sort(cda_trader_period[(cda_trader_period['direction'] == 'buy')]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_cda = np.sort(cda_trader_period[(cda_trader_period['direction'] == 'sell')]['price_dev_from_contract'].tolist())

sorted_price_dev_from_contract_buy_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_buy_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_buy_cda_first = np.sort(cda_trader_period[(cda_trader_period['direction'] == 'buy') & (cda_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_cda_first = np.sort(cda_trader_period[(cda_trader_period['direction'] == 'sell') & (cda_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())    

sorted_price_dev_from_contract_buy_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_buy_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_buy_cda_last = np.sort(cda_trader_period[(cda_trader_period['direction'] == 'buy') & (cda_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())
sorted_price_dev_from_contract_sell_cda_last = np.sort(cda_trader_period[(cda_trader_period['direction'] == 'sell') & (cda_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['price_dev_from_contract'].tolist())

cumulative_prob_price_dev_from_contract_buy_flow_r = np.arange(1, len(sorted_price_dev_from_contract_buy_flow_r) + 1) / len(sorted_price_dev_from_contract_buy_flow_r)
cumulative_prob_price_dev_from_contract_buy_flow_s = np.arange(1, len(sorted_price_dev_from_contract_buy_flow_s) + 1) / len(sorted_price_dev_from_contract_buy_flow_s)
cumulative_prob_price_dev_from_contract_sell_flow_r = np.arange(1, len(sorted_price_dev_from_contract_sell_flow_r) + 1) / len(sorted_price_dev_from_contract_sell_flow_r)
cumulative_prob_price_dev_from_contract_sell_flow_s = np.arange(1, len(sorted_price_dev_from_contract_sell_flow_s) + 1) / len(sorted_price_dev_from_contract_sell_flow_s)
cumulative_prob_price_dev_from_contract_buy_cda = np.arange(1, len(sorted_price_dev_from_contract_buy_cda) + 1) / len(sorted_price_dev_from_contract_buy_cda)
cumulative_prob_price_dev_from_contract_sell_cda = np.arange(1, len(sorted_price_dev_from_contract_sell_cda) + 1) / len(sorted_price_dev_from_contract_sell_cda)

cumulative_prob_price_dev_from_contract_buy_flow_r_first = np.arange(1, len(sorted_price_dev_from_contract_buy_flow_r_first) + 1) / len(sorted_price_dev_from_contract_buy_flow_r_first)
cumulative_prob_price_dev_from_contract_buy_flow_s_first = np.arange(1, len(sorted_price_dev_from_contract_buy_flow_s_first) + 1) / len(sorted_price_dev_from_contract_buy_flow_s_first)
cumulative_prob_price_dev_from_contract_sell_flow_r_first = np.arange(1, len(sorted_price_dev_from_contract_sell_flow_r_first) + 1) / len(sorted_price_dev_from_contract_sell_flow_r_first)
cumulative_prob_price_dev_from_contract_sell_flow_s_first = np.arange(1, len(sorted_price_dev_from_contract_sell_flow_s_first) + 1) / len(sorted_price_dev_from_contract_sell_flow_s_first)
cumulative_prob_price_dev_from_contract_buy_cda_first = np.arange(1, len(sorted_price_dev_from_contract_buy_cda_first) + 1) / len(sorted_price_dev_from_contract_buy_cda_first)
cumulative_prob_price_dev_from_contract_sell_cda_first = np.arange(1, len(sorted_price_dev_from_contract_sell_cda_first) + 1) / len(sorted_price_dev_from_contract_sell_cda_first)

cumulative_prob_price_dev_from_contract_buy_flow_r_last = np.arange(1, len(sorted_price_dev_from_contract_buy_flow_r_last) + 1) / len(sorted_price_dev_from_contract_buy_flow_r_last)
cumulative_prob_price_dev_from_contract_buy_flow_s_last = np.arange(1, len(sorted_price_dev_from_contract_buy_flow_s_last) + 1) / len(sorted_price_dev_from_contract_buy_flow_s_last)
cumulative_prob_price_dev_from_contract_sell_flow_r_last = np.arange(1, len(sorted_price_dev_from_contract_sell_flow_r_last) + 1) / len(sorted_price_dev_from_contract_sell_flow_r_last)
cumulative_prob_price_dev_from_contract_sell_flow_s_last = np.arange(1, len(sorted_price_dev_from_contract_sell_flow_s_last) + 1) / len(sorted_price_dev_from_contract_sell_flow_s_last)
cumulative_prob_price_dev_from_contract_buy_cda_last = np.arange(1, len(sorted_price_dev_from_contract_buy_cda_last) + 1) / len(sorted_price_dev_from_contract_buy_cda_last)
cumulative_prob_price_dev_from_contract_sell_cda_last = np.arange(1, len(sorted_price_dev_from_contract_sell_cda_last) + 1) / len(sorted_price_dev_from_contract_sell_cda_last)

plt.figure(figsize=(15, 10))
plt.plot(sorted_price_dev_from_contract_buy_flow_r, cumulative_prob_price_dev_from_contract_buy_flow_r, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer')
plt.plot(sorted_price_dev_from_contract_sell_flow_r, cumulative_prob_price_dev_from_contract_sell_flow_r, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller')
plt.plot(sorted_price_dev_from_contract_buy_flow_s, cumulative_prob_price_dev_from_contract_buy_flow_s, linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer')
plt.plot(sorted_price_dev_from_contract_sell_flow_s, cumulative_prob_price_dev_from_contract_sell_flow_s, linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller')
plt.plot(sorted_price_dev_from_contract_buy_cda, cumulative_prob_price_dev_from_contract_buy_cda, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_price_dev_from_contract_sell_cda, cumulative_prob_price_dev_from_contract_sell_cda, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.title('CDF of the Price Deviation from Contract Price (T1-T20)')
plt.xlabel('Price Minimum Margin')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_price_dev_from_contract_cdf_full.png'))
plt.close()

zero_counts = {
    "FlowR Buyer T1-T10": (np.count_nonzero(sorted_price_dev_from_contract_buy_flow_r_first == 0), len(sorted_price_dev_from_contract_buy_flow_r_first)),
    "FlowS Buyer T1-T10": (np.count_nonzero(sorted_price_dev_from_contract_buy_flow_s_first == 0), len(sorted_price_dev_from_contract_buy_flow_s_first)), 
    "FlowR Seller T1-T10": (np.count_nonzero(sorted_price_dev_from_contract_sell_flow_r_first == 0), len(sorted_price_dev_from_contract_sell_flow_r_first)),
    "FlowS Seller T1-T10": (np.count_nonzero(sorted_price_dev_from_contract_sell_flow_s_first == 0), len(sorted_price_dev_from_contract_sell_flow_s_first)),
    "FlowR Buyer T11-T20": (np.count_nonzero(sorted_price_dev_from_contract_buy_flow_r_last == 0), len(sorted_price_dev_from_contract_buy_flow_r_last)),
    "FlowS Buyer T11-T20": (np.count_nonzero(sorted_price_dev_from_contract_buy_flow_s_last == 0), len(sorted_price_dev_from_contract_buy_flow_s_last)),
    "FlowR Seller T11-T20": (np.count_nonzero(sorted_price_dev_from_contract_sell_flow_r_last == 0), len(sorted_price_dev_from_contract_sell_flow_r_last)),
    "FlowS Seller T11-T20": (np.count_nonzero(sorted_price_dev_from_contract_sell_flow_s_last == 0), len(sorted_price_dev_from_contract_sell_flow_s_last)),
    "CDA Buyer T1-T10": (np.count_nonzero(sorted_price_dev_from_contract_buy_cda_first == 0), len(sorted_price_dev_from_contract_buy_cda_first)),
    "CDA Seller T1-T10": (np.count_nonzero(sorted_price_dev_from_contract_sell_cda_first == 0), len(sorted_price_dev_from_contract_sell_cda_first)),
    "CDA Buyer T11-T20": (np.count_nonzero(sorted_price_dev_from_contract_buy_cda_last == 0), len(sorted_price_dev_from_contract_buy_cda_last)),
    "CDA Seller T11-T20": (np.count_nonzero(sorted_price_dev_from_contract_sell_cda_last == 0), len(sorted_price_dev_from_contract_sell_cda_last))
}

# Print the results
print("Counts of zero price deviation from contract price:")
print("-------------------------------------------------")
for label, count in zero_counts.items():
    print(f"{label}: {count}")

plt.figure(figsize=(15, 10))
plt.plot(sorted_price_dev_from_contract_buy_flow_r_first, cumulative_prob_price_dev_from_contract_buy_flow_r_first, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T1-T10')
plt.plot(sorted_price_dev_from_contract_sell_flow_r_first, cumulative_prob_price_dev_from_contract_sell_flow_r_first, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T1-T10')
plt.plot(sorted_price_dev_from_contract_buy_flow_s_first, cumulative_prob_price_dev_from_contract_buy_flow_s_first, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T1-T10')
plt.plot(sorted_price_dev_from_contract_sell_flow_s_first, cumulative_prob_price_dev_from_contract_sell_flow_s_first, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T1-T10')
plt.plot(sorted_price_dev_from_contract_buy_flow_r_last, cumulative_prob_price_dev_from_contract_buy_flow_r_last, marker=',', linestyle='dashdot', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T11-T20')
plt.plot(sorted_price_dev_from_contract_sell_flow_r_last, cumulative_prob_price_dev_from_contract_sell_flow_r_last, marker=',', linestyle='dashdot', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T11-T20')
plt.plot(sorted_price_dev_from_contract_buy_flow_s_last, cumulative_prob_price_dev_from_contract_buy_flow_s_last, linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T11-T20')
plt.plot(sorted_price_dev_from_contract_sell_flow_s_last, cumulative_prob_price_dev_from_contract_sell_flow_s_last, linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T11-T20')
plt.title('CDF of the Price Deviation from Contract Price')
plt.xlabel('Price Minimum Margin')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_price_dev_from_contract_cdf_all.png'))
plt.close()


# max rate 
sorted_max_rate_buy_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['max_rate'].tolist())
sorted_max_rate_buy_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['max_rate'].tolist())
sorted_max_rate_sell_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['max_rate'].tolist())
sorted_max_rate_sell_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['max_rate'].tolist())

sorted_max_rate_buy_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())
sorted_max_rate_buy_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())
sorted_max_rate_sell_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())
sorted_max_rate_sell_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())

sorted_max_rate_buy_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())
sorted_max_rate_buy_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())
sorted_max_rate_sell_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())
sorted_max_rate_sell_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate'].tolist())

cumulative_prob_max_rate_buy_flow_r = np.arange(1, len(sorted_max_rate_buy_flow_r) + 1) / len(sorted_max_rate_buy_flow_r)
cumulative_prob_max_rate_buy_flow_s = np.arange(1, len(sorted_max_rate_buy_flow_s) + 1) / len(sorted_max_rate_buy_flow_s)
cumulative_prob_max_rate_sell_flow_r = np.arange(1, len(sorted_max_rate_sell_flow_r) + 1) / len(sorted_max_rate_sell_flow_r)
cumulative_prob_max_rate_sell_flow_s = np.arange(1, len(sorted_max_rate_sell_flow_s) + 1) / len(sorted_max_rate_sell_flow_s)

cumulative_prob_max_rate_buy_flow_r_first = np.arange(1, len(sorted_max_rate_buy_flow_r_first) + 1) / len(sorted_max_rate_buy_flow_r_first)
cumulative_prob_max_rate_buy_flow_s_first = np.arange(1, len(sorted_max_rate_buy_flow_s_first) + 1) / len(sorted_max_rate_buy_flow_s_first)
cumulative_prob_max_rate_sell_flow_r_first = np.arange(1, len(sorted_max_rate_sell_flow_r_first) + 1) / len(sorted_max_rate_sell_flow_r_first)
cumulative_prob_max_rate_sell_flow_s_first = np.arange(1, len(sorted_max_rate_sell_flow_s_first) + 1) / len(sorted_max_rate_sell_flow_s_first)

cumulative_prob_max_rate_buy_flow_r_last = np.arange(1, len(sorted_max_rate_buy_flow_r_last) + 1) / len(sorted_max_rate_buy_flow_r_last)
cumulative_prob_max_rate_buy_flow_s_last = np.arange(1, len(sorted_max_rate_buy_flow_s_last) + 1) / len(sorted_max_rate_buy_flow_s_last)
cumulative_prob_max_rate_sell_flow_r_last = np.arange(1, len(sorted_max_rate_sell_flow_r_last) + 1) / len(sorted_max_rate_sell_flow_r_last)
cumulative_prob_max_rate_sell_flow_s_last = np.arange(1, len(sorted_max_rate_sell_flow_s_last) + 1) / len(sorted_max_rate_sell_flow_s_last)

plt.figure(figsize=(15, 10))
plt.plot(sorted_max_rate_buy_flow_r, cumulative_prob_max_rate_buy_flow_r, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer')
plt.plot(sorted_max_rate_sell_flow_r, cumulative_prob_max_rate_sell_flow_r, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller')
plt.plot(sorted_max_rate_buy_flow_s, cumulative_prob_max_rate_buy_flow_s, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer')
plt.plot(sorted_max_rate_sell_flow_s, cumulative_prob_max_rate_sell_flow_s, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller')
plt.title('CDF of the Max Rate (T1-T20)')
plt.xlabel('Max Rate')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir,  'groups_flow_max_rate_cdf_full.png'))
plt.close()

plt.figure(figsize=(15, 10))
plt.plot(sorted_max_rate_buy_flow_r_first, cumulative_prob_max_rate_buy_flow_r_first, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T1-T10')
plt.plot(sorted_max_rate_sell_flow_r_first, cumulative_prob_max_rate_sell_flow_r_first, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T1-T10')
plt.plot(sorted_max_rate_buy_flow_s_first, cumulative_prob_max_rate_buy_flow_s_first, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T1-T10')
plt.plot(sorted_max_rate_sell_flow_s_first, cumulative_prob_max_rate_sell_flow_s_first, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T1-T10')
plt.plot(sorted_max_rate_buy_flow_r_last, cumulative_prob_max_rate_buy_flow_r_last, marker=',', linestyle='dashdot', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T11-T20')
plt.plot(sorted_max_rate_sell_flow_r_last, cumulative_prob_max_rate_sell_flow_r_last, marker=',', linestyle='dashdot', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T11-T20')
plt.plot(sorted_max_rate_buy_flow_s_last, cumulative_prob_max_rate_buy_flow_s_last, linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T11-T20')
plt.plot(sorted_max_rate_sell_flow_s_last, cumulative_prob_max_rate_sell_flow_s_last, linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T11-T20')
plt.title('CDF of the Max Rate')
plt.xlabel('Max Rate')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_max_rate_cdf_all.png'))
plt.close()

# max rate percent 
sorted_max_rate_percent_buy_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['max_rate_percent'].tolist())
sorted_max_rate_percent_buy_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['max_rate_percent'].tolist())
sorted_max_rate_percent_sell_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['max_rate_percent'].tolist())
sorted_max_rate_percent_sell_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['max_rate_percent'].tolist())

sorted_max_rate_percent_buy_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())
sorted_max_rate_percent_buy_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())
sorted_max_rate_percent_sell_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())
sorted_max_rate_percent_sell_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())

sorted_max_rate_percent_buy_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())
sorted_max_rate_percent_buy_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())
sorted_max_rate_percent_sell_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())
sorted_max_rate_percent_sell_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['max_rate_percent'].tolist())

cumulative_prob_max_rate_percent_buy_flow_r = np.arange(1, len(sorted_max_rate_percent_buy_flow_r) + 1) / len(sorted_max_rate_percent_buy_flow_r)
cumulative_prob_max_rate_percent_buy_flow_s = np.arange(1, len(sorted_max_rate_percent_buy_flow_s) + 1) / len(sorted_max_rate_percent_buy_flow_s)
cumulative_prob_max_rate_percent_sell_flow_r = np.arange(1, len(sorted_max_rate_percent_sell_flow_r) + 1) / len(sorted_max_rate_percent_sell_flow_r)
cumulative_prob_max_rate_percent_sell_flow_s = np.arange(1, len(sorted_max_rate_percent_sell_flow_s) + 1) / len(sorted_max_rate_percent_sell_flow_s)

cumulative_prob_max_rate_percent_buy_flow_r_first = np.arange(1, len(sorted_max_rate_percent_buy_flow_r_first) + 1) / len(sorted_max_rate_percent_buy_flow_r_first)
cumulative_prob_max_rate_percent_buy_flow_s_first = np.arange(1, len(sorted_max_rate_percent_buy_flow_s_first) + 1) / len(sorted_max_rate_percent_buy_flow_s_first)
cumulative_prob_max_rate_percent_sell_flow_r_first = np.arange(1, len(sorted_max_rate_percent_sell_flow_r_first) + 1) / len(sorted_max_rate_percent_sell_flow_r_first)
cumulative_prob_max_rate_percent_sell_flow_s_first = np.arange(1, len(sorted_max_rate_percent_sell_flow_s_first) + 1) / len(sorted_max_rate_percent_sell_flow_s_first)

cumulative_prob_max_rate_percent_buy_flow_r_last = np.arange(1, len(sorted_max_rate_percent_buy_flow_r_last) + 1) / len(sorted_max_rate_percent_buy_flow_r_last)
cumulative_prob_max_rate_percent_buy_flow_s_last = np.arange(1, len(sorted_max_rate_percent_buy_flow_s_last) + 1) / len(sorted_max_rate_percent_buy_flow_s_last)
cumulative_prob_max_rate_percent_sell_flow_r_last = np.arange(1, len(sorted_max_rate_percent_sell_flow_r_last) + 1) / len(sorted_max_rate_percent_sell_flow_r_last)
cumulative_prob_max_rate_percent_sell_flow_s_last = np.arange(1, len(sorted_max_rate_percent_sell_flow_s_last) + 1) / len(sorted_max_rate_percent_sell_flow_s_last)

plt.figure(figsize=(15, 10))
plt.plot(sorted_max_rate_percent_buy_flow_r, cumulative_prob_max_rate_percent_buy_flow_r, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer')
plt.plot(sorted_max_rate_percent_sell_flow_r, cumulative_prob_max_rate_percent_sell_flow_r, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller')
plt.plot(sorted_max_rate_percent_buy_flow_s, cumulative_prob_max_rate_percent_buy_flow_s, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer')
plt.plot(sorted_max_rate_percent_sell_flow_s, cumulative_prob_max_rate_percent_sell_flow_s, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller')
plt.title('CDF of the Max Rate (T1-T20)')
plt.xlabel('Max Rate')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_max_rate_percent_cdf_full.png'))
plt.close()


plt.figure(figsize=(15, 10))
plt.plot(sorted_max_rate_percent_buy_flow_r_first, cumulative_prob_max_rate_percent_buy_flow_r_first, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T1-T10')
plt.plot(sorted_max_rate_percent_sell_flow_r_first, cumulative_prob_max_rate_percent_sell_flow_r_first, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T1-T10')
plt.plot(sorted_max_rate_percent_buy_flow_s_first, cumulative_prob_max_rate_percent_buy_flow_s_first, linestyle='solid', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T1-T10')
plt.plot(sorted_max_rate_percent_sell_flow_s_first, cumulative_prob_max_rate_percent_sell_flow_s_first, linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T1-T10')
plt.plot(sorted_max_rate_percent_buy_flow_r_last, cumulative_prob_max_rate_percent_buy_flow_r_last, marker=',', linestyle='dashdot', color=(0, 128/255, 0), markersize=5, label='FlowR Buyer T11-T20')
plt.plot(sorted_max_rate_percent_sell_flow_r_last, cumulative_prob_max_rate_percent_sell_flow_r_last, marker=',', linestyle='dashdot', color=(128/255, 0, 128/255), markersize=5, label='FlowR Seller T11-T20')
plt.plot(sorted_max_rate_percent_buy_flow_s_last, cumulative_prob_max_rate_percent_buy_flow_s_last, linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='FlowS Buyer T11-T20')
plt.plot(sorted_max_rate_percent_sell_flow_s_last, cumulative_prob_max_rate_percent_sell_flow_s_last, linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='FlowS Seller T11-T20')
plt.title('CDF of the Max Rate')
plt.xlabel('Max Rate')
plt.ylabel('Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'groups_flow_max_rate_percent_cdf_all.png'))
plt.close()

# excess profit
sorted_excess_profit_buy_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['excess_profit'].tolist())
sorted_excess_profit_buy_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy')]['excess_profit'].tolist())
sorted_excess_profit_sell_flow_r = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['excess_profit'].tolist())
sorted_excess_profit_sell_flow_s = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell')]['excess_profit'].tolist())
sorted_excess_profit_buy_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
sorted_excess_profit_buy_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
sorted_excess_profit_sell_flow_r_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
sorted_excess_profit_sell_flow_s_first = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] <= (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
sorted_excess_profit_buy_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
sorted_excess_profit_buy_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'buy') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
sorted_excess_profit_sell_flow_r_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] <= num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
sorted_excess_profit_sell_flow_s_last = np.sort(flow_trader_period[(flow_trader_period['group_id'] > num_groups_flow_low) & (flow_trader_period['direction'] == 'sell') & (flow_trader_period['round'] > (num_rounds - prac_rounds) // 2)]['excess_profit'].tolist())
cumulative_prob_excess_profit_buy_flow_r = np.arange(1, len(sorted_excess_profit_buy_flow_r) + 1) / len(sorted_excess_profit_buy_flow_r)
cumulative_prob_excess_profit_buy_flow_s = np.arange(1, len(sorted_excess_profit_buy_flow_s) + 1) / len(sorted_excess_profit_buy_flow_s)
cumulative_prob_excess_profit_sell_flow_r = np.arange(1, len(sorted_excess_profit_sell_flow_r) + 1) / len(sorted_excess_profit_sell_flow_r)
cumulative_prob_excess_profit_sell_flow_s = np.arange(1, len(sorted_excess_profit_sell_flow_s) + 1) / len(sorted_excess_profit_sell_flow_s)
cumulative_prob_excess_profit_buy_flow_r_first = np.arange(1, len(sorted_excess_profit_buy_flow_r_first) + 1) / len(sorted_excess_profit_buy_flow_r_first)
cumulative_prob_excess_profit_buy_flow_s_first = np.arange(1, len(sorted_excess_profit_buy_flow_s_first) + 1) / len(sorted_excess_profit_buy_flow_s_first)
cumulative_prob_excess_profit_sell_flow_r_first = np.arange(1, len(sorted_excess_profit_sell_flow_r_first) + 1) / len(sorted_excess_profit_sell_flow_r_first)
cumulative_prob_excess_profit_sell_flow_s_first = np.arange(1, len(sorted_excess_profit_sell_flow_s_first) + 1) / len(sorted_excess_profit_sell_flow_s_first)
cumulative_prob_excess_profit_buy_flow_r_last = np.arange(1, len(sorted_excess_profit_buy_flow_r_last) + 1) / len(sorted_excess_profit_buy_flow_r_last)
cumulative_prob_excess_profit_buy_flow_s_last = np.arange(1, len(sorted_excess_profit_buy_flow_s_last) + 1) / len(sorted_excess_profit_buy_flow_s_last)
cumulative_prob_excess_profit_sell_flow_r_last = np.arange(1, len(sorted_excess_profit_sell_flow_r_last) + 1) / len(sorted_excess_profit_sell_flow_r_last)
cumulative_prob_excess_profit_sell_flow_s_last = np.arange(1, len(sorted_excess_profit_sell_flow_s_last) + 1) / len(sorted_excess_profit_sell_flow_s_last)


########## scatter plots ##########

flow_trader_period['category']  = flow_trader_period[['format', 'direction', 'time']].astype(str).agg(' '.join, axis=1)

markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P']
colors = plt.cm.tab10.colors

# max_price - min_price vs excess profit
fig, ax = plt.subplots(figsize=(15, 10))
for i, group in enumerate(flow_trader_period['category'].unique()):
    group_data = flow_trader_period[flow_trader_period['category'] == group]
    ax.scatter(group_data['order_price_diff'], 
               group_data['excess_profit'], 
               marker=markers[i % len(markers)], 
               color=colors[i % len(colors)], 
               label=group, 
               alpha=0.5)
    
    coeffs = np.polyfit(group_data['order_price_diff'], group_data['excess_profit'], 1)
    x_vals = np.linspace(group_data['order_price_diff'].min(), group_data['order_price_diff'].max(), 100)
    y_vals = coeffs[0] * x_vals + coeffs[1]
    ax.plot(x_vals, y_vals, color=colors[i % len(colors)], linestyle='--', alpha=0.5)

ax.set_title('pH - pL vs Excess Profit')
ax.set_xlabel('pH - pL')
ax.set_ylabel('Excess Profit')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'groups_flow_pH-pL_vs_excess_profit.png'))
plt.close()


# max_rate_percent vs excess profit
fig, ax = plt.subplots(figsize=(15, 10))
for i, group in enumerate(flow_trader_period['category'].unique()):
    group_data = flow_trader_period[flow_trader_period['category'] == group]
    ax.scatter(group_data['max_rate_percent'], 
               group_data['excess_profit'], 
               marker=markers[i % len(markers)], 
               color=colors[i % len(colors)], 
               label=group, 
               alpha=0.5)
    
    coeffs = np.polyfit(group_data['max_rate_percent'], group_data['excess_profit'], 1)
    x_vals = np.linspace(group_data['max_rate_percent'].min(), group_data['max_rate_percent'].max(), 100)
    y_vals = coeffs[0] * x_vals + coeffs[1]
    ax.plot(x_vals, y_vals, color=colors[i % len(colors)], linestyle='--', alpha=0.5)
ax.set_title('Max Rate Percent vs Excess Profit')
ax.set_xlabel('Max Rate Percent')
ax.set_ylabel('Excess Profit')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'groups_flow_max_rate_percent_vs_excess_profit.png'))
plt.close()

# price_dev_from_contract vs excess profit
fig, ax = plt.subplots(figsize=(15, 10))
for i, group in enumerate(flow_trader_period['category'].unique()):
    group_data = flow_trader_period[flow_trader_period['category'] == group]
    ax.scatter(group_data['price_dev_from_contract'], 
               group_data['excess_profit'], 
               marker=markers[i % len(markers)], 
               color=colors[i % len(colors)], 
               label=group, 
               alpha=0.5)
    
    coeffs = np.polyfit(group_data['price_dev_from_contract'], group_data['excess_profit'], 1)
    x_vals = np.linspace(group_data['price_dev_from_contract'].min(), group_data['price_dev_from_contract'].max(), 100)
    y_vals = coeffs[0] * x_vals + coeffs[1]
    ax.plot(x_vals, y_vals, color=colors[i % len(colors)], linestyle='--', alpha=0.5)
ax.set_title('Price Deviation from Contract Price vs Excess Profit')
ax.set_xlabel('Price Deviation from Contract Price')
ax.set_ylabel('Excess Profits')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'groups_flow_price_dev_from_contract_vs_excess_profit.png'))
plt.close()


########## correlation ########## 
pearson_1 = flow_trader_period.groupby('category').apply(lambda g: g['excess_profit'].corr(g['price_dev_from_contract']))
pearson_2 = flow_trader_period.groupby('category').apply(lambda g: g['excess_profit'].corr(g['max_rate_percent']))
pearson_3 = flow_trader_period.groupby('category').apply(lambda g: g['excess_profit'].corr(g['order_price_diff']))
pearson_4 = flow_trader_period.groupby('category').apply(lambda g: g['price_dev_from_contract'].corr(g['max_rate_percent']))
pearson_5 = flow_trader_period.groupby('category').apply(lambda g: g['price_dev_from_contract'].corr(g['order_price_diff']))
pearson_6 = flow_trader_period.groupby('category').apply(lambda g: g['max_rate_percent'].corr(g['order_price_diff']))

print("Pearson Correlation between excess profit and price deviation from contract price:")
print(pearson_1)
print("Correlation between excess profit and max rate percent:")
print(pearson_2)
print("Correlation between excess profit and order price difference:")
print(pearson_3)
print("Correlation between price deviation from contract price and max rate percent:")
print(pearson_4)
print("Correlation between price deviation from contract price and order price difference:")
print(pearson_5)
print("Correlation between order price difference and max rate percent:")
print(pearson_6)

spearman_1 = flow_trader_period.groupby('category').apply(lambda g: g['excess_profit'].corr(g['price_dev_from_contract'], method='spearman'))
spearman_2 = flow_trader_period.groupby('category').apply(lambda g: g['excess_profit'].corr(g['max_rate_percent'], method='spearman'))
spearman_3 = flow_trader_period.groupby('category').apply(lambda g: g['excess_profit'].corr(g['order_price_diff'], method='spearman'))
spearman_4 = flow_trader_period.groupby('category').apply(lambda g: g['price_dev_from_contract'].corr(g['max_rate_percent'], method='spearman'))
spearman_5 = flow_trader_period.groupby('category').apply(lambda g: g['price_dev_from_contract'].corr(g['order_price_diff'], method='spearman'))
spearman_6 = flow_trader_period.groupby('category').apply(lambda g: g['max_rate_percent'].corr(g['order_price_diff'], method='spearman'))

print("Spearman Correlation between excess profit and price deviation from contract price:")
print(spearman_1)
print("Correlation between excess profit and max rate percent:")
print(spearman_2)
print("Correlation between excess profit and order price difference:")
print(spearman_3)
print("Correlation between price deviation from contract price and max rate percent:")
print(spearman_4)
print("Correlation between price deviation from contract price and order price difference:")
print(spearman_5)
print("Correlation between order price difference and max rate percent:")
print(spearman_6)