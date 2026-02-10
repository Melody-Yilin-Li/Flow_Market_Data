from helpers import *
from common import *
from config import *

directory = '/Users/YilinLi/Documents/UCSC/Flow Data/Flow_Market_Data/data/'
plt.close()

def main():
    print("params imported")

if __name__ == "__main__":
     main()

     
regress_flow_ind = pd.DataFrame()

# read in data 
list_individual_flow = []


executed_percent_buy_flow30_first10, executed_percent_sell_flow30_first10, executed_percent_buy_flow30_last10, executed_percent_sell_flow30_last10 = [], [], [], []
executed_percent_buy_flow60_first10, executed_percent_sell_flow60_first10, executed_percent_buy_flow60_last10, executed_percent_sell_flow60_last10 = [], [], [], []

order_price_spread_buy_flow30_first10, order_price_spread_sell_flow30_first10, order_price_spread_buy_flow30_last10, order_price_spread_sell_flow30_last10 = [], [], [], []
order_price_spread_buy_flow60_first10, order_price_spread_sell_flow60_first10, order_price_spread_buy_flow60_last10, order_price_spread_sell_flow60_last10 = [], [], [], []

for g in range(1, num_flow + 1):
    name = 'group' + str(g)
    group_mkt = []
    for r in range(1, num_periods - prac_periods + 1): 
        path = directory + 'flow{}/{}/1_market.json'.format(g, r + prac_periods)
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
    for r in range(1, num_periods - prac_periods + 1):
        order_price_low_buy = []
        order_price_high_buy = []
        order_num_buy = 0
        order_quantity_buy = []
        order_rate_buy = []

        order_price_low_buy_initial = []
        order_price_high_buy_initial = []
        order_quantity_buy_initial = []
        order_rate_buy_initial = []

        order_price_low_sell = []
        order_price_high_sell = []
        order_num_sell = 0
        order_quantity_sell = []
        order_rate_sell = []

        order_price_low_sell_initial = []
        order_price_high_sell_initial = []
        order_quantity_sell_initial = []
        order_rate_sell_initial = []

        visited = set()
        
        path = directory + 'flow{}/{}/1_participant.json'.format(g, r + prac_periods)
        rnd = pd.read_json(path)
        rnd = pd.merge(rnd, group_mkt[r - 1], how='left', on='timestamp') # attach clearing price and clearing rate 
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)

        max_rate_orders_buy = max_rate_orders_sell = 0
        executed_percent_buy, executed_percent_sell = {}, {}
        spread_buy, spread_sell = {}, {} 

        for ind, row in rnd.iterrows():
            for order in row['active_orders']:
                if order['direction'] == 'sell':
                    if order['order_id'] not in visited:
                        if row['timestamp'] < early_period:
                            order_price_low_sell_initial.append(order['min_price'])
                            order_price_high_sell_initial.append(order['max_price'])
                            order_quantity_sell_initial.append(order['quantity'])
                            order_rate_sell_initial.append(order['max_rate'])
                        order_price_low_sell.append(order['min_price'])
                        order_price_high_sell.append(order['max_price'])
                        order_num_sell += 1
                        order_quantity_sell.append(order['quantity'])
                        order_rate_sell.append(order['max_rate'])
                        if g <= num_flow30 and order['max_rate'] == max_order_rate \
                            or g > num_flow30 and order['max_rate'] == max_order_rate60:
                            max_rate_orders_sell += 1
                        spread_sell[order['order_id']] = order['max_price'] - order['min_price']
                    executed_percent_sell[order['order_id']] = order['fill_quantity'] / order['quantity']
                elif order['direction'] == 'buy':
                    if order['order_id'] not in visited:
                        if row['timestamp'] < early_period:
                            order_price_low_buy_initial.append(order['min_price'])
                            order_price_high_buy_initial.append(order['max_price'])
                            order_quantity_buy_initial.append(order['quantity'])
                            order_rate_buy_initial.append(order['max_rate'])
                        order_price_low_buy.append(order['min_price'])
                        order_price_high_buy.append(order['max_price'])
                        order_num_buy += 1
                        order_quantity_buy.append(order['quantity'])
                        order_rate_buy.append(order['max_rate'])
                        if g <= num_flow30 and order['max_rate'] == max_order_rate30 \
                            or g > num_flow30 and order['max_rate'] == max_order_rate60:
                            max_rate_orders_buy += 1
                        spread_buy[order['order_id']] = order['max_price'] - order['min_price']
                    executed_percent_buy[order['order_id']] = order['fill_quantity'] / order['quantity']
                visited.add(order['order_id'])
            if g <= num_flow30 and r <= (num_periods - prac_periods) // 2:
                executed_percent_sell_flow30_first10.append(list(executed_percent_sell.values()))
                executed_percent_buy_flow30_first10.append(list(executed_percent_buy.values()))
                order_price_spread_sell_flow30_first10.append(list(spread_sell.values()))
                order_price_spread_buy_flow30_first10.append(list(spread_buy.values()))
            elif g <= num_flow30 and r > (num_periods - prac_periods) // 2:
                executed_percent_sell_flow30_last10.append(list(executed_percent_sell.values()))
                executed_percent_buy_flow30_last10.append(list(executed_percent_buy.values()))
                order_price_spread_sell_flow30_last10.append(list(spread_sell.values()))
                order_price_spread_buy_flow30_last10.append(list(spread_buy.values()))
            elif g > num_flow30 and r <= (num_periods - prac_periods) // 2:
                executed_percent_sell_flow60_first10.append(list(executed_percent_sell.values()))
                executed_percent_buy_flow60_first10.append(list(executed_percent_buy.values()))
                order_price_spread_sell_flow60_first10.append(list(spread_sell.values()))
                order_price_spread_buy_flow60_first10.append(list(spread_buy.values()))
            elif g > num_flow30 and r > (num_periods - prac_periods) // 2:
                executed_percent_sell_flow60_last10.append(list(executed_percent_sell.values()))
                executed_percent_buy_flow60_last10.append(list(executed_percent_buy.values()))
                order_price_spread_sell_flow60_last10.append(list(spread_sell.values()))
                order_price_spread_buy_flow60_last10.append(list(spread_buy.values()))
        
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
        rnd['contract_percent'] = rnd['inventory'] / rnd['quantity']
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
        rnd.loc[(rnd['ind_ce_profit'] == 0) & (rnd['projected_profit'] >= 0), 'realized_surplus'] = np.nan
        rnd.loc[(rnd['ind_ce_profit'] == 0) & (rnd['projected_profit'] < 0), 'realized_surplus'] = np.nan
        
        opposite_direction = 0
        same_direction = 0
        opposite_sell = set()
        opposite_buy = set()
        same_sell = set()
        same_buy = set()

        for ind, row in rnd.iterrows():
            if row['direction'] != row['order_direction']: 
                if row['order_direction'] == 'sell' and row['order_max_price'] < row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']: 
                    opposite_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    opposite_buy.add(row['id_in_group'])
                if row['order_direction'] == 'buy' and row['order_min_price'] > row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']: 
                    opposite_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    opposite_sell.add(row['id_in_group'])
            else:
                if row['order_direction'] == 'sell' and row['order_max_price'] < row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']:
                    same_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    same_sell.add(row['id_in_group'])
                elif row['order_direction'] == 'buy' and row['order_min_price'] > row['price'] and rnd.loc[ind - 1, 'projected_profit'] > row['projected_profit']:
                    same_direction += row['projected_profit'] - rnd.loc[ind - 1, 'projected_profit']
                    same_buy.add(row['id_in_group'])

        regress_df = rnd[['direction', 'cumulative_quantity', 'fill_quantity', 'ce_quantity', 'ce_price', 'cash', 'in_market_quantity', 'ind_ce_profit', 'excess_profit', 'no_trans']][-players_per_group:].copy()
        regress_df = regress_df.groupby('direction', as_index=False) \
                .aggregate({'cumulative_quantity': 'mean', 'fill_quantity': 'sum', 'ce_quantity': 'mean', 'ce_price': 'mean', 'cash': 'sum', 'in_market_quantity': 'sum', 'ind_ce_profit': 'sum', 'excess_profit': 'sum', 'no_trans': 'mean'})\
                .reset_index(drop=True)
        regress_df['period'] = r
        regress_df['group'] = g
        regress_df['block'] = regress_df['period'] // ((num_periods - prac_periods) // blocks) + (regress_df['period'] % ((num_periods - prac_periods) // blocks) != 0)
        regress_df['format'] = 'Flow30' if g <= num_flow30 else 'Flow60'
        regress_df.rename(columns={'ind_ce_profit': 'ce_profit'}, inplace=True)
        regress_df['gross_profits_norm'] = regress_df['cash'] / regress_df['ce_profit']
        regress_df['order_num'] = 0
        regress_df['order_price_low'] = 0
        regress_df['order_price_high'] = 0
        regress_df['order_quantity'] = 0
        regress_df['order_rate'] = 0
        regress_df['order_price_low_initial'] = 0
        regress_df['order_price_high_initial'] = 0
        regress_df['order_quantity_initial'] = 0
        regress_df['order_rate_initial'] = 0
        regress_df['max_quantity/rate_orders_buy'] = max_rate_orders_buy
        regress_df['max_quantity/rate_orders_sell'] = max_rate_orders_sell
        for ind, row in regress_df.iterrows():
            if row['direction'] == 'sell':
                regress_df.loc[ind, 'order_num'] = order_num_sell
                regress_df.loc[ind, 'order_price_low'] = np.mean(order_price_low_sell)
                regress_df.loc[ind, 'order_price_high'] = np.mean(order_price_high_sell)
                regress_df.loc[ind, 'order_quantity'] = np.mean(order_quantity_sell)
                regress_df.loc[ind, 'order_rate'] = np.mean(order_rate_sell)
                regress_df.loc[ind, 'order_price_low_initial'] = np.mean(order_price_low_sell_initial)
                regress_df.loc[ind, 'order_price_high_initial'] = np.mean(order_price_high_sell_initial)
                regress_df.loc[ind, 'order_quantity_initial'] = np.mean(order_quantity_sell_initial)
                regress_df.loc[ind, 'order_rate_initial'] = np.mean(order_rate_sell_initial)
            else:
                regress_df.loc[ind, 'order_num'] = order_num_buy
                regress_df.loc[ind, 'order_price_low'] = np.mean(order_price_low_buy)
                regress_df.loc[ind, 'order_price_high'] = np.mean(order_price_high_buy)
                regress_df.loc[ind, 'order_quantity'] = np.mean(order_quantity_buy)
                regress_df.loc[ind, 'order_rate'] = np.mean(order_rate_buy)
                regress_df.loc[ind, 'order_price_low_initial'] = np.mean(order_price_low_buy_initial)
                regress_df.loc[ind, 'order_price_high_initial'] = np.mean(order_price_high_buy_initial)
                regress_df.loc[ind, 'order_quantity_initial'] = np.mean(order_quantity_buy_initial)
                regress_df.loc[ind, 'order_rate_initial'] = np.mean(order_rate_buy_initial)
        regress_df['rational_shares'] = rational_shares
        regress_flow_ind = pd.concat([regress_flow_ind, regress_df], ignore_index=True)

        rnd = rnd[(rnd['timestamp'] >= leave_out_seconds) & (rnd['timestamp'] < round_length - leave_out_seconds_end)]
        del rnd['no_trans']
        rnd["group"] = g
        rnd["period"] = r
        rnd["block"] = r // ((num_periods - prac_periods) // blocks) + (r % ((num_periods - prac_periods) // blocks) != 0)
        group_par.append(rnd)

    df = pd.concat(group_par, ignore_index=True, sort=False)
    df['ind_ce_rate'] = df['ce_rate'].div(players_per_group / 2)

    df['timestamp'] = df.groupby(['id_in_subsession', 'id_in_group'])['id_in_group'].cumcount() + 1

    list_individual_flow.append(df)

individual_per_second_flow = pd.concat(list_individual_flow, ignore_index=True, sort=False)


# aggregate the individual data by direction
data_flow_contract_by_direction = individual_per_second_flow[individual_per_second_flow['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0]\
        .groupby(['timestamp', 'group', 'direction'], as_index=False)\
        .agg({'projected_profit': 'sum'})
data_flow_contract_by_direction['ce_profits'] = 0
data_flow_contract_by_direction['period'] = data_flow_contract_by_direction['timestamp'] // (round_length - leave_out_seconds - leave_out_seconds_end)
for ind, row in data_flow_contract_by_direction.iterrows():
    if row['direction'] == 'buy': 
        data_flow_contract_by_direction.loc[ind, 'ce_profits'] = ce_profit_buy[row['period'] - 1]
    else:
        data_flow_contract_by_direction.loc[ind, 'ce_profits'] = ce_profit_sell[row['period'] - 1]
data_flow_contract_by_direction['realized_surplus'] = data_flow_contract_by_direction['projected_profit'] / data_flow_contract_by_direction['ce_profits']
data_flow_contract_by_direction = data_flow_contract_by_direction.drop('timestamp', axis = 1)

data_mean_flow = individual_per_second_flow.groupby(['timestamp', 'group'], as_index=False)[['benchmark_pace', 'projected_profit', 'in_market_percent']].apply(lambda x: x.abs().mean())

data_mean_flow_by_direction = individual_per_second_flow.groupby(['timestamp', 'group', 'direction'], as_index=False)[['benchmark_pace', 'projected_profit', 'in_market_percent']]\
    .apply(lambda x: x.abs().mean())

data_sum_flow_by_direction = individual_per_second_flow.groupby(['timestamp', 'group', 'direction'], as_index=False)['projected_profit'].apply(lambda x: x.abs().sum())


end_of_period_profits_flow = data_sum_flow_by_direction[data_sum_flow_by_direction['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0].copy()
end_of_period_profits_flow['period'] = data_sum_flow_by_direction['timestamp'] // (round_length - leave_out_seconds - leave_out_seconds_end)

mean_end_of_period_profits_flow = end_of_period_profits_flow.groupby(['group', 'direction'], as_index=False)[['projected_profit']].mean()
direction_means = mean_end_of_period_profits_flow.groupby('direction')['projected_profit'].transform('mean')
mean_end_of_period_profits_flow['mean'] = direction_means

mean_end_of_period_profits_flow_last10 = end_of_period_profits_flow[end_of_period_profits_flow['period'] > (num_periods - prac_periods) // 2]\
    .groupby(['group', 'direction'], as_index=False)[['projected_profit']].mean()
direction_means_last10 = mean_end_of_period_profits_flow_last10.groupby('direction')['projected_profit'].transform('mean')
mean_end_of_period_profits_flow_last10['mean'] = direction_means_last10

individual_per_second_flow['in_market_percent'] = individual_per_second_flow['in_market_percent'].abs() 
individual_agg_data_flow = individual_per_second_flow[individual_per_second_flow['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0]\
    .groupby(['group', 'timestamp'], as_index=False)[['projected_profit', 'in_market_percent', 'realized_surplus']].mean()
individual_agg_data_flow['period'] = individual_agg_data_flow['timestamp'] // (round_length - leave_out_seconds - leave_out_seconds_end)

summary_flow_by_direction = data_flow_contract_by_direction
summary_flow_ind_by_direction = individual_per_second_flow[individual_per_second_flow['timestamp'] % (round_length - leave_out_seconds - leave_out_seconds_end) == 0]\
        [['period', 'group', 'direction', 'projected_profit', 'ind_ce_profit', 'excess_profit']]\
        .reset_index(drop=True)

realized_surplus_buy_flow_all20 = summary_flow_by_direction[summary_flow_by_direction['direction'] == 'buy']['realized_surplus'].mean()
realized_surplus_buy_flow_last10 = summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'buy') & (summary_flow_by_direction['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean()
realized_surplus_buy_flow_first10 = summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'buy') & (summary_flow_by_direction['period'] <= (num_periods - prac_periods) // 2)]['realized_surplus'].mean()
realized_surplus_buy_flow_test = summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'buy') & (summary_flow_by_direction['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].tolist()

realized_surplus_sell_flow_all20 = summary_flow_by_direction[summary_flow_by_direction['direction'] == 'sell']['realized_surplus'].mean()
realized_surplus_sell_flow_last10 = summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'sell') & (summary_flow_by_direction['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean()
realized_surplus_sell_flow_first10 = summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'sell') & (summary_flow_by_direction['period'] <= (num_periods - prac_periods) // 2)]['realized_surplus'].mean()
realized_surplus_sell_flow_test = summary_flow_by_direction[(summary_flow_by_direction['direction'] == 'sell') & (summary_flow_by_direction['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].tolist()

profits_buy_flow30_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                            & (summary_flow_ind_by_direction['group'] <= num_flow30)]['projected_profit'].tolist()
profits_buy_flow30_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                            & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_ind_by_direction['group'] <= num_flow30)]['projected_profit'].tolist()
profits_buy_flow30_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                            & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_ind_by_direction['group'] <= num_flow30)]['projected_profit'].tolist()

profits_buy_flow60_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                            & (summary_flow_ind_by_direction['group'] > num_flow30)]['projected_profit'].tolist()
profits_buy_flow60_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                            & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_ind_by_direction['group'] > num_flow30)]['projected_profit'].tolist()
profits_buy_flow60_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                            & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_ind_by_direction['group'] > num_flow30)]['projected_profit'].tolist()

profits_sell_flow30_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                             & (summary_flow_ind_by_direction['group'] <= num_flow30)]['projected_profit'].tolist()
profits_sell_flow30_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                            & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_by_direction['group'] <= num_flow30)]['projected_profit'].tolist()
profits_sell_flow30_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                            & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_ind_by_direction['group'] > num_flow30)]['projected_profit'].tolist()

profits_sell_flow60_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                             & (summary_flow_ind_by_direction['group'] > num_flow30)]['projected_profit'].tolist()
profits_sell_flow60_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                            & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_by_direction['group'] > num_flow30)]['projected_profit'].tolist()
profits_sell_flow60_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                            & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                            & (summary_flow_by_direction['group'] > num_flow30)]['projected_profit'].tolist()

excess_profits_buy_flow30_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                                   & (summary_flow_ind_by_direction['group'] <= num_flow30)]['excess_profit'].tolist()
excess_profits_buy_flow30_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                                & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2) \
                                                                & (summary_flow_ind_by_direction['group'] <= num_flow30)]['excess_profit'].tolist()
excess_profits_buy_flow30_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                                    & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                                    & (summary_flow_ind_by_direction['group'] <= num_flow30)]['excess_profit'].tolist()

excess_profits_buy_flow60_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                                   & (summary_flow_ind_by_direction['group'] > num_flow30)]['excess_profit'].tolist()
excess_profits_buy_flow60_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                                & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2) \
                                                                & (summary_flow_ind_by_direction['group'] > num_flow30)]['excess_profit'].tolist()
excess_profits_buy_flow60_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'buy') \
                                                                    & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                                    & (summary_flow_ind_by_direction['group'] > num_flow30)]['excess_profit'].tolist()

excess_profits_sell_flow30_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                                    & (summary_flow_ind_by_direction['group'] <= num_flow30)]['excess_profit'].tolist()
excess_profits_sell_flow30_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                                    & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2)\
                                                                    & (summary_flow_ind_by_direction['group'] <= num_flow30)]['excess_profit'].tolist()
excess_profits_sell_flow30_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                                    & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                                    & (summary_flow_ind_by_direction['group'] <= num_flow30)]['excess_profit'].tolist()

excess_profits_sell_flow60_ind_all20 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                                    & (summary_flow_ind_by_direction['group'] > num_flow30)]['excess_profit'].tolist()
excess_profits_sell_flow60_ind_last10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                                & (summary_flow_ind_by_direction['period'] > (num_periods - prac_periods) // 2) \
                                                                & (summary_flow_ind_by_direction['group'] > num_flow30)]['excess_profit'].tolist()
excess_profits_sell_flow60_ind_first10 = summary_flow_ind_by_direction[(summary_flow_ind_by_direction['direction'] == 'sell') \
                                                                    & (summary_flow_ind_by_direction['period'] <= (num_periods - prac_periods) // 2) \
                                                                    & (summary_flow_ind_by_direction['group'] > num_flow30)]['excess_profit'].tolist()                                                   


# cdf of excess profits at ind level
sorted_excess_profit_flow30_ind_all20 = np.sort(excess_profits_buy_flow30_ind_all20 + excess_profits_sell_flow60_ind_all20)
sorted_excess_profit_flow60_ind_all20 = np.sort(excess_profits_buy_flow60_ind_all20 + excess_profits_sell_flow30_ind_all20)
sorted_excess_profit_buy_flow30_ind_all20 = np.sort(excess_profits_buy_flow30_ind_all20)
sorted_excess_profit_buy_flow60_ind_all20 = np.sort(excess_profits_buy_flow60_ind_all20)
sorted_excess_profit_sell_flow30_ind_all20 = np.sort(excess_profits_sell_flow30_ind_all20)
sorted_excess_profit_sell_flow60_ind_all20 = np.sort(excess_profits_sell_flow60_ind_all20)

cumulative_prob_excess_profit_flow30_ind_all20 = np.arange(1, len(sorted_excess_profit_flow30_ind_all20) + 1) / len(sorted_excess_profit_flow30_ind_all20)
cumulative_prob_excess_profit_flow60_ind_all20 = np.arange(1, len(sorted_excess_profit_flow60_ind_all20) + 1) / len(sorted_excess_profit_flow60_ind_all20)
cumulative_prob_excess_profit_buy_flow30_ind_all20 = np.arange(1, len(sorted_excess_profit_buy_flow30_ind_all20) + 1) / len(sorted_excess_profit_buy_flow30_ind_all20)
cumulative_prob_excess_profit_buy_flow60_ind_all20 = np.arange(1, len(sorted_excess_profit_buy_flow60_ind_all20) + 1) / len(sorted_excess_profit_buy_flow60_ind_all20)
cumulative_prob_excess_profit_sell_flow30_ind_all20 = np.arange(1, len(sorted_excess_profit_sell_flow30_ind_all20) + 1) / len(sorted_excess_profit_sell_flow30_ind_all20)
cumulative_prob_excess_profit_sell_flow60_ind_all20 = np.arange(1, len(sorted_excess_profit_sell_flow60_ind_all20) + 1) / len(sorted_excess_profit_sell_flow60_ind_all20)

# cdf of percent of order volume executed 
sorted_executed_percent_buy_flow30_first10 = np.sort(np.concatenate(executed_percent_buy_flow30_first10))
sorted_executed_percent_buy_flow30_last10 = np.sort(np.concatenate(executed_percent_buy_flow30_last10))
sorted_executed_percent_sell_flow30_first10 = np.sort(np.concatenate(executed_percent_sell_flow30_first10))
sorted_executed_percent_sell_flow30_last10 = np.sort(np.concatenate(executed_percent_sell_flow30_last10))
sorted_executed_percent_buy_flow60_first10 = np.sort(np.concatenate(executed_percent_buy_flow60_first10))
sorted_executed_percent_buy_flow60_last10 = np.sort(np.concatenate(executed_percent_buy_flow60_last10))
sorted_executed_percent_sell_flow60_first10 = np.sort(np.concatenate(executed_percent_sell_flow60_first10))
sorted_executed_percent_sell_flow60_last10 = np.sort(np.concatenate(executed_percent_sell_flow60_last10))
cumulative_prob_executed_percent_buy_flow30_first10 = np.arange(1, len(sorted_executed_percent_buy_flow30_first10) + 1) / len(sorted_executed_percent_buy_flow30_first10)
cumulative_prob_executed_percent_buy_flow30_last10 = np.arange(1, len(sorted_executed_percent_buy_flow30_last10) + 1) / len(sorted_executed_percent_buy_flow30_last10)
cumulative_prob_executed_percent_sell_flow30_first10 = np.arange(1, len(sorted_executed_percent_sell_flow30_first10) + 1) / len(sorted_executed_percent_sell_flow30_first10)
cumulative_prob_executed_percent_sell_flow30_last10 = np.arange(1, len(sorted_executed_percent_sell_flow30_last10) + 1) / len(sorted_executed_percent_sell_flow30_last10)
cumulative_prob_executed_percent_buy_flow60_first10 = np.arange(1, len(sorted_executed_percent_buy_flow60_first10) + 1) / len(sorted_executed_percent_buy_flow60_first10)
cumulative_prob_executed_percent_buy_flow60_last10 = np.arange(1, len(sorted_executed_percent_buy_flow60_last10) + 1) / len(sorted_executed_percent_buy_flow60_last10)
cumulative_prob_executed_percent_sell_flow60_first10 = np.arange(1, len(sorted_executed_percent_sell_flow60_first10) + 1) / len(sorted_executed_percent_sell_flow60_first10)
cumulative_prob_executed_percent_sell_flow60_last10 = np.arange(1, len(sorted_executed_percent_sell_flow60_last10) + 1) / len(sorted_executed_percent_sell_flow60_last10)

sorted_executed_percent_buy_flow30_all20 = np.sort(np.concatenate(executed_percent_buy_flow30_first10 + executed_percent_buy_flow30_last10))
sorted_executed_percent_buy_flow60_all20 = np.sort(np.concatenate(executed_percent_buy_flow60_first10 + executed_percent_buy_flow60_last10))
sorted_executed_percent_sell_flow30_all20 = np.sort(np.concatenate(executed_percent_sell_flow30_first10 + executed_percent_sell_flow30_last10))
sorted_executed_percent_sell_flow60_all20 = np.sort(np.concatenate(executed_percent_sell_flow60_first10 + executed_percent_sell_flow60_last10))
cumulative_prob_executed_percent_buy_flow30_all20 = np.arange(1, len(sorted_executed_percent_buy_flow30_all20) + 1) / len(sorted_executed_percent_buy_flow30_all20)
cumulative_prob_executed_percent_buy_flow60_all20 = np.arange(1, len(sorted_executed_percent_buy_flow60_all20) + 1) / len(sorted_executed_percent_buy_flow60_all20)
cumulative_prob_executed_percent_sell_flow30_all20 = np.arange(1, len(sorted_executed_percent_sell_flow30_all20) + 1) / len(sorted_executed_percent_sell_flow30_all20)
cumulative_prob_executed_percent_sell_flow60_all20 = np.arange(1, len(sorted_executed_percent_sell_flow60_all20) + 1) / len(sorted_executed_percent_sell_flow60_all20)

# cdf of order price spread
sorted_order_price_spread_buy_flow30_first10 = np.sort(np.concatenate(order_price_spread_buy_flow30_first10))
sorted_order_price_spread_buy_flow30_last10 = np.sort(np.concatenate(order_price_spread_buy_flow30_last10))
sorted_order_price_spread_sell_flow30_first10 = np.sort(np.concatenate(order_price_spread_sell_flow30_first10))
sorted_order_price_spread_sell_flow30_last10 = np.sort(np.concatenate(order_price_spread_sell_flow30_last10))
sorted_order_price_spread_buy_flow60_first10 = np.sort(np.concatenate(order_price_spread_buy_flow60_first10))
sorted_order_price_spread_buy_flow60_last10 = np.sort(np.concatenate(order_price_spread_buy_flow60_last10))
sorted_order_price_spread_sell_flow60_first10 = np.sort(np.concatenate(order_price_spread_sell_flow60_first10))
sorted_order_price_spread_sell_flow60_last10 = np.sort(np.concatenate(order_price_spread_sell_flow60_last10))
cumulative_prob_order_price_spread_buy_flow30_first10 = np.arange(1, len(sorted_order_price_spread_buy_flow30_first10) + 1) / len(sorted_order_price_spread_buy_flow30_first10)
cumulative_prob_order_price_spread_buy_flow30_last10 = np.arange(1, len(sorted_order_price_spread_buy_flow30_last10) + 1) / len(sorted_order_price_spread_buy_flow30_last10)
cumulative_prob_order_price_spread_sell_flow30_first10 = np.arange(1, len(sorted_order_price_spread_sell_flow30_first10) + 1) / len(sorted_order_price_spread_sell_flow30_first10)
cumulative_prob_order_price_spread_sell_flow30_last10 = np.arange(1, len(sorted_order_price_spread_sell_flow30_last10) + 1) / len(sorted_order_price_spread_sell_flow30_last10)
cumulative_prob_order_price_spread_buy_flow60_first10 = np.arange(1, len(sorted_order_price_spread_buy_flow60_first10) + 1) / len(sorted_order_price_spread_buy_flow60_first10)
cumulative_prob_order_price_spread_buy_flow60_last10 = np.arange(1, len(sorted_order_price_spread_buy_flow60_last10) + 1) / len(sorted_order_price_spread_buy_flow60_last10)
cumulative_prob_order_price_spread_sell_flow60_first10 = np.arange(1, len(sorted_order_price_spread_sell_flow60_first10) + 1) / len(sorted_order_price_spread_sell_flow60_first10)
cumulative_prob_order_price_spread_sell_flow60_last10 = np.arange(1, len(sorted_order_price_spread_sell_flow60_last10) + 1) / len(sorted_order_price_spread_sell_flow60_last10)

sorted_order_price_spread_buy_flow30_all20 = np.sort(np.concatenate(order_price_spread_buy_flow30_first10 + order_price_spread_buy_flow30_last10))
sorted_order_price_spread_buy_flow60_all20 = np.sort(np.concatenate(order_price_spread_buy_flow60_first10 + order_price_spread_buy_flow60_last10))
sorted_order_price_spread_sell_flow30_all20 = np.sort(np.concatenate(order_price_spread_sell_flow30_first10 + order_price_spread_sell_flow30_last10))
sorted_order_price_spread_sell_flow60_all20 = np.sort(np.concatenate(order_price_spread_sell_flow60_first10 + order_price_spread_sell_flow60_last10))
cumulative_prob_order_price_spread_buy_flow30_all20 = np.arange(1, len(sorted_order_price_spread_buy_flow30_all20) + 1) / len(sorted_order_price_spread_buy_flow30_all20)
cumulative_prob_order_price_spread_buy_flow60_all20 = np.arange(1, len(sorted_order_price_spread_buy_flow60_all20) + 1) / len(sorted_order_price_spread_buy_flow60_all20)
cumulative_prob_order_price_spread_sell_flow30_all20 = np.arange(1, len(sorted_order_price_spread_sell_flow30_all20) + 1) / len(sorted_order_price_spread_sell_flow30_all20)
cumulative_prob_order_price_spread_sell_flow60_all20 = np.arange(1, len(sorted_order_price_spread_sell_flow60_all20) + 1) / len(sorted_order_price_spread_sell_flow60_all20)


# cdf of realized surplus
sorted_realized_surplus_buy_flow30_first10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'buy') & (individual_per_second_flow['period'] <= (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] <= num_flow30)]['realized_surplus'].tolist())
sorted_realized_surplus_buy_flow30_last10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'buy') & (individual_per_second_flow['period'] > (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] <= num_flow30)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow30_first10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'sell') & (individual_per_second_flow['period'] <= (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] <= num_flow30)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow30_last10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'sell') & (individual_per_second_flow['period'] > (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] <= num_flow30)]['realized_surplus'].tolist())
sorted_realized_surplus_buy_flow60_first10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'buy') & (individual_per_second_flow['period'] <= (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] > num_flow30)]['realized_surplus'].tolist())
sorted_realized_surplus_buy_flow60_last10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'buy') & (individual_per_second_flow['period'] > (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] > num_flow30)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow60_first10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'sell') & (individual_per_second_flow['period'] <= (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] > num_flow30)]['realized_surplus'].tolist())
sorted_realized_surplus_sell_flow60_last10 = np.sort(individual_per_second_flow[(individual_per_second_flow['direction'] == 'sell') & (individual_per_second_flow['period'] > (num_periods - prac_periods) // 2) & (individual_per_second_flow['group'] > num_flow30)]['realized_surplus'].tolist())
cumulative_prob_realized_surplus_buy_flow30_first10 = np.arange(1, len(sorted_realized_surplus_buy_flow30_first10) + 1) / len(sorted_realized_surplus_buy_flow30_first10)
cumulative_prob_realized_surplus_buy_flow30_last10 = np.arange(1, len(sorted_realized_surplus_buy_flow30_last10) + 1) / len(sorted_realized_surplus_buy_flow30_last10)
cumulative_prob_realized_surplus_sell_flow30_first10 = np.arange(1, len(sorted_realized_surplus_sell_flow30_first10) + 1) / len(sorted_realized_surplus_sell_flow30_first10)
cumulative_prob_realized_surplus_sell_flow30_last10 = np.arange(1, len(sorted_realized_surplus_sell_flow30_last10) + 1) / len(sorted_realized_surplus_sell_flow30_last10)
cumulative_prob_realized_surplus_buy_flow60_first10 = np.arange(1, len(sorted_realized_surplus_buy_flow60_first10) + 1) / len(sorted_realized_surplus_buy_flow60_first10)
cumulative_prob_realized_surplus_buy_flow60_last10 = np.arange(1, len(sorted_realized_surplus_buy_flow60_last10) + 1) / len(sorted_realized_surplus_buy_flow60_last10)
cumulative_prob_realized_surplus_sell_flow60_first10 = np.arange(1, len(sorted_realized_surplus_sell_flow60_first10) + 1) / len(sorted_realized_surplus_sell_flow60_first10)
cumulative_prob_realized_surplus_sell_flow60_last10 = np.arange(1, len(sorted_realized_surplus_sell_flow60_last10) + 1) / len(sorted_realized_surplus_sell_flow60_last10)
