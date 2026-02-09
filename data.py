import os 
import numpy as np 
import pandas as pd
from tabulate import tabulate 
import statistics
import statsmodels.api as sm
from stargazer.stargazer import Stargazer
from matplotlib import rc
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

rc('text',usetex=True)

data_directory = '/Users/YilinLi/Documents/UCSC/Flow Data/FLOW_MARKET_DATA/'

required_files= [
    'data_interval.csv', 
    'data_period.csv', 
    'data_second.csv', 
    'data_profits.csv', 
    'data_liquidity.csv', 

]

def abs_flat(list):
    res = [abs(item) for sublist in list for item in sublist 
        if not np.isnan(item) 
        and item != 0
        ]
    return res

def flat(list):
    res = [item for sublist in list for item in sublist 
        if not np.isnan(item) 
        and item != 0
        ]
    return res

def getFirstHalf(full, lastHalf):
    res = full
    for i in lastHalf:
        res.remove(i)
    return res

def calculate_diff(group):
    return group.iloc[0] - group.iloc[-1]

# period all floats to 2 decimals
def remove_paren(s, dec):
    if isinstance(s, list):
        return '[{}, {}]'.format(str(period(s[0], dec)), str(period(s[1], dec)))
    elif isinstance(s, str) and s and s[0] == '(' and s[-1] == ')':
        return '(' + str(period(float(s[1:-1]))) + ')' if dec == 0 else '(' + str(period(float(s[1:-1]), dec)) + ')'
    else:
        return s

from config import *

def main():
    print("params imported")

if __name__ == "__main__":
    main()

print('liquidity')
exec(open(data_directory + 'liquidity_cda.py').read())
exec(open(data_directory + 'liquidity_flow.py').read())

regress_data_liquidity = pd.concat([liquidity_cda_period[['period', 'format', 'block', 'group', 'interval', 'ppi']], liquidity_flow_period[['period', 'format', 'block', 'group', 'interval', 'ppi']]], ignore_index=True)
regress_data_liquidity['group_id'] = regress_data_liquidity['group']
regress_data_liquidity['group'] = regress_data_liquidity['format'] + regress_data_liquidity['group'].astype(str)
regress_data_liquidity = regress_data_liquidity.drop('format', axis=1)


liquidity = [
    ['PPI',
    liquidity_cda_period[liquidity_cda_period['format'] == 'CDA']['liquidity'].mean(), 
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['group'] <= num_groups_flow30)]['liquidity'].mean(),
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['group'] > num_groups_flow30)]['liquidity'].mean(),
    liquidity_cda_period[(liquidity_cda_period['format'] == 'CDA') & (liquidity_cda_period['period'] > (num_periods - prac_periods) // 2)]['liquidity'].mean(), 
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2) & (liquidity_flow_period['group'] <= num_groups_flow30)]['liquidity'].mean(),
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2) & (liquidity_flow_period['group'] > num_groups_flow30)]['liquidity'].mean()],
]

liquidity_all20 = [
    ['PPI',
    liquidity_cda_period[liquidity_cda_period['format'] == 'CDA']['liquidity'].mean(), 
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['group'] <= num_groups_flow30)]['liquidity'].mean(),
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['group'] > num_groups_flow30)]['liquidity'].mean(),
    liquidity_cda_period[(liquidity_cda_period['format'] == 'CDA') & (liquidity_cda_period['period'] <= (num_periods - prac_periods) // 2)]['liquidity'].mean(), 
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['period'] <= (num_periods - prac_periods) // 2) & (liquidity_flow_period['group'] <= num_groups_flow30)]['liquidity'].mean(),
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['period'] <= (num_periods - prac_periods) // 2) & (liquidity_flow_period['group'] > num_groups_flow30)]['liquidity'].mean(),
    liquidity_cda_period[(liquidity_cda_period['format'] == 'CDA') & (liquidity_cda_period['period'] > (num_periods - prac_periods) // 2)]['liquidity'].mean(), 
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2) & (liquidity_flow_period['group'] <= num_groups_flow30)]['liquidity'].mean(),
    liquidity_flow_period[(liquidity_flow_period['format'] == 'FLOW') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2) & (liquidity_flow_period['group'] > num_groups_flow30)]['liquidity'].mean()],
]

for i, sublist in enumerate(liquidity):
    for j, element in enumerate(sublist):
        if isinstance(element, float):
            liquidity[i][j] = period(element, 2)

print('FLOW trader/period')
exec(open(data_directory + 'flow_trader_period.py').read())

print('CDA RESULTS')
exec(open(data_directory + 'cda.py').read())

print('FLOW RESULTS')
exec(open(data_directory + 'flow.py').read())

print('CDA INDIVIDUAL RESULTS')
exec(open(data_directory + 'cda_ind.py').read())

print('FLOW INDIVIDUAL RESULTS')
exec(open(data_directory + 'flow_ind.py').read())


exec(open(data_directory + 'new_plots.py').read())

# temp
check_negative_flow = list(filter(lambda x: x < 0, profits_buy_flow30_ind_all20 + profits_sell_flow30_ind_all20 + profits_buy_flow60_ind_all20 + profits_sell_flow60_ind_all20))
check_negative_cda = list(filter(lambda x: x < 0, profits_buy_cda_ind_all20 + profits_sell_cda_ind_all20))
print(
    'overall negative end-of-period profits count: ', period((len(check_negative_flow) + len(check_negative_cda)) / len(profits_buy_cda_ind_all20 + profits_sell_cda_ind_all20 + profits_buy_flow30_ind_all20 + profits_sell_flow30_ind_all20 + profits_buy_flow60_ind_all20 + profits_sell_flow60_ind_all20), 4), 
    '\nCDA negative end-of-period profits count: ', period(len(check_negative_cda) / len(profits_buy_cda_ind_all20 + profits_sell_cda_ind_all20), 4),
    '\nFLOW negative end-of-period profits count: ', period(len(check_negative_flow) / len(profits_buy_flow30_ind_all20 + profits_sell_flow30_ind_all20 + profits_buy_flow60_ind_all20 + profits_sell_flow60_ind_all20), 4),
)

# create data frames for regressions 
regress_data_direction = pd.concat([regress_cda_ind, regress_flow_ind], ignore_index=True)

regress_data_direction.to_csv(os.path.join(tables_dir, 'regress_data_direction.csv'), index=False)

regress_data_profits = regress_data_direction.groupby(['format', 'period', 'group',]).apply(lambda x: pd.Series({
    'gross_profits': x['cash'].sum(),
    'gross_norm_diff': calculate_diff(x['gross_profits_norm']),
    'total_ce_profit': x['ce_profit'].sum(),
    'overall_order_num': x['order_num'].sum(),
    # 'overall_order_price_low': (x['order_num'] * x['order_price_low']).sum() / x['order_num'].sum(), # theoretical calculations
    # 'overall_order_price_high': (x['order_num'] * x['order_price_high']).sum() / x['order_num'].sum(),
    # 'overall_order_quantity': (x['order_num'] * x['order_quantity']).sum() / x['order_num'].sum(),
    # 'overall_order_rate': (x['order_num'] * x['order_rate']).sum() / x['order_num'].sum(),
    'overall_order_price_low': x['order_price_low'].mean(), # simple avg between buy and sell
    'overall_order_price_high':  x['order_price_high'].mean(),
    'overall_order_quantity': x['order_quantity'].mean(),
    'overall_order_rate': x['order_rate'].mean(),
    'overall_order_price_low_initial': x['order_price_low_initial'].mean(), 
    'overall_order_price_high_initial': x['order_price_high_initial'].mean(),
    'overall_order_quantity_initial': x['order_quantity_initial'].mean(),
    'overall_order_rate_initial': x['order_rate_initial'].mean(),
    'max_quantity/rate_orders_buy': x['max_quantity/rate_orders_buy'].mean(), 
    'max_quantity/rate_orders_sell': x['max_quantity/rate_orders_sell'].mean(), 
    '%no_trans': x['no_trans'].mean() / round_length, 
})).reset_index()
regress_data_profits['max_quantity/rate_orders'] = regress_data_profits['max_quantity/rate_orders_buy'] + regress_data_profits['max_quantity/rate_orders_sell']
# regress_data_profits['%max_quantity/rate_orders'] = regress_data_profits['max_quantity/rate_orders'] / regress_data_profits['overall_order_num'] # theoretical calculations
regress_buy = regress_data_direction[regress_data_direction['direction'] == 'buy'][['format', 'period', 'group', 'cash', 'ce_profit', 'order_num', 'order_price_low', 'order_price_high', 'order_quantity', 'order_rate', 'order_price_low_initial', 'order_price_high_initial', 'order_quantity_initial', 'order_rate_initial']]
regress_sell = regress_data_direction[regress_data_direction['direction'] == 'sell'][['format', 'period', 'group', 'cash', 'ce_profit', 'order_num', 'order_price_low', 'order_price_high', 'order_quantity', 'order_rate', 'order_price_low_initial', 'order_price_high_initial', 'order_quantity_initial', 'order_rate_initial']]
regress_data_profits = pd.merge(regress_data_profits, regress_buy, on=['format', 'period', 'group'], how='left')
regress_data_profits.rename(columns={'cash': 'buyer_profit', 'ce_profit': 'buyer_ce_profit', 'order_num': 'buyer_order_num', 'order_price_low': 'buyer_order_price_low', 'order_price_high': 'buyer_order_price_high', 'order_quantity': 'buyer_order_quantity', 'order_rate': 'buyer_order_rate', 'order_price_low_initial': 'buyer_order_price_low_initial', 'order_price_high_initial': 'buyer_order_price_high_initial', 'order_quantity_initial': 'buyer_order_quantity_initial', 'order_rate_initial': 'buyer_order_rate_initial'}, inplace=True)
regress_data_profits = pd.merge(regress_data_profits, regress_sell, on=['format', 'period', 'group'], how='left')
regress_data_profits.rename(columns={'cash': 'seller_profit', 'ce_profit': 'seller_ce_profit', 'order_num': 'seller_order_num', 'order_price_low': 'seller_order_price_low', 'order_price_high': 'seller_order_price_high', 'order_quantity': 'seller_order_quantity', 'order_rate': 'seller_order_rate', 'order_price_low_initial': 'seller_order_price_low_initial', 'order_price_high_initial': 'seller_order_price_high_initial', 'order_quantity_initial': 'seller_order_quantity_initial', 'order_rate_initial': 'seller_order_rate_initial'}, inplace=True)
regress_data_profits['%max_quantity/rate_orders_buy'] = regress_data_profits['max_quantity/rate_orders_buy'] / regress_data_profits['buyer_order_num']
regress_data_profits['%max_quantity/rate_orders_sell'] = regress_data_profits['max_quantity/rate_orders_sell'] / regress_data_profits['seller_order_num']
regress_data_profits['%max_quantity/rate_orders'] = regress_data_profits[['%max_quantity/rate_orders_buy', '%max_quantity/rate_orders_sell']].mean(axis=1) # simple avg
regress_data_profits['group_id'] = regress_data_profits['group']
regress_data_profits['group'] = regress_data_profits['format'] + regress_data_profits['group'].astype(str)
regress_data_profits['block'] = regress_data_profits['period'] // ((num_periods - prac_periods) // blocks) + (regress_data_profits['period'] % ((num_periods - prac_periods) // blocks) != 0)
regress_data_profits['seller_realized_surplus'] = regress_data_profits['seller_profit'] / regress_data_profits['seller_ce_profit']
regress_data_profits['buyer_realized_surplus'] = regress_data_profits['buyer_profit'] / regress_data_profits['buyer_ce_profit']
regress_data_profits.rename(columns={'cash': 'profits'})

# df for each 5-second time interval
regress_data_interval = pd.concat([regress_cda, regress_flow], ignore_index=True)
interval_dummies = pd.get_dummies(regress_data_interval['block'], prefix='block')
regress_data_interval = regress_data_interval.join(interval_dummies)
regress_data_interval['group_id'] = regress_data_interval['group']
regress_data_interval['group'] = regress_data_interval['format'] + regress_data_interval['group'].astype(str)
categorical_cols = ['format']
regress_data_interval[categorical_cols] = regress_data_interval[categorical_cols].astype('category')
format_mapping = {'CDA': 0, 'Flow30': 1, 'Flow60': 2}
regress_data_interval['format_num'] = regress_data_interval['format'].map(format_mapping)
# regress_data_interval['test'] = (regress_data_interval['period'] - (num_periods - prac_periods)) * regress_data_interval['format_num']
# regress_data_interval['test_new'] = (1 - regress_data_interval['period'] / (num_periods - prac_periods)) * regress_data_interval['format_num']
regress_data_interval['price_deviation_log'] = np.log(np.where(regress_data_interval['price_deviation'] == 0, 0.001, regress_data_interval['price_deviation']))
regress_data_interval['price_change_log'] = np.log(np.abs(np.where(regress_data_interval['price_change'] == 0, 0.001, regress_data_interval['price_change'])))
regress_data_interval['log_wprice'] = np.log(regress_data_interval['weighted_price'])
regress_data_interval['change_log_wprice'] = regress_data_interval.groupby(['format', 'group', 'period'])['log_wprice'].diff()

# add ppi
regress_data_interval = regress_data_interval.merge(regress_data_liquidity, on=['period', 'block', 'group', 'interval', 'group_id'], how='left')

interval_std = regress_data_interval.groupby(['format', 'group', 'period', 'group_id'])['change_log_wprice'].std() # new price volatility measure


# df for each second
regress_data_second = pd.concat([regress_cda_second, regress_flow60econd], ignore_index=True)
second_dummies = pd.get_dummies(regress_data_second['block'], prefix='block')
regress_data_second = regress_data_second.join(second_dummies)
regress_data_second['group_id'] = regress_data_second['group']
regress_data_second['group'] = regress_data_second['format'] + regress_data_second['group'].astype(str)
regress_data_second['format_num'] = regress_data_second['format'].map(format_mapping)
# regress_data_second['test'] = (regress_data_second['period'] - (num_periods - prac_periods)) * regress_data_second['format_num']
# regress_data_second['test_new'] = (1 - regress_data_second['period'] / (num_periods - prac_periods)) * regress_data_second['format_num']
regress_data_second['log_price'] = np.log(regress_data_second['clearing_price'])
regress_data_second['change_log_price'] = regress_data_second.groupby(['format', 'group', 'period'])['log_price'].diff()
regress_data_second['price_change'] = regress_data_second.groupby(['format', 'group', 'period'])['clearing_price'].diff()
regress_data_second['price_deviation'] = regress_data_second['clearing_price'] - regress_data_second['ce_price']

second_std = regress_data_second.groupby(['format', 'group', 'period'])['change_log_price'].std() # new price volatility measure


# df for each period/period

regress_data_period = pd.concat([regress_cda_period, regress_flow_period], ignore_index=True)
period_dummies = pd.get_dummies(regress_data_period['block'], prefix='block')
regress_data_period = regress_data_period.join(period_dummies)
regress_data_period['group_id'] = regress_data_period['group']
regress_data_period['group'] = regress_data_period['format'] + regress_data_period['group'].astype(str)
regress_data_period[categorical_cols] = regress_data_period[categorical_cols].astype('category')
regress_data_period['format_num'] = regress_data_period['format'].map(format_mapping)
# regress_data_period['test'] = (regress_data_period['period'] - (num_periods - prac_periods)) * regress_data_period['format_num']
# regress_data_period['test_new'] = (1 - regress_data_period['period'] / (num_periods - prac_periods)) * regress_data_period['format_num']
regress_data_period['realized_surplus_percent'] = 100 * regress_data_period['realized_surplus']
regress_data_period['filled_contract_percent'] = 100 * regress_data_period['filled_contract']
regress_data_period = regress_data_period.merge(interval_std,  on = ['format', 'period', 'group'], how='left')
regress_data_period = regress_data_period.merge(second_std, on = ['format', 'period', 'group'], how='left')
regress_data_period['Traded/CE Quantity'] = regress_data_period['traded_volume'] / regress_data_period['ce_quantity']
regress_data_period = pd.merge(regress_data_period, regress_data_profits,  on=['format', 'period', 'group', 'block'], how='left')
regress_data_period['order_size'] = 2 * regress_data_period['traded_volume'] / regress_data_period['overall_order_num']
regress_data_period['gross_diff'] = (regress_data_period['buyer_profit'] - regress_data_period['buyer_ce_profit']) - (regress_data_period['seller_profit'] - regress_data_period['seller_ce_profit'])

regress_data_interval.to_csv(os.path.join(tables_dir, 'data_interval.csv'), index=False) 
regress_data_period.to_csv(os.path.join(tables_dir, 'data_period.csv'), index=False) 
regress_data_second.to_csv(os.path.join(tables_dir, 'data_second.csv'), index=False)
regress_data_profits.to_csv(os.path.join(tables_dir, 'data_profits.csv'), index=False)
regress_data_liquidity.to_csv(os.path.join(tables_dir, 'data_liquidity.csv'), index=False)


### use raw prices from each second
print("unweighted price statistics", 
    '\nAverage Price & {} & {} & {} & {} & {} & {} & {} & {} & {}'.format(
    regress_data_second[(regress_data_second['format'] == 'CDA')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['format'] == 'Flow30')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['format'] == 'Flow60')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'CDA')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow30')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow60')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'CDA')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow30')]['clearing_price'].mean().period(2),
    regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow60')]['clearing_price'].mean().period(2),
    ),
    '\n|Price - P_CE| & {} & {} & {} & {} & {} & {} & {} & {} & {}'.format(
    regress_data_second[(regress_data_second['format'] == 'CDA')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['format'] == 'Flow30')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['format'] == 'Flow60')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'CDA')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow30')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow60')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'CDA')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow30')]['price_deviation'].mean().period(2),
    regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow60')]['price_deviation'].mean().period(2),
    ),
    '\n|P_t - P_t - 1| & {} & {} & {} & {} & {} & {} & {} & {} & {}'.format(
    abs(regress_data_second[(regress_data_second['format'] == 'CDA')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['format'] == 'Flow30')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['format'] == 'Flow60')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'CDA')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow30')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['period'] <= (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow60')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'CDA')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow30')]['price_change']).mean().period(2),
    abs(regress_data_second[(regress_data_second['period'] > (num_periods - prac_periods) // 2) & (regress_data_second['format'] == 'Flow60')]['price_change']).mean().period(2),
    ),
    '\nStd(ln P_t - ln P_t - 1) & {} & {} & {} & {} & {} & {} & {} & {} & {}'.format(
    regress_data_period[(regress_data_period['format'] == 'CDA')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['format'] == 'Flow30')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['format'] == 'Flow60')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['period'] <= (num_periods - prac_periods) // 2) & (regress_data_period['format'] == 'CDA')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['period'] <= (num_periods - prac_periods) // 2) & (regress_data_period['format'] == 'Flow30')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['period'] <= (num_periods - prac_periods) // 2) & (regress_data_period['format'] == 'Flow60')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['period'] > (num_periods - prac_periods) // 2) & (regress_data_period['format'] == 'CDA')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['period'] > (num_periods - prac_periods) // 2) & (regress_data_period['format'] == 'Flow30')]['change_log_price'].mean().period(2),
    regress_data_period[(regress_data_period['period'] > (num_periods - prac_periods) // 2) & (regress_data_period['format'] == 'Flow60')]['change_log_price'].mean().period(2),
    )
)


summary_market_all20 = [
    [None, 'CDA', 'Flow30', 'Flow60', 'CDA', 'Flow30', 'FLOw_H', 'CDA', 'Flow30', 'Flow60', ], 
    [None, 'T1 - T20', 'T1 - T20', 'T1 - T20', 'T1 - T10', 'T1 - T10', 'T1 - T10', 'T11 - T20', 'T11 - T20', 'T11 - T20', ], 
    ['Clearing Price', 
        regress_data_interval[(regress_data_interval['format'] == 'CDA')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['weighted_price'].mean(),
    ],
    ['Std. Dev.', 
        '({})'.format(regress_data_interval[(regress_data_interval['format'] == 'CDA')]['weighted_price'].std()), 
        '({})'.format(regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['weighted_price'].std()),
        '({})'.format(regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['weighted_price'].std()),
        '({})'.format(regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['weighted_price'].std()), 
        '({})'.format(regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['weighted_price'].std()), 
        '({})'.format(regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['weighted_price'].std()),
        '({})'.format(regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['weighted_price'].std()), 
        '({})'.format(regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['weighted_price'].std()),
        '({})'.format(regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['weighted_price'].std()),
    ], 
    ['|P_{t} - P_{t - 1}|', 
        regress_data_interval[(regress_data_interval['format'] == 'CDA')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['price_change'].abs().mean(),
        ], 
    [r'Std. Dev. of (\ln P_{t} - \ln P_{t - 1})', 
        regress_data_period[regress_data_period['format'] == 'CDA']['change_log_wprice'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['change_log_wprice'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['change_log_wprice'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(),
        ],
    ['PPI',
        liquidity_cda_period[liquidity_cda_period['format'] == 'CDA']['ppi'].mean(), 
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow30')]['ppi'].mean(),
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow60')]['ppi'].mean(),
        liquidity_cda_period[(liquidity_cda_period['format'] == 'CDA') & (liquidity_cda_period['period'] <= (num_periods - prac_periods) // 2)]['ppi'].mean(), 
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow30') & (liquidity_flow_period['period'] <= (num_periods - prac_periods) // 2)]['ppi'].mean(),
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow60') & (liquidity_flow_period['period'] <= (num_periods - prac_periods) // 2)]['ppi'].mean(),
        liquidity_cda_period[(liquidity_cda_period['format'] == 'CDA') & (liquidity_cda_period['period'] > (num_periods - prac_periods) // 2)]['ppi'].mean(), 
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow30') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2)]['ppi'].mean(),
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow60') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2)]['ppi'].mean(), 
        ],
    ['Traded Volume (shares)', 
        regress_data_period[regress_data_period['format'] == 'CDA']['traded_volume'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow30']['traded_volume'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        ],  
    [r'Traded Vol. /Q\_{CE} (%)', 
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['Traded/CE Quantity'].mean(), 
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['Traded/CE Quantity'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['Traded/CE Quantity'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(), 
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(), 
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(),
        ],
    [r'Filled Contract / Q\_{CE} (%)', 
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['filled_contract'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['filled_contract'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        ],
    ['Num. of Transactions', 
        int(statistics.mean(transaction_number_cda_all20)), None, None,
        int(statistics.mean(transaction_number_cda_first10)), None, None,
        int(statistics.mean(transaction_number_cda_last10)), None, None,
        ],
    ['Mean Transaction Size', 
        int(statistics.mean(volume_volatility_cda_all20)), None, None,
        int(statistics.mean(volume_volatility_cda_first10)), None, None,
        int(statistics.mean(volume_volatility_cda_last10)), None, None,
        ],
    ['Std. Dev.', 
        '({})'.format(statistics.stdev(volume_volatility_cda_all20)), None, None,
        '({})'.format(statistics.stdev(volume_volatility_cda_first10)), None, None,
        '({})'.format(statistics.stdev(volume_volatility_cda_last10)), None, None,
        ],
    ['Clearing Rate', 
        None, statistics.mean(clearing_rate_flow30_all20), statistics.mean(clearing_rate_flow60_all20),
        None, statistics.mean(clearing_rate_flow30_first10), statistics.mean(clearing_rate_flow60_first10),
        None, statistics.mean(clearing_rate_flow30_last10), statistics.mean(clearing_rate_flow60_last10),
       ], 
    ['Std. Dev.', 
        None, '({})'.format(statistics.stdev(clearing_rate_flow30_all20)), '({})'.format(statistics.stdev(clearing_rate_flow60_all20)),
        None, '({})'.format(statistics.stdev(clearing_rate_flow30_first10)), '({})'.format(statistics.stdev(clearing_rate_flow60_first10)),
        None, '({})'.format(statistics.stdev(clearing_rate_flow30_last10)), '({})'.format(statistics.stdev(clearing_rate_flow60_last10)),
        ], 
    ['Time with no FLOW Trans (%)',
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['%no_trans'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%no_trans'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%no_trans'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%no_trans'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%no_trans'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%no_trans'].mean(),
    ],
    ['Info. Effic.: |Price - P_{CE}|', 
        regress_data_interval[(regress_data_interval['format'] == 'CDA')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] <= (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['price_deviation'].mean(),
        ], 
    [r'Alloc. Effic.: \Pi / \Pi_{CE} (x100)', 
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['realized_surplus'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['realized_surplus'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        ],
    [r'\Pi^{buy} / \Pi^{buy}_{CE} - \Pi^{sell} / \Pi^{sell}_{CE}',
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['gross_norm_diff'].mean(), 
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['gross_norm_diff'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['gross_norm_diff'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(), 
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(), 
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(),
        ],
]

summary_market_short = [
    [None, 'CDA', 'Flow30', 'Flow60', 'CDA', 'Flow30', 'Flow60', ], 
    [None, 'T1 - T20', 'T1 - T20', 'T1 - T20', 'T11 - T20', 'T11 - T20', 'T11 - T20', ], 
    ['Clearing Price', 
        regress_data_interval[(regress_data_interval['format'] == 'CDA')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['weighted_price'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['weighted_price'].mean(),
    ],
    ['Std. Dev.', 
        '({})'.format(regress_data_interval[(regress_data_interval['format'] == 'CDA')]['weighted_price'].std()), 
        '({})'.format(regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['weighted_price'].std()),
        '({})'.format(regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['weighted_price'].std()),
        '({})'.format(regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['weighted_price'].std()), 
        '({})'.format(regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['weighted_price'].std()),
        '({})'.format(regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['weighted_price'].std()),
    ], 
    ['|P_{t} - P_{t - 1}|', 
        regress_data_interval[(regress_data_interval['format'] == 'CDA')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['price_change'].abs().mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['price_change'].abs().mean(),
        ], 
    [r'Std. Dev. of (\ln P_{t} - \ln P_{t - 1})', 
        regress_data_period[regress_data_period['format'] == 'CDA']['change_log_wprice'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['change_log_wprice'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['change_log_wprice'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['change_log_wprice'].mean(),
        ],
    ['PPI',
        liquidity_cda_period[liquidity_cda_period['format'] == 'CDA']['ppi'].mean(), 
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow30') & (liquidity_flow_period['group'] <= num_groups_flow30)]['ppi'].mean(),
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow60') & (liquidity_flow_period['group'] > num_groups_flow30)]['ppi'].mean(),
        liquidity_cda_period[(liquidity_cda_period['format'] == 'CDA') & (liquidity_cda_period['period'] > (num_periods - prac_periods) // 2)]['ppi'].mean(), 
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow30') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2)]['ppi'].mean(),
        liquidity_flow_period[(liquidity_flow_period['format'] == 'Flow60') & (liquidity_flow_period['period'] > (num_periods - prac_periods) // 2)]['ppi'].mean(), 
        ],
    ['Traded Volume (shares)', 
        regress_data_period[regress_data_period['format'] == 'CDA']['traded_volume'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow30']['traded_volume'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['traded_volume'].mean(),
        ],  
    [r'Traded Vol. /Q\_{CE} (%)', 
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['Traded/CE Quantity'].mean(), 
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['Traded/CE Quantity'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['Traded/CE Quantity'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(), 
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['Traded/CE Quantity'].mean(),
        ],
    [r'Filled Contract / Q\_{CE} (%)', 
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['filled_contract'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['filled_contract'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['filled_contract'].mean(),
        ],
    ['Num. of Transactions', 
        int(statistics.mean(transaction_number_cda_all20)), None, None,
        int(statistics.mean(transaction_number_cda_last10)), None, None,
        ],
    ['Mean Transaction Size', 
        int(statistics.mean(volume_volatility_cda_all20)), None, None,
        int(statistics.mean(volume_volatility_cda_last10)), None, None,
        ],
    ['Std. Dev.', 
        '({})'.format(statistics.stdev(volume_volatility_cda_all20)), None, None,
        '({})'.format(statistics.stdev(volume_volatility_cda_last10)), None, None,
        ],
    ['Clearing Rate', 
        None, statistics.mean(clearing_rate_flow30_all20), statistics.mean(clearing_rate_flow60_all20),
        None, statistics.mean(clearing_rate_flow30_last10), statistics.mean(clearing_rate_flow30_last10),
       ], 
    ['Std. Dev.', 
        None, '({})'.format(statistics.stdev(clearing_rate_flow30_all20)), '({})'.format(statistics.stdev(clearing_rate_flow60_all20)),
        None, '({})'.format(statistics.stdev(clearing_rate_flow30_last10)), '({})'.format(statistics.stdev(clearing_rate_flow60_last10)),
        ], 
    ['Time with no FLOW Trans (%)',
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['%no_trans'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%no_trans'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%no_trans'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%no_trans'].mean(),
    ],
    ['Info. Effic.: |Price - P_{CE}|', 
        regress_data_interval[(regress_data_interval['format'] == 'CDA')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow30')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['format'] == 'Flow60')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'CDA')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow30')]['price_deviation'].mean(),
        regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['format'] == 'Flow60')]['price_deviation'].mean(),
        ], 
    [r'Alloc. Effic.: \Pi / \Pi_{CE} (x100)', 
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['realized_surplus'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['realized_surplus'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['realized_surplus'].mean(),
        ],
    [r'\Pi^{buy} / \Pi^{buy}_{CE} - \Pi^{sell} / \Pi^{sell}_{CE}',
        100 * regress_data_period[regress_data_period['format'] == 'CDA']['gross_norm_diff'].mean(), 
        100 * regress_data_period[regress_data_period['format'] == 'Flow30']['gross_norm_diff'].mean(),
        100 * regress_data_period[regress_data_period['format'] == 'Flow60']['gross_norm_diff'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(), 
        100 * regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(),
        100 * regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['gross_norm_diff'].mean(),
        ],
]

summary_trader_all20 = [
    [None, 'CDA', 'Flow30', 'Flow60', 'CDA', 'Flow30', 'FLOw_H', 'CDA', 'Flow30', 'Flow60', ], 
    [None, 'T1 - T20', 'T1 - T20', 'T1 - T20', 'T1 - T10', 'T1 - T10', 'T1 - T10', 'T11 - T20', 'T11 - T20', 'T11 - T20', ], 
    ['Number of Orders', 
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_num'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_num'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_num'].mean(),
        ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_num'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_num'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_num'].mean(),
    ], 
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_num'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_num'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(),
    ], 
    ['Quant. per Order', 
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_quantity'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_quantity'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(),
    ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_quantity'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_quantity'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(),
    ], 
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_quantity'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_quantity'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(),
    ], 
    ['Order Price', 
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_price_low'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_price_high'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_high'].mean()],
    ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_price_low'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_price_high'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_high'].mean()],
    ],
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_price_low'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_price_high'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_high'].mean()],
    ],
    ['Order Max Rate', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_rate'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['overall_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_rate'].mean(),
    ],
    ['Seller', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_rate'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['seller_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_rate'].mean(),
    ],
    ['Buyer', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_rate'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['buyer_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_rate'].mean(),
    ],
    ['%Order at Max Quantity', 
        regress_data_period[regress_data_period['format'] == 'CDA']['%max_quantity/rate_orders'].mean(), None, None, 
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(), None, None,
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(), None, None, 
    ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['%max_quantity/rate_orders_sell'].mean(), None, None,
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(), None, None,
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(), None, None,
    ],
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['%max_quantity/rate_orders_buy'].mean(), None, None,
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(), None, None,
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(), None, None,
    ],
    ['%Order at Max Rate', 
        None, 
        regress_data_period[regress_data_period['format'] == 'Flow30']['%max_quantity/rate_orders'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%max_quantity/rate_orders'].mean(),
        None,
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(),
        None,        
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(),
    ],
    ['Seller', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['%max_quantity/rate_orders_sell'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%max_quantity/rate_orders_sell'].mean(),
        None,
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(),
        None,
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(),
    ],
    ['Buyer', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['%max_quantity/rate_orders_buy'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%max_quantity/rate_orders_buy'].mean(),
        None,
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] <= (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(),
    ],
]

summary_trader_short = [
    [None, 'CDA', 'Flow30', 'Flow60', 'CDA', 'Flow30', 'Flow60',], 
    [None, 'T1 - T20', 'T1 - T20', 'T1 - T20', 'T11 - T20', 'T11 - T20', 'T11 - T20', ], 
    ['Number of Orders', 
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_num'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_num'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_num'].mean(),
        ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_num'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_num'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_num'].mean(),
    ], 
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_num'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_num'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_num'].mean(),
    ], 
    ['Quant. per Order', 
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_quantity'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_quantity'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity'].mean(),
    ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_quantity'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_quantity'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity'].mean(),
    ], 
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_quantity'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_quantity'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity'].mean(),
    ], 

    ['Quant. per Order (early)', 
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_quantity_initial'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_quantity_initial'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_quantity_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity_initial'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_quantity_initial'].mean(),
    ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_quantity_initial'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_quantity_initial'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_quantity_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity_initial'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_quantity_initial'].mean(),
    ], 
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_quantity_initial'].mean(), 
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_quantity_initial'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_quantity_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity_initial'].mean(), 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_quantity_initial'].mean(),
    ], 


    ['Order Price', 
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_price_low'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_price_high'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_high'].mean()],
    ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_price_low'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_price_high'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_high'].mean()],
    ],
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_price_low'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_price_high'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_price_low'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_price_high'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_high'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_high'].mean()],
    ],

    ['Order Price (early)',
        regress_data_period[regress_data_period['format'] == 'CDA']['overall_order_price_low_initial'].mean(),
        [regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_price_low_initial'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_price_high_initial'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_price_low_initial'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_price_high_initial'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low_initial'].mean(),
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low_initial'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_high_initial'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_low_initial'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_price_high_initial'].mean()],
    ],
    ['Seller',
        regress_data_period[regress_data_period['format'] == 'CDA']['seller_order_price_low_initial'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_price_low_initial'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_price_high_initial'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_price_low_initial'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_price_high_initial'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low_initial'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low_initial'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_high_initial'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_low_initial'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_price_high_initial'].mean()],
    ],
    ['Buyer',
        regress_data_period[regress_data_period['format'] == 'CDA']['buyer_order_price_low_initial'].mean(), 
        [regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_price_low_initial'].mean(), regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_price_high_initial'].mean()],
        [regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_price_low_initial'].mean(), regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_price_high_initial'].mean()],
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low_initial'].mean(), 
        [regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low_initial'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_high_initial'].mean()],
        [regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_low_initial'].mean(), regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_price_high_initial'].mean()],
    ],
    
    ['Order Max Rate', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_rate'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_rate'].mean(),
    ],
    ['Seller', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_rate'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_rate'].mean(),
    ],
    ['Buyer', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_rate'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_rate'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_rate'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_rate'].mean(),
    ],

    ['Order Max Rate (early)',
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['overall_order_rate_initial'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['overall_order_rate_initial'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_rate_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['overall_order_rate_initial'].mean(),
    ],
    ['Seller',
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['seller_order_rate_initial'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['seller_order_rate_initial'].mean(),
        None,   
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_rate_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_order_rate_initial'].mean(),
    ],
    ['Buyer',
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_order_rate_initial'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_order_rate_initial'].mean(),
        None,
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_rate_initial'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_order_rate_initial'].mean(),
    ],

    ['%Order at Max Quantity', 
        regress_data_period[regress_data_period['format'] == 'CDA']['%max_quantity/rate_orders'].mean(), None, None, 
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(), None, None, 
    ],
    ['Seller', 
        regress_data_period[regress_data_period['format'] == 'CDA']['%max_quantity/rate_orders_sell'].mean(), None, None,
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(), None, None,
    ],
    ['Buyer', 
        regress_data_period[regress_data_period['format'] == 'CDA']['%max_quantity/rate_orders_buy'].mean(), None, None,
        regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(), None, None,
    ],
    ['%Order at Max Rate', 
        None, 
        regress_data_period[regress_data_period['format'] == 'Flow30']['%max_quantity/rate_orders'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%max_quantity/rate_orders'].mean(),
        None,        
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders'].mean(),
    ],
    ['Seller', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['%max_quantity/rate_orders_sell'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%max_quantity/rate_orders_sell'].mean(),
        None,
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_sell'].mean(),
    ],
    ['Buyer', 
        None,
        regress_data_period[regress_data_period['format'] == 'Flow30']['%max_quantity/rate_orders_buy'].mean(),
        regress_data_period[regress_data_period['format'] == 'Flow60']['%max_quantity/rate_orders_buy'].mean(),
        None, 
        regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(),
        regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['%max_quantity/rate_orders_buy'].mean(),
    ],
]


summary_market_short = [[period(num, 2) if isinstance(num, float) else remove_paren(num, 2) for num in sublist] for sublist in summary_market_short]
summary_market_all20 = [[period(num, 2) if isinstance(num, float) else remove_paren(num, 2) for num in sublist] for sublist in summary_market_all20]  
summary_trader_short = [[period(num, 2) if isinstance(num, float) else remove_paren(num, 2) for num in sublist] for sublist in summary_trader_short]
summary_trader_all20 = [[period(num, 2) if isinstance(num, float) else remove_paren(num, 2) for num in sublist] for sublist in summary_trader_all20]
summary_market_short_int = [[period(num) if isinstance(num, float) else remove_paren(num, 0) for num in sublist] for sublist in summary_market_short]
summary_market_all20_int = [[period(num) if isinstance(num, float) else remove_paren(num, 0) for num in sublist] for sublist in summary_market_all20]
summary_trader_short_int = [[period(num) if isinstance(num, float) else remove_paren(num, 0) for num in sublist] for sublist in summary_trader_short]
summary_trader_all20_int = [[period(num) if isinstance(num, float) else remove_paren(num, 0) for num in sublist] for sublist in summary_trader_all20]

# output to a latex table 
summary_trader_short_table = tabulate(summary_trader_short, headers='firstrow', tablefmt='latex')
print('summary_trader_short_table')
with open(os.path.join(tables_dir, 'summary_trader_short_table.tex'), 'w') as f:
    f.write(summary_trader_short_table)

summary_trader_short_table_int = tabulate(summary_trader_short_int, headers='firstrow', tablefmt='latex')
print('summary_trader_short_table_int')
with open(os.path.join(tables_dir, 'summary_trader_short_table_int.tex'), 'w') as f:
    f.write(summary_trader_short_table_int)

summary_market_short_table = tabulate(summary_market_short, headers='firstrow', tablefmt='latex')
print('summary_market_short_table')
with open(os.path.join(tables_dir, 'summary_market_short_table.tex'), 'w') as f:
    f.write(summary_market_short_table)

summary_market_short_table_int = tabulate(summary_market_short_int, headers='firstrow', tablefmt='latex')
print('summary_market_short_table_int')
with open(os.path.join(tables_dir, 'summary_market_short_table_int.tex'), 'w') as f:
    f.write(summary_market_short_table_int)

summary_trader_all20_table = tabulate(summary_trader_all20, headers='firstrow', tablefmt='latex')
print('summary_trader_all20_table')
with open(os.path.join(tables_dir, 'summary_trader_all20_table.tex'), 'w') as f:
    f.write(summary_trader_all20_table)

summary_trader_all20_table_int = tabulate(summary_trader_all20_int, headers='firstrow', tablefmt='latex')
print('summary_trader_all20_table_int')
with open(os.path.join(tables_dir, 'summary_trader_all20_table_int.tex'), 'w') as f:
    f.write(summary_trader_all20_table_int)

summary_market_all20_table = tabulate(summary_market_all20, headers='firstrow', tablefmt='latex')
print('summary_market_all20_table')
with open(os.path.join(tables_dir, 'summary_market_all20_table.tex'), 'w') as f:
    f.write(summary_market_all20_table)

summary_market_all20_table_int = tabulate(summary_market_all20_int, headers='firstrow', tablefmt='latex')
with open(os.path.join(tables_dir, 'summary_market_all20_table_int.tex'), 'w') as f:
    f.write(summary_market_all20_table_int)




block_names = ['block_{}'.format(i) for i in range(1, blocks + 1)]

# regression for price deviation 
price_dev_formula = 'price_deviation ~ format + period + interval'
for dummy in block_names[:-1]:
    price_dev_formula += ' + ' + dummy
weights = 'quantity'
price_dev_model = sm.WLS.from_formula(price_dev_formula, data=regress_data_interval, weights=regress_data_interval[weights])
price_dev_res = price_dev_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_interval['group']), 'maxlags': 1})
price_dev_wald = price_dev_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False) 


price_dev_last10_formula = 'price_deviation ~ format + period + interval'
for dummy in block_names[:-1]:
    price_dev_last10_formula += ' + ' + dummy
weights = 'quantity'
price_dev_last10_model = sm.WLS.from_formula(price_dev_last10_formula, data=regress_data_interval[regress_data_interval['period'] > (num_periods - prac_periods) // 2], weights=regress_data_interval[regress_data_interval['period'] > (num_periods - prac_periods) // 2][weights])
price_dev_last10_res = price_dev_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_interval[regress_data_interval['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
price_dev_last10_wald = price_dev_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# regression for ppi
ppi_formula = 'ppi ~ format + period + interval'
for dummy in block_names[:-1]:
    ppi_formula += ' + ' + dummy
ppi_model = sm.OLS.from_formula(ppi_formula, data=regress_data_interval)
ppi_res = ppi_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_interval['group']), 'maxlags': 1})
ppi_wald = ppi_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


ppi_last10_formula = 'ppi ~ format + period + interval'
for dummy in block_names[:-1]:
    ppi_last10_formula += ' + ' + dummy
ppi_last10_model = sm.OLS.from_formula(ppi_last10_formula, data=regress_data_interval[regress_data_interval['period'] > (num_periods - prac_periods) // 2])
ppi_last10_res = ppi_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_interval[regress_data_interval['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
ppi_last10_wald = ppi_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)



# regression for price volatility 
# use quantity-weighted prices for each 5-second interval
price_vol_formula = 'change_log_wprice ~ format + period'
for dummy in block_names[:-1]:
    price_vol_formula += ' + ' + dummy
price_vol_model = sm.OLS.from_formula(price_vol_formula, data=regress_data_period)
price_vol_res = price_vol_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[regress_data_period['change_log_wprice'].notna()]['group']), 'maxlags': 1})
price_vol_wald = price_vol_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

price_vol_last10_formula = 'change_log_wprice ~ format + period'
for dummy in block_names[:-1]:
    price_vol_last10_formula += ' + ' + dummy
price_vol_last10_model = sm.OLS.from_formula(price_vol_last10_formula, data=regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2])
price_vol_last10_res = price_vol_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[(regress_data_period['period'] > (num_periods - prac_periods) // 2) & (regress_data_period['change_log_wprice'].notna())]['group']), 'maxlags': 1})
price_vol_last10_wald = price_vol_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# use absolute price change
price_vol_abs_formula = 'np.abs(price_change) ~ format + period + interval'
for dummy in block_names[:-1]:
    price_vol_abs_formula += ' + ' + dummy
weights = 'quantity'
price_vol_abs_model = sm.WLS.from_formula(price_vol_abs_formula, data=regress_data_interval, weights=regress_data_interval[weights])
price_vol_abs_res = price_vol_abs_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_interval[regress_data_interval['price_change'].notna()]['group']), 'maxlags': 1})
price_vol_abs_wald = price_vol_abs_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

price_vol_abs_last10_formula = 'np.abs(price_change) ~ format + period + interval'
for dummy in block_names[:-1]:
    price_vol_abs_last10_formula += ' + ' + dummy
weights = 'quantity'
price_vol_abs_last10_model = sm.WLS.from_formula(price_vol_abs_last10_formula, data=regress_data_interval[regress_data_interval['period'] > (num_periods - prac_periods) // 2], weights=regress_data_interval[regress_data_interval['period'] > (num_periods - prac_periods) // 2][weights])
price_vol_abs_last10_res = price_vol_abs_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_interval[(regress_data_interval['period'] > (num_periods - prac_periods) // 2) & (regress_data_interval['price_change'].notna())]['group']), 'maxlags': 1})
price_vol_abs_last10_wald = price_vol_abs_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# regression for realized surplus 
surplus_formula = 'realized_surplus ~ format + period'
for dummy in block_names[:-1]:
    surplus_formula += ' + ' + dummy
surplus_model = sm.OLS.from_formula(surplus_formula, data=regress_data_period)
surplus_res = surplus_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period['group']), 'maxlags': 1})
surplus_wald = surplus_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

surplus_last10_formula = 'realized_surplus ~ format + period'
for dummy in block_names[:-1]:
    surplus_last10_formula += ' + ' + dummy
surplus_last10_model = sm.OLS.from_formula(surplus_last10_formula, data=regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2])
surplus_last10_res = surplus_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
surplus_last10_wald = surplus_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# regression for traded volume
volume_formula = 'traded_volume ~ format + period'
for dummy in block_names[:-1]:
    volume_formula += ' + ' + dummy
volume_model = sm.OLS.from_formula(volume_formula, data=regress_data_period)
volume_res = volume_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period['group']), 'maxlags': 1})
volume_wald = volume_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

volume_last10_formula = 'traded_volume ~ format + period'
for dummy in block_names[:-1]:
    volume_last10_formula += ' + ' + dummy
volume_last10_model = sm.OLS.from_formula(volume_last10_formula, data=regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2])
volume_last10_res = volume_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
volume_last10_wald = volume_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# regression for filled CE quantity
filled_volume_formula = 'filled_contract ~ format + period'
for dummy in block_names[:-1]:
    filled_volume_formula += ' + ' + dummy
filled_volume_model = sm.OLS.from_formula(filled_volume_formula, data=regress_data_period)
filled_volume_res = filled_volume_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period['group']), 'maxlags': 1})
filled_volume_wald = filled_volume_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

filled_volume_last10_formula = 'filled_contract ~ format + period'
for dummy in block_names[:-1]:
    filled_volume_last10_formula += ' + ' + dummy
filled_volume_last10_model = sm.OLS.from_formula(filled_volume_last10_formula, data=regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2])
filled_volume_last10_res = filled_volume_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
filled_volume_last10_wald = filled_volume_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# regression for number of orders
order_number_formula = 'overall_order_num ~ format + period'
for dummy in block_names[:-1]:
    order_number_formula += ' + ' + dummy
order_number_model = sm.OLS.from_formula(order_number_formula, data=regress_data_period)
order_number_res = order_number_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period['group']), 'maxlags': 1})
order_number_wald = order_number_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

order_number_last10_formula = 'overall_order_num ~ format + period'
for dummy in block_names[:-1]:
    order_number_last10_formula += ' + ' + dummy
order_number_last10_model = sm.OLS.from_formula(order_number_last10_formula, data=regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2])
order_number_last10_res = order_number_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
order_number_last10_wald = order_number_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# regression for order quantity
order_size_formula = 'overall_order_quantity ~ format + period'
for dummy in block_names[:-1]:
    order_size_formula += ' + ' + dummy
order_size_model = sm.OLS.from_formula(order_size_formula, data=regress_data_period)
order_size_res = order_size_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period['group']), 'maxlags': 1})
order_size_wald = order_size_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

order_size_last10_formula = 'overall_order_quantity ~ format + period'
for dummy in block_names[:-1]:
    order_size_last10_formula += ' + ' + dummy
order_size_last10_model = sm.OLS.from_formula(order_size_last10_formula, data=regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2])
order_size_last10_res = order_size_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
order_size_last10_wald = order_size_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


# regression for gross diff
gross_diff_formula = 'gross_diff ~ format + period'
for dummy in block_names[:-1]:
    gross_diff_formula += ' + ' + dummy
gross_diff_model = sm.OLS.from_formula(gross_diff_formula, data=regress_data_period)
gross_diff_res = gross_diff_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period['group']), 'maxlags': 1})
gross_diff_wald = gross_diff_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)

gross_diff_last10_formula = 'gross_diff ~ format + period'
for dummy in block_names[:-1]:
    gross_diff_last10_formula += ' + ' + dummy
gross_diff_last10_model = sm.OLS.from_formula(gross_diff_last10_formula, data=regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2])
gross_diff_last10_res = gross_diff_last10_model.fit(cov_type='hac-panel', cov_kwds={'groups': np.asarray(regress_data_period[regress_data_period['period'] > (num_periods - prac_periods) // 2]['group']), 'maxlags': 1})
gross_diff_last10_wald = gross_diff_last10_res.wald_test('format[T.Flow30] - format[T.Flow60] = 0', scalar=False)


model_choice_price_vol = [
    price_vol_abs_res, price_vol_abs_last10_res, 
    price_vol_res, price_vol_last10_res, 
    ppi_res, ppi_last10_res, 
    ]

model_choice_trader_behavior = [
    order_number_res, order_number_last10_res,
    order_size_res, order_size_last10_res,
    ]

model_choice_volume = [
    volume_res, volume_last10_res,
    filled_volume_res, filled_volume_last10_res, 
]

model_choice_efficiency = [
    price_dev_res, price_dev_last10_res, 
    surplus_res, surplus_last10_res, 
    gross_diff_res, gross_diff_last10_res,
    ]

stargazer_price = Stargazer(model_choice_price_vol)
stargazer_trader = Stargazer(model_choice_trader_behavior)
stargazer_volume = Stargazer(model_choice_volume)
stargazer_efficiency = Stargazer(model_choice_efficiency)

print('WALD TEST')
print('trader behavior\n',
      order_number_wald.pvalue, order_number_last10_wald.pvalue,
      order_size_wald.pvalue, order_size_last10_wald.pvalue
      )

print('price\n',
      price_vol_abs_wald.pvalue, price_vol_abs_last10_wald.pvalue, 
      price_vol_wald.pvalue, price_vol_last10_wald.pvalue, 
      ppi_wald.pvalue, ppi_last10_wald.pvalue
      )

print('trade volume\n',
      volume_wald.pvalue, volume_last10_wald.pvalue,
      filled_volume_wald.pvalue, filled_volume_last10_wald.pvalue
      )

print('efficiency\n',
      price_dev_wald.pvalue, price_dev_last10_wald.pvalue,
      surplus_wald.pvalue, surplus_last10_wald.pvalue,
      gross_diff_wald.pvalue, gross_diff_last10_wald.pvalue
      )


print('trader behavior\n', 
    #   stargazer_trader.render_latex()
      )
trader_behavior_regression_table = stargazer_trader.render_latex()
with open(os.path.join(tables_dir, 'trader_behavior_regression_table.tex'), 'w') as f:
    f.write(trader_behavior_regression_table)
print('price\n', 
    #   stargazer_price.render_latex()
      )
price_regression_table = stargazer_price.render_latex()
with open(os.path.join(tables_dir, 'price_regression_table.tex'), 'w') as f:
    f.write(price_regression_table)
print('trade volume\n', 
    #   stargazer_volume.render_latex()
      )
trade_volume_regression_table = stargazer_volume.render_latex()
with open(os.path.join(tables_dir, 'trade_volume_regression_table.tex'), 'w') as f:
    f.write(trade_volume_regression_table)
print('efficiency\n', 
    #   stargazer_efficiency.render_latex()
      )
efficiency_regression_table = stargazer_efficiency.render_latex()
with open(os.path.join(tables_dir, 'efficiency_regression_table.tex'), 'w') as f:
    f.write(efficiency_regression_table)






cumsum_cda_mean = regress_data_interval[regress_data_interval['format'] == 'CDA'].groupby(['period', 'interval'])['%cumsum'].mean()
cumsum_flow30_mean = regress_data_interval[regress_data_interval['format'] == 'Flow30'].groupby(['period', 'interval'])['%cumsum'].mean()
cumsum_flow60_mean = regress_data_interval[regress_data_interval['format'] == 'Flow60'].groupby(['period', 'interval'])['%cumsum'].mean()

cumsum_cda_mean_agg = regress_data_interval[regress_data_interval['format'] == 'CDA'].groupby(['interval'])['%cumsum'].mean()
cumsum_flow30_mean_agg = regress_data_interval[regress_data_interval['format'] == 'Flow30'].groupby(['interval'])['%cumsum'].mean()
cumsum_flow60_mean_agg = regress_data_interval[regress_data_interval['format'] == 'Flow60'].groupby(['interval'])['%cumsum'].mean()

plot_df = pd.DataFrame(
    {
        'interval': [i for i in range(1, len(cumsum_cda_mean.values) + 1)],
        'cda_mean': cumsum_cda_mean.values,
        'flow30_mean': cumsum_flow30_mean.values,
        'flow60_mean': cumsum_flow60_mean.values,
    }
)

agg_plot_df = pd.DataFrame(
    {
        'interval': cumsum_cda_mean_agg.index,
        'cda_mean': cumsum_cda_mean_agg.values,
        'flow30_mean': cumsum_flow30_mean_agg.values,
        'flow60_mean': cumsum_flow60_mean_agg.values,
    }
)



##### profits distribution #####
# plot the cumulative distribution function
#### individuals ####
# all periods
sorted_gross_cda_buy = np.sort(profits_buy_cda_ind_all20)
cumulative_gross_cda_buy = np.arange(len(sorted_gross_cda_buy)) / len(sorted_gross_cda_buy)
sorted_gross_cda_sell = np.sort(profits_sell_cda_ind_all20)
cumulative_gross_cda_sell = np.arange(len(sorted_gross_cda_sell)) / len(sorted_gross_cda_sell)
sorted_gross_flow30_buy = np.sort(profits_buy_flow30_ind_all20)
cumulative_gross_flow30_buy = np.arange(len(sorted_gross_flow30_buy)) / len(sorted_gross_flow30_buy)
sorted_gross_flow30_sell = np.sort(profits_sell_flow30_ind_all20)
cumulative_gross_flow30_sell = np.arange(len(sorted_gross_flow30_sell)) / len(sorted_gross_flow30_sell)
sorted_gross_flow60_buy = np.sort(profits_buy_flow60_ind_all20)
cumulative_gross_flow60_buy = np.arange(len(sorted_gross_flow60_buy)) / len(sorted_gross_flow60_buy)
sorted_gross_flow60_sell = np.sort(profits_sell_flow60_ind_all20)
cumulative_gross_flow60_sell = np.arange(len(sorted_gross_flow60_sell)) / len(sorted_gross_flow60_sell)


plt.plot(sorted_gross_cda_buy, cumulative_gross_cda_buy, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_gross_cda_sell, cumulative_gross_cda_sell, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_gross_flow30_buy, cumulative_gross_flow30_buy, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_gross_flow30_sell, cumulative_gross_flow30_sell, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_gross_flow60_buy, cumulative_gross_flow60_buy, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_gross_flow60_sell, cumulative_gross_flow60_sell, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Gross Profits')
plt.xlabel('Gross Profits')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig(os.path.join(figures_dir, 'gross_profits_cdf.png'))
plt.close()

sorted_excess_cda_buy = np.sort(excess_profits_buy_cda_ind_all20)
cumulative_excess_cda_buy = np.arange(len(sorted_excess_cda_buy)) / len(sorted_excess_cda_buy)
sorted_excess_cda_sell = np.sort(excess_profits_sell_cda_ind_all20)
cumulative_excess_cda_sell = np.arange(len(sorted_excess_cda_sell)) / len(sorted_excess_cda_sell)
sorted_excess_flow30_buy = np.sort(excess_profits_buy_flow30_ind_all20)
cumulative_excess_flow30_buy = np.arange(len(sorted_excess_flow30_buy)) / len(sorted_excess_flow30_buy)
sorted_excess_flow30_sell = np.sort(excess_profits_sell_flow30_ind_all20)
cumulative_excess_flow30_sell = np.arange(len(sorted_excess_flow30_sell)) / len(sorted_excess_flow30_sell)
sorted_excess_flow60_buy = np.sort(excess_profits_buy_flow60_ind_all20)
cumulative_excess_flow60_buy = np.arange(len(sorted_excess_flow60_buy)) / len(sorted_excess_flow60_buy)
sorted_excess_flow60_sell = np.sort(excess_profits_sell_flow60_ind_all20)
cumulative_excess_flow60_sell = np.arange(len(sorted_excess_flow60_sell)) / len(sorted_excess_flow60_sell)

plt.plot(sorted_excess_cda_buy, cumulative_excess_cda_buy, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_excess_cda_sell, cumulative_excess_cda_sell, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_excess_flow30_buy, cumulative_excess_flow30_buy, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_excess_flow30_sell, cumulative_excess_flow30_sell, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_excess_flow60_buy, cumulative_excess_flow60_buy, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_excess_flow60_sell, cumulative_excess_flow60_sell, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Excess Profits')
plt.xlabel('Excess Profits')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig(os.path.join(figures_dir, 'excess_profits_cdf.png'))
plt.close()


# plot the cumulative distribution function
# last half periods
sorted_gross_cda_buy_last10 = np.sort(profits_buy_cda_ind_last10)
cumulative_gross_cda_buy_last10 = np.arange(len(sorted_gross_cda_buy_last10)) / len(sorted_gross_cda_buy_last10)
sorted_gross_cda_sell_last10 = np.sort(profits_sell_cda_ind_last10)
cumulative_gross_cda_sell_last10 = np.arange(len(sorted_gross_cda_sell_last10)) / len(sorted_gross_cda_sell_last10)
sorted_gross_flow30_buy_last10 = np.sort(profits_buy_flow30_ind_last10)
cumulative_gross_flow30_buy_last10 = np.arange(len(sorted_gross_flow30_buy_last10)) / len(sorted_gross_flow30_buy_last10)
sorted_gross_flow30_sell_last10 = np.sort(profits_sell_flow30_ind_last10)
cumulative_gross_flow30_sell_last10 = np.arange(len(sorted_gross_flow30_sell_last10)) / len(sorted_gross_flow30_sell_last10)
sorted_gross_flow60_buy_last10 = np.sort(profits_buy_flow60_ind_last10)
cumulative_gross_flow60_buy_last10 = np.arange(len(sorted_gross_flow60_buy_last10)) / len(sorted_gross_flow60_buy_last10)
sorted_gross_flow60_sell_last10 = np.sort(profits_sell_flow60_ind_last10)
cumulative_gross_flow60_sell_last10 = np.arange(len(sorted_gross_flow60_sell_last10)) / len(sorted_gross_flow60_sell_last10)

plt.plot(sorted_gross_cda_buy_last10, cumulative_gross_cda_buy_last10, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_gross_cda_sell_last10, cumulative_gross_cda_sell_last10, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_gross_flow30_buy_last10, cumulative_gross_flow30_buy_last10, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_gross_flow30_sell_last10, cumulative_gross_flow30_sell_last10, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_gross_flow60_buy_last10, cumulative_gross_flow60_buy_last10, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_gross_flow60_sell_last10, cumulative_gross_flow60_sell_last10, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Gross Profits')
plt.xlabel('Gross Profits')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig(os.path.join(figures_dir, 'gross_profits_last10_cdf.png'))
plt.close()

sorted_excess_cda_buy_last10 = np.sort(excess_profits_buy_cda_ind_last10)
cumulative_excess_cda_buy_last10 = np.arange(len(sorted_excess_cda_buy_last10)) / len(sorted_excess_cda_buy_last10)
sorted_excess_cda_sell_last10 = np.sort(excess_profits_sell_cda_ind_last10)
cumulative_excess_cda_sell_last10 = np.arange(len(sorted_excess_cda_sell_last10)) / len(sorted_excess_cda_sell_last10)
sorted_excess_flow30_buy_last10 = np.sort(excess_profits_buy_flow30_ind_last10)
cumulative_excess_flow30_buy_last10 = np.arange(len(sorted_excess_flow30_buy_last10)) / len(sorted_excess_flow30_buy_last10)
sorted_excess_flow30_sell_last10 = np.sort(excess_profits_sell_flow30_ind_last10)
cumulative_excess_flow30_sell_last10 = np.arange(len(sorted_excess_flow30_sell_last10)) / len(sorted_excess_flow30_sell_last10)
sorted_excess_flow60_buy_last10 = np.sort(excess_profits_buy_flow60_ind_last10)
cumulative_excess_flow60_buy_last10 = np.arange(len(sorted_excess_flow60_buy_last10)) / len(sorted_excess_flow60_buy_last10)
sorted_excess_flow60_sell_last10 = np.sort(excess_profits_sell_flow60_ind_last10)
cumulative_excess_flow60_sell_last10 = np.arange(len(sorted_excess_flow60_sell_last10)) / len(sorted_excess_flow60_sell_last10)

plt.plot(sorted_excess_cda_buy_last10, cumulative_excess_cda_buy_last10, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_excess_cda_sell_last10, cumulative_excess_cda_sell_last10, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_excess_flow30_buy_last10, cumulative_excess_flow30_buy_last10, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_excess_flow30_sell_last10, cumulative_excess_flow30_sell_last10, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_excess_flow60_buy_last10, cumulative_excess_flow60_buy_last10, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_excess_flow60_sell_last10, cumulative_excess_flow60_sell_last10, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Excess Profits')
plt.xlabel('Excess Profits')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend(loc='upper left')
plt.savefig(os.path.join(figures_dir, 'excess_profits_last10_cdf.png'))
plt.close()


#### groups #### 
# all periods
sorted_gross_cda_buy_group = np.sort(regress_data_period[regress_data_period['format'] == 'CDA']['buyer_realized_surplus'].tolist())
cumulative_gross_cda_buy_group = np.arange(len(sorted_gross_cda_buy_group)) / len(sorted_gross_cda_buy_group)
sorted_gross_cda_sell_group = np.sort(regress_data_period[regress_data_period['format'] == 'CDA']['seller_realized_surplus'].tolist())
cumulative_gross_cda_sell_group = np.arange(len(sorted_gross_cda_sell_group)) / len(sorted_gross_cda_sell_group)
sorted_gross_flow30_buy_group = np.sort(regress_data_period[regress_data_period['format'] == 'Flow30']['buyer_realized_surplus'].tolist())
cumulative_gross_flow30_buy_group = np.arange(len(sorted_gross_flow30_buy_group)) / len(sorted_gross_flow30_buy_group)
sorted_gross_flow30_sell_group = np.sort(regress_data_period[regress_data_period['format'] == 'Flow30']['seller_realized_surplus'].tolist())
cumulative_gross_flow30_sell_group = np.arange(len(sorted_gross_flow30_sell_group)) / len(sorted_gross_flow30_sell_group)
sorted_gross_flow60_buy_group = np.sort(regress_data_period[regress_data_period['format'] == 'Flow60']['buyer_realized_surplus'].tolist())
cumulative_gross_flow60_buy_group = np.arange(len(sorted_gross_flow60_buy_group)) / len(sorted_gross_flow60_buy_group)
sorted_gross_flow60_sell_group = np.sort(regress_data_period[regress_data_period['format'] == 'Flow60']['seller_realized_surplus'].tolist())
cumulative_gross_flow60_sell_group = np.arange(len(sorted_gross_flow60_sell_group)) / len(sorted_gross_flow60_sell_group)   


plt.plot(sorted_gross_cda_buy_group, cumulative_gross_cda_buy_group, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA buyer')
plt.plot(sorted_gross_cda_sell_group, cumulative_gross_cda_sell_group, marker=',', linestyle='solid', color=(128/255, 0,  128/255), markersize=5, label='CDA seller')
plt.plot(sorted_gross_flow30_buy_group, cumulative_gross_flow30_buy_group, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 buyer')
plt.plot(sorted_gross_flow30_sell_group, cumulative_gross_flow30_sell_group, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 seller')
plt.plot(sorted_gross_flow60_buy_group, cumulative_gross_flow60_buy_group, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 buyer')
plt.plot(sorted_gross_flow60_sell_group, cumulative_gross_flow60_sell_group, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 seller')
plt.title('CDF of Normalized Gross Profits')
plt.xlabel('Gross / CE Profits')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig(os.path.join(figures_dir, 'group_gross_profits_cdf.png'))
plt.close()


sorted_gross_cda_buy_group_last10 = np.sort(regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_realized_surplus'].tolist())
cumulative_gross_cda_buy_group_last10 = np.arange(len(sorted_gross_cda_buy_group_last10)) / len(sorted_gross_cda_buy_group_last10)
sorted_gross_cda_sell_group_last10 = np.sort(regress_data_period[(regress_data_period['format'] == 'CDA') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_realized_surplus'].tolist())
cumulative_gross_cda_sell_group_last10 = np.arange(len(sorted_gross_cda_sell_group_last10)) / len(sorted_gross_cda_sell_group_last10)
sorted_gross_flow30_buy_group_last10 = np.sort(regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_realized_surplus'].tolist())
cumulative_gross_flow30_buy_group_last10 = np.arange(len(sorted_gross_flow30_buy_group_last10)) / len(sorted_gross_flow30_buy_group_last10)
sorted_gross_flow30_sell_group_last10 = np.sort(regress_data_period[(regress_data_period['format'] == 'Flow30') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_realized_surplus'].tolist())
cumulative_gross_flow30_sell_group_last10 = np.arange(len(sorted_gross_flow30_sell_group_last10)) / len(sorted_gross_flow30_sell_group_last10)
sorted_gross_flow60_buy_group_last10 = np.sort(regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['buyer_realized_surplus'].tolist())
cumulative_gross_flow60_buy_group_last10 = np.arange(len(sorted_gross_flow60_buy_group_last10)) / len(sorted_gross_flow60_buy_group_last10)
sorted_gross_flow60_sell_group_last10 = np.sort(regress_data_period[(regress_data_period['format'] == 'Flow60') & (regress_data_period['period'] > (num_periods - prac_periods) // 2)]['seller_realized_surplus'].tolist())
cumulative_gross_flow60_sell_group_last10 = np.arange(len(sorted_gross_flow60_sell_group_last10)) / len(sorted_gross_flow60_sell_group_last10)

plt.plot(sorted_gross_cda_buy_group_last10, cumulative_gross_cda_buy_group_last10, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_gross_cda_sell_group_last10, cumulative_gross_cda_sell_group_last10, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_gross_flow30_buy_group_last10, cumulative_gross_flow30_buy_group_last10, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_gross_flow30_sell_group_last10, cumulative_gross_flow30_sell_group_last10, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_gross_flow60_buy_group_last10, cumulative_gross_flow60_buy_group_last10, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_gross_flow60_sell_group_last10, cumulative_gross_flow60_sell_group_last10, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Normalized Gross Profits')
plt.xlabel('Gross / CE Profits')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig(os.path.join(figures_dir, 'group_gross_profits_last10_cdf.png'))
plt.close()

# excess profits
# all periods
sorted_excess_cda_group = np.sort(regress_data_period[regress_data_period['format'] == 'CDA']['gross_diff'].tolist())
cumulative_excess_cda_group = np.arange(len(sorted_excess_cda_group)) / len(sorted_excess_cda_group)
sorted_excess_flow30_group = np.sort(regress_data_period[regress_data_period['format'] == 'Flow30']['gross_diff'].tolist())
cumulative_excess_flow30_group = np.arange(len(sorted_excess_flow30_group)) / len(sorted_excess_flow30_group)
sorted_excess_flow60_group = np.sort(regress_data_period[regress_data_period['format'] == 'Flow60']['gross_diff'].tolist())
cumulative_excess_flow60_group = np.arange(len(sorted_excess_flow60_group)) / len(sorted_excess_flow60_group)

plt.plot(sorted_excess_cda_group, cumulative_excess_cda_group, marker=',', linestyle='solid', markersize=5, label='CDA')
plt.plot(sorted_excess_flow30_group, cumulative_excess_flow30_group, marker=',', linestyle='dashed', markersize=5, label='Flow30')
plt.plot(sorted_excess_flow60_group, cumulative_excess_flow60_group, marker=',', linestyle='dotted', markersize=5, label='Flow60')

plt.title('CDF of Normalized Gross Profits')
plt.xlabel('Gross - CE Profits (buy - sell)')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig(os.path.join(figures_dir, 'group_excess_profits_cdf.png'))
plt.close()


########## percent of order volume executed  ##########

plt.plot(sorted_executed_percent_buy_cda_all20, cumulative_prob_executed_percent_buy_cda_all20, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_executed_percent_sell_cda_all20, cumulative_prob_executed_percent_sell_cda_all20, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_executed_percent_buy_flow30_all20, cumulative_prob_executed_percent_buy_flow30_all20, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_executed_percent_sell_flow30_all20, cumulative_prob_executed_percent_sell_flow30_all20, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_executed_percent_buy_flow60_all20, cumulative_prob_executed_percent_buy_flow60_all20, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_executed_percent_sell_flow60_all20, cumulative_prob_executed_percent_sell_flow60_all20, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Executed Order Volume (T1 - T20)')
plt.xlabel('%Executed Order Volume')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'executed_order_volume_percent_cdf.png'))
plt.close()

plt.plot(sorted_executed_percent_buy_cda_first10, cumulative_prob_executed_percent_buy_cda_first10, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_executed_percent_sell_cda_first10, cumulative_prob_executed_percent_sell_cda_first10, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_executed_percent_buy_flow30_first10, cumulative_prob_executed_percent_buy_flow30_first10, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_executed_percent_sell_flow30_first10, cumulative_prob_executed_percent_sell_flow30_first10, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_executed_percent_buy_flow60_first10, cumulative_prob_executed_percent_buy_flow60_first10, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_executed_percent_sell_flow60_first10, cumulative_prob_executed_percent_sell_flow60_first10, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Executed Order Volume (T1 - T10)')
plt.xlabel('%Executed Order Volume')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'executed_order_volume_percent_first10_cdf.png'))
plt.close()

plt.plot(sorted_executed_percent_buy_cda_last10, cumulative_prob_executed_percent_buy_cda_last10, marker=',', linestyle='solid', color=(0, 128/255, 0), markersize=5, label='CDA Buyer')
plt.plot(sorted_executed_percent_sell_cda_last10, cumulative_prob_executed_percent_sell_cda_last10, marker=',', linestyle='solid', color=(128/255, 0, 128/255), markersize=5, label='CDA Seller')
plt.plot(sorted_executed_percent_buy_flow30_last10, cumulative_prob_executed_percent_buy_flow30_last10, marker=',', linestyle='dashed', color=(0, 128/255, 0), markersize=5, label='Flow30 Buyer')
plt.plot(sorted_executed_percent_sell_flow30_last10, cumulative_prob_executed_percent_sell_flow30_last10, marker=',', linestyle='dashed', color=(128/255, 0, 128/255), markersize=5, label='Flow30 Seller')
plt.plot(sorted_executed_percent_buy_flow60_last10, cumulative_prob_executed_percent_buy_flow60_last10, marker=',', linestyle='dotted', color=(0, 128/255, 0), markersize=5, label='Flow60 Buyer')
plt.plot(sorted_executed_percent_sell_flow60_last10, cumulative_prob_executed_percent_sell_flow60_last10, marker=',', linestyle='dotted', color=(128/255, 0, 128/255), markersize=5, label='Flow60 Seller')
plt.title('CDF of Executed Order Volume (T11 - T20)')
plt.xlabel('%Executed Order Volume')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.savefig(os.path.join(figures_dir, 'executed_order_volume_percent_last10_cdf.png'))
plt.close()




# compress to 120s by taking the mean of each timestamp
compress_df_cda = market_per_second_cda[(market_per_second_cda['group_id'] == 1)].copy()
compress_df_cda['timestamp'] = compress_df_cda['timestamp'] % round_length
compress_df_cda['timestamp'] = compress_df_cda['timestamp'].replace(0, round_length)
compress_df_cda['mean_cumulative_quantity_percent'] = compress_df_cda['mean_cumulative_quantity'] / compress_df_cda['ce_quantity']
summary_cda = compress_df_cda.groupby('timestamp').agg({'mean_cumulative_quantity_percent': 'mean'}).reset_index()

compress_df_flow30 = market_per_second_flow[(market_per_second_flow['group_id'] == 1)].copy()
compress_df_flow30['timestamp'] = compress_df_flow30['timestamp'] % round_length
compress_df_flow30['timestamp'] = compress_df_flow30['timestamp'].replace(0, round_length)
compress_df_flow30['mean_cumulative_quantity_percent'] = compress_df_flow30['mean_cumulative_quantity'] / compress_df_flow30['ce_quantity']
summary_flow30 = compress_df_flow30.groupby('timestamp').agg({'mean_cumulative_quantity_percent': 'mean'}).reset_index()

compress_df_flow60 = market_per_second_flow[(market_per_second_flow['group_id'] == 6)].copy()
compress_df_flow60['timestamp'] = compress_df_flow60['timestamp'] % round_length
compress_df_flow60['timestamp'] = compress_df_flow60['timestamp'].replace(0, round_length)
compress_df_flow60['mean_cumulative_quantity_percent'] = compress_df_flow60['mean_cumulative_quantity'] / compress_df_flow60['ce_quantity']
summary_flow60 = compress_df_flow60.groupby('timestamp').agg({'mean_cumulative_quantity_percent': 'mean'}).reset_index()

plt.figure(figsize=(8, 5))
plt.plot(summary_cda['timestamp'], summary_cda['mean_cumulative_quantity_percent'], linestyle='solid', c='green', label='CDA')
plt.plot(summary_flow30['timestamp'], summary_flow30['mean_cumulative_quantity_percent'], c='green', linestyle='dashed', label='Flow30')
plt.plot(summary_flow60['timestamp'], summary_flow60['mean_cumulative_quantity_percent'], c='green', linestyle='dotted', label='Flow60')
plt.hlines(y=1, xmin=1, xmax=round_length, colors='plum', linestyles='--')
plt.xticks(np.arange(1, round_length + 2, 10), np.arange(0, round_length + 1, 10))
plt.xlabel('Time')
plt.ylabel('Percent')
plt.ylim(0, 1.1)
plt.legend(loc='lower right')
# plt.title('Flow60 Cumulative / CE Quantity vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_cumsum_compress_all.png'))
plt.close()

