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
import heapq
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


# input session constants 
from config import *

def main():
    print("params imported")
    
if __name__ == "__main__":
     main()

     
liquidity_flow_period = pd.DataFrame()

colors = ['lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'lightcoral', 'lightskyblue', 'pink'] # add more colors with more than 6 groups

 # Define the integral of the function mx + b from x1 to x2
def integral_line(m, b, start, end):
    return (m / 2) * (end ** 2 - start ** 2) + b * (end - start)


# Define the integral of the horizontal line y = h from x1 to x2
def integral_horizontal(h, start, end):
    return h * (end - start)


def integral_of_line_and_horizontal(p1, p2, h, start, end):
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return 0
    # Calculate the slope (m) and y-intercept (b) of the line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    # Calculate the integrals
    integral_line_result = integral_line(m, b, start, end)
    integral_horizontal_result = integral_horizontal(h, start, end)
    # Calculate the area between the two curves
    area_between_curves = abs(integral_line_result - integral_horizontal_result)
    return area_between_curves

def find_line_point(p1, p2, x): # return y coordinate
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2:
        return 0
    # Calculate the slope (m) and y-intercept (b) of the line
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b

def calculate_intersection(p1, p2, p3, p4):
    # Calculate the slopes and intercepts
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    if x1 == x2 == x3 == x4 and y2 < y4:
        return (x1, (y2 + y4) / 2)

    # Calculate slopes
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')

    # Calculate intercepts
    b1 = y1 - m1 * x1 if m1 != float('inf') else None 
    b2 = y3 - m2 * x3 if m2 != float('inf') else None 

    # If the lines are parallel (slopes are equal), there's no intersection
    if m1 == m2:
        return None

    # Calculate intersection point
    if m1 == float('inf'):  # Line 1 is vertical
        x = x1
        y = m2 * x + b2
    elif m2 == float('inf'):  # Line 2 is vertical
        x = x3
        y = m1 * x + b1
    else:
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

    # Check if the intersection is within the bounds of both segments
    if (min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and
        min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4)):
        return (x, y)
    else:
        return None

def calculate_vertical_diff(bids_kinks, asks_kinks):
    l_bids, l_asks = len(bids_kinks), len(asks_kinks)
    inter_x = inter_y = None
    # find intersection if any 
    for i in range(l_bids - 1):
        for j in range(l_asks - 1):
            intersection = calculate_intersection(bids_kinks[i], bids_kinks[i + 1], asks_kinks[j], asks_kinks[j + 1])
            if intersection is not None and inter_x is None and inter_y is None:  # Check if intersection is not None
                inter_x, inter_y = intersection
    i = j = 1
    y_bid = y_ask = 0
    x_new = inter_x + small_rate_change
    while i < l_bids and bids_kinks[i][0] <= x_new:
        i += 1
    if i < l_bids:
        y_bid = find_line_point(bids_kinks[i - 1], bids_kinks[i], x_new)
    else:
        y_bid = min_order_price
    while j < l_asks and asks_kinks[j][0] <= x_new:
        j += 1
    if j < l_asks:
        y_ask = find_line_point(asks_kinks[j - 1], asks_kinks[j], x_new)
    else:
        y_ask = max_order_price
    return y_ask - y_bid

def calculate_integral(bids_kinks, asks_kinks):
    l_bids, l_asks = len(bids_kinks), len(asks_kinks)
    inter_x = inter_y = None
    # find intersection if any 
    for i in range(l_bids - 1):
        for j in range(l_asks - 1):
            intersection = calculate_intersection(bids_kinks[i], bids_kinks[i + 1], asks_kinks[j], asks_kinks[j + 1])
            if intersection is not None and inter_x is None and inter_y is None:  # Check if intersection is not None
                inter_x, inter_y = intersection
    bid_area = ask_area = 0
    i = j = 1

    while i < l_bids and bids_kinks[i][0] <= inter_x:
        i += 1
    integral_range = small_rate_change
    start = inter_x
    while integral_range > 0:
        if i < l_bids:
            end = min(bids_kinks[i][0], start + integral_range)
            bid_area += integral_of_line_and_horizontal(bids_kinks[i - 1], bids_kinks[i], inter_y, start, end)
            integral_range -= (end - start)
            start = end 
            i += 1
        elif i == l_bids:
            end = start + integral_range
            bid_area += integral_horizontal(inter_y - min_order_price, start, end)
            integral_range = 0

    while j < l_asks and asks_kinks[j][0] <= inter_x:
        j += 1
    integral_range = small_rate_change
    start = inter_x
    while integral_range > 0:
        if j < l_asks:
            end = min(asks_kinks[j][0], start + integral_range)
            ask_area += integral_of_line_and_horizontal(asks_kinks[j - 1], asks_kinks[j], inter_y, start, end)
            integral_range -= (end - start)
            start = end 
            j += 1
        elif j == l_asks:
            end = start + integral_range
            ask_area += integral_horizontal(max_order_price - inter_y, start, end)
            integral_range = 0
    return bid_area + ask_area 


def get_slope(slopes):
    ans = 0
    for s in slopes:
        ans += 1 / s
    return 1 / ans if ans else 0

def get_price_slope(orders): # takes raw data input and return (ph, pl, slope) for bids and (pl, ph, slope) for asks
    bids = []
    asks = []
    for order in orders:
        if order['direction'] == 'buy':
            slope = (order['min_price'] - order['max_price']) / order['max_rate']
            bids.append([order['max_price'], order['min_price'], slope])
        else:
            slope = (order['max_price'] - order['min_price']) / order['max_rate']
            asks.append([order['min_price'], order['max_price'], slope])
    bids.sort(key=lambda x: x[0], reverse=True)
    asks.sort(key=lambda x: x[0])
    return bids, asks 

def get_prices(orders, bid): # takes output from get_price_slope() and returns 
    prices = set()
    for p1, p2, s in orders:
        prices.add(p1)
        prices.add(p2)
    prices = list(prices)
    if bid: 
        prices.sort(reverse=True)
    else:
        prices.sort()
    price_ranges = {(prices[i - 1], prices[i]): [] for i in range(len(prices))}
    return price_ranges 


def get_bids_kinks(bids): # return bids kinks (rate, price)
    slopes = get_prices(bids, 1)
    kinks = {20: 0}
    for ph, pl, s in bids:
        for k, v in slopes.items():
            if k[0] >= ph > k[1] or k[0] > pl >= k[1]:
                v.append(s)
    x = 0
    for (a, b), s in slopes.items():
        new_slope = get_slope(s)
        kinks[a] = x
        if new_slope:
            x += (b - a) / new_slope 
        kinks[b] = x
    kink = [(r, p) for p, r in kinks.items()]
    kink.sort(key=lambda x: (x[0], -x[1]))
    if kink[-1][1] != min_order_price:
        kink.append((kink[-1][0], min_order_price))
    return kink


def get_asks_kinks(asks): # return asks kinks (rate, price)
    slopes = get_prices(asks, 0)
    kinks = {0: 0}
    for pl, ph, s in asks:
        for k, v in slopes.items():
            if k[0] <= pl < k[1] or k[0] < ph <= k[1]:
                v.append(s)
    x = 0
    for (a, b), s in slopes.items():
        new_slope = get_slope(s)
        kinks[a] = x
        if new_slope:
            x += (b - a) / new_slope
        kinks[b] = x
    kink = [(r, p) for p, r in kinks.items()]
    kink.sort()
    if kink[-1][1] != max_order_price:
        kink.append((kink[-1][0], max_order_price))
    return kink


def aggregate_slope(pairs): # returns aggregate slope and total quantity
    ans = 0
    rem_q = 0
    for s, q in pairs:
        ans += 1 / s 
        rem_q += q
    if ans: 
        return 1 / ans, rem_q 
    else:
        return 0, rem_q

def get_trans_slopes(orders, clearingPrice): # returns flow demand slope, flow supply slope, remaining quantity
    bids_in_mkt = []
    asks_in_mkt = []

    bids_out_of_mkt = []
    asks_out_of_mkt = []

    for order in orders:
        if order['direction'] == 'buy':
            slope = (order['min_price'] - order['max_price']) / order['max_rate']
            rem_quantity = order['quantity'] - order['fill_quantity']
            if order['min_price'] <= clearingPrice <= order['max_price']: 
                bids_in_mkt.append((slope, rem_quantity))
            else:
                bids_out_of_mkt.append((order['max_price'], slope, rem_quantity))
        else:
            slope = (order['max_price'] - order['min_price']) / order['max_rate']
            rem_quantity = order['quantity'] - order['fill_quantity']
            if order['min_price'] <= clearingPrice <= order['max_price']: 
                asks_in_mkt.append((slope, rem_quantity))
            else:
                asks_out_of_mkt.append((order['min_price'], slope, rem_quantity))
    if bids_in_mkt: 
        flow_demand_slope, demand_q = aggregate_slope(bids_in_mkt)
    elif bids_out_of_mkt: 
        bids_out_of_mkt.sort(reverse=True)
        flow_demand_slope, demand_q = bids_out_of_mkt[0][1], bids_out_of_mkt[0][2]
    else:
        flow_demand_slope = demand_q = np.nan

    if asks_in_mkt: 
        flow_supply_slope, supply_q = aggregate_slope(asks_in_mkt)
    elif asks_out_of_mkt:
        asks_out_of_mkt.sort()
        flow_supply_slope, supply_q = asks_out_of_mkt[0][1], asks_out_of_mkt[0][2]
    else:
        flow_supply_slope = supply_q = np.nan

    if np.isnan(supply_q) or np.isnan(demand_q):
        rem_quantity = np.nan
    else:
        rem_quantity = min(demand_q, supply_q)
    
    if np.isnan(flow_demand_slope) and np.isnan(flow_supply_slope):
        flow_demand_slope = (min_order_price - max_order_price) / small_rate_change
        flow_supply_slope = (max_order_price - min_order_price) / small_rate_change
    elif np.isnan(flow_demand_slope):
        best_ask = asks_out_of_mkt[0][0]
        flow_supply_slope += (best_ask - min_order_price) / small_rate_change
        flow_demand_slope = 0
    elif np.isnan(flow_supply_slope):
        best_bid = bids_out_of_mkt[0][0]
        flow_demand_slope -= (max_order_price - best_bid) / small_rate_change
        flow_supply_slope = 0

    return flow_demand_slope, flow_supply_slope, rem_quantity


for g in range(1, num_groups_flow + 1):
    name = 'group' + str(g)
    group_mkt = []
    for r in range(1, num_rounds - prac_rounds + 1): 
        path = directory + 'flow{}/{}/1_market.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(path)
        rnd = rnd[(leave_out_seconds <= rnd['timestamp']) & (rnd['timestamp'] < round_length - leave_out_seconds_end) & (rnd['before_transaction'] == False)].reset_index(drop=True)
        rnd = rnd.drop(columns=['id_in_subsession', 'before_transaction'])
        rnd['cumulative_quantity'] = rnd['clearing_rate'].cumsum()
        rnd['ce_rate'] =  ce_rate[r - 1]
        rnd['ce_quantity'] = ce_quantity[r - 1]
        rnd['ce_price'] = ce_price[r - 1]
        rnd['timestamp'] = np.arange(leave_out_seconds, round_length - leave_out_seconds_end, 1)
        group_mkt.append(rnd) 

    group_par = []
    for r in range(1, num_rounds - prac_rounds + 1):
        path = directory + 'flow{}/{}/1_participant.json'.format(g, r + prac_rounds)
        rnd = pd.read_json(
            path,
        )
        rnd = rnd[(leave_out_seconds <= rnd['timestamp']) & (rnd['timestamp'] < round_length - leave_out_seconds_end) & (rnd['before_transaction'] == False)].reset_index(drop=True)
        rnd = pd.merge(rnd, group_mkt[r - 1], how='left', on='timestamp') # attache clearing price and clearing rate 
        rnd = rnd[(rnd['before_transaction'] == False)].reset_index(drop=True)
        result = rnd.groupby('timestamp')['active_orders'].agg(lambda x: [order for orders in x for order in orders]).reset_index()
        result['round'] = r
        result['group'] = g
        result['block'] = result['round'] // ((num_rounds - prac_rounds) // blocks) + (result['round'] % ((num_rounds - prac_rounds) // blocks) != 0)
        result['interval'] = (result['timestamp'] // price_interval_size) + 1
        result = result[(result['timestamp'] + 1) % price_interval_size == 0]
        result['demand_slope'] = 0
        result['supply_slope'] = 0
        result['remaining_quantity'] = 0
        result['ppi'] = 0
        result = pd.merge(result, group_mkt[r - 1], how='left', on='timestamp') # attache clearing price and clearing rate 
        for ind, row in result.iterrows():
            demand_slope, supply_slope, remaining_quantity = get_trans_slopes(row['active_orders'], row['clearing_price'])
            result.at[ind, 'demand_slope'] = demand_slope
            result.at[ind, 'supply_slope'] = supply_slope
            result.at[ind, 'remaining_quantity'] = remaining_quantity

            bids, asks = get_price_slope(row['active_orders'])
            bids_kinks = get_bids_kinks(bids)
            asks_kinks = get_asks_kinks(asks)
            
            integral = calculate_integral(bids_kinks, asks_kinks)
            diff = calculate_vertical_diff(bids_kinks, asks_kinks)
            # ppi = integral / liquidity_shares
            result.at[ind, 'ppi'] = diff / small_rate_change
        result['format'] = 'FlowR' if g <= num_groups_flow_low else 'FlowS'
        liquidity_flow_period = pd.concat([liquidity_flow_period, result])

liquidity_flow_period['liquidity'] = liquidity_flow_period['supply_slope'] - liquidity_flow_period['demand_slope']
pd.set_option('display.max_rows', None)