from helpers import *
from common import *
from config import *

directory = '/Users/YilinLi/Documents/UCSC/Flow Data/'
plt.close()

def main():
    print("params imported")
    
if __name__ == "__main__":
     main()

liquidity_cda_period = pd.DataFrame()

def get_best_bids_asks(orders):
    bids = []
    asks = []
    for order in orders:
        if order['direction'] == 'buy':
            bids.append((order['price'], order['quantity']))
        else:
            asks.append((order['price'], order['quantity']))
    bids.sort(reverse=True)
    asks.sort()
    best_bid_ask_spread = asks[0][0] - bids[0][0] if bids and asks else np.nan
    
    weighted_prices_to_buy = cum_share_to_buy = 0
    for p, q in asks:
        if cum_share_to_buy + q <= liquidity_shares:
            weighted_prices_to_buy += p * q
            cum_share_to_buy += q
        else:
            weighted_prices_to_buy += p * (liquidity_shares - cum_share_to_buy)
            cum_share_to_buy = liquidity_shares
            break
    if cum_share_to_buy < liquidity_shares:
        weighted_prices_to_buy += max_order_price * (liquidity_shares - cum_share_to_buy)
        cum_share_to_buy = liquidity_shares

    price_to_buy_liquidity_shares = weighted_prices_to_buy / min(cum_share_to_buy, liquidity_shares) if weighted_prices_to_buy != 0 else max_order_price

    weighted_prices_to_sell = cum_share_to_sell = 0
    for p, q in bids:
        if cum_share_to_sell + q <= liquidity_shares:
            weighted_prices_to_sell += p * q
            cum_share_to_sell += q
        else:
            weighted_prices_to_sell += p * (liquidity_shares - cum_share_to_sell)
            cum_share_to_sell = liquidity_shares
            break
    if cum_share_to_sell < liquidity_shares:
        weighted_prices_to_sell += min_order_price * (liquidity_shares - cum_share_to_sell)
        cum_share_to_sell = liquidity_shares
    
    price_to_sell_liquidity_shares = weighted_prices_to_sell / min(cum_share_to_sell, liquidity_shares) if weighted_prices_to_sell != 0 else min_order_price
    
    return best_bid_ask_spread, price_to_buy_liquidity_shares, price_to_sell_liquidity_shares

for g in range(1, num_groups_cda + 1):
    group_mkt = []
    for r in range(1, num_periods - prac_periods + 1): 
        path = directory + 'Flow_Market_Data/data/cda{}/{}/1_market.json'.format(g, r + prac_periods)
        rnd = pd.read_json(
            path,
        )
        # rnd['clearing_price'].fillna(method='bfill', inplace=True)
        # rnd.fillna(0, inplace=True)
        rnd = rnd[(leave_out_seconds <= rnd['timestamp']) & (rnd['timestamp'] < round_length - leave_out_seconds_end) & (rnd['before_transaction'] == False)].reset_index(drop=True)
        rnd = rnd.drop(columns=['id_in_subsession', 'before_transaction'])
        rnd['ce_quantity'] = ce_quantity[r - 1]
        rnd['ce_price'] = ce_price[r - 1]
        group_mkt.append(rnd) 


    for r in range(1, num_periods - prac_periods + 1):
        path = directory + 'Flow_Market_Data/data/cda{}/{}/1_participant.json'.format(g, r + prac_periods)
        rnd = pd.read_json(
            path,
        )
        # rnd.fillna(0, inplace=True)
        rnd = rnd[(leave_out_seconds <= rnd['timestamp']) & (rnd['timestamp'] < round_length - leave_out_seconds_end) & (rnd['before_transaction'] == False)].reset_index(drop=True)
        rnd = pd.merge(rnd, group_mkt[r - 1], how='left', on='timestamp')
        result = rnd.groupby('timestamp')['active_orders'].agg(lambda x: [order for orders in x for order in orders]).reset_index()
        result['period'] = r
        result['group'] = g
        result['block'] = result['period'] // ((num_periods - prac_periods) // blocks) + (result['period'] % ((num_periods - prac_periods) // blocks) != 0)
        result['interval'] = (result['timestamp'] // price_interval_size) + 1
        result = result[(result['timestamp'] + 1) % price_interval_size == 0]
        result['best_bid_ask_spread'] = 0
        result['weighted_price (buy_liquidity_shares)'] = 0
        result['weighted_price (sell_liquidity_shares)'] = 0
        
        for ind, row in result.iterrows():
            spread, p_buy, p_sell = get_best_bids_asks(row['active_orders'])
            result.at[ind, 'best_bid_ask_spread'] = spread
            result.at[ind, 'weighted_price (buy_liquidity_shares)'] = p_buy
            result.at[ind, 'weighted_price (sell_liquidity_shares)'] = p_sell

        result['format'] = 'CDA'
        result = pd.merge(result, group_mkt[r - 1], how='left', on='timestamp') # attache clearing price and clearing rate 
        
        liquidity_cda_period = pd.concat([liquidity_cda_period, result])

liquidity_cda_period['liquidity'] = liquidity_cda_period['weighted_price (buy_liquidity_shares)'] - liquidity_cda_period['weighted_price (sell_liquidity_shares)']
liquidity_cda_period['ppi'] = liquidity_cda_period['weighted_price (buy_liquidity_shares)'] - liquidity_cda_period['weighted_price (sell_liquidity_shares)']
