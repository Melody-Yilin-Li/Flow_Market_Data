# input session constants 
import os

# Define directories for tables and figures 
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create directories for tables and figures if they don't exist
tables_dir = os.path.join(current_dir, "tables")
figures_dir = os.path.join(current_dir, "figures")
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

moving_average_size = 5 # number of seconds to calculate moving average
price_interval_size = 5 # number of seconds to calculate price interval
liquidity_shares = 20 # number of shares to calculate PPI in price interval
small_rate_change = liquidity_shares / price_interval_size # minimum rate change to consider for PPI calculation

min_order_price = 0 # minimum price per share 
max_order_price = 20 # maximum price per share

num_groups_cda = 5 # number of groups for the CDA treatment in the experiment
num_groups_flow = 10 # number of groups for the Flow treatment in the experiment
num_groups_flow60 = 5 # number of groups for the high Flow60 treatment in the experiment
num_groups_flow30 = 5 # number of groups for the low Flow30 treatment in the experiment
players_per_group = 8 # number of players in each group
prac_periods = 2 # number of practice periods before the main experiment
num_periods = 22 # total number of periods in the experiment (including practice periods)
round_length = 120 # number of seconds for each period
leave_out_seconds = 0 # number of seconds to leave out at the beginning of each period for data analysis (to exclude initial volatility)
leave_out_seconds_end = 0 # number of seconds to leave out at the end of each period for data analysis (to exclude end-of-period effects)
blocks = 5 # number of contract blocks in the experiment 
price = [14, 6, 9, 6, 14] # CE price for each contract block
ce_price = [p for p in price for _ in range(players_per_group // 2)] # CE price for each period
quantity = [1100, 1200, 1500, 1200, 1100] # CE quantity for each contract block
ce_quantity = [q for q in quantity for _ in range(players_per_group // 2)] # CE quantity for each period
profits = [11700, 10700, 13200, 10700, 11700] # CE profit for each contract block 
ce_profit = [p for p in profits for _ in range(players_per_group // 2)] # CE profit for each period

profits_buy = [2400, 8800, 6900, 8800, 2400] # CE profit for buyers for each contract block
profits_sell = [9300, 1900, 6300, 1900, 9300] # CE profit for sellers for each contract block
ce_profit_buy = [p for p in profits_buy for _ in range(players_per_group // 2)] # CE profit for buyers for each period
ce_profit_sell = [p for p in profits_sell for _ in range(players_per_group // 2)] # CE profit for sellers for each period

max_order_quantity = 200 # maximum quantity per order for CDA orders
max_order_rate = 30 # baseline maximum number of shares per second for FLOW orders 

max_order_rate30 = max_order_rate # maximum number of shares per second for Flow30 treatment 
max_order_rate60 = max_order_rate * 2 # maximum number of shares per second for Flow60 treatment 

early_period = 20 # number of seconds at the beginning of each period to consider for early period analysis (to capture initial volatility and order flow)

# contract specifications for sellers and buyers in each contract block 
# {
#   block: {contract price: contract quantity}
# }
contract_sell = {
    1: {3: 400, 4: 400, 11: 300, 14: 0},
    2: {3: 300, 4: 300, 5: 400, 6: 200},
    3: {3: 400, 4: 400, 5: 400, 8: 300},
    4: {3: 300, 4: 300, 5: 400, 6: 200},
    5: {3: 400, 4: 400, 11: 300, 14: 0}, 
    }
contract_buy = {
    1: {17: 300, 16: 300, 14: 200},
    2: {17: 400, 15: 400, 8: 400, 6: 0},
    3: {16: 300, 15: 400, 14: 400, 10: 400},
    4: {17: 400, 15: 400, 8: 400, 6: 0}, 
    5: {17: 300, 16: 300, 14: 200}, 
    }

ce_rate = [2 * i / round_length for i in ce_quantity] # uniform rate of shares per second for CE orders based on CE quantity and period length

# colors for visualizations
colors = [
    'lightgreen', 'lightblue', 'lavender', 'moccasin', 'lightsteelblue', 'lightcoral', 'lightskyblue', 'pink',
    'peachpuff', 'thistle', 'honeydew', 'powderblue', 'mistyrose', 'palegreen', 'paleturquoise', 'lightyellow',
    'cornsilk', 'lemonchiffon', 'azure', 'aliceblue', 'seashell', 'beige', 'oldlace', 'floralwhite'
]