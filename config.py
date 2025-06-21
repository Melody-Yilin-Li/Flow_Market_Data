# input session constants 
# high_flow_max_rate = int(input("Enter flow max rate (0 -> low = 30, 1 -> high = 60): "))

import os

current_dir = os.path.dirname(os.path.abspath(__file__))

tables_dir = os.path.join(current_dir, "tables")
figures_dir = os.path.join(current_dir, "figures")

os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

moving_average_size = 5
price_interval_size = 5
liquidity_shares = 20
small_rate_change = liquidity_shares / price_interval_size

min_order_price = 0
max_order_price = 20

num_groups_cda = 5
num_groups_flow = 10
num_groups_flow_high = 5
num_groups_flow_low = 5
players_per_group = 8
prac_rounds = 2 
num_rounds = 22
round_length = 120
leave_out_seconds = 0
leave_out_seconds_end = 0
blocks = 5
price = [14, 6, 9, 6, 14]
ce_price = [p for p in price for _ in range(players_per_group // 2)]
quantity = [1100, 1200, 1500, 1200, 1100]
ce_quantity = [q for q in quantity for _ in range(players_per_group // 2)]
profits = [11700, 10700, 13200, 10700, 11700]
ce_profit = [p for p in profits for _ in range(players_per_group // 2)]

profits_buy = [2400, 8800, 6900, 8800, 2400]
profits_sell = [9300, 1900, 6300, 1900, 9300]
ce_profit_buy = [p for p in profits_buy for _ in range(players_per_group // 2)]
ce_profit_sell = [p for p in profits_sell for _ in range(players_per_group // 2)]

max_order_quantity = 200
max_order_rate = 30

max_order_rate_low = max_order_rate
max_order_rate_high = max_order_rate * 2


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

ce_rate = [2 * i / round_length for i in ce_quantity]

initial_seconds = 20