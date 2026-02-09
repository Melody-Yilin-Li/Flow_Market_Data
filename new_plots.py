plt.figure(figsize=(18, 7))
plt.scatter(data=market_per_second_cda[(market_per_second_cda['clearing_price'] > 0) \
                                         & (market_per_second_cda['group_id'] == 1)], \
            x='timestamp', y='mean_clearing_price', c='green', label='CDA', s=5, linestyle='solid')
plt.step(data=market_per_second_flow[(market_per_second_flow['mean_clearing_price'] > 0) \
                                   & (market_per_second_flow['group_id'] == 1)], \
            x='timestamp', y='mean_clearing_price', where='pre', c='blue', label='Flow30', linestyle='solid')
plt.step(data=market_per_second_flow[(market_per_second_flow['mean_clearing_price'] > 0) \
                                   & (market_per_second_flow['group_id'] == 6)], \
            x='timestamp', y='mean_clearing_price', where='pre', c='orange', label='Flow60', linestyle='dotted')
plt.step(data=market_per_second_cda[(market_per_second_cda['group_id'] == 1)], \
         x='timestamp', y='ce_price', where='pre', c='plum', label='CE Price')
vline_xs = [(round_length - leave_out_seconds - leave_out_seconds_end) * i for i in range(1, num_periods - prac_periods)]
for i, x in enumerate(vline_xs, 1):
    color = 'slategray' if i in [4, 8, 12, 16] else 'lightgray'
    plt.vlines(x, ymin=0, ymax=20, colors=color, linestyles='dotted')
round_label_xs = list(range(60, 2401, 120))
for i, x in enumerate(round_label_xs, 1):
    plt.text(x, 2, str(i), color='slategray', ha='center', fontsize=12)
block_label_xs = list(range(240, 2161, 480))  # [240, 720, 1200, 1680, 2160]
for i, x in enumerate(block_label_xs, 1):
    plt.text(x, 18, f'Block {i}', color='slategray', ha='center', fontsize=14, fontweight='bold')

plt.legend(bbox_to_anchor=(1, 1),
    loc='upper left', 
    borderaxespad=0.5)
plt.ylim(0, 20)
plt.xlim(0, round_length * (num_periods - prac_periods) + 1)
plt.xlabel('Time')
plt.xticks(np.arange(1, round_length * (num_periods - prac_periods) + 2, round_length), np.arange(0, round_length * (num_periods - prac_periods) + 1, round_length))
plt.ylabel('Price')
# plt.title('CDA Transaction Prices vs Time')
plt.savefig(os.path.join(figures_dir, 'groups_mean_prices.png'))
plt.close()




plt.figure(figsize=(8, 5))
plt.plot(participant_per_second_cda[(participant_per_second_cda['group_id'] == 1)]['period'], \
        participant_per_second_cda[(participant_per_second_cda['group_id'] == 1)]['mean_quantity'], \
        linestyle='solid', c='green', label='CDA')
plt.plot(participant_per_second_flow[(participant_per_second_flow['group_id'] == 1)]['period'], \
        participant_per_second_flow[(participant_per_second_flow['group_id'] == 1)]['mean_quantity'], \
        linestyle='dashed', c='green', label='Flow30')
plt.plot(participant_per_second_flow[(participant_per_second_flow['group_id'] == 6)]['period'], \
        participant_per_second_flow[(participant_per_second_flow['group_id'] == 6)]['mean_quantity'], \
        linestyle='dotted', c='green', label='Flow60')
plt.step(data=participant_per_second_cda[(participant_per_second_cda['group_id'] == 1)], \
         x='period', y='ce_quantity', where='mid', c='plum', label='CE Quantity')
plt.legend(loc='lower right')
plt.ylim(0, 2000)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_periods - prac_periods + 1), np.arange(1, num_periods - prac_periods + 1))
plt.ylabel('Shares')
plt.title('Traded Volume vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_mean_quantity.png'))
plt.close()





plt.figure(figsize=(8, 5))
plt.plot(participant_per_second_cda[(participant_per_second_cda['group_id'] == 1)]['period'], \
        participant_per_second_cda[(participant_per_second_cda['group_id'] == 1)]['mean_realized_surplus'], \
        linestyle='solid', c='green', label='CDA')
plt.plot(participant_per_second_flow[(participant_per_second_flow['group_id'] == 1)]['period'], \
        participant_per_second_flow[(participant_per_second_flow['group_id'] == 1)]['mean_realized_surplus'], \
        linestyle='dashed', c='green', label='Flow30')
plt.plot(participant_per_second_flow[(participant_per_second_flow['group_id'] == 6)]['period'], \
        participant_per_second_flow[(participant_per_second_flow['group_id'] == 6)]['mean_realized_surplus'], \
        linestyle='dotted', c='green', label='Flow60')
plt.hlines(y=1, xmin=1, xmax=num_periods-prac_periods, colors='plum', linestyles='--')
plt.legend(loc='lower right')
plt.ylim(0, 1.2)
plt.xlabel('Period')
plt.xticks(np.arange(1, num_periods - prac_periods + 1), np.arange(1, num_periods - prac_periods + 1))
plt.ylabel('Percent')
plt.title('Realized Surplus vs Period')
plt.savefig(os.path.join(figures_dir, 'groups_mean_surplus.png'))
plt.close()
