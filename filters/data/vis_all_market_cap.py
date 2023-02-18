import pandas as pd
import matplotlib.pyplot as plt

def log_10_product(x, pos):
    """The two args are the value and tick position.
    Label ticks with the product of the exponentiation"""
    return '%0.3f%%' % (x)

formatter = plt.FuncFormatter(log_10_product)

all_market_cap = pd.read_csv('all_market_cap.csv')
all_market_cap.loc[:,"Date"] = pd.to_datetime(all_market_cap.loc[:,"Date"] )
topx = 5
max_market_cap = all_market_cap.groupby('symbol')["Market Cap"].agg("first")
sorted_market_cap = max_market_cap.sort_values()
sorted_market_cap_top_x = sorted_market_cap[-topx:]
tickers = set(sorted_market_cap_top_x.index)
gold_market_cap = 11e12
for ticker in tickers:
    df_to_plot = all_market_cap.loc[all_market_cap.loc[:,'symbol']==ticker,:]
    plt.plot(df_to_plot.loc[:,"Date"], 100*df_to_plot.loc[:,"Market Cap"]/gold_market_cap,label = ticker)

plt.yscale("log")
plt.legend()
plt.ylim([0.001,100])
plt.ylabel("% Gold market cap")
plt.grid()
ax = plt.gca()
ax.yaxis.set_major_formatter(formatter)
plt.show()