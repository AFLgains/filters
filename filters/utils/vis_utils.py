import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
from filters.filter_types.filters import RSI
import matplotlib.dates as mdates
import pandas as pd


def visualise_buy_sells(strat_name, actual_price_history, true_vel, pos, vel, res=None):
    if res is not None:
        buy_dates = [b[0] for b in res["buy_dates"]]
        sell_dates = [s[0] for s in res["sell_dates"]]
        buy_prices = [b[1] for b in res["buy_dates"]]
        sell_prices = [s[1] for s in res["sell_dates"]]
        buy_velocity = [b[2] for b in res["buy_dates"]]
        sell_velocity = [b[2] for b in res["buy_dates"]]
        portfolio_value_history_dates = [p[0] for p in res["porfolio_value_history"]]
        portfolio_value_history_value = [p[1] for p in res["porfolio_value_history"]]
    else:
        buy_dates = []
        sell_dates = []
        buy_prices = []
        sell_prices = []
        buy_velocity = []
        sell_velocity = []

    plt.style.use("dark_background")

    fig = plt.figure(figsize=(12 * 2, 6 * 2))
    ax1 = plt.subplot(2, 3, 1)
    ax2 = plt.subplot(2, 3, 4)
    ax3 = plt.subplot(2, 3, 2)
    ax4 = plt.subplot(2, 3, 5)
    ax5 = plt.subplot(1, 3, 3)
    axs = [ax1, ax2, ax3, ax4, ax5]

    fig.suptitle(
        f"{strat_name}: Price history, inferred velocity, portfolio gain, trade win rate"
    )

    axs[0].set_title("Price history")
    axs[0].plot(actual_price_history, "r", label="actual")
    axs[0].plot(pos, label="smoothed")
    axs[0].plot(buy_dates, buy_prices, "go", label="buy")
    axs[0].plot(sell_dates, sell_prices, "ro", label="sell")
    axs[0].set_xlabel("Time (days)")
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(color="grey", linestyle="--", linewidth=0.5)
    axs[0].legend()

    axs[1].set_title("Velocity history")
    axs[1].plot([0, len(actual_price_history)], [0, 0], "w--")
    axs[1].plot(true_vel, "r", label="actual")
    axs[1].plot(vel, label="smoothed")
    axs[1].plot(buy_dates, buy_velocity, "go", label="buy")
    axs[1].plot(sell_dates, sell_velocity, "ro", label="sell")
    axs[1].set_xlabel("Time (days)")
    axs[1].set_ylabel("Velocity (USD/day)")
    axs[1].grid(color="grey", linestyle="--", linewidth=0.5)
    axs[1].legend()

    table_data = [
        ["Win Rate", "{:.1f}%".format(100 * res["win_rate"])],
        ["Annualised return", "{:.1f}%".format(res["annualised_pct_gain"] * 100)],
        ["Total Trades", res["total_trades"]],
        ["Ave gain when win", "{:.1f}%".format(res["ave_gain_when_win"] * 100)],
        ["Ave loss when lose", f"{round(res['ave_loss_when_lose'],2)*100}%"],
    ]

    cellColours = [
        ["Black", "Black"],
        ["Black", "Black"],
        ["Black", "Black"],
        ["Black", "Black"],
        ["Black", "Black"],
    ]

    colors = []
    for c in res["monthly_pnl"]["PnL"]:
        if c > 0:
            colors.append("g")
        else:
            colors.append("r")

    axs[3].set_title("Porfolio value history")
    axs[3].step(portfolio_value_history_dates, portfolio_value_history_value, color="w")
    axs[3].set_xlabel("Time (days)")
    axs[3].set_ylabel("Value (USD)")
    axs[3].grid(color="grey", linestyle="--", linewidth=0.5)
    axs[3].set_ylim(0, 1500)

    axs[2].set_title("Porfolio monthly PnL")
    axs[2].bar(res["monthly_pnl"].index, res["monthly_pnl"]["PnL"], color=colors)
    axs[2].set_xlabel("Month")
    axs[2].set_ylabel("Profit/loss (%)")
    axs[2].grid(color="grey", linestyle="--", linewidth=0.5)

    axs[4].set_title("Key Stats")
    table = axs[4].table(
        cellText=table_data,
        loc="center",
        colWidths=[0.15, 0.25],
        cellColours=cellColours,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(24)
    table.scale(1, 4)

    axs[4].axis("off")

    plt.show()


def vis_live_price(
    price_history_df,
    dates,
    pos_estimate,
    vel_estimate,
    strats,
    n_days=50,
    ticker="",
    output_filename="output/live.png",
):

    price_history_list = list(price_history_df["close"])
    price_history_df["volume"] = price_history_df["volumeto"]
    price_history_df = price_history_df[
        ["date", "open", "high", "low", "close", "volume"]
    ]
    price_history_df["date"] = price_history_df["date"].apply(mdates.date2num)

    price_history_list = price_history_list[-n_days:]
    dates = dates[-n_days:]
    price_history_df = price_history_df.iloc[-n_days:, :]
    for i in range(len(pos_estimate)):
        pos_estimate[i] = pos_estimate[i][-n_days:]
        vel_estimate[i] = vel_estimate[i][-n_days:]

    dates_list = list(dates)
    title = f"Velocity chart for {ticker}, updated: {dates_list[len(dates_list) - 1]}"

    rsi_data = RSI(pd.Series(price_history_list), 14)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(12 * 2, 6 * 2))
    fig.suptitle(title, fontsize="xx-large")

    ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4)  # axisbg='#07000d')
    ax2 = plt.subplot2grid((6, 4), (5, 0), rowspan=1, colspan=4)  # axisbg='#07000d')
    ax3 = plt.subplot2grid((6, 4), (0, 0), rowspan=1, colspan=4)  # axisbg='#07000d')

    axs = [ax1, ax2, ax3]
    candlestick_ohlc(
        axs[0],
        price_history_df.values,
        width=0.6,
        colorup="#53c156",
        colordown="#ff1717",
        alpha=0.9,
    )
    for i in range(len(strats)):
        axs[0].plot(dates, pos_estimate[i], label=strats[i].name)
    axs[0].set_ylabel("Price (USD)")
    axs[0].grid(color="grey", linestyle="--", linewidth=0.5)
    axs[0].legend(loc="upper left")
    axs[0].tick_params(labelbottom=False)

    ax1v = axs[0].twinx()
    ax1v.fill_between(
        price_history_df["date"],
        0,
        price_history_df["volume"],
        facecolor="#00ffe8",
        alpha=0.4,
    )
    ax1v.axes.yaxis.set_ticklabels([])
    ax1v.grid(False)
    ###Edit this to 3, so it's a bit larger
    ax1v.set_ylim(0, 3 * price_history_df["volume"].max())
    ax1v.spines["bottom"].set_color("#5998ff")
    ax1v.spines["top"].set_color("#5998ff")
    ax1v.spines["left"].set_color("#5998ff")
    ax1v.spines["right"].set_color("#5998ff")
    ax1v.tick_params(axis="x", colors="w")
    ax1v.tick_params(axis="y", colors="w")

    dates = list(dates)
    for i in range(len(strats)):
        axs[1].plot(dates, vel_estimate[i], label=strats[i].name)
    axs[1].plot([dates[0], dates[len(dates) - 1]], [0, 0], "w--")
    axs[1].set_xlabel("Time (days)")
    axs[1].set_ylabel("Velocity (USD)")
    axs[1].grid(color="grey", linestyle="--", linewidth=0.5)

    axs[2].plot([dates[0], dates[len(dates) - 1]], [70, 70], "r--")
    axs[2].plot([dates[0], dates[len(dates) - 1]], [30, 30], "g--")
    axs[2].plot(dates, rsi_data, "r")
    axs[2].set_xlabel("Time (days)")
    axs[2].set_ylabel("RSI")
    axs[2].grid(color="grey", linestyle="--", linewidth=0.5)
    axs[2].tick_params(labelbottom=False)

    plt.savefig(output_filename)
