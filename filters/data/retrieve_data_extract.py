# Script to get an extract of daily ETH price
import cryptocompare
import datetime
import pandas as pd
import matplotlib.pyplot as plt

for ticker in ["XRP", "DOGE", "LTC", "BNB", "XLM"]:

    cryptocompare.cryptocompare._set_api_key_parameter(
        "6e89c7206509df377432f33c9359bd07e11cc556b3c0976e4107336a648f4460"
    )

    price_history = cryptocompare.get_historical_price_day(
        ticker, "USD", toTs=datetime.datetime(2021, 5, 5)
    )
    price_history_df = pd.DataFrame.from_dict(price_history)
    price_history_df["date"] = pd.to_datetime(price_history_df["time"], unit="s")

    split = 0.8
    n_rows = len(price_history_df)
    n_train = round(split * n_rows)
    n_test = n_rows - n_train
    train = price_history_df.iloc[0:n_train, :]
    test = price_history_df.iloc[-n_test:, :]

    plt.plot(train.date, train.close)
    plt.plot(test.date, test.close, "r")
    plt.show()

    # save
    train.to_csv("C:/projects/filters/filters/filters/data/" + ticker + "_train.csv")
    test.to_csv("C:/projects/filters/filters/filters/data/" + ticker + "_test.csv")
