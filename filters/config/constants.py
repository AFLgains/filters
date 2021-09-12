_BASE_CONFIG_FILE = "main.yml"
_BASE_FILE_NAME = "live.png"

PASSWORD = "PASSWORD"
API_KEY = "API_KEY"

TIME_TO_RUN = "0.01"


#Fake portfolios
EMPTY_PORTFOLIO = {"current_cash": 0}
FULL_PORTFOLIO = {"current_cash": 1}

#EMAIL
_EMAIL_TABLE_FORMAT="\n%-20s | %-20s | %-20s | %-20s | %-20s"
_EMAIL_TABLE_FORMAT_FLOAT="\n%-20s | %-20.1f | %-20.3f | %-20s | %-20s"
_EMAIL_TABLE_HEADERS=("Strat","Price","Vel","If in..","If out..(Cash)")
EMAIL_SUBJECT = "ETH PRICE VELOCITY UPDATE: {current_time_str}"
EMAIL_TICKER = "\nTicker: {ticker}"
EMAIL_TIMESTAMP="\nCurrent GMT time is: {current_time_str}"
EMAIL_DATAAGE="\nData pulled is {hrs} hrs old"
EMAIL_NEXTDATA="\nNew daily price pulled in {hrs} hrs"
EMAIL_CURRENTPRICE="\nCurrent price: {price}"
EMAIL_TABLEHEADER=_EMAIL_TABLE_FORMAT%_EMAIL_TABLE_HEADERS
EMAIL_LINE = "\n"+"-"*20*5
EMAIL_DISCLAIMER = ("\n\nThese emails are for education and entertainment purposes only,"
        "not financial advice. It is not intended as a substitute for professional"
        "financial, legal or tax advice. I am not a financial professional and"
        "am not aware of your personal financial circumstances. I don't recommend "
        "you follow any recommendations provided by the tracker, either in the past "
        "or in the future. In fact, it would be unreasonable to rely on this "
        "information for any purpose whatsoever.")