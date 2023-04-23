import requests
import pandas as pd
import numpy as np
import talib as ta
import time
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *

load_dotenv()

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')

client = Client(API_KEY, API_SECRET)


def get_data(pair, interval, limit):
    """
    تابعی برای دریافت داده های مربوط به یک رمزارز به صورت تاریخچه
    """
    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval={interval}&limit={limit}"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data)
    df.columns = ['open_time', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore']
    df = df.drop(columns=['open_time', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['o'] = df['o'].astype('float')
    df['h'] = df['h'].astype('float')
    df['l'] = df['l'].astype('float')
    df['c'] = df['c'].astype('float')
    df['v'] = df['v'].astype('float')
    df = df.set_index(pd.to_datetime(df.index, unit='ms'))
    return df


def calculate_indicators(df):
    """
    تابعی برای محاسبه نقاط اصلی ابزار Ichimoku Cloud
    """
    conversion = (ta.EMA(df['hdef main():
# تعریف تنظیمات اولیه
pair = "BTCUSDT"
interval = "1h"
limit = 500
buy_threshold = 0.5 # مقدار آستانه برای خرید
sell_threshold = -0.5 # مقدار آستانه برای فروش
# دریافت داده ها و محاسبه نقاط ابزار Ichimoku Cloud
df = get_data(pair, interval, limit)
calculate_indicators(df)

# محاسبه سیگنال ها و تولید نوتیفیکیشن ها
df['buy_signal'] = np.where(df['tenkan_sen'] > df['kijun_sen'], 1, 0)
df['sell_signal'] = np.where(df['tenkan_sen'] < df['kijun_sen'], 1, 0)
df['cumulative_buy_signal'] = df['buy_signal'].cumsum()
df['cumulative_sell_signal'] = df['sell_signal'].cumsum()
df['buy_notification'] = np.where(df['cumulative_buy_signal'] > df['cumulative_sell_signal'], 
                                  np.where(df['cumulative_buy_signal'] - df['cumulative_sell_signal'] >= buy_threshold, 
                                           f"BUY {pair} NOW! ({df['cumulative_buy_signal'][len(df)-1]} > {df['cumulative_sell_signal'][len(df)-1]})",
                                           ""), "")
df['sell_notification'] = np.where(df['cumulative_sell_signal'] > df['cumulative_buy_signal'], 
                                   np.where(df['cumulative_sell_signal'] - df['cumulative_buy_signal'] >= abs(sell_threshold), 
                                            f"SELL {pair} NOW! ({df['cumulative_sell_signal'][len(df)-1]} > {df['cumulative_buy_signal'][len(df)-1]})",
                                            ""), "")
notifications = [notification for notification in df['buy_notification'].append(df['sell_notification']) if notification != ""]

# چاپ نتایج
print("Notifications:")
for notification in notifications:
    print(notification)
print("Profit/Loss:")
print(f"{df['profit_loss'][len(df)-1]:.2f}%")
print("Risk:")
print(f"{df['risk'][len(df)-1]:.2f}%")
print("Buy Value:")
print(f"{df['buy_value'][len(df)-1]:.2f}%")
if name == "main":
main()
# محاسبه ابزار Ichimoku Cloud
high_prices = np.array(df['h'])
low_prices = np.array(df['l'])
close_prices = np.array(df['c'])
nine_period_high = ta.MAX(high_prices, timeperiod=9)
nine_period_low = ta.MIN(low_prices, timeperiod=9)
df['tenkan_sen'] = (nine_period_high + nine_period_low) /2
df['kijun_sen'] = (ta.MAX(high_prices, timeperiod=26) + ta.MIN(low_prices, timeperiod=26))/2
df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
df['senkou_span_b'] = ((ta.MAX(high_prices, timeperiod=52) + ta.MIN(low_prices, timeperiod=52))/2).shift(26)
df['chikou_span'] = close_prices.shift(-26)

# محاسبه مقادیر RSI
df['rsi'] = ta.RSI(close_prices, timeperiod=14)

# محاسبه مقدار موجودی و سود
balance = 1000
holdings = 0
buy_price = 0
sell_price = 0
profit_loss = 0
total_trades = 0
win_trades = 0
lose_trades = 0

# شرایط خرید و فروش بر اساس ابزار Ichimoku Cloud و RSI
for i in range(len(df)):
    if df['tenkan_sen'][i] > df['kijun_sen'][i] and df['c'][i] > df['senkou_span_a'][i] and df['c'][i] > df['senkou_span_b'][i] and df['rsi'][i] < 30:
        if holdings == 0:
            buy_price = df['c'][i]
            holdings = balance / buy_price
            balance = 0
            print(f"Buy at {buy_price:.2f}")
            
    elif df['tenkan_sen'][i] < df['kijun_sen'][i] and df['c'][i] < df['senkou_span_a'][i] and df['c'][i] < df['senkou_span_b'][i] and df['rsi'][i] > 70:
        if holdings > 0:
            sell_price = df['c'][i]
            balance = holdings * sell_price
            holdings = 0
            print(f"Sell at {sell_price:.2f}")
            
    # محاسبه مقدار سود و زیان و تعداد معاملات
    if holdings > 0:
        current_value = holdings * df['c'][i]
        profit_loss = current_value - balance
    else:
        profit_loss = balance - (holdings * df['c'][i])
        
    if profit_loss > 0:
        win_trades += 1
    elif profit_loss < 0:
        lose_trades += 1
    total_trades = win_trades + lose_trades
        
# نمایش مقادیر مورد نیاز
profit_percentage = (profit_loss / balance) * 100
risk_percentage = ((balance - abs(profit_loss)) / balance
def calculate_signals(df):
    """
    تابعی برای محاسبه سیگنال‌های خرید و فروش با استفاده از ابزار Ichimoku Cloud
    """
    # محاسبه خطوط ابتدایی Ichimoku Cloud
    df['conversion'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[0]
    df['base'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[1]
    df['lead_a'] = (df['conversion'] + df['base']) / 2
    df['lead_b'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[2]

    # محاسبه مقدار ابری (Kumo)
    df['cloud_top'] = (df['lead_a'] + df['lead_b']) / 2
    df['cloud_bottom'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[3]

    # سیگنال خرید
    buy_signal = (df['c'] > df['cloud_top']) & (df['c'].shift(1) <= df['cloud_top'].shift(1)) & (df['c'] > df['lead_a'])
    df.loc[buy_signal, 'buy_signal'] = 1
    df['buy_signal'].fillna(0, inplace=True)

    # سیگنال فروش
    sell_signal = (df['c'] < df['def calculate_signals(df):
"""
تابعی برای محاسبه سیگنال‌های خرید و فروش با استفاده از ابزار Ichimoku Cloud
"""
# محاسبه خطوط ابتدایی Ichimoku Cloud
df['conversion'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[0]
df['base'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[1]
df['lead_a'] = (df['conversion'] + df['base']) / 2
df['lead_b'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[2]
# محاسبه مقدار ابری (Kumo)
df['cloud_top'] = (df['lead_a'] + df['lead_b']) / 2
df['cloud_bottom'] = ta.ICHIMOKU(df['h'], df['l'], timeperiod1=9, timeperiod2=26, timeperiod3=52)[3]

# سیگنال خرید
buy_signal = (df['c'] > df['cloud_top']) & (df['c'].shift(1) <= df['cloud_top'].shift(1)) & (df['c'] > df['lead_a'])
df.loc[buy_signal, 'buy_signal'] = 1
df['buy_signal'].fillna(0, inplace=True)

# سیگنال فروش
sell_signal = (df['c'] < df['cloud_bottom']) & (df['c'].shift(1) >= df['cloud_bottom'].shift(1)) & (df['c'] < df['lead_a'])
df.loc[sell_signal, 'sell_signal'] = 1
df['sell_signal'].fillna(0, inplace=True)

return df
def run_strategy(pair, interval, limit):
"""
تابعی برای اجرای روش Ichimoku Cloud بر روی داده های تاریخچه یک رمزارز
"""
# دریافت داده ها از API
df = get_data(pair, interval, limit)
# محاسبه نقاط اصلی ابزار Ichimoku Cloud
df = calculate_indicators(df)

# محاسبه سیگنال‌های خرید و فروش
df = calculate_signals(df)

# ارسال نوتیفیکیشن برای خرید و فروش
send_notification(df)

# نمایش نمودار سود و زیان
plot_profit_loss(df)
def send_notification(df):
"""
تابعی برای ارسال نوتیفیکیشن برای سیگنال‌های خرید و فروش
"""
# ارسال نوتیفیکیشن برای سیگنال خرید
if df['buy_signal'].iloc[-1] ==def backtest(df, initial_capital=10000):
"""
تابعی برای بررسی عملکرد سیگنال‌های خرید و فروش در طول زمان
"""
# محاسبه سود و ضرر
df['profit_loss'] = 0
df.loc[df['buy_signal'] == 1, 'profit_loss'] = df['c'] - df['c'].shift(1)
df.loc[df['sell_signal'] == 1, 'profit_loss'] = df['c'].shift(1) - df['c']
df['cum_profit_loss'] = df['profit_loss'].cumsum()
# محاسبه ریسک وارد شده در هر تراکنش
df['risk'] = 0
buy_index = df.index[df['buy_signal'] == 1].tolist()
sell_index = df.index[df['sell_signal'] == 1].tolist()
if sell_index[0] < buy_index[0]:
    sell_index = sell_index[1:]
if len(sell_index) > len(buy_index):
    sell_index = sell_index[:-1]
for i, j in zip(buy_index, sell_index):
    price_at_buy = df.loc[i, 'c']
    price_at_sell = df.loc[j, 'c']
    stop_loss = df.loc[i:j, 'cloud_bottom'].min()
    risk = price_at_buy - stop_loss
    df.loc[i:j, 'risk'] = risk

# محاسبه نتایج بر اساس سرمایه اولیه
df['return_pct'] = (df['cum_profit_loss'] / initial_capital) * 100
df['risk_pct'] = (df['risk'] / initial_capital) * 100
df['position_size'] = (initial_capital * 0.02) / df['risk']
df['position_size'] = df['position_size'].fillna(method='ffill')
df['trade_cost'] = df['position_size'] * df['c'] * 0.001
df['net_profit'] = df['cum_profit_loss'] - df['trade_cost']
df['return_on_investment'] = (df['net_profit'] / initial_capital) * 100
return df.tail(1)[['return_pct', 'risk_pct', 'net_profit', 'return_on_investment']]
def backtest_strategy(df):
"""
تابعی برای بررسی عملکرد استراتژی خرید و فروش بر اساس سیگنال‌های محاسبه شده
"""
# محاسبه سود و ضرر
df['profit'] = 0
df['loss'] = 0
df['buy_price'] = 0
df['sell_price'] = 0
df['position'] = 0
df['position'] = np.where(df['buy_signal'] == 1, 1, df['position'])
df['position'] = np.where(df['sell_signal'] == 1, -1, df['position'])
for i in range(1, len(df)):
    # پایین ترین قیمت خرید
    if df['position'][i] == 1 and df['position'][i-1] == 0:
        df['buy_price'][i] = df['c'][i]
    else:
        df['buy_price'][i] = df['buy_price'][i-1]

    # بالاترین قیمت فروش
    if df['position'][i] == -1 and df['position'][i-1] == 0:
        df['sell_price'][i] = df['c'][i]
    else:
        df['sell_price'][i] = df['sell_price'][i-1]

    # محاسبه سود و ضرر
    if df['position'][i] == 1 and df['position'][i-1] == 0:
        df['profit'][i] = 0
        df['loss'][i] = 0
    elif df['position'][i] == -1 and df['position'][i-1] == 0:
        df['profit'][i] = 0
        df['loss'][i] = 0
    elif df['position'][i] == 0 and df['position'][i-1] != 0:
        if df['position'][i-1] == 1:
            df['profit'][i] = (df['c'][i] - df['buy_price'][i]) / df['buy_price'][i]
            df['loss'][i] = 0
        else:
            df['profit'][i] = (df['sell_price'][i] - df['c'][i]) / df['sell_price'][i]
            df['loss'][i] = 0
    else:
        if df['position'][i] == 1:
            df['profit'][i] = (df['c'][i] - df['buy_price'][i-1]) / df['buy_price'][i-1]
            df['loss'][i] = 0
        else:
            df['profit'][i] = (df['sell_price'][i-1] - df['c'][i]) / df['sell_price'][i-1]
            df['loss'][i] = 0

# محاسبه مجموع سود و ضرر
total_profit = df['profit'].sum()
total_loss = df['loss'].sum()
net_profit = total_profit + total_loss

# محاسبه درصد سود و ضرر
total_trades = len(df[df['position'] != 0])
win_trades = len(df
def calculate_profit_loss(df):
"""
تابعی برای محاسبه سود یا ضرر تجارت‌های انجام شده
"""
# محاسبه سود یا ضرر هر تجارت
df['pl'] = np.where(df['position'].shift(1) == 1, df['c'] - df['buy_price'], df['sell_price'] - df['c'])
df['pl'].fillna(0, inplace=True)
# محاسبه سود یا ضرر کل تجارت‌های انجام شده
df['cum_pl'] = df['pl'].cumsum()

return df
def main(pair, interval, limit):
"""
تابع اصلی برای اجرای ربات تحلیل گر
"""
# دریافت داده‌های تاریخی
df = get_data(pair, interval, limit)
# محاسبه نقاط اصلی Ichimoku Cloud
df = calculate_indicators(df)

# محاسبه سیگنال‌های خرید و فروش
df = calculate_signals(df)

# محاسبه سود یا ضرر تجارت‌های انجام شده
df = calculate_profit_loss(df)

# نمایش تمامی تجارت‌های انجام شده به همراه سیگنال‌های خرید و فروش
print(df[['c', 'buy_signal', 'sell_signal', 'position', 'buy_price', 'sell_price', 'pl', 'cum_pl']])

# محاسبه مقدار سود، ضرر، ریسک و ارزش خرید به صورت درصدی
profit = round((df['cum_pl'].iloc[-1] / df['buy_price'].iloc[0]) * 100, 2)
loss = round((abs(df['cum_pl'].iloc[-1]) / df['buy_price'].iloc[0]) * 100, 2)
risk = round((abs(df['pl']) / df['buy_price'].iloc[0]).mean() * 100, 2)
buy_value = round(df['buy_price'].iloc[-1] / df['buy_price'].iloc[0] * 100, 2)

# نمایش مقادیر سود، ضرر، ریسک و ارزش خرید به صورت درصدی
print(f"Profit: {profit}%")
print(f"Loss: {loss}%")
print(f"Risk: {risk}%")
print(f"Buy Value: {buy_value}%")

# ارسال نوتیفیکیشن به کاربر برای اخطار خرید یا فروش
if df['buy_signal'].iloc[-1] == 1:
    message = f"BUY signal detected for {pair} at {df['c'].iloc[-1]}"
    send_notification(message)
elif df['sell_signal'].iloc[-1] == 1:
    message = f"SELL signal detected for {pair} at {df['
Import required libraries
import pandas as pd
import numpy as np
import talib as ta
from binance.client import Client
from datetime import datetime, timedelta
import pytz
import time
import os

Set API keys and pair
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)
pair = 'BTCUSDT'

Set trading parameters
interval = Client.KLINE_INTERVAL_1HOUR # 1-hour candles
quantity = 0.001 # quantity to trade
stop_loss = 0.02 # stop loss percentage
take_profit = 0.04 # take profit percentage

Set notification parameters
send_notification = True
if send_notification:
from pushbullet import Pushbullet
pb = Pushbullet("YOUR_PUSHBULLET_API_KEY")

Define functions to calculate indicators, signals, and profit/loss
def calculate_indicators(df):
# Add Ichimoku Cloud indicators
high_prices = df['h'].values
low_prices = df['l'].values
close_prices = df['c'].values
tenkan_sen = ta.ICHIMOKU(high_prices, low_prices, tenkan_period=9, tenkan_shift=9, price='high')
kijun_sen = ta.ICHIMOKU(high_prices, low_prices, kijun_period=26, kijun_shift=26, price='high')
senkou_span_a = (tenkan_sen + kijun_sen) / 2
senkou_span_b = ta.ICHIMOKU(high_prices, low_prices, senkou_period=52, senkou_shift=26, price='high')
chikou_span = close_prices

df['tenkan_sen'] = tenkan_sen
df['kijun_sen'] = kijun_sen
df['senkou_span_a'] = senkou_span_a
df['senkou_span_b'] = senkou_span_b
df['chikou_span'] = chikou_span

return df
def calculate_signals(df):
# Add buy and sell signals
buy_signal = (df['tenkan_sen'].shift(1) < df['kijun_sen'].shift(1)) &
(df['tenkan_sen'] > df['kijun_sen']) &
(df['c'] > df['senkou_span_a']) &
(df['c'] > df['senkou_span_b'])
sell_signal = (df['tenkan_sen'].shift(1) > df['kijun_sen'].shift(1)) & \
              (df['tenkan_sen'] < df['kijun_sen']) & \
              (df['c'] < df['senkou_span_a']) & \
              (df['c'] < df['senkou_span_b'])

df['buy_signal'] = buy_signal.astype(int)
df['sell_signal'] = sell_signal.astype(int)

# Calculate position based on signals
position = df['buy_signal'].copy()
position[df['sell_signal'] == 1] = -1
position = position.fillna(0)

df['position'] = position

# Calculate buy and sell prices
buy_price = df['c'][df['buy_signal'] == 1]
sell_price = df['c'][df['sell_signal'] == 1]

# Fill forward for buy and sell prices
buy_price =return df
تابع محاسبه سیگنال‌های خرید و فروش
def calculate_signals(df):
buy_signal = []
sell_signal = []
position = []
buy_price = []
sell_price = []
for i in range(len(df)):
    if df['tenkan_sen'].iloc[i] > df['kijun_sen'].iloc[i] and df['c'].iloc[i] > df['senkou_span_a'].iloc[i] and \
            df['c'].iloc[i] > df['senkou_span_b'].iloc[i] and df['chikou_span'].iloc[i] > df['c'].iloc[i]:
        buy_signal.append(1)
        sell_signal.append(0)
        position.append(1)
        buy_price.append(df['c'].iloc[i])
        sell_price.append(0)
    elif df['tenkan_sen'].iloc[i] < df['kijun_sen'].iloc[i] and df['c'].iloc[i] < df['senkou_span_a'].iloc[i] and \
            df['c'].iloc[i] < df['senkou_span_b'].iloc[i] and df['chikou_span'].iloc[i] < df['c'].iloc[i]:
        buy_signal.append(0)
        sell_signal.append(1)
        position.append(-1)
        buy_price.append(0)
        sell_price.append(df['c'].iloc[i])
    else:
        buy_signal.append(0)
        sell_signal.append(0)
        position.append(0)
        buy_price.append(0)
        sell_price.append(0)

df['buy_signal'] = buy_signal
df['sell_signal'] = sell_signal
df['position'] = position
df['buy_price'] = buy_price
df['sell_price'] = sell_price

return df
تابع محاسبه سود یا ضرر تجارت‌های انجام شده
def calculate_profit_loss(df):
pl = []
cum_pl = 0
for i in range(len(df)):
    if i > 0 and df['position'].iloc[i - 1] == 1 and df['sell_signal'].iloc[i] == 1:
        pl.append(df['c'].iloc[i] - df['buy_price'].iloc[i - 1])
        cum_pl += df['c'].iloc[i] - df['buy_price'].iloc[i - 1]
    elif i > 0 and df['position'].iloc[i - 1] == -1 and df['buy_signal'].iloc[i] == 1:
        pl.append(df['sell_price'].iloc[i - 1] - df['c'].iloc[i])
        cum_pl += df['sell_price'].iloc[i - 1] - df['c'].iloc[i]
    else:
        pl.append(0)

df['pl'] = pl
df['cum_pl'] = cum_pl + df['pl'].cumsum()

return df
تابع ارسال نوتیفیکیشن
def send_notification(message):
# محتوای پیام به تلگرام، ایمیل و یا پیامک ارسال می‌شود
print(message)

استفاده از تابع‌ها برای پیاده‌سازی و استفاده از الگوریتم
pair = 'BTC/USD'
ارسال ایمیل به کاربر با خلاصه‌ای از نتایج
import smtplib
from email.mime.text import MIMEText

تنظیمات ایمیل
smtp_server = 'smtp.gmail.com'
smtp_port = 587
smtp_username = 'your_email@gmail.com'
smtp_password = 'your_email_password'
email_from = 'your_email@gmail.com'
email_to = 'recipient_email@gmail.com'

محتوای ایمیل
email_subject = f"Trading results for {pair}"
email_body = f"Profit: {profit}%\nLoss: {loss}%\nRisk: {risk}%\nBuy Value: {buy_value}%\n\nTrade Details:\n{df[['c', 'buy_signal', 'sell_signal', 'position', 'buy_price', 'sell_price', 'pl', 'cum_pl']]}"

تنظیمات ایمیل
msg = MIMEText(email_body)
msg['Subject'] = email_subject
msg['From'] = email_from
msg['To'] = email_to

ارسال ایمیل
try:
server = smtplib.SMTP(smtp_server, smtp_port)
server.ehlo()
server.starttls()
server.login(smtp_username, smtp_password)
server.sendmail(email_from, email_to, msg.as_string())
server.close()
print("Email sent successfully!")
except Exception as e:
print(f"Error: {e}")
