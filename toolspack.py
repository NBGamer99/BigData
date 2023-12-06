import pandas as pd
import numpy as np
import json

# Open the JSON file
with open('./Stock_Data.json', 'r') as f:
  # Load the JSON data from the file
  data = json.load(f)

# Now you can use the data in your program
# print(['Price History']['open'])
# INFOS = {}
COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
# user_input = input("Enter Stock Symbol : ")

# user_input = input('Enter Symbol : ')
# print(user)
# df = pd.read_csv('./INFOS.csv', delimiter=',')
# df_aapl = df[df['Stock Symbol'] == user]

def convert_to_float(list):
    return [float(x) for x in list]

# print(tuple(df['Stock Symbol']))
def get_Stocks(SYMBOL):
  STOCK = data[SYMBOL]
  PRICES = STOCK['Price History']
  # print(type(PRICES['open']))

  DATA = [
    STOCK['Date'],
    PRICES['open'],
    PRICES['high'],
    PRICES['low'],
    PRICES['close'],
    STOCK['volume']
  ]
  df = pd.DataFrame(np.array(DATA).T, columns=COLUMNS)
  df['Date'] = pd.to_datetime(df['Date'])
  df = df.set_index('Date')
  df['Open'] = pd.to_numeric(df['Open'], downcast='float')
  df['High'] = pd.to_numeric(df['High'], downcast='float')
  df['Low'] = pd.to_numeric(df['Low'], downcast='float')
  df['Close'] = pd.to_numeric(df['Close'], downcast='float')
  df['Volume'] = pd.to_numeric(df['Volume'], downcast='float')
  return df
# print(type(get_Stocks(user_input).Date[25]))

def get_infos(SYMBOL):
  STOCK = data[SYMBOL]
  INFOS = {}
  INFOS["SYMBOL"] = SYMBOL
  INFOS["NAME"] = STOCK["Name"]
  INFOS["Description"] = STOCK["Description"]
  INFOS["Founder"] = STOCK["Founder"]
  INFOS["Country"] = STOCK["Country"]
  INFOS["Foundation Date"] = STOCK["Date of Birth"]
  INFOS["Industry"] = STOCK["Industry"]
  INFOS["Market Value"] = STOCK["Market Value"]
  return INFOS

def get_Symbols(data):
  return (tuple(data.keys()))

# print(get_stocks(data))