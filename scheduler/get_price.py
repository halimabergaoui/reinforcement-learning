import pandas as pd
import numpy as np
import time
import math
import datetime
from dydx3 import Client
from dydx3.constants import MARKET_SOL_USD
from requests.exceptions import ConnectionError, Timeout, RequestException
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas_ta as ta
from datetime import datetime, timedelta

def convert_to_iso_format(date_string):
    try:
        input_datetime = datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')
        
        iso_format = input_datetime.isoformat()
        
        return iso_format
    except ValueError:
        return "Invalid date format. Please provide a date in the format 'YYYY-MM-DDTHH:mm:SS'."

def fetch_data(from_time, to_time):
    def fetch_data_with_retry(client, market, resolution, max_retries=10000, retry_delay=1):
            attempts = 0
            while attempts < max_retries:
                try:
                    candles = client.public.get_candles(
                        market=market,
                        resolution=resolution,
                        from_iso=convert_to_iso_format(from_time),
                        to_iso=convert_to_iso_format(to_time)


                    )

                    return candles
                except (ConnectionError, Timeout, RequestException) as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed: {e}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

            raise Exception("Max retries exceeded. Unable to fetch data from the API.")
    
    client = Client(host='https://api.dydx.exchange')
    try:
        print("Fetching OHLCV data from dydx....")
        candles = fetch_data_with_retry(client, MARKET_SOL_USD, '1DAY')
    except Exception as e:
        print(e)
    candles_data = candles.data
    if 'candles' in candles_data:
        dydx_data = pd.DataFrame(candles_data['candles'])
        dydx_data = dydx_data[["startedAt","low","high","open","close","baseTokenVolume"]]
        dydx_data['startedAt'] = pd.to_datetime(dydx_data['startedAt'])
        dydx_data['startedAt'] = dydx_data['startedAt'].dt.strftime('%Y-%m-%d %H:%M:%S')
        dydx_data = dydx_data.rename(columns = {'startedAt':'timestamp','baseTokenVolume':'volume'})
        dydx_data = dydx_data.iloc[::-1]
        dydx_data = dydx_data.sort_values(by='timestamp', ascending=False)
        dydx_data['close'] = dydx_data['close'].astype(float)
        dydx_data['high'] = dydx_data['high'].astype(float)
        dydx_data['open'] = dydx_data['open'].astype(float)
        dydx_data['low'] = dydx_data['low'].astype(float)
        dydx_data['volume'] = dydx_data['volume'].astype(float)

        return dydx_data['close']

    else:
        print("The 'candles' key was not found in the response.")
        return -1

def extract_prices(n_days):
    iterations_needed = calculate_iterations(n_days)
    i = 0
    dfs = []
    end_time = datetime.now()
    while (i<iterations_needed):
            start_time = end_time - timedelta(days=90)
            sample = fetch_data(start_time.strftime('%Y-%m-%dT%H:%M:%S'), end_time.strftime('%Y-%m-%dT%H:%M:%S'))
            dfs.append(sample)
            end_time = start_time
            i = i+1
            sample = [] 
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.iloc[::-1]
    last_n_values = combined_df.tail(n_days)

    array = last_n_values.to_numpy()
    return array


def calculate_iterations(num_days):
    return math.ceil(num_days / 90)

if __name__ == "__main__":
    n_days = 110
    prices = extract_prices(n_days)
    print(prices)
    