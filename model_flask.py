import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from prophet import Prophet
from flask import Flask, jsonify
from datetime import datetime, timedelta

def fetch_gold_data(start_date='2010-01-01', end_date='2025-04-02'):
    df = yf.Ticker('GC=F').history(period='1d', start=start_date, end=end_date)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def prepare_data(df):
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = (df['Return'] > 0).astype(int)
    df.dropna(inplace=True)
    df.index = df.index.tz_localize(None)
    return df

df = prepare_data(fetch_gold_data())

ts_data = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
ts_data['ds'] = ts_data['ds'].dt.tz_localize(None)
prophet_model = Prophet().fit(ts_data)
forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=7))
forecast_7d = forecast[['ds', 'yhat']].tail(7)

def get_buy_signal():
    reference_date = datetime(2025, 4, 4)
    available_dates = df.index[df.index <= reference_date]
    if not available_dates.empty:
        last_close = df.loc[available_dates.max(), 'Close']
    else:
        last_close = df['Close'].iloc[-1]
    trend = forecast_7d['yhat'].mean()
    return "Buy" if trend > last_close else "Hold/Sell"


def get_support_resistance():
    reference_date = datetime(2025, 4, 4)
    last_10 = df.loc[df.index <= reference_date].tail(10)['Close']
    support = last_10.min()
    resistance = last_10.max()
    return support, resistance

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    support, resistance = get_support_resistance()
    return jsonify({
        "prediction": forecast_7d.to_dict(orient='records'),
        "recommendation": get_buy_signal(),
        "support": support,
        "resistance": resistance
    })

if __name__ == '__main__':
    app.run(debug=True)