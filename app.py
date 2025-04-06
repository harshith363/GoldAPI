# app.py

import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

def fetch_gold_data(start_date='2010-01-01', end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    df = yf.Ticker('GC=F').history(period='1d', start=start_date, end=end_date)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def prepare_data(df):
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = (df['Return'] > 0).astype(int)
    df.dropna(inplace=True)
    df.index = df.index.tz_localize(None)
    return df

def make_forecast():
    df = prepare_data(fetch_gold_data())
    ts_data = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    ts_data['ds'] = ts_data['ds'].dt.tz_localize(None)
    
    prophet_model = Prophet()
    prophet_model.fit(ts_data)
    forecast = prophet_model.predict(prophet_model.make_future_dataframe(periods=7))
    forecast_7d = forecast[['ds', 'yhat']].tail(7)

    reference_date = datetime.today()
    available_dates = df.index[df.index <= reference_date]
    last_close = df.loc[available_dates.max(), 'Close'] if not available_dates.empty else df['Close'].iloc[-1]
    trend = forecast_7d['yhat'].mean()
    recommendation = "Buy" if trend > last_close else "Hold/Sell"

    last_10 = df.loc[df.index <= reference_date].tail(10)['Close']
    support = last_10.min()
    resistance = last_10.max()

    return forecast_7d.to_dict(orient='records'), recommendation, support, resistance

@app.route('/predict', methods=['GET'])
def predict():
    forecast, recommendation, support, resistance = make_forecast()
    return jsonify({
        "prediction": forecast,
        "recommendation": recommendation,
        "support": support,
        "resistance": resistance
    })

if __name__ == '__main__':
    app.run(debug=True)
