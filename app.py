# app.py

import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

def fetch_data(ticker, start_date='2010-01-01', end_date='2025-04-01'):
    df = yf.Ticker(ticker).history(start=start_date, end=end_date)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

def prepare_data(df):
    df['Return'] = df['Close'].pct_change()
    df['Direction'] = (df['Return'] > 0).astype(int)
    df.dropna(inplace=True)
    df.index = df.index.tz_localize(None)
    return df

def forecast_ticker(ticker='GC=F', days=7):
    try:
        df = fetch_data(ticker)
        df = prepare_data(df)

        # --- RSI ---
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # --- Moving Averages ---
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # --- Forecasting ---
        ts_data = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        ts_data['ds'] = ts_data['ds'].dt.tz_localize(None)

        model = Prophet()
        model.fit(ts_data)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        forecast_n = forecast[['ds', 'yhat']].tail(days)

        # Recommendation Logic
        last_close = df['Close'].iloc[-1]
        trend = forecast_n['yhat'].mean()
        recommendation = "Buy" if trend > last_close else "Hold/Sell"

        # Support & Resistance
        last_10 = df['Close'].tail(10)
        support = last_10.min()
        resistance = last_10.max()

        # Trend Strength from RSI
        latest_rsi = df['RSI'].iloc[-1]
        if latest_rsi < 30:
            trend_strength = "Strong Uptrend Likely (Oversold)"
        elif latest_rsi < 45:
            trend_strength = "Weak Uptrend"
        elif latest_rsi <= 55:
            trend_strength = "Neutral / Sideways"
        elif latest_rsi <= 70:
            trend_strength = "Weak Downtrend"
        else:
            trend_strength = "Strong Downtrend Likely (Overbought)"

        return {
            "ticker": ticker,
            "prediction": forecast_n.to_dict(orient='records'),
            "recommendation": recommendation,
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "latest_RSI": round(latest_rsi, 2),
            "SMA_20": round(df['SMA_20'].iloc[-1], 2),
            "EMA_20": round(df['EMA_20'].iloc[-1], 2),
            "trend_strength": trend_strength
        }

    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e)
        }

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', default='GC=F')
    days = request.args.get('days', default=7, type=int)

    if days < 1:
        days = 1
    elif days > 30:
        days = 30

    result = forecast_ticker(ticker=ticker, days=days)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
