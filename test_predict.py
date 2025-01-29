import joblib
import yfinance as yf
import pandas as pd
from datetime import datetime
import pytz

wib = pytz.timezone('Asia/Jakarta')

def is_market_open():
    """
    Mengecek apakah pasar sedang buka (sebelum jam 16:30 WIB).
    """
    now = datetime.now(wib)
    market_close_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return now < market_close_time

def predict(stock_symbol):
    model_path = f"models/{stock_symbol}.JK_model.pkl" 

    data = yf.download(stock_symbol + '.JK', period='1y', interval='1d')
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['Return'] = data['Close'].pct_change()

    model = joblib.load(model_path)

    features = data[['Close', 'EMA_10', 'Return']].iloc[-1:]
    features.columns = ['Close', 'EMA_10', 'Return'] 

    prediction = model.predict(features)
    prediction_value = prediction[0]

    if is_market_open():
        print(f"Prediksi harga {stock_symbol} hari ini ({datetime.now(wib).strftime('%d/%m/%Y')}): Rp {prediction_value:,.2f}")
    else:
        print(f"Prediksi harga {stock_symbol} untuk hari berikutnya: Rp {prediction_value:,.2f}")
    return prediction_value

# Contoh penggunaan
predict('PTBA')  # Ganti dengan simbol saham yang sesuai
