from flask import Flask, render_template, request
import joblib
import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import locale

locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')

app = Flask(__name__)

@app.template_filter('format_price')
def format_price(value):
    try:
        formatted_value = locale.currency(value, grouping=True)
        return formatted_value.replace("Rp", "Rp ")
    except (ValueError, TypeError):
        return value

stocks = {
    'ADRO.JK': 'PT Alamtri Resources Indonesia Tbk (ADRO)',
    'ASII.JK': 'Astra International Tbk (ASII)',
    'BBCA.JK': 'Bank Central Asia Tbk (BBCA)',
    'BBNI.JK': 'Bank Negara Indonesia Persero Tbk (BBNI)',
    'BBRI.JK': 'Bank Rakyat Indonesia Persero Tbk (BBRI)',
    'BMRI.JK': 'Bank Mandiri Persero Tbk (BMRI)',
    'ICBP.JK': 'Indofood CBP Sukses Makmur Tbk (ICBP)',
    'PTBA.JK': 'Bukit Asam Tbk (PTBA)',
    'TLKM.JK': 'Telkom Indonesia Persero Tbk (TLKM)',
    'TOWR.JK': 'Sarana Menara Nusantara Tbk (TOWR)',
}

wib = pytz.timezone('Asia/Jakarta')

def is_market_open():
    now = datetime.now(wib)
    market_close_time = now.replace(hour=16, minute=30, second=0, microsecond=0)
    return now < market_close_time

def get_close_price(stock_symbol):
    try:
        if is_market_open():
            return "Bursa belum tutup, harga Close belum final."
        else:
            data = yf.download(stock_symbol, period='1d', interval='1d')
            if data.empty:
                return "Data tidak tersedia untuk saham ini."
            close_price = float(data['Close'].iloc[-1])
            return close_price
    except Exception as e:
        print(f"Error di get_close_price: {e}")
        return "Terjadi kesalahan saat mengambil data harga close."


def calculate_accuracy(actual, predicted):
    try:
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if actual.size == 0 or predicted.size == 0:
            return "Data kosong tidak dapat dihitung akurasi."
        
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return round(mape, 2)
    except ZeroDivisionError:
        return "Tidak dapat menghitung MAPE (divisi dengan nol)."
    except Exception as e:
        return f"Terjadi error saat menghitung akurasi: {e}"


@app.route('/')
def index():
    return render_template('index.html', stocks=stocks)

def download_and_check_data(stock_symbol):
    print(f"Mengunduh data untuk {stock_symbol} ...")
    data = yf.download(stock_symbol, period='1y', interval='1d')
    
    if data.empty:
        return None, "Data tidak tersedia."
    
    print(f"Kolom yang tersedia dalam data:\n{data.columns}")
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    print(f"Data setelah meratakan kolom:\n{data.tail()}")
    
    return data, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.form['stock_symbol']
        model_path = f'models/{stock_symbol}_model.pkl'

        try:
            model = joblib.load(model_path)
        except FileNotFoundError:
            return f"Model untuk {stock_symbol} tidak ditemukan."

        data, error_message = download_and_check_data(stock_symbol)
        if error_message:
            return error_message

        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['Return'] = data['Close'].pct_change()

        data_clean = data.dropna(subset=['Close', 'EMA_10', 'Return'])
        if data_clean.empty:
            return "Data yang cukup untuk prediksi tidak tersedia."

        input_data = data_clean[['Close', 'EMA_10', 'Return']].iloc[-1:].copy()
        prediction = model.predict(input_data)
        prediction_value = prediction[0]

        now = datetime.now(wib)
        if is_market_open():
            close_price = float(data['Close'].iloc[-2]) 
        else:
            close_price = float(data['Close'].iloc[-1])

        percentage_change = ((prediction_value - close_price) / close_price) * 100
        percentage_change = round(percentage_change, 2)

        actual_values = data_clean['Close'][-10:].values 
        predicted_values = model.predict(data_clean[['Close', 'EMA_10', 'Return']][-10:])
        accuracy = calculate_accuracy(actual_values, predicted_values)

        if is_market_open():
            prediction_date = now.strftime('%d/%m/%Y') 
        else:
            prediction_date = (now + timedelta(days=1)).strftime('%d/%m/%Y') 

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Harga Aktual'))
        fig.add_trace(go.Scatter(x=[data.index[-1]], y=[prediction_value], mode='markers+text', 
                                name='Prediksi', text="Prediksi", textposition="top center"))
        fig.update_layout(title=f'Harga Saham {stocks[stock_symbol]}', xaxis_title="Tanggal", yaxis_title="Harga Saham (Rp)")

        graph_html = fig.to_html(full_html=False)

        return render_template(
            'result.html',
            stock_name=stocks[stock_symbol],
            prediction=prediction_value,
            close_price=close_price,
            percentage_change=percentage_change,
            accuracy=accuracy,
            graph_html=graph_html,
            prediction_date=prediction_date
        )
    
    except Exception as e:
        print(f"Terjadi error: {e}")
        return f"Terjadi error: {str(e)}"


# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
