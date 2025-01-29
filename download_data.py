import yfinance as yf
import pandas as pd
import os

def download_and_process_stock_data(ticker, start_date, end_date, output_folder):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if stock_data.empty:
        print(f"Data untuk {ticker} tidak tersedia.")
        return

    stock_data = stock_data.dropna()

    stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
    stock_data['Return'] = stock_data['Close'].pct_change()

    stock_data = stock_data.dropna()

    output_file = os.path.join(output_folder, f"{ticker}.xlsx")
    stock_data.to_excel(output_file, index=True, sheet_name="Processed Data")
    print(f"Data untuk {ticker} telah diproses dan disimpan ke {output_file}.")

stocks = ['ADRO.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK', 
          'BMRI.JK', 'ICBP.JK', 'PTBA.JK', 'TLKM.JK', 'TOWR.JK']
start_date = "2022-10-31" 
end_date = "2024-10-31"
output_folder = "data"

os.makedirs(output_folder, exist_ok=True)

for stock in stocks:
    download_and_process_stock_data(stock, start_date, end_date, output_folder)
