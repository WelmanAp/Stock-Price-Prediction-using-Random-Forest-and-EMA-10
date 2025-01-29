import yfinance as yf
import pandas as pd
import os

def download_and_process_stock_data(ticker, start_date, end_date, output_folder):
    # Unduh data saham dari yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    if stock_data.empty:
        print(f"Data untuk {ticker} tidak tersedia.")
        return

    # Data Cleaning: Hapus baris dengan missing values (NaN) pada semua kolom
    stock_data = stock_data.dropna()

    # Data Transformation: Tambahkan kolom EMA_10 dan Return
    stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
    stock_data['Return'] = stock_data['Close'].pct_change()

    # Data Cleaning (Setelah Transformasi): Hapus baris yang memiliki nilai NaN baru
    stock_data = stock_data.dropna()

    # Simpan data yang telah dibersihkan dan diproses ke file Excel
    output_file = os.path.join(output_folder, f"{ticker}.xlsx")
    stock_data.to_excel(output_file, index=True, sheet_name="Processed Data")
    print(f"Data untuk {ticker} telah diproses dan disimpan ke {output_file}.")

# Daftar saham (ticker) dan pengaturan waktu
stocks = ['ADRO.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK', 
          'BMRI.JK', 'ICBP.JK', 'PTBA.JK', 'TLKM.JK', 'TOWR.JK']
start_date = "2022-10-31" 
end_date = "2024-10-31"
output_folder = "data"

# Pastikan folder output ada
os.makedirs(output_folder, exist_ok=True)

# Unduh dan proses data untuk setiap saham
for stock in stocks:
    download_and_process_stock_data(stock, start_date, end_date, output_folder)
