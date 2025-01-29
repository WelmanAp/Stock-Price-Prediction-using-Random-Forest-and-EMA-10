import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

def train_model(file_path):
    data = pd.read_excel(file_path)

    X = data[['Close', 'EMA_10', 'Return']]
    y = data['Close'].shift(-1).dropna() 
    X = X.iloc[:-1] 

    if len(X) < 1:
        print(f"Tidak ada data untuk pelatihan di {file_path}.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    model_name = file_path.split('/')[-1].replace('.xlsx', '_model.pkl')
    joblib.dump(model, f'models/{model_name}')
    print(f"Model untuk {file_path} telah disimpan.")

# Daftar file saham
files = [
    'data/ADRO.JK.xlsx',
    'data/ASII.JK.xlsx',
    'data/BBCA.JK.xlsx', 
    'data/BBNI.JK.xlsx', 
    'data/BBRI.JK.xlsx',
    'data/BMRI.JK.xlsx',
    'data/ICBP.JK.xlsx',
    'data/PTBA.JK.xlsx',
    'data/TLKM.JK.xlsx', 
    'data/TOWR.JK.xlsx'
]

for file in files:
    train_model(file)
