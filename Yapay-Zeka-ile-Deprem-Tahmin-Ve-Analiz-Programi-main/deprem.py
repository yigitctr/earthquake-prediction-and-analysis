import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import requests
import json
import pickle

def fetch_and_save_earthquake_data(filename='ogren.txt'):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
    
    url = f"https://deprem.afad.gov.tr/apiv2/event/filter?start={start_date_str}&end={end_date_str}"
    response = requests.get(url)
    
    if response.status_code == 200:
        deprem_data = response.json()
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(deprem_data, file, ensure_ascii=False, indent=4)
        print(f"Veriler {filename} dosyasına başarıyla kaydedildi.")
    else:
        print("API'den veri çekme işlemi başarısız oldu:", response.status_code)

# LSTM modeli
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def prepare_data(data, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
        y.append(data[i+timesteps])
    return np.array(X), np.array(y)

def main():
    filename = 'ogren.txt'
    fetch_and_save_earthquake_data(filename)
    
    with open(filename, 'r', encoding='utf-8') as file:
        deprem_data = json.load(file)

    df = pd.DataFrame(deprem_data)
    df['magnitude'] = pd.to_numeric(df['magnitude'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['magnitude']).sort_values('date')

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['scaled_magnitude'] = scaler.fit_transform(df[['magnitude']])

    timesteps = 30
    X, y = prepare_data(df['scaled_magnitude'].values, timesteps)
    X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(2)
    y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

    input_dim = 1
    hidden_dim = 50
    num_layers = 1
    output_dim = 1
    num_epochs = 100
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Model Eğitimi
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model.pth')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    model.eval()
    with torch.no_grad():
        predicted_magnitudes = model(X_train).detach().numpy()
    predicted_magnitudes = scaler.inverse_transform(predicted_magnitudes).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['magnitude'], label='Gerçek Değerler', color='blue')
    plt.title('Mevcut Deprem Verileri - Kaya AI')
    plt.xlabel('Tarih')
    plt.ylabel('Büyüklük')
    plt.legend()
    plt.grid(True)
    plt.show()

    # extra.py betiğini çalıştırıyoruz
    os.system('python extra.py')

if __name__ == "__main__":
    main()
