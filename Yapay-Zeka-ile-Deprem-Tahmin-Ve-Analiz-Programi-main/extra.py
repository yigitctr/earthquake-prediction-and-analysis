import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import pickle

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

def load_model_and_predict():
    model = LSTMModel(input_dim=1, hidden_dim=50, num_layers=1, output_dim=1)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    last_data = np.array([0.5]) 
    future_predictions = []
    for _ in range(30):
        with torch.no_grad():
            input_tensor = torch.tensor(last_data[-30:]).float().unsqueeze(0).unsqueeze(2)
            prediction = model(input_tensor).cpu().numpy()[0, 0]
            future_predictions.append(prediction)
            last_data = np.append(last_data, prediction)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    future_dates = [datetime.now() + timedelta(days=i) for i in range(30)]

    return future_dates, future_predictions

def main():
    future_dates, future_predictions = load_model_and_predict()

    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_predictions, marker='o', linestyle='-', label='Tahmini Büyüklükler', color='red')
    plt.title('Gelecekteki Deprem Tahminleri - Kaya AI')
    plt.xlabel('Tarih')
    plt.ylabel('Büyüklük')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
