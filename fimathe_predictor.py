from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

app = Flask(__name__)

def load_history(pair, interval='1h', months=12):
    end = datetime.now()
    start = end - timedelta(days=30*months)
    symbol = pair.replace('USD','USDT') if 'USD' in pair and pair != 'XAUUSD' else pair
    if pair == "XAUUSD":
        symbol = 'GC=F'
    else:
        symbol = pair[:3] + pair[3:] + '=X'
    df = yf.download(symbol, start=start, end=end, interval=interval)
    return df.dropna()

def fimathe_logic(df):
    if len(df) < 201: return None
    last = df.iloc[-1]
    week = df.iloc[-168:]
    high = week['High'].max()
    low = week['Low'].min()
    canal = high - low
    mid = (high + low) / 2
    zona_sup = mid + 0.1*canal
    zona_inf = mid - 0.1*canal
    mm200 = df['Close'].rolling(200).mean().iloc[-1]
    # Se mm200 for NaN, retorna None
    if pd.isna(mm200):
        return None
    # Aqui garantimos que mm200 é float
    mm200 = float(mm200)
    price = last['Close']
    sinal = None
    tendencia = 'up' if price > mm200 else 'down'
    if price > zona_sup and tendencia == 'up': sinal = 'buy'
    if price < zona_inf and tendencia == 'down': sinal = 'sell'
    return {
        'price': float(price), 'sinal': sinal, 'canal': float(canal),
        'zona_sup': float(zona_sup), 'zona_inf': float(zona_inf),
        'tendencia': tendencia, 'timestamp': last.name.strftime('%Y-%m-%d %H:%M')
    }

def stat_predict(df, sinal):
    # Pega todos os momentos parecidos com o contexto atual e mede acerto
    df['mm200'] = df['Close'].rolling(200).mean()
    df['tend'] = np.where(df['Close'] > df['mm200'], 'up', 'down')
    # Sinal FIMATHE: as regras aqui são customizáveis por versão!
    highs = df['High'].rolling(168).max()
    lows  = df['Low'].rolling(168).min()
    mid   = (highs + lows) / 2
    canal = highs - lows
    zona_sup = mid + 0.1*canal
    zona_inf = mid - 0.1*canal

    df['fimathe_buy']  = (df['Close'] > zona_sup) & (df['tend']=='up')
    df['fimathe_sell'] = (df['Close'] < zona_inf) & (df['tend']=='down')

    df['signal'] = None
    df.loc[df['fimathe_buy'], 'signal'] = 'buy'
    df.loc[df['fimathe_sell'], 'signal'] = 'sell'

    results = []
    for i in range(200, len(df)-2):
        sig = df.iloc[i]['signal']
        if not sig: continue
        entry = df.iloc[i]['Close']
        # Saída simulada: SL/TP = 2 níveis do canal (~canal/4)
        canal_size = df.iloc[i]['High'] - df.iloc[i]['Low']
        sl = entry - 2*(canal_size/8) if sig == 'buy' else entry + 2*(canal_size/8)
        tp = entry + 2*(canal_size/8) if sig == 'buy' else entry - 2*(canal_size/8)
        window = df.iloc[i+1:i+7]['Close'] # próximos 6 horas
        gain = None
        for v in window:
            if sig == 'buy':
                if v >= tp: gain = 1; break
                if v <= sl: gain = 0; break
            else:
                if v <= tp: gain = 1; break
                if v >= sl: gain = 0; break
        if gain is not None:
            results.append(gain)
    if not results: return 0.0, 0
    acerto = sum(results)/len(results)
    return acerto, len(results)

@app.route('/predict', methods=['POST'])
def predict():
    req = request.json
    pair = req.get('pair')
    df = load_history(pair)
    atual = fimathe_logic(df)
    if not atual or not atual['sinal']:
        return jsonify({'score': 0, 'suggestion': 'ignore'})
    score, ntrades = stat_predict(df, atual['sinal'])
    # Você pode definir o limiar: ex, só operar se acerto >0.65 e >30 casos históricos
    suggestion = 'execute' if score > 0.65 and ntrades > 30 else 'ignore'
    return jsonify({**atual, 'score': score, 'ntrades': ntrades, 'suggestion': suggestion})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Usa a porta que o Railway define, ou 5000 localmente
    app.run(host='0.0.0.0', port=port)

