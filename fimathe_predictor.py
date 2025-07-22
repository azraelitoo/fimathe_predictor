from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import traceback

app = Flask(__name__)

# Gestão de risco: nunca arrisca mais de 2% por trade, metas diárias ajustáveis
META_DIARIA = 1000  # Alvo diário de lucro (R$)
STOP_DIARIO = -300  # Stop loss diário (R$ negativo)
RISCO_POR_TRADE = 0.02  # 2% do saldo máximo por operação

# Função para calcular lote dinâmico baseado no saldo, risco e distância do SL
def calcula_lote(saldo, price, sl, risco=RISCO_POR_TRADE):
    dist = abs(price - sl)
    if dist == 0: dist = 1  # Para evitar divisão por zero
    lote = max(min(round((saldo * risco) / dist, 2), saldo / 100), 0.01)  # Limita mínimo 0.01
    return lote

# Múltiplos tipos de entradas FIMATHE (Nivel 1, 2, Reversão, Quebra, Micro)
def todas_entradas(df, saldo):
    sinais = []
    if len(df) < 201:
        return sinais
    last = df.iloc[-1]
    week = df.iloc[-168:]
    high = week['High'].max()
    low = week['Low'].min()
    canal = high - low
    mid = (high + low) / 2
    zona_sup = mid + 0.1 * canal
    zona_inf = mid - 0.1 * canal
    mm200 = df['Close'].rolling(200).mean().iloc[-1]
    if pd.isna(mm200):
        return sinais
    mm200 = float(mm200)
    price = float(last['Close'])
    tendencia = 'up' if price > mm200 else 'down'

    # ---- Nível 1: padrão
    if price > zona_sup and tendencia == 'up':
        tp = price + canal * 0.2
        sl = price - canal * 0.1
        lote = calcula_lote(saldo, price, sl)
        sinais.append({
            'variant': 'nivel_1', 'sinal': 'buy', 'tp': round(tp,2), 'sl': round(sl,2),
            'score': 1, 'suggestion': 'execute', 'lot_size': lote
        })
    if price < zona_inf and tendencia == 'down':
        tp = price - canal * 0.2
        sl = price + canal * 0.1
        lote = calcula_lote(saldo, price, sl)
        sinais.append({
            'variant': 'nivel_1', 'sinal': 'sell', 'tp': round(tp,2), 'sl': round(sl,2),
            'score': 1, 'suggestion': 'execute', 'lot_size': lote
        })

    # ---- Nível 2: reversão zona neutra
    candle_ant = df.iloc[-2]
    if price < zona_sup and price > zona_inf:
        if candle_ant['Close'] > zona_sup and tendencia == 'down':
            tp = price - canal * 0.1
            sl = price + canal * 0.1
            lote = calcula_lote(saldo, price, sl)
            sinais.append({
                'variant': 'nivel_2', 'sinal': 'sell', 'tp': round(tp,2), 'sl': round(sl,2),
                'score': 0.8, 'suggestion': 'execute', 'lot_size': lote
            })
        if candle_ant['Close'] < zona_inf and tendencia == 'up':
            tp = price + canal * 0.1
            sl = price - canal * 0.1
            lote = calcula_lote(saldo, price, sl)
            sinais.append({
                'variant': 'nivel_2', 'sinal': 'buy', 'tp': round(tp,2), 'sl': round(sl,2),
                'score': 0.8, 'suggestion': 'execute', 'lot_size': lote
            })

    # ---- Reversão extremos canal (scalper)
    if price >= high:
        tp = price - canal * 0.1
        sl = price + canal * 0.1
        lote = calcula_lote(saldo, price, sl, 0.01)  # menos risco
        sinais.append({
            'variant': 'reversao', 'sinal': 'sell', 'tp': round(tp,2), 'sl': round(sl,2),
            'score': 0.6, 'suggestion': 'execute', 'lot_size': lote
        })
    if price <= low:
        tp = price + canal * 0.1
        sl = price - canal * 0.1
        lote = calcula_lote(saldo, price, sl, 0.01)
        sinais.append({
            'variant': 'reversao', 'sinal': 'buy', 'tp': round(tp,2), 'sl': round(sl,2),
            'score': 0.6, 'suggestion': 'execute', 'lot_size': lote
        })

    # ---- Quebra de canal (breakout)
    if price > high * 1.001 and tendencia == 'up':
        tp = price + canal * 0.3
        sl = price - canal * 0.1
        lote = calcula_lote(saldo, price, sl)
        sinais.append({
            'variant': 'quebra_canal', 'sinal': 'buy', 'tp': round(tp,2), 'sl': round(sl,2),
            'score': 0.7, 'suggestion': 'execute', 'lot_size': lote
        })
    if price < low * 0.999 and tendencia == 'down':
        tp = price - canal * 0.3
        sl = price + canal * 0.1
        lote = calcula_lote(saldo, price, sl)
        sinais.append({
            'variant': 'quebra_canal', 'sinal': 'sell', 'tp': round(tp,2), 'sl': round(sl,2),
            'score': 0.7, 'suggestion': 'execute', 'lot_size': lote
        })

    # ---- Micro-canal: lateralização dentro do canal
    micro_mid = (zona_sup + zona_inf) / 2
    if abs(price - micro_mid) < canal * 0.05:
        if tendencia == 'up':
            tp = price + canal * 0.05
            sl = price - canal * 0.05
            lote = calcula_lote(saldo, price, sl, 0.01)
            sinais.append({
                'variant': 'micro', 'sinal': 'buy', 'tp': round(tp,2), 'sl': round(sl,2),
                'score': 0.5, 'suggestion': 'execute', 'lot_size': lote
            })
        if tendencia == 'down':
            tp = price - canal * 0.05
            sl = price + canal * 0.05
            lote = calcula_lote(saldo, price, sl, 0.01)
            sinais.append({
                'variant': 'micro', 'sinal': 'sell', 'tp': round(tp,2), 'sl': round(sl,2),
                'score': 0.5, 'suggestion': 'execute', 'lot_size': lote
            })
    return sinais

def load_history(pair, interval='1h', months=12):
    try:
        end = datetime.now()
        start = end - timedelta(days=30*months)
        if pair == "XAUUSD":
            symbol = 'GC=F'
        else:
            symbol = pair[:3] + pair[3:] + '=X'
        df = yf.download(symbol, start=start, end=end, interval=interval)
        if df is None or df.empty:
            return None
        return df.dropna()
    except Exception as e:
        print(f"Erro ao carregar histórico para {pair}: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        req = request.get_json()
        pair = req.get('pair')
        saldo = float(req.get('saldo', 200))  # Permite saldo customizável por requisição
        if not pair:
            return jsonify({"error": "Pair not provided"}), 400
        df = load_history(pair)
        sinais = todas_entradas(df, saldo)
        if not sinais:
            return jsonify({'signals': [], 'suggestion': 'ignore', 'error': 'No FIMATHE signal', 'saldo': saldo})
        # Gestão de banca automática pode ser acoplada aqui se desejar
        return jsonify({'pair': pair, 'signals': sinais, 'saldo': saldo, 'suggestion': 'execute' if sinais else 'ignore'})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
