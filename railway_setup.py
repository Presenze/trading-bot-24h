#!/usr/bin/env python3
"""
🚀 TRADING BOT 24/7 - RAILWAY HOSTING
Sistema ottimizzato per Railway.app
"""

import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import json
import logging
from flask import Flask, render_template_string, jsonify
import requests

# Configurazione logging per Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurazione MT5
MT5_CONFIG = {
    'login': 101595237,
    'password': 'ahs8wuus!U',
    'server': 'Ava-Demo 1-MT5'
}

# Flask app per Railway
app = Flask(__name__)

class RailwayTradingBot:
    def __init__(self):
        self.running = False
        self.positions = []
        self.stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0.0,
            'last_update': datetime.now().isoformat()
        }
        self.mt5_connected = False
    
    def connect_mt5(self):
        """Connessione MT5 per Railway"""
        try:
            if not mt5.initialize():
                logger.error("❌ MT5 non inizializzato")
                return False
            
            if not mt5.login(MT5_CONFIG['login'], 
                           password=MT5_CONFIG['password'], 
                           server=MT5_CONFIG['server']):
                logger.error("❌ Login MT5 fallito")
                return False
            
            logger.info("✅ MT5 connesso su Railway!")
            self.mt5_connected = True
            return True
        except Exception as e:
            logger.error(f"❌ Errore connessione: {e}")
            return False
    
    def get_account_info(self):
        """Ottieni info account"""
        try:
            if not self.mt5_connected:
                return None
                
            account_info = mt5.account_info()
            if account_info:
                return {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free,
                    'margin_level': account_info.margin_level
                }
        except Exception as e:
            logger.error(f"❌ Errore account info: {e}")
        return None
    
    def analyze_market(self):
        """Analisi mercato avanzata"""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        signals = []
        
        for symbol in symbols:
            try:
                if not self.mt5_connected:
                    continue
                    
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    
                    # Analisi tecnica avanzata
                    current_price = df['close'].iloc[-1]
                    sma_20 = df['close'].rolling(20).mean().iloc[-1]
                    sma_50 = df['close'].rolling(50).mean().iloc[-1]
                    
                    # RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # MACD
                    ema_12 = df['close'].ewm(span=12).mean()
                    ema_26 = df['close'].ewm(span=26).mean()
                    macd = ema_12 - ema_26
                    signal_line = macd.ewm(span=9).mean()
                    current_macd = macd.iloc[-1]
                    current_signal = signal_line.iloc[-1]
                    
                    # Logica di trading
                    confidence = 0.5
                    signal_type = 'HOLD'
                    
                    if (current_price > sma_20 and current_price > sma_50 and 
                        current_rsi < 70 and current_macd > current_signal):
                        signal_type = 'BUY'
                        confidence = 0.8
                    elif (current_price < sma_20 and current_price < sma_50 and 
                          current_rsi > 30 and current_macd < current_signal):
                        signal_type = 'SELL'
                        confidence = 0.8
                    
                    signals.append({
                        'symbol': symbol,
                        'signal': signal_type,
                        'confidence': confidence,
                        'price': current_price,
                        'rsi': current_rsi,
                        'macd': current_macd
                    })
            except Exception as e:
                logger.error(f"❌ Errore analisi {symbol}: {e}")
        
        return signals
    
    def get_positions(self):
        """Ottieni posizioni aperte"""
        try:
            if not self.mt5_connected:
                return []
                
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'direction': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': pos.volume,
                    'price': pos.price_open,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'commission': pos.commission
                })
            
            return result
        except Exception as e:
            logger.error(f"❌ Errore posizioni: {e}")
            return []
    
    def run_trading_loop(self):
        """Loop principale di trading per Railway"""
        self.running = True
        logger.info("🚀 Bot di trading avviato su Railway!")
        
        while self.running:
            try:
                # Aggiorna statistiche
                self.stats['last_update'] = datetime.now().isoformat()
                
                # Analizza mercato
                signals = self.analyze_market()
                positions = self.get_positions()
                account_info = self.get_account_info()
                
                # Aggiorna stats
                self.positions = positions
                self.stats['total_trades'] = len(positions)
                self.stats['total_profit'] = sum(pos['profit'] for pos in positions)
                self.stats['profitable_trades'] = len([p for p in positions if p['profit'] > 0])
                
                # Log periodico
                if len(signals) > 0:
                    active_signals = [s for s in signals if s['signal'] != 'HOLD']
                    logger.info(f"📊 Analisi completata: {len(active_signals)} segnali attivi, {len(positions)} posizioni")
                
                time.sleep(60)  # Controlla ogni minuto
                
            except Exception as e:
                logger.error(f"❌ Errore nel loop principale: {e}")
                time.sleep(30)
    
    def start(self):
        """Avvia il bot"""
        if self.connect_mt5():
            # Avvia loop in thread separato
            trading_thread = threading.Thread(target=self.run_trading_loop)
            trading_thread.daemon = True
            trading_thread.start()
            return True
        return False

# Inizializza bot
bot = RailwayTradingBot()

# Template HTML per dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Trading Bot 24/7 - Railway</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .status-card h3 {
            color: #ffd700;
            margin-bottom: 10px;
        }
        .value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .signals {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .signal {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
        }
        .signal.sell {
            border-left-color: #f44336;
        }
        .signal.hold {
            border-left-color: #ff9800;
        }
        .pulse {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Trading Bot 24/7 - Railway</h1>
            <p>Hosting gratuito 24/7 su Railway.app</p>
            <div class="pulse">🔄 Sistema Attivo</div>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>💰 Saldo</h3>
                <div class="value" id="balance">€0.00</div>
            </div>
            <div class="status-card">
                <h3>📊 Equity</h3>
                <div class="value" id="equity">€0.00</div>
            </div>
            <div class="status-card">
                <h3>🎯 Posizioni</h3>
                <div class="value" id="positions">0</div>
            </div>
            <div class="status-card">
                <h3>📈 Profitto</h3>
                <div class="value" id="profit">€0.00</div>
            </div>
        </div>
        
        <div class="signals">
            <h2>🎯 Segnali di Trading</h2>
            <div id="signals-list">
                <p>Caricamento segnali...</p>
            </div>
        </div>
        
        <div class="status-card">
            <h3>⏰ Ultimo Aggiornamento</h3>
            <div id="last-update">Caricamento...</div>
        </div>
    </div>

    <script>
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('balance').textContent = '€' + (data.account?.balance || 0).toFixed(2);
                    document.getElementById('equity').textContent = '€' + (data.account?.equity || 0).toFixed(2);
                    document.getElementById('positions').textContent = data.positions?.length || 0;
                    document.getElementById('profit').textContent = '€' + (data.stats?.total_profit || 0).toFixed(2);
                    document.getElementById('last-update').textContent = new Date().toLocaleString('it-IT');
                    
                    // Aggiorna segnali
                    const signalsDiv = document.getElementById('signals-list');
                    if (data.signals && data.signals.length > 0) {
                        signalsDiv.innerHTML = data.signals.map(signal => `
                            <div class="signal ${signal.signal.toLowerCase()}">
                                <strong>${signal.symbol}</strong> - ${signal.signal} 
                                (Confidenza: ${(signal.confidence * 100).toFixed(0)}%) 
                                - Prezzo: ${signal.price.toFixed(5)}
                            </div>
                        `).join('');
                    } else {
                        signalsDiv.innerHTML = '<p>Nessun segnale attivo</p>';
                    }
                })
                .catch(error => {
                    console.error('Errore aggiornamento:', error);
                });
        }
        
        // Aggiorna ogni 30 secondi
        updateDashboard();
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    """Dashboard principale"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    """API per status del bot"""
    try:
        account_info = bot.get_account_info()
        positions = bot.get_positions()
        signals = bot.analyze_market()
        
        return jsonify({
            'account': account_info,
            'positions': positions,
            'signals': signals,
            'stats': bot.stats,
            'mt5_connected': bot.mt5_connected,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"❌ Errore API status: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/health')
def health_check():
    """Health check per Railway"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'mt5_connected': bot.mt5_connected
    })

if __name__ == '__main__':
    # Avvia il bot
    logger.info("🎯 Inizializzazione Trading Bot per Railway...")
    if bot.start():
        logger.info("✅ Bot avviato con successo!")
    else:
        logger.warning("⚠️ Bot avviato senza connessione MT5")
    
    # Avvia Flask app
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Avviando server Flask su porta {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
