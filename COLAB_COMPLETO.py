# 🚀 TRADING BOT 24/7 - GOOGLE COLAB COMPLETO
# Copia e incolla questo codice in Google Colab per avviare il bot!

print("🚀 Avviando Trading Bot 24/7 su Google Colab...")

# Installa dipendenze
!pip install MetaTrader5 pandas numpy scikit-learn flask matplotlib seaborn plotly requests

# Import librerie
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime
import json
import requests
from flask import Flask, render_template_string

print("✅ Librerie installate e importate!")

# Configurazione MT5
MT5_CONFIG = {
    'login': 101595237,
    'password': 'ahs8wuus!U',
    'server': 'Ava-Demo 1-MT5'
}

# Classe Trading Bot
class ColabTradingBot:
    def __init__(self):
        self.running = False
        self.positions = []
        self.stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0.0
        }
    
    def connect_mt5(self):
        """Connessione MT5"""
        try:
            if not mt5.initialize():
                print("❌ MT5 non inizializzato")
                return False
            
            if not mt5.login(MT5_CONFIG['login'], 
                           password=MT5_CONFIG['password'], 
                           server=MT5_CONFIG['server']):
                print("❌ Login MT5 fallito")
                return False
            
            print("✅ MT5 connesso su Colab!")
            return True
        except Exception as e:
            print(f"❌ Errore connessione: {e}")
            return False
    
    def get_account_info(self):
        """Ottieni info account"""
        try:
            account_info = mt5.account_info()
            if account_info:
                return {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'free_margin': account_info.margin_free
                }
        except:
            pass
        return None
    
    def analyze_market(self):
        """Analisi mercato semplificata"""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        signals = []
        
        for symbol in symbols:
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    
                    # Analisi semplice
                    current_price = df['close'].iloc[-1]
                    sma_20 = df['close'].rolling(20).mean().iloc[-1]
                    sma_50 = df['close'].rolling(50).mean().iloc[-1]
                    
                    # Calcola RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    current_rsi = rsi.iloc[-1]
                    
                    # Segnale di trading
                    if current_price > sma_20 and current_price > sma_50 and current_rsi < 70:
                        signals.append({'symbol': symbol, 'signal': 'BUY', 'confidence': 0.75, 'price': current_price})
                    elif current_price < sma_20 and current_price < sma_50 and current_rsi > 30:
                        signals.append({'symbol': symbol, 'signal': 'SELL', 'confidence': 0.75, 'price': current_price})
                    else:
                        signals.append({'symbol': symbol, 'signal': 'HOLD', 'confidence': 0.5, 'price': current_price})
            except Exception as e:
                print(f"❌ Errore analisi {symbol}: {e}")
        
        return signals
    
    def get_positions(self):
        """Ottieni posizioni aperte"""
        try:
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
                    'profit': pos.profit
                })
            
            return result
        except Exception as e:
            print(f"❌ Errore posizioni: {e}")
            return []
    
    def run_trading_loop(self):
        """Loop principale di trading"""
        self.running = True
        print("🚀 Bot di trading avviato su Colab!")
        print("=" * 60)
        
        while self.running:
            try:
                # Mostra info account
                account_info = self.get_account_info()
                if account_info:
                    print(f"💰 Saldo: €{account_info['balance']:.2f}")
                    print(f"📊 Equity: €{account_info['equity']:.2f}")
                    print(f"🛡️ Margine: €{account_info['margin']:.2f}")
                    print(f"💵 Margine Libero: €{account_info['free_margin']:.2f}")
                
                # Mostra posizioni aperte
                positions = self.get_positions()
                if positions:
                    print(f"📈 Posizioni Aperte: {len(positions)}")
                    for pos in positions:
                        profit_color = "🟢" if pos['profit'] >= 0 else "🔴"
                        print(f"  {profit_color} {pos['symbol']} {pos['direction']} - P&L: €{pos['profit']:.2f}")
                else:
                    print("📈 Nessuna posizione aperta")
                
                # Analizza mercato
                signals = self.analyze_market()
                print("\n🎯 Segnali di Trading:")
                for signal in signals:
                    if signal['signal'] != 'HOLD':
                        emoji = "📈" if signal['signal'] == 'BUY' else "📉"
                        print(f"  {emoji} {signal['symbol']}: {signal['signal']} (Confidenza: {signal['confidence']*100:.0f}%) - Prezzo: {signal['price']:.5f}")
                
                # Statistiche
                total_profit = sum(pos['profit'] for pos in positions)
                print(f"\n📊 Statistiche:")
                print(f"  💰 Profitto Totale: €{total_profit:.2f}")
                print(f"  🎯 Posizioni Attive: {len(positions)}")
                print(f"  📈 Segnali Generati: {len([s for s in signals if s['signal'] != 'HOLD'])}")
                
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Bot attivo")
                print("=" * 60)
                
                time.sleep(60)  # Controlla ogni minuto
                
            except Exception as e:
                print(f"❌ Errore nel loop principale: {e}")
                time.sleep(30)
    
    def start(self):
        """Avvia il bot"""
        if self.connect_mt5():
            self.run_trading_loop()

# Avvia il bot
print("🎯 Inizializzazione Trading Bot...")
bot = ColabTradingBot()
bot.start()
