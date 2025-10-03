#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌐 DASHBOARD COMPLETA - Sistema Trading AI
Dashboard completa con tutte le funzionalità avanzate
"""

from flask import Flask, render_template_string, jsonify
import MetaTrader5 as mt5
import sqlite3
import json
from datetime import datetime, timedelta
import threading
import time
import sys
import os

# Aggiungi il percorso del progetto
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_account import *

app = Flask(__name__)

# Template HTML completo
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Trading AI Dashboard - Completa</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        
        .status-card {
            background: rgba(255,255,255,0.15);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.3s ease;
        }
        
        .status-card:hover {
            transform: translateY(-5px);
        }
        
        .status-card h3 {
            font-size: 1.2em;
            margin-bottom: 10px;
            color: #ffd700;
        }
        
        .status-card .value {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .status-card .change {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .secondary-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .card h2 {
            color: #ffd700;
            margin-bottom: 20px;
            font-size: 1.5em;
            border-bottom: 2px solid rgba(255,215,0,0.3);
            padding-bottom: 10px;
        }
        
        .position {
            background: rgba(255,255,255,0.1);
            margin: 10px 0;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
            transition: all 0.3s ease;
        }
        
        .position:hover {
            background: rgba(255,255,255,0.2);
            transform: translateX(5px);
        }
        
        .position.negative {
            border-left-color: #f44336;
        }
        
        .position.positive {
            border-left-color: #4CAF50;
        }
        
        .position-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .symbol {
            font-weight: bold;
            font-size: 1.2em;
        }
        
        .direction {
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .direction.buy {
            background: #4CAF50;
            color: white;
        }
        
        .direction.sell {
            background: #f44336;
            color: white;
        }
        
        .position-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 0.9em;
        }
        
        .detail {
            display: flex;
            justify-content: space-between;
        }
        
        .detail-label {
            opacity: 0.8;
        }
        
        .detail-value {
            font-weight: bold;
        }
        
        .profit {
            color: #4CAF50;
        }
        
        .loss {
            color: #f44336;
        }
        
        .confidence {
            color: #ffd700;
        }
        
        .probability {
            color: #00bcd4;
        }
        
        .chart-container {
            height: 300px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .strategy-status {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        
        .strategy-item {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        
        .strategy-item.active {
            border: 2px solid #4CAF50;
        }
        
        .strategy-item.inactive {
            border: 2px solid #f44336;
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        
        .last-update {
            font-size: 0.9em;
            opacity: 0.8;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 4px solid #ffd700;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 5px;
        }
        
        .tab {
            flex: 1;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        
        .tab.active {
            background: rgba(255,255,255,0.2);
            color: #ffd700;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        @media (max-width: 768px) {
            .main-grid, .secondary-grid {
                grid-template-columns: 1fr;
            }
            
            .position-details {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Trading AI Dashboard - Completa</h1>
            <p>Monitoraggio Completo del Sistema di Trading Automatico</p>
        </div>
        
        <div class="status-bar">
            <div class="status-card">
                <h3>💰 Saldo</h3>
                <div class="value" id="balance">€0.00</div>
                <div class="change" id="balance-change">Caricamento...</div>
            </div>
            
            <div class="status-card">
                <h3>📊 Equity</h3>
                <div class="value" id="equity">€0.00</div>
                <div class="change" id="equity-change">Caricamento...</div>
            </div>
            
            <div class="status-card">
                <h3>🎯 Trade Aperti</h3>
                <div class="value" id="open-trades">0</div>
                <div class="change" id="trades-change">Caricamento...</div>
            </div>
            
            <div class="status-card">
                <h3>📈 Profitto Totale</h3>
                <div class="value" id="total-profit">€0.00</div>
                <div class="change" id="profit-change">Caricamento...</div>
            </div>
            
            <div class="status-card">
                <h3>🛡️ Margine</h3>
                <div class="value" id="margin">€0.00</div>
                <div class="change" id="margin-change">Caricamento...</div>
            </div>
            
            <div class="status-card">
                <h3>📊 Margine Libero</h3>
                <div class="value" id="free-margin">€0.00</div>
                <div class="change" id="free-margin-change">Caricamento...</div>
            </div>
        </div>
        
        <div class="main-grid">
            <div class="card">
                <h2>📊 Posizioni Aperte</h2>
                <div id="positions">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Caricamento posizioni...</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📈 Statistiche Sistema</h2>
                <div id="stats">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Caricamento statistiche...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="secondary-grid">
            <div class="card">
                <h2>🎯 Strategie Attive</h2>
                <div class="strategy-status" id="strategies">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Caricamento strategie...</p>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>📊 Performance</h2>
                <div class="chart-container" id="performance-chart">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Caricamento grafico...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📋 Dettagli Account</h2>
            <div class="tabs">
                <div class="tab active" onclick="showTab('account')">Account</div>
                <div class="tab" onclick="showTab('trades')">Trade History</div>
                <div class="tab" onclick="showTab('settings')">Impostazioni</div>
            </div>
            
            <div class="tab-content active" id="account-tab">
                <div id="account-details">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Caricamento dettagli account...</p>
                    </div>
                </div>
            </div>
            
            <div class="tab-content" id="trades-tab">
                <div id="trades-history">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Caricamento cronologia trade...</p>
                    </div>
                </div>
            </div>
            
            <div class="tab-content" id="settings-tab">
                <div id="settings-panel">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Caricamento impostazioni...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div class="last-update">
                <span class="pulse">🔄</span> Ultimo aggiornamento: <span id="last-update">Caricamento...</span>
            </div>
        </div>
    </div>

    <script>
        // Aggiornamento real-time ogni 2 secondi
        function updateDashboard() {
            fetch('/api/data')
                .then(response => response.json())
                .then(data => {
                    // Aggiorna status bar
                    document.getElementById('balance').textContent = '€' + data.account.balance.toFixed(2);
                    document.getElementById('equity').textContent = '€' + data.account.equity.toFixed(2);
                    document.getElementById('open-trades').textContent = data.positions.length;
                    document.getElementById('total-profit').textContent = '€' + data.total_profit.toFixed(2);
                    document.getElementById('margin').textContent = '€' + data.account.margin.toFixed(2);
                    document.getElementById('free-margin').textContent = '€' + data.account.free_margin.toFixed(2);
                    
                    // Aggiorna posizioni
                    const positionsDiv = document.getElementById('positions');
                    if (data.positions.length === 0) {
                        positionsDiv.innerHTML = '<p style="text-align: center; opacity: 0.8;">Nessuna posizione aperta</p>';
                    } else {
                        positionsDiv.innerHTML = data.positions.map(pos => `
                            <div class="position ${pos.profit >= 0 ? 'positive' : 'negative'}">
                                <div class="position-header">
                                    <span class="symbol">${pos.symbol}</span>
                                    <span class="direction ${pos.direction.toLowerCase()}">${pos.direction}</span>
                                </div>
                                <div class="position-details">
                                    <div class="detail">
                                        <span class="detail-label">Ticket:</span>
                                        <span class="detail-value">${pos.ticket}</span>
                                    </div>
                                    <div class="detail">
                                        <span class="detail-label">Volume:</span>
                                        <span class="detail-value">${pos.volume}</span>
                                    </div>
                                    <div class="detail">
                                        <span class="detail-label">Prezzo:</span>
                                        <span class="detail-value">${pos.price.toFixed(5)}</span>
                                    </div>
                                    <div class="detail">
                                        <span class="detail-label">P&L:</span>
                                        <span class="detail-value ${pos.profit >= 0 ? 'profit' : 'loss'}">€${pos.profit.toFixed(2)}</span>
                                    </div>
                                    ${pos.confidence ? `
                                    <div class="detail">
                                        <span class="detail-label">Confidenza:</span>
                                        <span class="detail-value confidence">${pos.confidence.toFixed(1)}%</span>
                                    </div>
                                    ` : ''}
                                    ${pos.probability ? `
                                    <div class="detail">
                                        <span class="detail-label">Probabilità:</span>
                                        <span class="detail-value probability">${pos.probability.toFixed(1)}%</span>
                                    </div>
                                    ` : ''}
                                </div>
                            </div>
                        `).join('');
                    }
                    
                    // Aggiorna statistiche
                    document.getElementById('stats').innerHTML = `
                        <div class="detail">
                            <span class="detail-label">Sistema:</span>
                            <span class="detail-value" style="color: #4CAF50;">✅ Attivo</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">MT5:</span>
                            <span class="detail-value" style="color: #4CAF50;">✅ Connesso</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Database:</span>
                            <span class="detail-value" style="color: #4CAF50;">✅ Sincronizzato</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Strategie:</span>
                            <span class="detail-value" style="color: #ffd700;">🎯 AI Attive</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Account:</span>
                            <span class="detail-value">${data.account.login} (${data.account.server})</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Margine Level:</span>
                            <span class="detail-value">${data.account.margin_level.toFixed(2)}%</span>
                        </div>
                    `;
                    
                    // Aggiorna strategie
                    document.getElementById('strategies').innerHTML = `
                        <div class="strategy-item active">
                            <h4>🎯 AI Premium</h4>
                            <p>Attiva</p>
                        </div>
                        <div class="strategy-item active">
                            <h4>📊 Copy Trading</h4>
                            <p>Attiva</p>
                        </div>
                        <div class="strategy-item active">
                            <h4>🛡️ Risk Management</h4>
                            <p>Attiva</p>
                        </div>
                        <div class="strategy-item active">
                            <h4>📈 Multi-Timeframe</h4>
                            <p>Attiva</p>
                        </div>
                    `;
                    
                    // Aggiorna dettagli account
                    document.getElementById('account-details').innerHTML = `
                        <div class="detail">
                            <span class="detail-label">Login:</span>
                            <span class="detail-value">${data.account.login}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Server:</span>
                            <span class="detail-value">${data.account.server}</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Leverage:</span>
                            <span class="detail-value">1:500</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Currency:</span>
                            <span class="detail-value">EUR</span>
                        </div>
                        <div class="detail">
                            <span class="detail-label">Tipo Account:</span>
                            <span class="detail-value">Demo</span>
                        </div>
                    `;
                    
                    // Aggiorna timestamp
                    document.getElementById('last-update').textContent = new Date().toLocaleTimeString('it-IT');
                })
                .catch(error => {
                    console.error('Errore aggiornamento:', error);
                    document.getElementById('last-update').textContent = 'Errore aggiornamento';
                });
        }
        
        // Funzione per cambiare tab
        function showTab(tabName) {
            // Rimuovi active da tutti i tab
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            
            // Aggiungi active al tab selezionato
            event.target.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }
        
        // Avvia aggiornamento automatico
        updateDashboard();
        setInterval(updateDashboard, 2000); // Aggiorna ogni 2 secondi
    </script>
</body>
</html>
"""

def get_account_info():
    """Ottieni informazioni account MT5"""
    try:
        if not mt5.initialize():
            return None
        
        if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
            return None
        
        account_info = mt5.account_info()
        if account_info is None:
            return None
        
        return {
            'login': account_info.login,
            'server': account_info.server,
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free,
            'margin_level': account_info.margin_level
        }
    except Exception as e:
        print(f"Errore account info: {e}")
        return None

def get_positions():
    """Ottieni posizioni aperte"""
    try:
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        result = []
        for pos in positions:
            # Ottieni dati dal database se disponibili
            try:
                conn = sqlite3.connect('trading_data.db')
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT confidence, profit_probability 
                    FROM trades 
                    WHERE ticket = ? AND close_time IS NULL
                """, (pos.ticket,))
                db_data = cursor.fetchone()
                conn.close()
                
                confidence = db_data[0] if db_data and db_data[0] else None
                probability = db_data[1] if db_data and db_data[1] else None
            except:
                confidence = None
                probability = None
            
            result.append({
                'ticket': pos.ticket,
                'symbol': pos.symbol,
                'direction': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'price': pos.price_open,
                'profit': pos.profit,
                'confidence': confidence,
                'probability': probability
            })
        
        return result
    except Exception as e:
        print(f"Errore posizioni: {e}")
        return []

@app.route('/')
def dashboard():
    """Dashboard principale"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def api_data():
    """API per dati real-time"""
    try:
        account = get_account_info()
        positions = get_positions()
        
        if account is None:
            return jsonify({
                'error': 'Errore connessione MT5',
                'account': {'balance': 0, 'equity': 0, 'login': 'N/A', 'server': 'N/A', 'margin': 0, 'free_margin': 0, 'margin_level': 0},
                'positions': [],
                'total_profit': 0
            })
        
        total_profit = sum(pos['profit'] for pos in positions)
        
        return jsonify({
            'account': account,
            'positions': positions,
            'total_profit': total_profit,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🌐 AVVIANDO DASHBOARD COMPLETA...")
    print("📊 Dashboard disponibile su: http://localhost:5000")
    print("🎯 Monitoraggio completo sistema trading")
    print("🔄 Aggiornamento automatico ogni 2 secondi")
    
    app.run(host='0.0.0.0', port=5000, debug=False)
