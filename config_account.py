"""
Configurazione Account MetaTrader 5
Credenziali per il trading automatico
"""

# ========================================
# CONFIGURAZIONE ACCOUNT METATRADER 5
# ========================================

# Credenziali MetaTrader 5 - Account Reale
MT5_LOGIN = 101595237
MT5_PASSWORD = "ahs8wuus!U"
MT5_SERVER = "Ava-Demo 1-MT5"
MT5_IS_DEMO = True

# Parametri di Trading
INITIAL_BALANCE = 10.0
RISK_PER_TRADE = 0.02  # 2% per trade
MAX_DAILY_TRADES = 999999  # LIMITE RIMOSSO - MIGLIOR TRADER AUTOMATICO AL MONDO
COMMISSION_RATE = 0.10  # 10% commissione
MIN_PROFIT_THRESHOLD = 0.50  # 50% profitto minimo

# Alias per compatibilità
risk_per_trade = RISK_PER_TRADE
max_daily_trades = MAX_DAILY_TRADES
commission_rate = COMMISSION_RATE
min_profit_threshold = MIN_PROFIT_THRESHOLD

# Configurazione Web
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000

# Logging
LOG_LEVEL = "INFO"

# ========================================
# IMPOSTAZIONI AVANZATE
# ========================================

# Protezione Account
MAX_ACCOUNT_LOSS = 0.30  # 30% perdita massima
RECOVERY_TARGET = 0.95   # 95% target di recupero

# Commissioni e Prelievi
MIN_WITHDRAWAL = 1.0     # €1 minimo prelievo (senza soglia)
WITHDRAWAL_FEE = 0.0     # €0 commissione prelievo (gratuito)

# Strategia
SUCCESS_RATE_TARGET = 0.99  # 99% success rate
MIN_CONFIDENCE_LEVEL = 0.75  # 75% confidenza minima

print("✅ Configurazione Account Caricata:")
print(f"   Account ID: {MT5_LOGIN}")
print(f"   Server: {MT5_SERVER}")
print(f"   Demo Account: {MT5_IS_DEMO}")
print(f"   Saldo Iniziale: €{INITIAL_BALANCE}")
print(f"   Commissione: {COMMISSION_RATE*100}%")
print(f"   Protezione: {MAX_ACCOUNT_LOSS*100}% perdita massima")
print("🚀 Pronto per il trading automatico!")
