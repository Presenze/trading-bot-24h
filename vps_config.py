#!/usr/bin/env python3
"""
🚀 TRADING BOT 24/7 - VPS CONFIGURATION
Configurazione ottimizzata per VPS gratuiti
"""

import os
import sys
import logging
from datetime import datetime

# Configurazione logging per VPS
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/trading-bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configurazione VPS
VPS_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 5000)),
    'debug': False,
    'auto_restart': True,
    'log_level': 'INFO'
}

# Configurazione MT5 per VPS
MT5_CONFIG = {
    'login': 101595237,
    'password': 'ahs8wuus!U',
    'server': 'Ava-Demo 1-MT5',
    'timeout': 10000,
    'retry_attempts': 3
}

# Configurazione trading per VPS
TRADING_CONFIG = {
    'max_positions': 5,
    'risk_per_trade': 0.02,
    'stop_loss_pips': 50,
    'take_profit_pips': 100,
    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD'],
    'timeframes': ['H1', 'H4', 'D1']
}

def setup_vps_environment():
    """Configura ambiente VPS"""
    try:
        # Crea directory per logs
        os.makedirs('/var/log', exist_ok=True)
        
        # Crea directory per dati
        os.makedirs('/opt/trading-bot/data', exist_ok=True)
        
        # Configura permessi
        os.chmod('/var/log/trading-bot.log', 0o644)
        
        logger.info("✅ Ambiente VPS configurato")
        return True
    except Exception as e:
        logger.error(f"❌ Errore configurazione VPS: {e}")
        return False

def check_vps_resources():
    """Controlla risorse VPS"""
    try:
        import psutil
        
        # Controlla CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Controlla RAM
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Controlla disco
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        
        logger.info(f"📊 Risorse VPS - CPU: {cpu_percent}%, RAM: {memory_percent}%, Disco: {disk_percent}%")
        
        # Avvisi se risorse basse
        if cpu_percent > 80:
            logger.warning("⚠️ CPU usage alto!")
        if memory_percent > 80:
            logger.warning("⚠️ RAM usage alto!")
        if disk_percent > 80:
            logger.warning("⚠️ Disco usage alto!")
        
        return True
    except ImportError:
        logger.warning("⚠️ psutil non installato, monitoraggio risorse disabilitato")
        return True
    except Exception as e:
        logger.error(f"❌ Errore controllo risorse: {e}")
        return False

def get_vps_info():
    """Ottieni informazioni VPS"""
    try:
        import platform
        import socket
        
        info = {
            'hostname': socket.gethostname(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'uptime': datetime.now().isoformat(),
            'public_ip': None
        }
        
        # Prova a ottenere IP pubblico
        try:
            import requests
            response = requests.get('https://ifconfig.me', timeout=5)
            info['public_ip'] = response.text.strip()
        except:
            pass
        
        return info
    except Exception as e:
        logger.error(f"❌ Errore info VPS: {e}")
        return {}

if __name__ == "__main__":
    print("🚀 Trading Bot VPS Configuration")
    print("=" * 50)
    
    # Setup ambiente
    if setup_vps_environment():
        print("✅ Ambiente VPS configurato")
    
    # Controlla risorse
    if check_vps_resources():
        print("✅ Risorse VPS controllate")
    
    # Mostra info VPS
    info = get_vps_info()
    if info:
        print(f"🖥️ Hostname: {info.get('hostname', 'N/A')}")
        print(f"🌐 IP Pubblico: {info.get('public_ip', 'N/A')}")
        print(f"🐍 Python: {info.get('python_version', 'N/A')}")
    
    print("=" * 50)
    print("🚀 VPS pronto per Trading Bot!")
