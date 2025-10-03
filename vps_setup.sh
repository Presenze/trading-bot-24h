#!/bin/bash
# 🚀 TRADING BOT 24/7 - SETUP VPS GRATUITO
# Script per configurazione su VPS gratuito

echo "🚀 Configurazione Trading Bot 24/7 su VPS..."

# Aggiorna sistema
sudo apt update && sudo apt upgrade -y

# Installa Python e dipendenze
sudo apt install python3 python3-pip git -y

# Installa dipendenze Python
pip3 install -r requirements.txt

# Crea servizio systemd per avvio automatico
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Trading Bot 24/7
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/python3 sistema_integrato_24h.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Abilita e avvia servizio
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

echo "✅ Trading Bot configurato e avviato!"
echo "📊 Dashboard: http://$(curl -s ifconfig.me):5000"
echo "🔧 Controlla status: sudo systemctl status trading-bot"
