#!/bin/bash
# 🚀 TRADING BOT 24/7 - ORACLE CLOUD SETUP
# VPS sempre gratuito con 2 VM

echo "🚀 Configurazione Trading Bot su Oracle Cloud..."

# Aggiorna sistema
sudo apt update && sudo apt upgrade -y

# Installa Python e dipendenze
sudo apt install python3 python3-pip git curl wget -y

# Installa dipendenze Python
pip3 install -r requirements.txt

# Crea directory per il bot
mkdir -p /opt/trading-bot
cp *.py /opt/trading-bot/
cp *.txt /opt/trading-bot/
cp *.json /opt/trading-bot/
cd /opt/trading-bot

# Crea servizio systemd per avvio automatico
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Trading Bot 24/7
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/trading-bot
ExecStart=/usr/bin/python3 railway_setup.py
Restart=always
RestartSec=10
Environment=PORT=5000

[Install]
WantedBy=multi-user.target
EOF

# Abilita e avvia servizio
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot

# Configura firewall
sudo ufw allow 5000/tcp
sudo ufw allow ssh
sudo ufw --force enable

# Ottieni IP pubblico
PUBLIC_IP=$(curl -s ifconfig.me)

echo "✅ Trading Bot configurato e avviato!"
echo "📊 Dashboard: http://$PUBLIC_IP:5000"
echo "🔧 Controlla status: sudo systemctl status trading-bot"
echo "📋 Logs: sudo journalctl -u trading-bot -f"
