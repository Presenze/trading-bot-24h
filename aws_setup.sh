#!/bin/bash
# 🚀 TRADING BOT 24/7 - AWS SETUP
# 12 mesi gratuiti

echo "🚀 Configurazione Trading Bot su AWS..."

# Aggiorna sistema
sudo yum update -y

# Installa Python e dipendenze
sudo yum install python3 python3-pip git curl wget -y

# Installa dipendenze Python
pip3 install -r requirements.txt

# Crea directory per il bot
sudo mkdir -p /opt/trading-bot
sudo cp *.py /opt/trading-bot/
sudo cp *.txt /opt/trading-bot/
sudo cp *.json /opt/trading-bot/
cd /opt/trading-bot

# Crea servizio systemd
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Trading Bot 24/7
After=network.target

[Service]
Type=simple
User=ec2-user
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
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload

# Ottieni IP pubblico
PUBLIC_IP=$(curl -s ifconfig.me)

echo "✅ Trading Bot configurato su AWS!"
echo "📊 Dashboard: http://$PUBLIC_IP:5000"
echo "💰 Tier gratuito: 12 mesi"
