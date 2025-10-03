@echo off
title Trading Bot 24/7 - Avvio Semplificato
color 0A

echo.
echo ████████████████████████████████████████████████████████████████████
echo ██                                                                ██
echo ██    🚀 TRADING BOT 24/7 - AVVIO SEMPLIFICATO                  ██
echo ██                                                                ██
echo ████████████████████████████████████████████████████████████████████
echo.

echo 🔧 Installando dipendenze...
pip install -r requirements.txt

echo.
echo 🚀 Avviando Trading Bot 24/7...
echo 📊 Dashboard: http://localhost:5000
echo.

python sistema_integrato_24h.py

echo.
echo ⚠️ Sistema terminato. Premi un tasto per chiudere...
pause >nul
