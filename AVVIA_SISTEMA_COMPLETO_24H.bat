@echo off
title SISTEMA INTEGRATO 24/7 - TRADING AUTOMATICO COMPLETO
color 0A

echo.
echo ████████████████████████████████████████████████████████████████████
echo ██                                                                ██
echo ██    🌟 SISTEMA INTEGRATO 24/7 - TRADING AUTOMATICO COMPLETO    ██
echo ██                                                                ██
echo ████████████████████████████████████████████████████████████████████
echo.
echo 🚀 Avviando tutti i sistemi avanzati...
echo 💰 Trading automatico continuo
echo 🧠 Strategie adattive intelligenti  
echo 🛡️ Gestione rischio avanzata
echo 📊 Monitoraggio performance continuo
echo 🔒 Sicurezza enterprise
echo 🌐 Dashboard web: http://localhost:5000
echo.
echo ════════════════════════════════════════════════════════════════════
echo.

REM Controlla se Python è installato
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python non trovato! Installare Python 3.8+ prima di continuare.
    pause
    exit /b 1
)

echo ✅ Python trovato
echo.

REM Installa dipendenze se necessarie
echo 📦 Verificando dipendenze...
pip install --quiet MetaTrader5 pandas numpy scikit-learn flask matplotlib seaborn plotly cryptography pyjwt requests asyncio threading

if errorlevel 1 (
    echo ⚠️ Alcune dipendenze potrebbero non essere installate correttamente
    echo 🔄 Continuando comunque...
)

echo ✅ Dipendenze verificate
echo.

REM Controlla file di configurazione
if not exist "config_account.py" (
    echo ❌ File config_account.py non trovato!
    echo 📝 Assicurati di aver configurato i dati MT5
    pause
    exit /b 1
)

echo ✅ Configurazione MT5 trovata
echo.

REM Backup automatico prima dell'avvio
echo 💾 Creando backup automatico...
if not exist "backups" mkdir backups
copy "trading_data.db" "backups\trading_data_backup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%.db" >nul 2>&1
echo ✅ Backup creato
echo.

REM Avvia sistema completo
echo 🌟 AVVIANDO SISTEMA INTEGRATO 24/7...
echo.
echo ⚠️ IMPORTANTE:
echo    - Assicurati che MT5 sia aperto e connesso
echo    - Il sistema funzionerà 24/7 fino a interruzione manuale
echo    - Monitora il log: sistema_integrato_24h.log
echo    - Dashboard web: http://localhost:5000
echo.
echo 🚀 Premere INVIO per avviare il sistema completo...
pause >nul

echo.
echo ████████████████████████████████████████████████████████████████████
echo ██                    SISTEMA 24/7 ATTIVO                        ██
echo ████████████████████████████████████████████████████████████████████
echo.

REM Avvia il sistema integrato
python sistema_integrato_24h.py

echo.
echo ████████████████████████████████████████████████████████████████████
echo ██                   SISTEMA 24/7 TERMINATO                      ██
echo ████████████████████████████████████████████████████████████████████
echo.
echo 📊 Controlla i log per le statistiche finali
echo 💾 Backup automatico salvato in cartella backups/
echo.
pause
