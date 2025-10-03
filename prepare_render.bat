@echo off
title Preparazione File per Render
color 0A

echo.
echo ████████████████████████████████████████████████████████████████████
echo ██                                                                ██
echo ██    🚀 PREPARAZIONE FILE PER RENDER                            ██
echo ██                                                                ██
echo ████████████████████████████████████████████████████████████████████
echo.

echo 📁 Controllo file necessari...

REM Controlla file essenziali
if not exist "Dockerfile" (
    echo ❌ Dockerfile non trovato!
    pause
    exit /b 1
)

if not exist "render.yaml" (
    echo ❌ render.yaml non trovato!
    pause
    exit /b 1
)

if not exist "railway_setup.py" (
    echo ❌ railway_setup.py non trovato!
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo ❌ requirements.txt non trovato!
    pause
    exit /b 1
)

if not exist "config_account.py" (
    echo ❌ config_account.py non trovato!
    pause
    exit /b 1
)

echo ✅ Tutti i file essenziali trovati!

echo.
echo 📦 Creazione archivio per Render...

REM Crea archivio ZIP
powershell -command "Compress-Archive -Path '*.py', '*.txt', '*.yaml', 'Dockerfile', '.dockerignore' -DestinationPath 'trading-bot-render.zip' -Force"

if exist "trading-bot-render.zip" (
    echo ✅ Archivio creato: trading-bot-render.zip
) else (
    echo ❌ Errore creazione archivio!
    pause
    exit /b 1
)

echo.
echo 🚀 FILE PRONTI PER RENDER!
echo.
echo 📋 PROSSIMI PASSI:
echo 1. Vai su https://render.com
echo 2. Crea account gratuito
echo 3. New Web Service
echo 4. Build and deploy from a Dockerfile
echo 5. Carica il file trading-bot-render.zip
echo 6. Deploy!
echo.
echo 📁 File da caricare: trading-bot-render.zip
echo.

pause
