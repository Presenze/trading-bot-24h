# 🚀 SETUP GITHUB PER RENDER

## 📋 **PROCEDURA RAPIDA**

### **PASSO 1: Crea Account GitHub**
1. Vai su: https://github.com
2. Clicca "Sign up"
3. Inserisci email e password
4. Verifica email

### **PASSO 2: Crea Repository**
1. **GitHub** → "New repository"
2. **Name**: `trading-bot-24h`
3. **Description**: `Trading Bot 24/7`
4. **Public** ✅ (importante!)
5. **Add README** ❌ (non necessario)
6. **Clicca** "Create repository"

### **PASSO 3: Carica File**
1. **Clicca** "uploading an existing file"
2. **Trascina** tutti i file della cartella `Trading_Bot_24h`
3. **Commit message**: `Initial commit - Trading Bot 24/7`
4. **Clicca** "Commit changes"

### **PASSO 4: Connetti a Render**
1. **Torna su Render**
2. **Clicca** "GitHub"
3. **Autorizza** Render ad accedere a GitHub
4. **Seleziona** il repository `trading-bot-24h`
5. **Deploy** automatico!

## 🚀 **ALTERNATIVA: REPOSITORY PUBBLICO**

Se non vuoi creare account, puoi usare questo repository pubblico:
`https://github.com/username/trading-bot-template`

## 📊 **CONFIGURAZIONE RENDER**

### **Build Settings:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python railway_setup.py`
- **Environment**: `Python 3`

### **Environment Variables:**
- `PORT`: `5000`
- `PYTHON_VERSION`: `3.10`

## 🎯 **RISULTATO**

Una volta deployato:
- **URL**: `https://trading-bot-24h.onrender.com`
- **Dashboard**: Accesso web al bot
- **Deploy automatico** ad ogni push su GitHub

## 🚀 **VANTAGGI GITHUB + RENDER**

- ✅ **Deploy automatico** ad ogni modifica
- ✅ **Version control** del codice
- ✅ **Backup automatico** su GitHub
- ✅ **Facile aggiornamento** del bot
- ✅ **Collaborazione** con altri

---
**🚀 GitHub + Render = Deploy automatico!**
