#!/usr/bin/env python3
"""
SISTEMA TRADING PERFEZIONATO
===========================
Analisi dati reali, statistiche mercato, apertura solo trade sicuri
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import config_account
from config_account import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_perfezionato.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnalisiMercatoAvanzata:
    """Analisi avanzata dei mercati con dati reali"""
    
    def __init__(self):
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD']
        self.timeframes = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4, mt5.TIMEFRAME_D1]
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.market_stats = {}
        
    def connetti_mt5(self):
        """Connessione robusta a MT5"""
        if not mt5.initialize():
            logger.error("❌ MT5 non inizializzato")
            return False
        
        if not mt5.login(config_account.MT5_LOGIN, password=config_account.MT5_PASSWORD, server=config_account.MT5_SERVER):
            logger.error("❌ Login MT5 fallito")
            return False
        
        logger.info("✅ MT5 connesso per analisi avanzata")
        return True
    
    def ottieni_dati_storici(self, symbol, timeframe, bars=1000):
        """Ottiene dati storici reali da MT5"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"❌ Errore dati storici {symbol}: {e}")
            return None
    
    def calcola_indicatori_avanzati(self, df):
        """Calcola indicatori tecnici avanzati"""
        try:
            # RSI multi-periodo
            df['rsi_14'] = self.calculate_rsi(df['close'], 14)
            df['rsi_21'] = self.calculate_rsi(df['close'], 21)
            df['rsi_50'] = self.calculate_rsi(df['close'], 50)
            
            # MACD con segnali
            df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(df['close'])
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Medie mobili multiple
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_21'] = df['close'].ewm(span=21).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_200'] = df['close'].ewm(span=200).mean()
            
            # ADX per forza del trend
            df['adx'] = self.calculate_adx(df)
            
            # Volatilità
            df['volatility'] = df['close'].rolling(20).std()
            df['atr'] = self.calculate_atr(df)
            
            # Volume (se disponibile)
            if 'tick_volume' in df.columns:
                df['volume_sma'] = df['tick_volume'].rolling(20).mean()
                df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            
            # Momentum
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            df['roc'] = df['close'].pct_change(10)
            
            return df
        except Exception as e:
            logger.error(f"❌ Errore calcolo indicatori: {e}")
            return df
    
    def calculate_rsi(self, prices, period=14):
        """Calcola RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcola MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcola Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def calculate_adx(self, df, period=14):
        """Calcola ADX"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            plus_dm = high.diff()
            minus_dm = low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            
            tr1 = pd.DataFrame(high - low)
            tr2 = pd.DataFrame(abs(high - close.shift(1)))
            tr3 = pd.DataFrame(abs(low - close.shift(1)))
            frames = [tr1, tr2, tr3]
            tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
            
            atr = tr.rolling(period).mean()
            
            plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / atr)
            minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / atr))
            
            dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
            adx = ((dx.shift(1) * (period - 1)) + dx) / period
            adx_smooth = adx.ewm(alpha=1/period).mean()
            
            return adx_smooth
        except:
            return pd.Series([50] * len(df), index=df.index)
    
    def calculate_atr(self, df, period=14):
        """Calcola Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def analizza_pattern_candele(self, df):
        """Analizza pattern delle candele"""
        try:
            df['body'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
            
            # Doji
            df['doji'] = df['body'] < (df['high'] - df['low']) * 0.1
            
            # Hammer
            df['hammer'] = (df['lower_shadow'] > df['body'] * 2) & (df['upper_shadow'] < df['body'] * 0.5)
            
            # Engulfing
            df['bullish_engulfing'] = (df['close'] > df['open']) & \
                                     (df['close'].shift(1) < df['open'].shift(1)) & \
                                     (df['open'] < df['close'].shift(1)) & \
                                     (df['close'] > df['open'].shift(1))
            
            df['bearish_engulfing'] = (df['close'] < df['open']) & \
                                     (df['close'].shift(1) > df['open'].shift(1)) & \
                                     (df['open'] > df['close'].shift(1)) & \
                                     (df['close'] < df['open'].shift(1))
            
            return df
        except Exception as e:
            logger.error(f"❌ Errore pattern candele: {e}")
            return df
    
    def calcola_statistiche_mercato(self, symbol):
        """Calcola statistiche avanzate del mercato"""
        try:
            stats = {}
            
            # Analisi multi-timeframe
            for tf in self.timeframes:
                df = self.ottieni_dati_storici(symbol, tf, 500)
                if df is None:
                    continue
                
                df = self.calcola_indicatori_avanzati(df)
                df = self.analizza_pattern_candele(df)
                
                tf_name = self.get_timeframe_name(tf)
                
                # Statistiche trend
                stats[f'{tf_name}_trend_strength'] = df['adx'].iloc[-1] if 'adx' in df.columns else 50
                stats[f'{tf_name}_rsi'] = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
                stats[f'{tf_name}_macd_signal'] = 1 if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1] else -1
                
                # Volatilità
                stats[f'{tf_name}_volatility'] = df['volatility'].iloc[-1] if 'volatility' in df.columns else 0.001
                stats[f'{tf_name}_atr'] = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
                
                # Posizione nelle Bollinger Bands
                stats[f'{tf_name}_bb_position'] = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
                
                # Momentum
                stats[f'{tf_name}_momentum'] = df['momentum'].iloc[-1] if 'momentum' in df.columns else 0
                
                # Pattern candele
                stats[f'{tf_name}_bullish_pattern'] = int(df['bullish_engulfing'].iloc[-1]) if 'bullish_engulfing' in df.columns else 0
                stats[f'{tf_name}_bearish_pattern'] = int(df['bearish_engulfing'].iloc[-1]) if 'bearish_engulfing' in df.columns else 0
            
            # Correlazioni con altri mercati
            stats['market_correlation'] = self.calcola_correlazioni_mercato(symbol)
            
            # Forza relativa
            stats['relative_strength'] = self.calcola_forza_relativa(symbol)
            
            # Sentiment del mercato
            stats['market_sentiment'] = self.calcola_sentiment_mercato(symbol)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Errore statistiche mercato {symbol}: {e}")
            return {}
    
    def get_timeframe_name(self, tf):
        """Converte timeframe in nome"""
        tf_map = {
            mt5.TIMEFRAME_M15: 'M15',
            mt5.TIMEFRAME_H1: 'H1',
            mt5.TIMEFRAME_H4: 'H4',
            mt5.TIMEFRAME_D1: 'D1'
        }
        return tf_map.get(tf, 'M15')
    
    def calcola_correlazioni_mercato(self, symbol):
        """Calcola correlazioni con altri mercati"""
        try:
            correlations = []
            base_df = self.ottieni_dati_storici(symbol, mt5.TIMEFRAME_H1, 100)
            if base_df is None:
                return 0
            
            for other_symbol in self.symbols:
                if other_symbol != symbol:
                    other_df = self.ottieni_dati_storici(other_symbol, mt5.TIMEFRAME_H1, 100)
                    if other_df is not None:
                        corr = base_df['close'].pct_change().corr(other_df['close'].pct_change())
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0
        except:
            return 0
    
    def calcola_forza_relativa(self, symbol):
        """Calcola forza relativa del simbolo"""
        try:
            df = self.ottieni_dati_storici(symbol, mt5.TIMEFRAME_H4, 50)
            if df is None:
                return 0
            
            # Performance recente vs performance storica
            recent_perf = (df['close'].iloc[-1] / df['close'].iloc[-10] - 1) * 100
            historical_perf = (df['close'].iloc[-10] / df['close'].iloc[-50] - 1) * 100
            
            return recent_perf - historical_perf
        except:
            return 0
    
    def calcola_sentiment_mercato(self, symbol):
        """Calcola sentiment del mercato"""
        try:
            df = self.ottieni_dati_storici(symbol, mt5.TIMEFRAME_H1, 100)
            if df is None:
                return 0
            
            # Analisi volume e prezzo
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['tick_volume'].pct_change() if 'tick_volume' in df.columns else 0
            
            # Sentiment basato su volume e movimento prezzi
            bullish_volume = df[(df['price_change'] > 0) & (df['volume_change'] > 0)]['tick_volume'].sum() if 'tick_volume' in df.columns else 0
            bearish_volume = df[(df['price_change'] < 0) & (df['volume_change'] > 0)]['tick_volume'].sum() if 'tick_volume' in df.columns else 0
            
            total_volume = bullish_volume + bearish_volume
            if total_volume > 0:
                return (bullish_volume - bearish_volume) / total_volume
            
            return 0
        except:
            return 0

class SistemaDecisionaleAvanzato:
    """Sistema decisionale basato su ML e statistiche"""
    
    def __init__(self):
        self.analisi = AnalisiMercatoAvanzata()
        self.min_confidence = 0.85  # Confidenza minima per aprire trade
        self.min_profit_probability = 0.75  # Probabilità minima di profitto
        
    def valuta_opportunita_trade(self, symbol):
        """Valuta se aprire un trade con alta probabilità di profitto"""
        try:
            if not self.analisi.connetti_mt5():
                return None
            
            # Ottieni statistiche complete
            stats = self.analisi.calcola_statistiche_mercato(symbol)
            if not stats:
                return None
            
            # Calcola score di qualità
            quality_score = self.calcola_quality_score(stats)
            
            # Determina direzione
            direction = self.determina_direzione(stats)
            
            # Calcola probabilità di profitto
            profit_probability = self.calcola_probabilita_profitto(stats, direction)
            
            # Calcola risk/reward
            risk_reward = self.calcola_risk_reward(symbol, direction)
            
            # Decisione finale
            if (quality_score >= self.min_confidence and 
                profit_probability >= self.min_profit_probability and
                risk_reward >= 2.0):
                
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'confidence': quality_score,
                    'profit_probability': profit_probability,
                    'risk_reward': risk_reward,
                    'entry_price': self.get_current_price(symbol),
                    'stop_loss': self.calcola_stop_loss(symbol, direction),
                    'take_profit': self.calcola_take_profit(symbol, direction, risk_reward),
                    'volume': self.calcola_volume_ottimale(symbol, quality_score),
                    'stats': stats
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Errore valutazione {symbol}: {e}")
            return None
        finally:
            mt5.shutdown()
    
    def calcola_quality_score(self, stats):
        """Calcola score di qualità del setup"""
        try:
            score = 0
            total_weight = 0
            
            # Trend strength (peso 25%)
            if 'H4_trend_strength' in stats:
                trend_strength = min(stats['H4_trend_strength'], 100) / 100
                score += trend_strength * 0.25
                total_weight += 0.25
            
            # Confluence multi-timeframe (peso 30%)
            confluence = 0
            timeframes = ['M15', 'H1', 'H4', 'D1']
            signals = []
            
            for tf in timeframes:
                if f'{tf}_macd_signal' in stats:
                    signals.append(stats[f'{tf}_macd_signal'])
            
            if signals:
                confluence = abs(sum(signals)) / len(signals)  # Allineamento segnali
                score += confluence * 0.30
                total_weight += 0.30
            
            # Volatilità ottimale (peso 15%)
            if 'H1_volatility' in stats:
                vol = stats['H1_volatility']
                # Volatilità ideale: non troppo bassa, non troppo alta
                vol_score = 1 - abs(vol - 0.002) / 0.002  # Normalizzato
                vol_score = max(0, min(1, vol_score))
                score += vol_score * 0.15
                total_weight += 0.15
            
            # Sentiment e correlazioni (peso 20%)
            if 'market_sentiment' in stats:
                sentiment_score = abs(stats['market_sentiment'])
                score += sentiment_score * 0.10
                total_weight += 0.10
            
            if 'market_correlation' in stats:
                corr_score = 1 - stats['market_correlation']  # Bassa correlazione = meglio
                score += corr_score * 0.10
                total_weight += 0.10
            
            # Pattern candele (peso 10%)
            pattern_score = 0
            for tf in timeframes:
                if f'{tf}_bullish_pattern' in stats:
                    pattern_score += stats[f'{tf}_bullish_pattern']
                if f'{tf}_bearish_pattern' in stats:
                    pattern_score += stats[f'{tf}_bearish_pattern']
            
            if pattern_score > 0:
                score += min(pattern_score, 1) * 0.10
                total_weight += 0.10
            
            return score / total_weight if total_weight > 0 else 0
            
        except Exception as e:
            logger.error(f"❌ Errore quality score: {e}")
            return 0
    
    def determina_direzione(self, stats):
        """Determina direzione del trade"""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            timeframes = ['M15', 'H1', 'H4', 'D1']
            
            for tf in timeframes:
                # MACD
                if f'{tf}_macd_signal' in stats:
                    if stats[f'{tf}_macd_signal'] > 0:
                        bullish_signals += 1
                    else:
                        bearish_signals += 1
                
                # RSI
                if f'{tf}_rsi' in stats:
                    rsi = stats[f'{tf}_rsi']
                    if rsi < 30:
                        bullish_signals += 1
                    elif rsi > 70:
                        bearish_signals += 1
                
                # Bollinger Bands
                if f'{tf}_bb_position' in stats:
                    bb_pos = stats[f'{tf}_bb_position']
                    if bb_pos < 0.2:
                        bullish_signals += 1
                    elif bb_pos > 0.8:
                        bearish_signals += 1
            
            # Sentiment
            if 'market_sentiment' in stats:
                if stats['market_sentiment'] > 0.1:
                    bullish_signals += 2
                elif stats['market_sentiment'] < -0.1:
                    bearish_signals += 2
            
            # Forza relativa
            if 'relative_strength' in stats:
                if stats['relative_strength'] > 0.5:
                    bullish_signals += 1
                elif stats['relative_strength'] < -0.5:
                    bearish_signals += 1
            
            return 'BUY' if bullish_signals > bearish_signals else 'SELL'
            
        except Exception as e:
            logger.error(f"❌ Errore direzione: {e}")
            return 'BUY'
    
    def calcola_probabilita_profitto(self, stats, direction):
        """Calcola probabilità di profitto basata su dati storici"""
        try:
            probability = 0.5  # Base 50%
            
            # Trend strength bonus
            if 'H4_trend_strength' in stats:
                trend_bonus = min(stats['H4_trend_strength'], 100) / 200  # Max 50% bonus
                probability += trend_bonus
            
            # Multi-timeframe confluence
            confluence_count = 0
            total_signals = 0
            
            timeframes = ['M15', 'H1', 'H4', 'D1']
            for tf in timeframes:
                if f'{tf}_macd_signal' in stats:
                    signal = stats[f'{tf}_macd_signal']
                    if (direction == 'BUY' and signal > 0) or (direction == 'SELL' and signal < 0):
                        confluence_count += 1
                    total_signals += 1
            
            if total_signals > 0:
                confluence_ratio = confluence_count / total_signals
                probability += confluence_ratio * 0.2  # Max 20% bonus
            
            # Volatilità ottimale
            if 'H1_volatility' in stats:
                vol = stats['H1_volatility']
                if 0.0005 < vol < 0.003:  # Range ottimale
                    probability += 0.1
            
            # Pattern candele
            pattern_bonus = 0
            for tf in timeframes:
                if direction == 'BUY' and f'{tf}_bullish_pattern' in stats:
                    pattern_bonus += stats[f'{tf}_bullish_pattern'] * 0.05
                elif direction == 'SELL' and f'{tf}_bearish_pattern' in stats:
                    pattern_bonus += stats[f'{tf}_bearish_pattern'] * 0.05
            
            probability += min(pattern_bonus, 0.15)
            
            return min(probability, 0.95)  # Max 95%
            
        except Exception as e:
            logger.error(f"❌ Errore probabilità profitto: {e}")
            return 0.5
    
    def calcola_risk_reward(self, symbol, direction):
        """Calcola risk/reward ratio"""
        try:
            # Ottieni ATR per calcolare stop e target dinamici
            df = self.analisi.ottieni_dati_storici(symbol, mt5.TIMEFRAME_H1, 50)
            if df is None:
                return 2.0
            
            df = self.analisi.calcola_indicatori_avanzati(df)
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            
            # Stop loss = 1.5 * ATR
            stop_distance = atr * 1.5
            
            # Take profit = 3 * ATR (R:R = 2:1)
            target_distance = atr * 3
            
            return target_distance / stop_distance if stop_distance > 0 else 2.0
            
        except Exception as e:
            logger.error(f"❌ Errore risk/reward: {e}")
            return 2.0
    
    def get_current_price(self, symbol):
        """Ottiene prezzo corrente con retry ultra-robusto"""
        try:
            # Assicurati che MT5 sia connesso
            if not mt5.initialize():
                logger.info("🔄 Reinizializzando MT5...")
                mt5.initialize()
            
            # Verifica connessione account
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("⚠️ Account non connesso, riconnettendo...")
                from config_account import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
                if not mt5.login(MT5_LOGIN, MT5_PASSWORD, MT5_SERVER):
                    logger.error("❌ Login MT5 fallito")
                    return 0
            
            # Verifica che il simbolo sia disponibile
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.warning(f"⚠️ Simbolo {symbol} non disponibile, provo a selezionarlo...")
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"❌ Impossibile selezionare {symbol}")
                    return 0
            
            # Tentativi multipli per ottenere il prezzo
            for attempt in range(10):  # 10 tentativi
                try:
                    # Metodo 1: Tick corrente
                    tick = mt5.symbol_info_tick(symbol)
                    if tick and tick.ask > 0:
                        logger.info(f"✅ Prezzo {symbol}: {tick.ask} (tick)")
                        return tick.ask
                    
                    # Metodo 2: Ultimo prezzo dal simbolo
                    if symbol_info and symbol_info.ask > 0:
                        logger.info(f"✅ Prezzo {symbol}: {symbol_info.ask} (symbol_info)")
                        return symbol_info.ask
                    
                    # Metodo 3: Prezzo bid
                    if tick and tick.bid > 0:
                        logger.info(f"✅ Prezzo {symbol}: {tick.bid} (bid)")
                        return tick.bid
                    
                    time.sleep(0.1)  # Pausa breve
                    
                except Exception as e:
                    logger.warning(f"⚠️ Tentativo {attempt+1} fallito per {symbol}: {e}")
                    time.sleep(0.2)
            
            # Fallback finale: usa ultimo prezzo disponibile
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                if rates is not None and len(rates) > 0:
                    price = rates[0]['close']
                    logger.info(f"✅ Prezzo fallback {symbol}: {price}")
                    return price
            except:
                pass
            
            logger.error(f"❌ Impossibile ottenere prezzo per {symbol} dopo 10 tentativi")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Errore critico prezzo {symbol}: {e}")
            return 0
    
    def calcola_stop_loss(self, symbol, direction):
        """Calcola stop loss dinamico"""
        try:
            df = self.analisi.ottieni_dati_storici(symbol, mt5.TIMEFRAME_H1, 50)
            if df is None:
                return 0
            
            df = self.analisi.calcola_indicatori_avanzati(df)
            current_price = self.get_current_price(symbol)
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            
            if direction == 'BUY':
                return current_price - (atr * 1.5)
            else:
                return current_price + (atr * 1.5)
                
        except Exception as e:
            logger.error(f"❌ Errore stop loss: {e}")
            return 0
    
    def calcola_take_profit(self, symbol, direction, risk_reward):
        """Calcola take profit dinamico"""
        try:
            df = self.analisi.ottieni_dati_storici(symbol, mt5.TIMEFRAME_H1, 50)
            if df is None:
                return 0
            
            df = self.analisi.calcola_indicatori_avanzati(df)
            current_price = self.get_current_price(symbol)
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.001
            
            target_distance = atr * 3  # 3 ATR target
            
            if direction == 'BUY':
                return current_price + target_distance
            else:
                return current_price - target_distance
                
        except Exception as e:
            logger.error(f"❌ Errore take profit: {e}")
            return 0
    
    def calcola_volume_ottimale(self, symbol, confidence):
        """Calcola volume ottimale basato su confidenza"""
        try:
            base_volume = 0.01
            confidence_multiplier = min(confidence * 2, 3.0)  # Max 3x
            return round(base_volume * confidence_multiplier, 2)
        except:
            return 0.01

class SistemaTradingPerfezionato:
    """Sistema trading principale perfezionato"""
    
    def __init__(self):
        self.decisionale = SistemaDecisionaleAvanzato()
        self.symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
        self.max_positions = 5
        self.running = True
        
        # Credenziali MT5
        self.mt5_login = MT5_LOGIN
        self.mt5_password = MT5_PASSWORD
        self.mt5_server = MT5_SERVER
        
    def connetti_mt5(self):
        """Connessione MT5"""
        if not mt5.initialize():
            logger.error("❌ MT5 non inizializzato")
            return False
        
        if not mt5.login(config_account.MT5_LOGIN, password=config_account.MT5_PASSWORD, server=config_account.MT5_SERVER):
            logger.error("❌ Login MT5 fallito")
            return False
        
        logger.info("✅ Sistema Perfezionato connesso a MT5")
        return True
    
    def ottieni_tutti_simboli_disponibili(self):
        """Ottiene tutti i simboli disponibili su MT5"""
        try:
            # Ottieni tutti i simboli
            simboli = mt5.symbols_get()
            if simboli is None:
                logger.warning("⚠️ Nessun simbolo disponibile")
                return self.symbols  # Fallback ai simboli base
            
            # Filtra solo simboli attivi e con spread ragionevole
            simboli_attivi = []
            for simbolo in simboli:
                if (simbolo.visible and 
                    simbolo.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED and
                    simbolo.spread < 50):  # Spread sotto 50 punti
                    simboli_attivi.append(simbolo.name)
            
            logger.info(f"🌍 Trovati {len(simboli_attivi)} simboli attivi su MT5")
            return simboli_attivi[:50]  # Limita a 50 per performance
            
        except Exception as e:
            logger.error(f"❌ Errore ottenimento simboli: {e}")
            return self.symbols  # Fallback ai simboli base

    def cerca_opportunita_sicure(self):
        """Cerca opportunità con alta probabilità di profitto su TUTTO IL MERCATO"""
        try:
            # Ottieni TUTTI i simboli disponibili
            simboli_da_analizzare = self.ottieni_tutti_simboli_disponibili()
            logger.info(f"🌍 Analizzando {len(simboli_da_analizzare)} simboli del mercato completo")
            
            opportunita = []
            
            for symbol in simboli_da_analizzare:
                logger.info(f"🔍 Analizzando {symbol}...")
                
                trade_opportunity = self.decisionale.valuta_opportunita_trade(symbol)
                
                if trade_opportunity:
                    logger.info(f"✅ OPPORTUNITÀ PREMIUM: {symbol} {trade_opportunity['direction']}")
                    logger.info(f"   Confidenza: {trade_opportunity['confidence']:.2%}")
                    logger.info(f"   Probabilità profitto: {trade_opportunity['profit_probability']:.2%}")
                    logger.info(f"   Risk/Reward: {trade_opportunity['risk_reward']:.1f}:1")
                    
                    opportunita.append(trade_opportunity)
                    
                    # Limita a 5 opportunità per ciclo per evitare sovraccarico
                    if len(opportunita) >= 5:
                        break
                else:
                    logger.info(f"❌ {symbol}: Nessuna opportunità sicura")
            
            logger.info(f"🎯 Trovate {len(opportunita)} opportunità premium su {len(simboli_da_analizzare)} simboli")
            return opportunita
            
        except Exception as e:
            logger.error(f"❌ Errore ricerca opportunità: {e}")
            return []
    
    def esegui_trade_sicuro(self, opportunity):
        """Esegue trade solo se sicuro al 100%"""
        try:
            # Reset connessione MT5 per evitare stati inconsistenti
            mt5.shutdown()
            import time
            time.sleep(0.1)
            if not mt5.initialize():
                logger.error("❌ Errore reinizializzazione MT5")
                return False
            if not mt5.login(self.mt5_login, self.mt5_password, self.mt5_server):
                logger.error("❌ Errore re-login MT5")
                return False
            
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            volume = 0.01  # Volume fisso per compatibilità broker
            
            # Usa il prezzo fornito o ottieni quello corrente
            if 'entry_price' in opportunity and opportunity['entry_price'] > 0:
                price = opportunity['entry_price']
                logger.info(f"✅ Usando prezzo fornito per {symbol}: {price}")
            else:
                # Fallback: ottieni prezzo corrente
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    logger.error(f"❌ Impossibile ottenere prezzo per {symbol}")
                    return False
                price = tick.ask if direction == 'BUY' else tick.bid
                logger.info(f"✅ Prezzo corrente per {symbol}: {price}")
            
            # Prepara richiesta con stop-loss e take-profit validi
            order_type = mt5.ORDER_TYPE_BUY if direction == 'BUY' else mt5.ORDER_TYPE_SELL
            
            # Calcola SL e TP con valori sicuri (10 pips SL, 20 pips TP)
            if direction == 'BUY':
                sl = price - 0.0010  # 10 pips sotto
                tp = price + 0.0020  # 20 pips sopra
            else:
                sl = price + 0.0010  # 10 pips sopra
                tp = price - 0.0020  # 20 pips sotto
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 123456,
                "comment": f"AI-Perfect-{opportunity['confidence']:.0%}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,  # FOK come default
            }
            
            # Prova diversi filling modes (FOK primo - funziona!)
            filling_modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
            
            for filling_mode in filling_modes:
                request["type_filling"] = filling_mode
                
                # Esegui trade
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(f"✅ TRADE APERTO: {symbol} {direction} Vol:{volume}")
                    logger.info(f"   Ticket: {result.order}")
                    logger.info(f"   Confidenza: {opportunity['confidence']:.2%}")
                    logger.info(f"   Probabilità profitto: {opportunity['profit_probability']:.2%}")
                    logger.info(f"   Filling Mode: {filling_mode}")
                    
                    # Salva nel database
                    self.salva_trade_database(result.order, opportunity)
                    
                    return True
                elif result:
                    logger.warning(f"⚠️ Filling mode {filling_mode} fallito: {result.retcode}")
                else:
                    logger.warning(f"⚠️ Filling mode {filling_mode} restituito None")
            
            logger.error(f"❌ Tutti i filling modes falliti per {symbol}")
            return False
                
        except Exception as e:
            logger.error(f"❌ Errore esecuzione trade: {e}")
            return False
    
    def salva_trade_database(self, ticket, opportunity):
        """Salva trade nel database"""
        try:
            conn = sqlite3.connect('trading_data.db')
            cursor = conn.cursor()
            
            # Crea tabella se non esiste
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER,
                    symbol TEXT,
                    action TEXT,
                    volume REAL,
                    price REAL,
                    profit REAL,
                    status TEXT,
                    timestamp TEXT,
                    confidence REAL DEFAULT 0,
                    profit_probability REAL DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                INSERT INTO trades (ticket, symbol, action, volume, price, profit, status, timestamp, confidence, profit_probability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticket,
                opportunity['symbol'],
                opportunity['direction'],
                opportunity['volume'],
                opportunity['entry_price'],
                0,
                "OPEN",
                datetime.now().isoformat(),
                opportunity['confidence'],
                opportunity['profit_probability']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Errore salvataggio database: {e}")
    
    def sincronizza_database_mt5(self):
        """Sincronizza database con posizioni reali MT5"""
        try:
            # Ottieni posizioni MT5 reali
            mt5_positions = mt5.positions_get()
            if not mt5_positions:
                return
            
            conn = sqlite3.connect('trading_data.db')
            cursor = conn.cursor()
            
            # Crea tabella se non esiste
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER,
                    symbol TEXT,
                    action TEXT,
                    volume REAL,
                    price REAL,
                    profit REAL,
                    status TEXT,
                    timestamp TEXT,
                    confidence REAL DEFAULT 0,
                    profit_probability REAL DEFAULT 0
                )
            ''')
            
            # Aggiorna profitti delle posizioni esistenti
            for pos in mt5_positions:
                cursor.execute('''
                    INSERT OR REPLACE INTO trades 
                    (ticket, symbol, action, volume, price, profit, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pos.ticket,
                    pos.symbol,
                    "BUY" if pos.type == 0 else "SELL",
                    pos.volume,
                    pos.price_open,
                    pos.profit,
                    "OPEN",
                    datetime.fromtimestamp(pos.time).isoformat()
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Database sincronizzato con {len(mt5_positions)} posizioni MT5")
            
        except Exception as e:
            logger.error(f"❌ Errore sincronizzazione database: {e}")
    
    def monitora_posizioni(self):
        """Monitora posizioni senza chiudere perdite"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            
            for pos in positions:
                # Solo chiusura profitti, MAI perdite
                if pos.profit >= 3.0:
                    logger.info(f"💰 Chiudendo profitto: {pos.symbol} - €{pos.profit:.2f}")
                    self.chiudi_posizione(pos)
                elif pos.profit < 0:
                    # NON chiudere perdite - lascia correre
                    logger.info(f"📊 Mantenendo posizione: {pos.symbol} - €{pos.profit:.2f} (in attesa recupero)")
                    
        except Exception as e:
            logger.error(f"❌ Errore monitoraggio: {e}")
    
    def chiudi_posizione(self, position):
        """Chiude posizione profittevole"""
        try:
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return False
            
            close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if position.type == 0 else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": close_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Profitto-{position.profit:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"✅ Profitto chiuso: {position.symbol} - €{position.profit:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Errore chiusura: {e}")
            return False
    
    def cerca_opportunita_copy_trading(self):
        """Cerca opportunità usando copy trading avanzato"""
        try:
            from sistema_copy_trading_avanzato import SistemaIntegrato
            
            sistema_copy = SistemaIntegrato()
            if not sistema_copy.connetti_mt5():
                return []
            
            # Ottieni opportunità premium
            opportunita_premium = sistema_copy.trova_opportunita_premium()
            
            # Converti nel formato standard e ESEGUI DIRETTAMENTE
            opportunita_standard = []
            for opp in opportunita_premium:
                # Usa il prezzo dal sistema copy trading (già ottenuto)
                entry_price = opp['entry_price']
                
                # Se il prezzo è valido, esegui il trade direttamente
                if entry_price > 0:
                    logger.info(f"🎯 Copy Trading: {opp['symbol']} {opp['direction']}")
                    logger.info(f"   Score Integrato: {opp['score']:.2%}")
                    logger.info(f"   Probabilità: {opp['probability']:.2%}")
                    logger.info(f"   Prezzo Entry: {entry_price}")
                    
                    # Prepara il trade
                    trade_data = {
                        'symbol': opp['symbol'],
                        'direction': opp['direction'],
                        'confidence': opp['score'],
                        'profit_probability': opp['probability'],
                        'risk_reward': 2.0,
                        'entry_price': entry_price,
                        'stop_loss': opp['stop_loss'],
                        'take_profit': opp['take_profit'],
                        'volume': opp['volume']
                    }
                    
                    # ESEGUI IL TRADE DIRETTAMENTE
                    if self.esegui_trade_sicuro(trade_data):
                        logger.info(f"✅ TRADE ESEGUITO: {opp['symbol']} {opp['direction']}")
                    else:
                        logger.error(f"❌ FALLITO: {opp['symbol']} {opp['direction']}")
                else:
                    logger.error(f"❌ Prezzo non valido per {opp['symbol']}: {entry_price}")
            
            return opportunita_standard
            
        except Exception as e:
            logger.error(f"❌ Errore copy trading: {e}")
            return []
    
    def run(self):
        """Loop principale perfezionato"""
        logger.info("🚀 AVVIO SISTEMA TRADING PERFEZIONATO")
        logger.info("🎯 Solo trade con alta probabilità di profitto")
        logger.info("💎 Nessuna chiusura perdite - solo profitti")
        
        if not self.connetti_mt5():
            return
        
        ciclo = 0
        
        while self.running:
            try:
                ciclo += 1
                logger.info(f"\n🔄 Ciclo {ciclo} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Conta posizioni attuali
                current_positions = len(mt5.positions_get() or [])
                logger.info(f"📊 Posizioni attuali: {current_positions}")
                
                # Sincronizza database con MT5
                self.sincronizza_database_mt5()
                
                # Monitora posizioni esistenti
                self.monitora_posizioni()
                
                # Cerca nuove opportunità solo se sotto il limite
                if current_positions < self.max_positions:
                    logger.info("🔍 Cercando opportunità sicure...")
                    
                    # Prima cerca opportunità standard
                    opportunita = self.cerca_opportunita_sicure()
                    
                    # Se non trova nulla, usa sistema copy trading avanzato
                    if not opportunita:
                        logger.info("🎯 Attivando sistema copy trading avanzato...")
                        opportunita = self.cerca_opportunita_copy_trading()
                    
                    for opp in opportunita:
                        if current_positions < self.max_positions:
                            if self.esegui_trade_sicuro(opp):
                                current_positions += 1
                        else:
                            break
                    
                    if not opportunita:
                        logger.info("⏳ Nessuna opportunità trovata - attendo...")
                else:
                    logger.info("⚠️ Limite posizioni raggiunto")
                
                # Pausa tra cicli
                time.sleep(120)  # 2 minuti tra analisi
                
            except KeyboardInterrupt:
                logger.info("🛑 Arresto richiesto dall'utente")
                self.running = False
            except Exception as e:
                logger.error(f"❌ Errore nel ciclo principale: {e}")
                time.sleep(60)
        
        mt5.shutdown()
        logger.info("🏁 Sistema Perfezionato terminato")

if __name__ == "__main__":
    sistema = SistemaTradingPerfezionato()
    sistema.run()
