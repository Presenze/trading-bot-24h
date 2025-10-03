#!/usr/bin/env python3
"""
MOTORE DI STRATEGIA ADATTIVA
- Cambia strategia automaticamente in base alle condizioni di mercato
- Riconosce trend, range, alta/bassa volatilità
- Adatta parametri in tempo reale
- Machine Learning per ottimizzazione continua
"""

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCondition(Enum):
    """Condizioni di mercato"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class StrategyType(Enum):
    """Tipi di strategia"""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"
    GRID_TRADING = "grid_trading"
    MOMENTUM = "momentum"
    CONTRARIAN = "contrarian"

@dataclass
class StrategyParameters:
    """Parametri per ogni strategia"""
    strategy_type: StrategyType
    entry_threshold: float
    exit_threshold: float
    stop_loss_pips: int
    take_profit_pips: int
    position_size_multiplier: float
    max_positions: int
    timeframe_minutes: int
    indicators_config: Dict[str, Any]
    risk_multiplier: float

@dataclass
class MarketAnalysis:
    """Analisi delle condizioni di mercato"""
    condition: MarketCondition
    trend_strength: float
    volatility_level: float
    momentum: float
    support_resistance_levels: List[float]
    confidence: float
    timestamp: datetime

class AdaptiveStrategyEngine:
    """
    Motore di strategia adattiva che cambia approccio in base al mercato
    """
    
    def __init__(self, db_path: str = 'trading_data.db'):
        self.db_path = db_path
        self.logger = logger
        
        # Strategia attualmente attiva
        self.current_strategy = StrategyType.TREND_FOLLOWING
        self.current_parameters = None
        
        # Modelli ML
        self.market_classifier = None
        self.strategy_optimizer = None
        self.scaler = StandardScaler()
        
        # Configurazioni strategia
        self.strategy_configs = self.initialize_strategy_configs()
        
        # Performance tracking per strategia
        self.strategy_performance = {}
        
        # Market analysis cache
        self.market_analysis_cache = {}
        
        # Setup database
        self.setup_database()
        
        # Carica o inizializza modelli ML
        self.initialize_ml_models()
        
        self.logger.info("Motore strategia adattiva inizializzato")
    
    def setup_database(self):
        """Setup database per tracking adattivo"""
        try:
            conn = sqlite3.connect('adaptive_strategy.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_conditions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    condition TEXT,
                    trend_strength REAL,
                    volatility_level REAL,
                    momentum REAL,
                    confidence REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_switches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    from_strategy TEXT,
                    to_strategy TEXT,
                    reason TEXT,
                    market_condition TEXT,
                    expected_improvement REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    strategy_type TEXT,
                    symbol TEXT,
                    trades_count INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    total_profit REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    market_condition TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore setup database adattivo: {e}")
    
    def initialize_strategy_configs(self) -> Dict[StrategyType, StrategyParameters]:
        """Inizializza configurazioni per ogni strategia"""
        configs = {}
        
        # Trend Following Strategy
        configs[StrategyType.TREND_FOLLOWING] = StrategyParameters(
            strategy_type=StrategyType.TREND_FOLLOWING,
            entry_threshold=0.7,
            exit_threshold=0.3,
            stop_loss_pips=50,
            take_profit_pips=100,
            position_size_multiplier=1.0,
            max_positions=3,
            timeframe_minutes=60,
            indicators_config={
                'ema_fast': 12,
                'ema_slow': 26,
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            risk_multiplier=1.0
        )
        
        # Mean Reversion Strategy
        configs[StrategyType.MEAN_REVERSION] = StrategyParameters(
            strategy_type=StrategyType.MEAN_REVERSION,
            entry_threshold=0.8,
            exit_threshold=0.5,
            stop_loss_pips=30,
            take_profit_pips=60,
            position_size_multiplier=1.2,
            max_positions=5,
            timeframe_minutes=15,
            indicators_config={
                'bb_period': 20,
                'bb_std': 2.0,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            },
            risk_multiplier=0.8
        )
        
        # Breakout Strategy
        configs[StrategyType.BREAKOUT] = StrategyParameters(
            strategy_type=StrategyType.BREAKOUT,
            entry_threshold=0.6,
            exit_threshold=0.4,
            stop_loss_pips=40,
            take_profit_pips=120,
            position_size_multiplier=1.5,
            max_positions=2,
            timeframe_minutes=30,
            indicators_config={
                'bb_period': 20,
                'bb_std': 2.0,
                'volume_ma': 20,
                'atr_period': 14
            },
            risk_multiplier=1.2
        )
        
        # Scalping Strategy
        configs[StrategyType.SCALPING] = StrategyParameters(
            strategy_type=StrategyType.SCALPING,
            entry_threshold=0.5,
            exit_threshold=0.3,
            stop_loss_pips=15,
            take_profit_pips=25,
            position_size_multiplier=0.8,
            max_positions=8,
            timeframe_minutes=5,
            indicators_config={
                'ema_fast': 5,
                'ema_slow': 13,
                'rsi_period': 7,
                'stoch_k': 5,
                'stoch_d': 3
            },
            risk_multiplier=0.6
        )
        
        # Grid Trading Strategy
        configs[StrategyType.GRID_TRADING] = StrategyParameters(
            strategy_type=StrategyType.GRID_TRADING,
            entry_threshold=0.4,
            exit_threshold=0.2,
            stop_loss_pips=100,
            take_profit_pips=50,
            position_size_multiplier=0.5,
            max_positions=10,
            timeframe_minutes=60,
            indicators_config={
                'grid_spacing': 20,  # pips
                'grid_levels': 5,
                'bb_period': 20,
                'bb_std': 1.5
            },
            risk_multiplier=0.4
        )
        
        # Momentum Strategy
        configs[StrategyType.MOMENTUM] = StrategyParameters(
            strategy_type=StrategyType.MOMENTUM,
            entry_threshold=0.75,
            exit_threshold=0.4,
            stop_loss_pips=45,
            take_profit_pips=90,
            position_size_multiplier=1.1,
            max_positions=4,
            timeframe_minutes=30,
            indicators_config={
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'momentum_period': 10,
                'volume_ma': 20
            },
            risk_multiplier=0.9
        )
        
        # Contrarian Strategy
        configs[StrategyType.CONTRARIAN] = StrategyParameters(
            strategy_type=StrategyType.CONTRARIAN,
            entry_threshold=0.8,
            exit_threshold=0.5,
            stop_loss_pips=35,
            take_profit_pips=70,
            position_size_multiplier=1.0,
            max_positions=3,
            timeframe_minutes=60,
            indicators_config={
                'rsi_period': 14,
                'rsi_extreme_oversold': 20,
                'rsi_extreme_overbought': 80,
                'bb_period': 20,
                'bb_std': 2.5
            },
            risk_multiplier=0.7
        )
        
        return configs
    
    def analyze_market_conditions(self, symbol: str) -> MarketAnalysis:
        """Analizza le condizioni di mercato correnti"""
        try:
            # Controlla cache
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if cache_key in self.market_analysis_cache:
                return self.market_analysis_cache[cache_key]
            
            # Ottieni dati di mercato
            df = self.get_market_data(symbol, mt5.TIMEFRAME_M15, 200)
            if df.empty:
                return self.get_default_market_analysis()
            
            # Calcola indicatori per analisi
            indicators = self.calculate_market_indicators(df)
            
            # Determina condizione di mercato
            condition = self.classify_market_condition(indicators)
            
            # Calcola metriche
            trend_strength = self.calculate_trend_strength(indicators)
            volatility_level = self.calculate_volatility_level(indicators)
            momentum = self.calculate_momentum(indicators)
            support_resistance = self.find_support_resistance_levels(df)
            confidence = self.calculate_analysis_confidence(indicators)
            
            analysis = MarketAnalysis(
                condition=condition,
                trend_strength=trend_strength,
                volatility_level=volatility_level,
                momentum=momentum,
                support_resistance_levels=support_resistance,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
            # Salva in cache
            self.market_analysis_cache[cache_key] = analysis
            
            # Salva nel database
            self.save_market_analysis(symbol, analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Errore nell'analisi condizioni mercato: {e}")
            return self.get_default_market_analysis()
    
    def get_market_data(self, symbol: str, timeframe: int, count: int) -> pd.DataFrame:
        """Ottiene dati di mercato"""
        try:
            if not mt5.initialize():
                return pd.DataFrame()
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None:
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Errore nel recuperare dati mercato: {e}")
            return pd.DataFrame()
    
    def calculate_market_indicators(self, df: pd.DataFrame) -> Dict:
        """Calcola indicatori per analisi mercato"""
        try:
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = df['close'].rolling(20).mean()
            indicators['sma_50'] = df['close'].rolling(50).mean()
            indicators['ema_12'] = df['close'].ewm(span=12).mean()
            indicators['ema_26'] = df['close'].ewm(span=26).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            indicators['macd'] = indicators['ema_12'] - indicators['ema_26']
            indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
            indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
            
            # Bollinger Bands
            bb_middle = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            indicators['bb_upper'] = bb_middle + (bb_std * 2)
            indicators['bb_lower'] = bb_middle - (bb_std * 2)
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / bb_middle
            
            # ATR (Average True Range)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            indicators['atr'] = true_range.rolling(14).mean()
            
            # Volatilità
            indicators['volatility'] = df['close'].pct_change().rolling(20).std()
            
            # Volume (se disponibile)
            if 'tick_volume' in df.columns:
                indicators['volume_ma'] = df['tick_volume'].rolling(20).mean()
                indicators['volume_ratio'] = df['tick_volume'] / indicators['volume_ma']
            
            # Momentum
            indicators['momentum'] = df['close'] / df['close'].shift(10) - 1
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo indicatori: {e}")
            return {}
    
    def classify_market_condition(self, indicators: Dict) -> MarketCondition:
        """Classifica le condizioni di mercato"""
        try:
            if not indicators:
                return MarketCondition.RANGING
            
            # Ottieni valori più recenti
            current_values = {}
            for key, series in indicators.items():
                if hasattr(series, 'iloc') and len(series) > 0:
                    current_values[key] = series.iloc[-1]
                else:
                    current_values[key] = 0
            
            # Logica di classificazione
            
            # Controlla trend
            sma_20 = current_values.get('sma_20', 0)
            sma_50 = current_values.get('sma_50', 0)
            ema_12 = current_values.get('ema_12', 0)
            ema_26 = current_values.get('ema_26', 0)
            
            # Trend strength
            if sma_20 > sma_50 and ema_12 > ema_26:
                trend_direction = 1  # Uptrend
            elif sma_20 < sma_50 and ema_12 < ema_26:
                trend_direction = -1  # Downtrend
            else:
                trend_direction = 0  # Sideways
            
            # Volatilità
            volatility = current_values.get('volatility', 0)
            atr = current_values.get('atr', 0)
            bb_width = current_values.get('bb_width', 0)
            
            high_volatility = volatility > 0.02 or bb_width > 0.05
            
            # MACD per momentum
            macd = current_values.get('macd', 0)
            macd_signal = current_values.get('macd_signal', 0)
            macd_histogram = current_values.get('macd_histogram', 0)
            
            # RSI per overbought/oversold
            rsi = current_values.get('rsi', 50)
            
            # Momentum
            momentum = current_values.get('momentum', 0)
            
            # Classificazione
            if high_volatility:
                if abs(momentum) > 0.02:
                    return MarketCondition.BREAKOUT
                else:
                    return MarketCondition.HIGH_VOLATILITY
            
            if trend_direction == 1 and macd > macd_signal and momentum > 0.01:
                return MarketCondition.TRENDING_UP
            elif trend_direction == -1 and macd < macd_signal and momentum < -0.01:
                return MarketCondition.TRENDING_DOWN
            
            if (rsi > 70 and momentum < 0) or (rsi < 30 and momentum > 0):
                return MarketCondition.REVERSAL
            
            if volatility < 0.01 and bb_width < 0.02:
                return MarketCondition.LOW_VOLATILITY
            
            return MarketCondition.RANGING
            
        except Exception as e:
            self.logger.error(f"Errore nella classificazione mercato: {e}")
            return MarketCondition.RANGING
    
    def calculate_trend_strength(self, indicators: Dict) -> float:
        """Calcola forza del trend (0-1)"""
        try:
            if not indicators:
                return 0.5
            
            # Ottieni valori recenti
            sma_20 = indicators.get('sma_20', pd.Series()).iloc[-1] if 'sma_20' in indicators else 0
            sma_50 = indicators.get('sma_50', pd.Series()).iloc[-1] if 'sma_50' in indicators else 0
            ema_12 = indicators.get('ema_12', pd.Series()).iloc[-1] if 'ema_12' in indicators else 0
            ema_26 = indicators.get('ema_26', pd.Series()).iloc[-1] if 'ema_26' in indicators else 0
            macd = indicators.get('macd', pd.Series()).iloc[-1] if 'macd' in indicators else 0
            macd_signal = indicators.get('macd_signal', pd.Series()).iloc[-1] if 'macd_signal' in indicators else 0
            
            # Calcola score trend
            score = 0
            total_indicators = 0
            
            # MA alignment
            if sma_20 != 0 and sma_50 != 0:
                if sma_20 > sma_50:
                    score += 1
                total_indicators += 1
            
            # EMA alignment
            if ema_12 != 0 and ema_26 != 0:
                if ema_12 > ema_26:
                    score += 1
                total_indicators += 1
            
            # MACD
            if macd != 0 and macd_signal != 0:
                if macd > macd_signal:
                    score += 1
                total_indicators += 1
            
            if total_indicators == 0:
                return 0.5
            
            strength = score / total_indicators
            return strength
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo trend strength: {e}")
            return 0.5
    
    def calculate_volatility_level(self, indicators: Dict) -> float:
        """Calcola livello di volatilità (0-1)"""
        try:
            if not indicators:
                return 0.5
            
            volatility = indicators.get('volatility', pd.Series()).iloc[-1] if 'volatility' in indicators else 0
            bb_width = indicators.get('bb_width', pd.Series()).iloc[-1] if 'bb_width' in indicators else 0
            atr = indicators.get('atr', pd.Series()).iloc[-1] if 'atr' in indicators else 0
            
            # Normalizza volatilità (valori tipici per forex)
            vol_score = min(volatility / 0.03, 1.0) if volatility > 0 else 0
            bb_score = min(bb_width / 0.08, 1.0) if bb_width > 0 else 0
            atr_score = min(atr / 0.002, 1.0) if atr > 0 else 0
            
            # Media pesata
            total_score = (vol_score * 0.4 + bb_score * 0.4 + atr_score * 0.2)
            return min(total_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo volatility level: {e}")
            return 0.5
    
    def calculate_momentum(self, indicators: Dict) -> float:
        """Calcola momentum (-1 to 1)"""
        try:
            if not indicators:
                return 0.0
            
            momentum = indicators.get('momentum', pd.Series()).iloc[-1] if 'momentum' in indicators else 0
            macd_histogram = indicators.get('macd_histogram', pd.Series()).iloc[-1] if 'macd_histogram' in indicators else 0
            rsi = indicators.get('rsi', pd.Series()).iloc[-1] if 'rsi' in indicators else 50
            
            # Normalizza momentum
            mom_score = np.tanh(momentum * 50)  # Tanh per limitare a [-1, 1]
            
            # MACD histogram contribution
            macd_score = np.tanh(macd_histogram * 1000) if macd_histogram != 0 else 0
            
            # RSI contribution
            rsi_score = (rsi - 50) / 50  # Normalizza RSI a [-1, 1]
            
            # Media pesata
            total_momentum = (mom_score * 0.5 + macd_score * 0.3 + rsi_score * 0.2)
            return np.clip(total_momentum, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo momentum: {e}")
            return 0.0
    
    def find_support_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Trova livelli di supporto e resistenza"""
        try:
            if df.empty:
                return []
            
            # Usa ultimi 50 periodi
            recent_df = df.tail(50)
            
            # Trova pivot points
            highs = recent_df['high'].rolling(window=5, center=True).max()
            lows = recent_df['low'].rolling(window=5, center=True).min()
            
            # Identifica pivot highs e lows
            pivot_highs = recent_df[recent_df['high'] == highs]['high'].values
            pivot_lows = recent_df[recent_df['low'] == lows]['low'].values
            
            # Combina e rimuovi duplicati
            levels = np.concatenate([pivot_highs, pivot_lows])
            levels = np.unique(np.round(levels, 5))
            
            # Ordina e prendi i più significativi
            levels = sorted(levels)[-10:]  # Ultimi 10 livelli
            
            return levels.tolist()
            
        except Exception as e:
            self.logger.error(f"Errore nel trovare supporti/resistenze: {e}")
            return []
    
    def calculate_analysis_confidence(self, indicators: Dict) -> float:
        """Calcola confidenza dell'analisi (0-1)"""
        try:
            if not indicators:
                return 0.5
            
            # Controlla completezza dati
            required_indicators = ['sma_20', 'sma_50', 'rsi', 'macd', 'volatility']
            available_indicators = sum(1 for ind in required_indicators if ind in indicators and len(indicators[ind]) > 0)
            
            completeness_score = available_indicators / len(required_indicators)
            
            # Controlla consistenza segnali
            consistency_score = 0.5  # Base score
            
            # Se abbiamo abbastanza dati, calcola consistenza
            if available_indicators >= 3:
                # Logica semplificata per consistenza
                consistency_score = 0.8
            
            # Combina scores
            confidence = (completeness_score * 0.6 + consistency_score * 0.4)
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo confidence: {e}")
            return 0.5
    
    def get_default_market_analysis(self) -> MarketAnalysis:
        """Restituisce analisi di default"""
        return MarketAnalysis(
            condition=MarketCondition.RANGING,
            trend_strength=0.5,
            volatility_level=0.5,
            momentum=0.0,
            support_resistance_levels=[],
            confidence=0.3,
            timestamp=datetime.now()
        )
    
    def select_optimal_strategy(self, market_analysis: MarketAnalysis, symbol: str) -> StrategyType:
        """Seleziona la strategia ottimale per le condizioni di mercato"""
        try:
            condition = market_analysis.condition
            volatility = market_analysis.volatility_level
            trend_strength = market_analysis.trend_strength
            momentum = abs(market_analysis.momentum)
            
            # Logica di selezione strategia
            if condition in [MarketCondition.TRENDING_UP, MarketCondition.TRENDING_DOWN]:
                if trend_strength > 0.7:
                    return StrategyType.TREND_FOLLOWING
                elif momentum > 0.5:
                    return StrategyType.MOMENTUM
                else:
                    return StrategyType.TREND_FOLLOWING
            
            elif condition == MarketCondition.RANGING:
                if volatility < 0.3:
                    return StrategyType.GRID_TRADING
                else:
                    return StrategyType.MEAN_REVERSION
            
            elif condition == MarketCondition.HIGH_VOLATILITY:
                if momentum > 0.6:
                    return StrategyType.BREAKOUT
                else:
                    return StrategyType.SCALPING
            
            elif condition == MarketCondition.LOW_VOLATILITY:
                return StrategyType.GRID_TRADING
            
            elif condition == MarketCondition.BREAKOUT:
                return StrategyType.BREAKOUT
            
            elif condition == MarketCondition.REVERSAL:
                return StrategyType.CONTRARIAN
            
            # Default
            return StrategyType.TREND_FOLLOWING
            
        except Exception as e:
            self.logger.error(f"Errore nella selezione strategia: {e}")
            return StrategyType.TREND_FOLLOWING
    
    def should_switch_strategy(self, current_strategy: StrategyType, 
                             optimal_strategy: StrategyType,
                             market_analysis: MarketAnalysis) -> Tuple[bool, str]:
        """Determina se cambiare strategia"""
        try:
            if current_strategy == optimal_strategy:
                return False, "Strategia già ottimale"
            
            # Controlla performance strategia corrente
            current_performance = self.get_strategy_performance(current_strategy)
            
            # Soglie per cambio strategia
            min_confidence = 0.7
            min_performance_degradation = 0.2
            
            # Controlla confidence dell'analisi
            if market_analysis.confidence < min_confidence:
                return False, f"Confidence troppo bassa: {market_analysis.confidence:.2f}"
            
            # Controlla se performance attuale è degradata
            if current_performance.get('win_rate', 50) < 40:
                return True, f"Performance degradata: Win Rate {current_performance.get('win_rate', 0):.1f}%"
            
            # Controlla se nuova strategia è significativamente migliore
            optimal_performance = self.get_strategy_performance(optimal_strategy)
            
            current_score = self.calculate_strategy_score(current_performance)
            optimal_score = self.calculate_strategy_score(optimal_performance)
            
            if optimal_score > current_score + min_performance_degradation:
                return True, f"Strategia migliore disponibile: Score {optimal_score:.2f} vs {current_score:.2f}"
            
            return False, "Nessun miglioramento significativo previsto"
            
        except Exception as e:
            self.logger.error(f"Errore nel controllo switch strategia: {e}")
            return False, f"Errore: {e}"
    
    def get_strategy_performance(self, strategy_type: StrategyType) -> Dict:
        """Ottiene performance di una strategia"""
        try:
            # Controlla cache
            if strategy_type in self.strategy_performance:
                return self.strategy_performance[strategy_type]
            
            # Calcola da database
            conn = sqlite3.connect('adaptive_strategy.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(win_rate), AVG(profit_factor), AVG(total_profit), 
                       AVG(max_drawdown), AVG(sharpe_ratio), COUNT(*)
                FROM strategy_performance 
                WHERE strategy_type = ? AND timestamp > datetime('now', '-30 days')
            ''', (strategy_type.value,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result and result[5] > 0:  # Se ci sono dati
                performance = {
                    'win_rate': result[0] or 50,
                    'profit_factor': result[1] or 1.0,
                    'total_profit': result[2] or 0,
                    'max_drawdown': result[3] or 0,
                    'sharpe_ratio': result[4] or 0,
                    'sample_size': result[5]
                }
            else:
                # Performance di default basata su tipo strategia
                performance = self.get_default_strategy_performance(strategy_type)
            
            # Salva in cache
            self.strategy_performance[strategy_type] = performance
            return performance
            
        except Exception as e:
            self.logger.error(f"Errore nel recuperare performance strategia: {e}")
            return self.get_default_strategy_performance(strategy_type)
    
    def get_default_strategy_performance(self, strategy_type: StrategyType) -> Dict:
        """Performance di default per strategia"""
        defaults = {
            StrategyType.TREND_FOLLOWING: {'win_rate': 55, 'profit_factor': 1.3, 'total_profit': 0, 'max_drawdown': 15, 'sharpe_ratio': 0.8, 'sample_size': 0},
            StrategyType.MEAN_REVERSION: {'win_rate': 65, 'profit_factor': 1.2, 'total_profit': 0, 'max_drawdown': 12, 'sharpe_ratio': 0.9, 'sample_size': 0},
            StrategyType.BREAKOUT: {'win_rate': 45, 'profit_factor': 1.8, 'total_profit': 0, 'max_drawdown': 20, 'sharpe_ratio': 1.1, 'sample_size': 0},
            StrategyType.SCALPING: {'win_rate': 70, 'profit_factor': 1.1, 'total_profit': 0, 'max_drawdown': 8, 'sharpe_ratio': 0.7, 'sample_size': 0},
            StrategyType.GRID_TRADING: {'win_rate': 80, 'profit_factor': 1.05, 'total_profit': 0, 'max_drawdown': 25, 'sharpe_ratio': 0.6, 'sample_size': 0},
            StrategyType.MOMENTUM: {'win_rate': 50, 'profit_factor': 1.4, 'total_profit': 0, 'max_drawdown': 18, 'sharpe_ratio': 0.9, 'sample_size': 0},
            StrategyType.CONTRARIAN: {'win_rate': 60, 'profit_factor': 1.25, 'total_profit': 0, 'max_drawdown': 14, 'sharpe_ratio': 0.8, 'sample_size': 0}
        }
        
        return defaults.get(strategy_type, defaults[StrategyType.TREND_FOLLOWING])
    
    def calculate_strategy_score(self, performance: Dict) -> float:
        """Calcola score complessivo strategia"""
        try:
            win_rate = performance.get('win_rate', 50)
            profit_factor = performance.get('profit_factor', 1.0)
            max_drawdown = performance.get('max_drawdown', 20)
            sharpe_ratio = performance.get('sharpe_ratio', 0)
            
            # Normalizza metriche (0-1)
            win_rate_score = min(win_rate / 80, 1.0)  # 80% = perfect
            profit_factor_score = min(profit_factor / 2.0, 1.0)  # 2.0 = perfect
            drawdown_score = max(0, 1.0 - max_drawdown / 30)  # 30% = worst
            sharpe_score = min(sharpe_ratio / 2.0, 1.0)  # 2.0 = perfect
            
            # Media pesata
            score = (win_rate_score * 0.3 + 
                    profit_factor_score * 0.3 + 
                    drawdown_score * 0.2 + 
                    sharpe_score * 0.2)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo score strategia: {e}")
            return 0.5
    
    def switch_strategy(self, new_strategy: StrategyType, reason: str, 
                       market_condition: MarketCondition):
        """Cambia strategia attiva"""
        try:
            old_strategy = self.current_strategy
            
            # Aggiorna strategia corrente
            self.current_strategy = new_strategy
            self.current_parameters = self.strategy_configs[new_strategy]
            
            # Log del cambio
            self.logger.info(f"🔄 STRATEGIA CAMBIATA: {old_strategy.value} → {new_strategy.value}")
            self.logger.info(f"   Motivo: {reason}")
            self.logger.info(f"   Condizione mercato: {market_condition.value}")
            
            # Salva nel database
            self.save_strategy_switch(old_strategy, new_strategy, reason, market_condition)
            
            # Pulisci cache performance
            self.strategy_performance.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore nel cambio strategia: {e}")
            return False
    
    def adapt_strategy_parameters(self, market_analysis: MarketAnalysis) -> StrategyParameters:
        """Adatta parametri strategia alle condizioni di mercato"""
        try:
            base_params = self.strategy_configs[self.current_strategy]
            adapted_params = StrategyParameters(**base_params.__dict__)
            
            volatility = market_analysis.volatility_level
            trend_strength = market_analysis.trend_strength
            momentum = abs(market_analysis.momentum)
            
            # Adatta stop loss e take profit basato su volatilità
            if volatility > 0.7:  # Alta volatilità
                adapted_params.stop_loss_pips = int(base_params.stop_loss_pips * 1.5)
                adapted_params.take_profit_pips = int(base_params.take_profit_pips * 1.3)
            elif volatility < 0.3:  # Bassa volatilità
                adapted_params.stop_loss_pips = int(base_params.stop_loss_pips * 0.7)
                adapted_params.take_profit_pips = int(base_params.take_profit_pips * 0.8)
            
            # Adatta position size basato su trend strength
            if trend_strength > 0.8:
                adapted_params.position_size_multiplier = base_params.position_size_multiplier * 1.2
            elif trend_strength < 0.3:
                adapted_params.position_size_multiplier = base_params.position_size_multiplier * 0.8
            
            # Adatta soglie entry/exit basato su momentum
            if momentum > 0.6:
                adapted_params.entry_threshold = base_params.entry_threshold * 0.9  # Più aggressivo
            elif momentum < 0.2:
                adapted_params.entry_threshold = base_params.entry_threshold * 1.1  # Più conservativo
            
            # Adatta max positions basato su condizioni
            if market_analysis.condition == MarketCondition.HIGH_VOLATILITY:
                adapted_params.max_positions = max(1, int(base_params.max_positions * 0.5))
            elif market_analysis.condition == MarketCondition.LOW_VOLATILITY:
                adapted_params.max_positions = int(base_params.max_positions * 1.5)
            
            return adapted_params
            
        except Exception as e:
            self.logger.error(f"Errore nell'adattamento parametri: {e}")
            return self.strategy_configs[self.current_strategy]
    
    def run_adaptive_cycle(self, symbol: str) -> Dict:
        """Esegue ciclo completo di adattamento strategia"""
        try:
            # 1. Analizza condizioni mercato
            market_analysis = self.analyze_market_conditions(symbol)
            
            # 2. Seleziona strategia ottimale
            optimal_strategy = self.select_optimal_strategy(market_analysis, symbol)
            
            # 3. Controlla se cambiare strategia
            should_switch, switch_reason = self.should_switch_strategy(
                self.current_strategy, optimal_strategy, market_analysis
            )
            
            # 4. Cambia strategia se necessario
            if should_switch:
                self.switch_strategy(optimal_strategy, switch_reason, market_analysis.condition)
            
            # 5. Adatta parametri alle condizioni correnti
            adapted_params = self.adapt_strategy_parameters(market_analysis)
            self.current_parameters = adapted_params
            
            # 6. Prepara risultato
            result = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'market_condition': market_analysis.condition.value,
                'trend_strength': market_analysis.trend_strength,
                'volatility_level': market_analysis.volatility_level,
                'momentum': market_analysis.momentum,
                'confidence': market_analysis.confidence,
                'current_strategy': self.current_strategy.value,
                'strategy_switched': should_switch,
                'switch_reason': switch_reason if should_switch else None,
                'adapted_parameters': {
                    'entry_threshold': adapted_params.entry_threshold,
                    'stop_loss_pips': adapted_params.stop_loss_pips,
                    'take_profit_pips': adapted_params.take_profit_pips,
                    'position_size_multiplier': adapted_params.position_size_multiplier,
                    'max_positions': adapted_params.max_positions
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nel ciclo adattivo: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def save_market_analysis(self, symbol: str, analysis: MarketAnalysis):
        """Salva analisi mercato nel database"""
        try:
            conn = sqlite3.connect('adaptive_strategy.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_conditions (
                    timestamp, symbol, condition, trend_strength, 
                    volatility_level, momentum, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis.timestamp, symbol, analysis.condition.value,
                analysis.trend_strength, analysis.volatility_level,
                analysis.momentum, analysis.confidence
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore nel salvare analisi mercato: {e}")
    
    def save_strategy_switch(self, from_strategy: StrategyType, to_strategy: StrategyType,
                           reason: str, market_condition: MarketCondition):
        """Salva cambio strategia nel database"""
        try:
            conn = sqlite3.connect('adaptive_strategy.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO strategy_switches (
                    timestamp, from_strategy, to_strategy, reason, market_condition
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(), from_strategy.value, to_strategy.value,
                reason, market_condition.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore nel salvare switch strategia: {e}")
    
    def initialize_ml_models(self):
        """Inizializza modelli Machine Learning"""
        try:
            # Percorsi modelli
            model_dir = Path("ml_models")
            model_dir.mkdir(exist_ok=True)
            
            classifier_path = model_dir / "market_classifier.joblib"
            scaler_path = model_dir / "scaler.joblib"
            
            # Carica modelli esistenti o crea nuovi
            if classifier_path.exists():
                self.market_classifier = joblib.load(classifier_path)
                self.scaler = joblib.load(scaler_path)
                self.logger.info("Modelli ML caricati")
            else:
                # Crea e addestra nuovi modelli
                self.train_ml_models()
                
        except Exception as e:
            self.logger.error(f"Errore nell'inizializzazione modelli ML: {e}")
    
    def train_ml_models(self):
        """Addestra modelli Machine Learning"""
        try:
            # Per ora, crea modelli base
            # In produzione, useresti dati storici per training
            self.market_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Dati di esempio per training (sostituire con dati reali)
            X_sample = np.random.rand(1000, 10)  # 10 features
            y_sample = np.random.randint(0, len(MarketCondition), 1000)
            
            # Addestra
            X_scaled = self.scaler.fit_transform(X_sample)
            self.market_classifier.fit(X_scaled, y_sample)
            
            # Salva modelli
            model_dir = Path("ml_models")
            joblib.dump(self.market_classifier, model_dir / "market_classifier.joblib")
            joblib.dump(self.scaler, model_dir / "scaler.joblib")
            
            self.logger.info("Modelli ML addestrati e salvati")
            
        except Exception as e:
            self.logger.error(f"Errore nell'addestramento modelli ML: {e}")
    
    def get_current_strategy_info(self) -> Dict:
        """Ottiene informazioni strategia corrente"""
        return {
            'strategy_type': self.current_strategy.value,
            'parameters': self.current_parameters.__dict__ if self.current_parameters else None,
            'performance': self.get_strategy_performance(self.current_strategy)
        }

def main():
    """Test del motore strategia adattiva"""
    print("🧠 MOTORE DI STRATEGIA ADATTIVA")
    print("=" * 50)
    
    # Inizializza engine
    engine = AdaptiveStrategyEngine()
    
    # Test su simbolo
    symbol = "EURUSD"
    
    print(f"🔍 Analizzando condizioni mercato per {symbol}...")
    
    # Esegui ciclo adattivo
    result = engine.run_adaptive_cycle(symbol)
    
    if 'error' not in result:
        print(f"\n📊 ANALISI MERCATO:")
        print(f"   Condizione: {result['market_condition']}")
        print(f"   Trend Strength: {result['trend_strength']:.2f}")
        print(f"   Volatilità: {result['volatility_level']:.2f}")
        print(f"   Momentum: {result['momentum']:.2f}")
        print(f"   Confidence: {result['confidence']:.2f}")
        
        print(f"\n🎯 STRATEGIA ATTIVA:")
        print(f"   Tipo: {result['current_strategy']}")
        print(f"   Cambiata: {'Sì' if result['strategy_switched'] else 'No'}")
        if result['strategy_switched']:
            print(f"   Motivo: {result['switch_reason']}")
        
        print(f"\n⚙️  PARAMETRI ADATTATI:")
        params = result['adapted_parameters']
        print(f"   Entry Threshold: {params['entry_threshold']:.2f}")
        print(f"   Stop Loss: {params['stop_loss_pips']} pips")
        print(f"   Take Profit: {params['take_profit_pips']} pips")
        print(f"   Position Size Multiplier: {params['position_size_multiplier']:.2f}")
        print(f"   Max Positions: {params['max_positions']}")
        
        # Mostra info strategia corrente
        strategy_info = engine.get_current_strategy_info()
        performance = strategy_info['performance']
        
        print(f"\n📈 PERFORMANCE STRATEGIA:")
        print(f"   Win Rate: {performance['win_rate']:.1f}%")
        print(f"   Profit Factor: {performance['profit_factor']:.2f}")
        print(f"   Max Drawdown: {performance['max_drawdown']:.1f}%")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        
    else:
        print(f"❌ Errore: {result['error']}")
    
    print("\n✅ Test motore adattivo completato!")

if __name__ == "__main__":
    main()
