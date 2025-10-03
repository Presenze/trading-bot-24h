#!/usr/bin/env python3
"""
GESTIONE DEL RISCHIO AVANZATA E DINAMICA
- Stop-loss dinamici basati su volatilità e ATR
- Position sizing adattivo con Kelly Criterion
- Correlazione tra asset per diversificazione
- Protezione account multi-livello
- Risk parity e volatility targeting
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
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Livelli di rischio"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"

class ProtectionLevel(Enum):
    """Livelli di protezione account"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class RiskParameters:
    """Parametri di rischio dinamici"""
    max_risk_per_trade: float
    max_portfolio_risk: float
    max_correlation_exposure: float
    volatility_target: float
    max_drawdown_limit: float
    position_size_multiplier: float
    stop_loss_multiplier: float
    take_profit_multiplier: float
    max_positions: int
    risk_level: RiskLevel

@dataclass
class PositionRisk:
    """Rischio di una singola posizione"""
    symbol: str
    position_size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    risk_amount: float
    risk_percentage: float
    volatility: float
    correlation_risk: float

class EnhancedRiskManager:
    """
    Sistema di gestione del rischio avanzato e dinamico
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.logger = logger
        
        # Parametri di rischio per livello
        self.risk_configs = self.initialize_risk_configs()
        self.current_risk_level = RiskLevel.MODERATE
        self.current_protection_level = ProtectionLevel.NORMAL
        
        # Tracking delle performance
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.consecutive_losses = 0
        self.total_trades = 0
        self.winning_trades = 0
        
        # Correlazioni tra asset
        self.correlation_matrix = {}
        self.correlation_cache_time = None
        
        # Volatilità target
        self.volatility_target = 0.15  # 15% annuale
        self.volatility_lookback = 20   # giorni
        
        # Setup database
        self.setup_database()
        
        self.logger.info("Enhanced Risk Manager inizializzato")
    
    def setup_database(self):
        """Setup database per risk management"""
        try:
            conn = sqlite3.connect('enhanced_risk.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    event_type TEXT,
                    risk_level TEXT,
                    protection_level TEXT,
                    description TEXT,
                    action_taken TEXT,
                    impact REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_risks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    position_size REAL,
                    risk_amount REAL,
                    risk_percentage REAL,
                    volatility REAL,
                    correlation_risk REAL,
                    stop_loss REAL,
                    take_profit REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    total_risk REAL,
                    portfolio_volatility REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    var_95 REAL,
                    expected_shortfall REAL,
                    risk_adjusted_return REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore setup database risk: {e}")
    
    def initialize_risk_configs(self) -> Dict[RiskLevel, RiskParameters]:
        """Inizializza configurazioni di rischio"""
        configs = {}
        
        # Conservative
        configs[RiskLevel.CONSERVATIVE] = RiskParameters(
            max_risk_per_trade=0.005,  # 0.5%
            max_portfolio_risk=0.02,   # 2%
            max_correlation_exposure=0.3,
            volatility_target=0.10,
            max_drawdown_limit=0.05,   # 5%
            position_size_multiplier=0.5,
            stop_loss_multiplier=1.5,
            take_profit_multiplier=0.8,
            max_positions=3,
            risk_level=RiskLevel.CONSERVATIVE
        )
        
        # Moderate
        configs[RiskLevel.MODERATE] = RiskParameters(
            max_risk_per_trade=0.01,   # 1%
            max_portfolio_risk=0.05,   # 5%
            max_correlation_exposure=0.5,
            volatility_target=0.15,
            max_drawdown_limit=0.10,   # 10%
            position_size_multiplier=1.0,
            stop_loss_multiplier=1.0,
            take_profit_multiplier=1.0,
            max_positions=5,
            risk_level=RiskLevel.MODERATE
        )
        
        # Aggressive
        configs[RiskLevel.AGGRESSIVE] = RiskParameters(
            max_risk_per_trade=0.02,   # 2%
            max_portfolio_risk=0.10,   # 10%
            max_correlation_exposure=0.7,
            volatility_target=0.20,
            max_drawdown_limit=0.15,   # 15%
            position_size_multiplier=1.5,
            stop_loss_multiplier=0.8,
            take_profit_multiplier=1.2,
            max_positions=8,
            risk_level=RiskLevel.AGGRESSIVE
        )
        
        # Ultra Aggressive
        configs[RiskLevel.ULTRA_AGGRESSIVE] = RiskParameters(
            max_risk_per_trade=0.03,   # 3%
            max_portfolio_risk=0.15,   # 15%
            max_correlation_exposure=0.8,
            volatility_target=0.25,
            max_drawdown_limit=0.20,   # 20%
            position_size_multiplier=2.0,
            stop_loss_multiplier=0.6,
            take_profit_multiplier=1.5,
            max_positions=12,
            risk_level=RiskLevel.ULTRA_AGGRESSIVE
        )
        
        return configs
    
    def calculate_dynamic_position_size(self, symbol: str, entry_price: float, 
                                      stop_loss: float, confidence: float = 0.7) -> float:
        """Calcola position size dinamico"""
        try:
            # Ottieni parametri di rischio correnti
            risk_params = self.risk_configs[self.current_risk_level]
            
            # 1. Kelly Criterion
            kelly_size = self.calculate_kelly_position_size(symbol)
            
            # 2. Volatility-based sizing
            volatility_size = self.calculate_volatility_based_size(symbol, entry_price)
            
            # 3. Risk-based sizing
            risk_amount = self.current_balance * risk_params.max_risk_per_trade
            stop_loss_distance = abs(entry_price - stop_loss)
            
            if stop_loss_distance > 0:
                risk_based_size = risk_amount / stop_loss_distance
            else:
                risk_based_size = 0
            
            # 4. Correlation adjustment
            correlation_multiplier = self.calculate_correlation_adjustment(symbol)
            
            # 5. Confidence adjustment
            confidence_multiplier = confidence
            
            # 6. Volatility targeting
            vol_target_multiplier = self.calculate_volatility_target_adjustment(symbol)
            
            # Combina tutti i fattori
            base_size = min(kelly_size, volatility_size, risk_based_size)
            
            final_size = (base_size * 
                         risk_params.position_size_multiplier *
                         correlation_multiplier *
                         confidence_multiplier *
                         vol_target_multiplier)
            
            # Applica limiti
            max_position_value = self.current_balance * 0.1  # Max 10% del balance
            max_size_by_value = max_position_value / entry_price
            
            final_size = min(final_size, max_size_by_value)
            final_size = max(final_size, 0.01)  # Minimum size
            
            # Log del calcolo
            self.logger.info(f"Position size calculation for {symbol}:")
            self.logger.info(f"  Kelly: {kelly_size:.4f}, Volatility: {volatility_size:.4f}, Risk: {risk_based_size:.4f}")
            self.logger.info(f"  Correlation adj: {correlation_multiplier:.2f}, Confidence: {confidence_multiplier:.2f}")
            self.logger.info(f"  Final size: {final_size:.4f}")
            
            return round(final_size, 2)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo position size: {e}")
            return 0.01
    
    def calculate_kelly_position_size(self, symbol: str) -> float:
        """Calcola position size usando Kelly Criterion"""
        try:
            # Ottieni statistiche storiche
            win_rate, avg_win, avg_loss = self.get_symbol_statistics(symbol)
            
            if avg_loss == 0 or win_rate == 0:
                return 0.01
            
            # Kelly formula: f = (bp - q) / b
            # dove b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
            b = avg_win / abs(avg_loss)
            p = win_rate / 100.0
            q = 1 - p
            
            kelly_fraction = (b * p - q) / b
            
            # Applica safety factor (25% del Kelly)
            safe_kelly = max(0, kelly_fraction * 0.25)
            
            # Converti in position size
            position_size = (self.current_balance * safe_kelly) / 1000  # Assumendo prezzo ~1
            
            return max(position_size, 0.01)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo Kelly: {e}")
            return 0.01
    
    def calculate_volatility_based_size(self, symbol: str, price: float) -> float:
        """Calcola position size basato su volatilità"""
        try:
            # Ottieni volatilità del simbolo
            volatility = self.get_symbol_volatility(symbol)
            
            if volatility == 0:
                return 0.01
            
            # Volatility targeting
            risk_params = self.risk_configs[self.current_risk_level]
            target_vol = risk_params.volatility_target
            
            # Calcola size per raggiungere target volatility
            vol_multiplier = target_vol / volatility
            base_size = (self.current_balance * 0.01) / price  # 1% base
            
            volatility_size = base_size * vol_multiplier
            
            return max(volatility_size, 0.01)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo volatility size: {e}")
            return 0.01
    
    def calculate_correlation_adjustment(self, symbol: str) -> float:
        """Calcola aggiustamento per correlazione"""
        try:
            # Ottieni posizioni correnti
            current_positions = self.get_current_positions()
            
            if not current_positions:
                return 1.0
            
            # Calcola correlazione media con posizioni esistenti
            correlations = []
            for pos_symbol in current_positions:
                if pos_symbol != symbol:
                    corr = self.get_correlation(symbol, pos_symbol)
                    if corr is not None:
                        correlations.append(abs(corr))
            
            if not correlations:
                return 1.0
            
            avg_correlation = np.mean(correlations)
            
            # Riduci size se alta correlazione
            risk_params = self.risk_configs[self.current_risk_level]
            max_corr = risk_params.max_correlation_exposure
            
            if avg_correlation > max_corr:
                adjustment = max_corr / avg_correlation
            else:
                adjustment = 1.0
            
            return max(adjustment, 0.1)  # Minimo 10%
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo correlation adjustment: {e}")
            return 1.0
    
    def calculate_volatility_target_adjustment(self, symbol: str) -> float:
        """Calcola aggiustamento per volatility targeting"""
        try:
            # Ottieni volatilità portfolio corrente
            portfolio_vol = self.calculate_portfolio_volatility()
            
            risk_params = self.risk_configs[self.current_risk_level]
            target_vol = risk_params.volatility_target
            
            if portfolio_vol == 0:
                return 1.0
            
            # Se portfolio vol è sopra target, riduci size
            if portfolio_vol > target_vol:
                adjustment = target_vol / portfolio_vol
            else:
                adjustment = 1.0
            
            return max(adjustment, 0.2)  # Minimo 20%
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo vol target adjustment: {e}")
            return 1.0
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, 
                                  direction: str, atr_multiplier: float = 2.0) -> float:
        """Calcola stop loss dinamico basato su ATR e volatilità"""
        try:
            # Ottieni ATR
            atr = self.get_symbol_atr(symbol)
            
            if atr == 0:
                # Fallback: usa percentuale fissa
                atr = entry_price * 0.01  # 1%
            
            # Ottieni parametri di rischio
            risk_params = self.risk_configs[self.current_risk_level]
            
            # Calcola distanza stop loss
            stop_distance = atr * atr_multiplier * risk_params.stop_loss_multiplier
            
            # Applica direzione
            if direction.upper() == "BUY":
                stop_loss = entry_price - stop_distance
            else:
                stop_loss = entry_price + stop_distance
            
            # Verifica che non sia troppo vicino o lontano
            min_distance = entry_price * 0.005  # Min 0.5%
            max_distance = entry_price * 0.05   # Max 5%
            
            actual_distance = abs(entry_price - stop_loss)
            
            if actual_distance < min_distance:
                if direction.upper() == "BUY":
                    stop_loss = entry_price - min_distance
                else:
                    stop_loss = entry_price + min_distance
            elif actual_distance > max_distance:
                if direction.upper() == "BUY":
                    stop_loss = entry_price - max_distance
                else:
                    stop_loss = entry_price + max_distance
            
            return round(stop_loss, 5)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo stop loss dinamico: {e}")
            # Fallback
            if direction.upper() == "BUY":
                return entry_price * 0.98  # 2% stop loss
            else:
                return entry_price * 1.02
    
    def calculate_dynamic_take_profit(self, symbol: str, entry_price: float, 
                                    stop_loss: float, direction: str, 
                                    risk_reward_ratio: float = 2.0) -> float:
        """Calcola take profit dinamico"""
        try:
            # Calcola distanza stop loss
            stop_distance = abs(entry_price - stop_loss)
            
            # Ottieni parametri di rischio
            risk_params = self.risk_configs[self.current_risk_level]
            
            # Calcola take profit basato su risk/reward ratio
            tp_distance = stop_distance * risk_reward_ratio * risk_params.take_profit_multiplier
            
            # Applica direzione
            if direction.upper() == "BUY":
                take_profit = entry_price + tp_distance
            else:
                take_profit = entry_price - tp_distance
            
            # Considera supporti/resistenze
            sr_levels = self.get_support_resistance_levels(symbol)
            if sr_levels:
                # Aggiusta take profit se vicino a livelli importanti
                take_profit = self.adjust_tp_for_sr_levels(take_profit, sr_levels, direction)
            
            return round(take_profit, 5)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo take profit dinamico: {e}")
            # Fallback
            stop_distance = abs(entry_price - stop_loss)
            if direction.upper() == "BUY":
                return entry_price + (stop_distance * 2)
            else:
                return entry_price - (stop_distance * 2)
    
    def adjust_tp_for_sr_levels(self, take_profit: float, sr_levels: List[float], 
                               direction: str) -> float:
        """Aggiusta take profit considerando supporti/resistenze"""
        try:
            threshold = take_profit * 0.002  # 0.2% threshold
            
            for level in sr_levels:
                if abs(take_profit - level) < threshold:
                    # TP troppo vicino a S/R, aggiusta
                    if direction.upper() == "BUY":
                        # Se long, metti TP leggermente sotto resistenza
                        take_profit = level - threshold
                    else:
                        # Se short, metti TP leggermente sopra supporto
                        take_profit = level + threshold
                    break
            
            return take_profit
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiustamento TP: {e}")
            return take_profit
    
    def calculate_portfolio_risk(self) -> Dict[str, float]:
        """Calcola rischio totale del portfolio"""
        try:
            positions = self.get_current_positions_detailed()
            
            if not positions:
                return {
                    'total_risk': 0.0,
                    'risk_percentage': 0.0,
                    'var_95': 0.0,
                    'expected_shortfall': 0.0,
                    'portfolio_volatility': 0.0
                }
            
            # Calcola rischio individuale
            individual_risks = []
            position_values = []
            
            for pos in positions:
                risk_amount = abs(pos.unrealized_pnl) if pos.unrealized_pnl < 0 else 0
                individual_risks.append(risk_amount)
                position_values.append(abs(pos.position_size * pos.current_price))
            
            total_risk = sum(individual_risks)
            total_value = sum(position_values)
            
            # Calcola VaR (Value at Risk) 95%
            if len(individual_risks) > 1:
                var_95 = np.percentile(individual_risks, 95)
                expected_shortfall = np.mean([r for r in individual_risks if r >= var_95])
            else:
                var_95 = total_risk
                expected_shortfall = total_risk
            
            # Calcola volatilità portfolio
            portfolio_volatility = self.calculate_portfolio_volatility()
            
            return {
                'total_risk': total_risk,
                'risk_percentage': (total_risk / self.current_balance) * 100,
                'var_95': var_95,
                'expected_shortfall': expected_shortfall,
                'portfolio_volatility': portfolio_volatility,
                'position_count': len(positions),
                'total_exposure': total_value
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo portfolio risk: {e}")
            return {'total_risk': 0.0, 'risk_percentage': 0.0}
    
    def calculate_portfolio_volatility(self) -> float:
        """Calcola volatilità del portfolio"""
        try:
            positions = self.get_current_positions_detailed()
            
            if not positions:
                return 0.0
            
            # Ottieni pesi delle posizioni
            total_value = sum(abs(pos.position_size * pos.current_price) for pos in positions)
            
            if total_value == 0:
                return 0.0
            
            weights = []
            volatilities = []
            symbols = []
            
            for pos in positions:
                weight = abs(pos.position_size * pos.current_price) / total_value
                volatility = pos.volatility
                
                weights.append(weight)
                volatilities.append(volatility)
                symbols.append(pos.symbol)
            
            # Calcola volatilità portfolio considerando correlazioni
            portfolio_variance = 0.0
            
            for i in range(len(positions)):
                for j in range(len(positions)):
                    if i == j:
                        # Varianza individuale
                        portfolio_variance += (weights[i] ** 2) * (volatilities[i] ** 2)
                    else:
                        # Covarianza
                        correlation = self.get_correlation(symbols[i], symbols[j])
                        if correlation is not None:
                            covariance = correlation * volatilities[i] * volatilities[j]
                            portfolio_variance += 2 * weights[i] * weights[j] * covariance
            
            portfolio_volatility = math.sqrt(max(portfolio_variance, 0))
            
            return portfolio_volatility
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo portfolio volatility: {e}")
            return 0.0
    
    def update_risk_level(self, new_level: RiskLevel, reason: str = "Manual"):
        """Aggiorna livello di rischio"""
        try:
            old_level = self.current_risk_level
            self.current_risk_level = new_level
            
            self.logger.info(f"🔄 Risk level changed: {old_level.value} → {new_level.value}")
            self.logger.info(f"   Reason: {reason}")
            
            # Log evento
            self.log_risk_event("RISK_LEVEL_CHANGE", new_level.value, 
                              ProtectionLevel.NORMAL, 
                              f"Risk level changed from {old_level.value} to {new_level.value}: {reason}",
                              "RISK_PARAMETERS_UPDATED")
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento risk level: {e}")
    
    def check_protection_levels(self) -> ProtectionLevel:
        """Controlla e aggiorna livelli di protezione"""
        try:
            # Calcola drawdown corrente
            self.current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            
            # Aggiorna max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
            
            # Determina livello protezione
            risk_params = self.risk_configs[self.current_risk_level]
            
            if self.current_drawdown >= 0.25:  # 25% drawdown
                new_level = ProtectionLevel.EMERGENCY
            elif self.current_drawdown >= 0.20:  # 20% drawdown
                new_level = ProtectionLevel.CRITICAL
            elif self.current_drawdown >= risk_params.max_drawdown_limit:
                new_level = ProtectionLevel.WARNING
            else:
                new_level = ProtectionLevel.NORMAL
            
            # Se livello è cambiato, prendi azioni
            if new_level != self.current_protection_level:
                self.activate_protection_measures(new_level)
            
            return new_level
            
        except Exception as e:
            self.logger.error(f"Errore nel controllo protection levels: {e}")
            return ProtectionLevel.NORMAL
    
    def activate_protection_measures(self, protection_level: ProtectionLevel):
        """Attiva misure di protezione"""
        try:
            old_level = self.current_protection_level
            self.current_protection_level = protection_level
            
            self.logger.warning(f"🛡️ Protection level: {old_level.value} → {protection_level.value}")
            
            if protection_level == ProtectionLevel.WARNING:
                # Riduci risk per trade del 50%
                self.reduce_position_sizes(0.5)
                self.log_risk_event("PROTECTION_WARNING", self.current_risk_level.value,
                                  protection_level, "Drawdown warning - reducing position sizes",
                                  "POSITION_SIZES_REDUCED_50%")
            
            elif protection_level == ProtectionLevel.CRITICAL:
                # Riduci risk per trade del 75%
                self.reduce_position_sizes(0.25)
                # Chiudi posizioni in perdita
                self.close_losing_positions()
                self.log_risk_event("PROTECTION_CRITICAL", self.current_risk_level.value,
                                  protection_level, "Critical drawdown - emergency measures",
                                  "POSITIONS_REDUCED_LOSSES_CLOSED")
            
            elif protection_level == ProtectionLevel.EMERGENCY:
                # Chiudi tutte le posizioni
                self.close_all_positions()
                # Passa a modalità conservativa
                self.update_risk_level(RiskLevel.CONSERVATIVE, "Emergency protection")
                self.log_risk_event("PROTECTION_EMERGENCY", self.current_risk_level.value,
                                  protection_level, "Emergency drawdown - all positions closed",
                                  "ALL_POSITIONS_CLOSED")
            
        except Exception as e:
            self.logger.error(f"Errore nell'attivazione protezioni: {e}")
    
    def reduce_position_sizes(self, multiplier: float):
        """Riduci dimensioni posizioni"""
        try:
            # Aggiorna moltiplicatore per nuove posizioni
            current_params = self.risk_configs[self.current_risk_level]
            current_params.position_size_multiplier *= multiplier
            
            self.logger.info(f"Position size multiplier reduced to {current_params.position_size_multiplier:.2f}")
            
        except Exception as e:
            self.logger.error(f"Errore nella riduzione position sizes: {e}")
    
    def close_losing_positions(self):
        """Chiudi posizioni in perdita"""
        try:
            positions = self.get_current_positions_detailed()
            
            for pos in positions:
                if pos.unrealized_pnl < 0:
                    self.close_position(pos.symbol)
                    self.logger.info(f"Closed losing position: {pos.symbol} (Loss: {pos.unrealized_pnl:.2f})")
            
        except Exception as e:
            self.logger.error(f"Errore nella chiusura posizioni in perdita: {e}")
    
    def close_all_positions(self):
        """Chiudi tutte le posizioni"""
        try:
            positions = self.get_current_positions_detailed()
            
            for pos in positions:
                self.close_position(pos.symbol)
                self.logger.info(f"Emergency close: {pos.symbol} (PnL: {pos.unrealized_pnl:.2f})")
            
        except Exception as e:
            self.logger.error(f"Errore nella chiusura tutte posizioni: {e}")
    
    def close_position(self, symbol: str):
        """Chiudi posizione specifica"""
        try:
            # Implementazione specifica per MT5
            if not mt5.initialize():
                return False
            
            positions = mt5.positions_get(symbol=symbol)
            if not positions:
                return False
            
            for position in positions:
                # Prepara richiesta chiusura
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "price": mt5.symbol_info_tick(symbol).bid if position.type == 0 else mt5.symbol_info_tick(symbol).ask,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "Risk Management Close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(close_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Errore nella chiusura posizione {symbol}: {e}")
            return False
    
    def validate_trade_request(self, symbol: str, direction: str, position_size: float,
                             entry_price: float, stop_loss: float, take_profit: float) -> Tuple[bool, str]:
        """Valida richiesta di trade"""
        try:
            # 1. Controlla livello protezione
            if self.current_protection_level == ProtectionLevel.EMERGENCY:
                return False, "Emergency protection active - no new trades"
            
            # 2. Controlla risk per trade
            risk_amount = abs(entry_price - stop_loss) * position_size
            risk_percentage = (risk_amount / self.current_balance) * 100
            
            risk_params = self.risk_configs[self.current_risk_level]
            max_risk_pct = risk_params.max_risk_per_trade * 100
            
            if risk_percentage > max_risk_pct:
                return False, f"Risk per trade too high: {risk_percentage:.2f}% > {max_risk_pct:.2f}%"
            
            # 3. Controlla portfolio risk
            portfolio_risk = self.calculate_portfolio_risk()
            total_risk_pct = portfolio_risk['risk_percentage']
            max_portfolio_risk_pct = risk_params.max_portfolio_risk * 100
            
            if total_risk_pct + risk_percentage > max_portfolio_risk_pct:
                return False, f"Portfolio risk limit exceeded: {total_risk_pct + risk_percentage:.2f}% > {max_portfolio_risk_pct:.2f}%"
            
            # 4. Controlla correlazione
            correlation_adj = self.calculate_correlation_adjustment(symbol)
            if correlation_adj < 0.5:  # Alta correlazione
                return False, f"High correlation risk - adjustment factor: {correlation_adj:.2f}"
            
            # 5. Controlla numero massimo posizioni
            current_positions = len(self.get_current_positions())
            if current_positions >= risk_params.max_positions:
                return False, f"Max positions limit reached: {current_positions}/{risk_params.max_positions}"
            
            # 6. Controlla stop loss e take profit
            if abs(entry_price - stop_loss) < entry_price * 0.002:  # Min 0.2%
                return False, "Stop loss too close to entry price"
            
            if abs(take_profit - entry_price) < abs(entry_price - stop_loss) * 0.5:  # Min 1:0.5 R/R
                return False, "Take profit too close - insufficient risk/reward ratio"
            
            return True, "Trade request validated"
            
        except Exception as e:
            self.logger.error(f"Errore nella validazione trade: {e}")
            return False, f"Validation error: {e}"
    
    def update_balance(self, new_balance: float):
        """Aggiorna balance e calcola metriche"""
        try:
            old_balance = self.current_balance
            self.current_balance = new_balance
            
            # Aggiorna peak balance
            if new_balance > self.peak_balance:
                self.peak_balance = new_balance
            
            # Calcola PnL giornaliero
            pnl_change = new_balance - old_balance
            self.daily_pnl += pnl_change
            
            # Controlla livelli protezione
            self.check_protection_levels()
            
            # Log se cambio significativo
            if abs(pnl_change) > self.current_balance * 0.01:  # >1% change
                self.logger.info(f"Balance updated: {old_balance:.2f} → {new_balance:.2f} (Change: {pnl_change:+.2f})")
            
        except Exception as e:
            self.logger.error(f"Errore nell'aggiornamento balance: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Ottieni riassunto completo del rischio"""
        try:
            portfolio_risk = self.calculate_portfolio_risk()
            risk_params = self.risk_configs[self.current_risk_level]
            
            return {
                'risk_level': self.current_risk_level.value,
                'protection_level': self.current_protection_level.value,
                'current_balance': self.current_balance,
                'peak_balance': self.peak_balance,
                'current_drawdown': self.current_drawdown * 100,
                'max_drawdown': self.max_drawdown * 100,
                'daily_pnl': self.daily_pnl,
                'portfolio_risk': portfolio_risk,
                'risk_parameters': {
                    'max_risk_per_trade': risk_params.max_risk_per_trade * 100,
                    'max_portfolio_risk': risk_params.max_portfolio_risk * 100,
                    'max_positions': risk_params.max_positions,
                    'volatility_target': risk_params.volatility_target * 100,
                    'position_size_multiplier': risk_params.position_size_multiplier
                },
                'consecutive_losses': self.consecutive_losses,
                'total_trades': self.total_trades,
                'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel risk summary: {e}")
            return {}
    
    # Helper methods per dati di mercato
    def get_symbol_statistics(self, symbol: str) -> Tuple[float, float, float]:
        """Ottieni statistiche simbolo (win_rate, avg_win, avg_loss)"""
        try:
            # Implementazione semplificata - in produzione usare dati reali
            return 60.0, 0.015, -0.010  # 60% win rate, 1.5% avg win, -1% avg loss
        except:
            return 50.0, 0.01, -0.01
    
    def get_symbol_volatility(self, symbol: str) -> float:
        """Ottieni volatilità simbolo"""
        try:
            # Implementazione semplificata
            return 0.15  # 15% annuale
        except:
            return 0.15
    
    def get_symbol_atr(self, symbol: str) -> float:
        """Ottieni ATR simbolo"""
        try:
            if not mt5.initialize():
                return 0.001
            
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 14)
            if rates is None:
                return 0.001
            
            df = pd.DataFrame(rates)
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(14).mean().iloc[-1]
            
            return atr if not np.isnan(atr) else 0.001
            
        except:
            return 0.001
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Ottieni correlazione tra simboli"""
        try:
            # Implementazione semplificata - in produzione calcolare da dati reali
            if symbol1 == symbol2:
                return 1.0
            
            # Correlazioni tipiche forex
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            if symbol1 in major_pairs and symbol2 in major_pairs:
                return 0.3  # Correlazione moderata
            
            return 0.1  # Bassa correlazione default
            
        except:
            return None
    
    def get_current_positions(self) -> List[str]:
        """Ottieni simboli posizioni correnti"""
        try:
            if not mt5.initialize():
                return []
            
            positions = mt5.positions_get()
            return [pos.symbol for pos in positions] if positions else []
            
        except:
            return []
    
    def get_current_positions_detailed(self) -> List[PositionRisk]:
        """Ottieni dettagli posizioni correnti"""
        try:
            if not mt5.initialize():
                return []
            
            positions = mt5.positions_get()
            if not positions:
                return []
            
            position_risks = []
            for pos in positions:
                # Calcola metriche rischio
                volatility = self.get_symbol_volatility(pos.symbol)
                correlation_risk = self.calculate_correlation_adjustment(pos.symbol)
                
                position_risk = PositionRisk(
                    symbol=pos.symbol,
                    position_size=pos.volume,
                    entry_price=pos.price_open,
                    current_price=pos.price_current,
                    stop_loss=pos.sl if pos.sl > 0 else 0,
                    take_profit=pos.tp if pos.tp > 0 else 0,
                    unrealized_pnl=pos.profit,
                    risk_amount=abs(pos.profit) if pos.profit < 0 else 0,
                    risk_percentage=(abs(pos.profit) / self.current_balance * 100) if pos.profit < 0 else 0,
                    volatility=volatility,
                    correlation_risk=correlation_risk
                )
                
                position_risks.append(position_risk)
            
            return position_risks
            
        except Exception as e:
            self.logger.error(f"Errore nel recuperare posizioni dettagliate: {e}")
            return []
    
    def get_support_resistance_levels(self, symbol: str) -> List[float]:
        """Ottieni livelli supporto/resistenza"""
        try:
            # Implementazione semplificata
            return []
        except:
            return []
    
    def log_risk_event(self, event_type: str, risk_level: str, protection_level: ProtectionLevel,
                      description: str, action_taken: str, impact: float = 0.0):
        """Log evento di rischio"""
        try:
            conn = sqlite3.connect('enhanced_risk.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_events (
                    timestamp, event_type, risk_level, protection_level,
                    description, action_taken, impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), event_type, risk_level, protection_level.value,
                description, action_taken, impact
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore nel log risk event: {e}")

def main():
    """Test del sistema di risk management avanzato"""
    print("🛡️ ENHANCED RISK MANAGEMENT SYSTEM")
    print("=" * 50)
    
    # Inizializza risk manager
    risk_manager = EnhancedRiskManager(initial_balance=10000)
    
    # Test calcolo position size
    symbol = "EURUSD"
    entry_price = 1.1000
    stop_loss = 1.0950
    
    print(f"🧮 Calcolando position size per {symbol}...")
    position_size = risk_manager.calculate_dynamic_position_size(symbol, entry_price, stop_loss, confidence=0.8)
    
    print(f"   Entry Price: {entry_price}")
    print(f"   Stop Loss: {stop_loss}")
    print(f"   Position Size: {position_size}")
    
    # Test calcolo stop loss dinamico
    dynamic_sl = risk_manager.calculate_dynamic_stop_loss(symbol, entry_price, "BUY")
    print(f"   Dynamic Stop Loss: {dynamic_sl}")
    
    # Test calcolo take profit dinamico
    dynamic_tp = risk_manager.calculate_dynamic_take_profit(symbol, entry_price, dynamic_sl, "BUY")
    print(f"   Dynamic Take Profit: {dynamic_tp}")
    
    # Test validazione trade
    is_valid, message = risk_manager.validate_trade_request(symbol, "BUY", position_size, entry_price, dynamic_sl, dynamic_tp)
    print(f"\n✅ Trade Validation: {is_valid}")
    print(f"   Message: {message}")
    
    # Test calcolo portfolio risk
    portfolio_risk = risk_manager.calculate_portfolio_risk()
    print(f"\n📊 Portfolio Risk:")
    print(f"   Total Risk: {portfolio_risk['total_risk']:.2f}")
    print(f"   Risk Percentage: {portfolio_risk['risk_percentage']:.2f}%")
    print(f"   Portfolio Volatility: {portfolio_risk['portfolio_volatility']:.2f}")
    
    # Test cambio risk level
    print(f"\n🔄 Testing risk level changes...")
    risk_manager.update_risk_level(RiskLevel.AGGRESSIVE, "Testing aggressive mode")
    
    # Simula drawdown per testare protezioni
    print(f"\n🛡️ Testing protection levels...")
    risk_manager.update_balance(8000)  # 20% drawdown
    
    # Mostra riassunto finale
    risk_summary = risk_manager.get_risk_summary()
    print(f"\n📋 RISK SUMMARY:")
    print(f"   Risk Level: {risk_summary['risk_level']}")
    print(f"   Protection Level: {risk_summary['protection_level']}")
    print(f"   Current Drawdown: {risk_summary['current_drawdown']:.1f}%")
    print(f"   Max Drawdown: {risk_summary['max_drawdown']:.1f}%")
    print(f"   Daily PnL: {risk_summary['daily_pnl']:+.2f}")
    
    print("\n✅ Enhanced Risk Management test completato!")

if __name__ == "__main__":
    main()
