#!/usr/bin/env python3
"""
ANALIZZATORE DI PERFORMANCE AVANZATO
- Calcolo metriche di performance in tempo reale
- Monitoraggio Win Rate, Profit Factor, Sharpe Ratio, Maximum Drawdown
- Analisi di robustezza della strategia
- Sistema di allerta per performance degradate
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Metriche di performance complete"""
    # Metriche base
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Metriche di profitto
    total_profit: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    
    # Metriche di rischio
    max_drawdown: float
    current_drawdown: float
    recovery_factor: float
    calmar_ratio: float
    
    # Metriche statistiche
    sharpe_ratio: float
    sortino_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Metriche di consistenza
    consecutive_wins: int
    consecutive_losses: int
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # Metriche temporali
    avg_trade_duration: float
    profit_per_day: float
    trades_per_day: float
    
    # Metriche di efficienza
    expectancy: float
    kelly_percentage: float
    optimal_f: float
    
    # Timestamp
    calculation_time: datetime

class PerformanceAnalyzer:
    """
    Analizzatore di performance avanzato per trading bot
    """
    
    def __init__(self, db_path: str = 'trading_data.db'):
        self.db_path = db_path
        self.logger = logger
        
        # Soglie di performance
        self.performance_thresholds = {
            'excellent_win_rate': 70.0,
            'good_win_rate': 60.0,
            'minimum_win_rate': 50.0,
            'excellent_profit_factor': 2.0,
            'good_profit_factor': 1.5,
            'minimum_profit_factor': 1.2,
            'maximum_drawdown': 20.0,
            'warning_drawdown': 15.0,
            'minimum_sharpe': 1.0,
            'good_sharpe': 1.5,
            'excellent_sharpe': 2.0
        }
        
        # Setup database per metriche
        self.setup_performance_database()
        
    def setup_performance_database(self):
        """Setup database per tracking performance"""
        try:
            conn = sqlite3.connect('performance_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    total_profit REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    calmar_ratio REAL,
                    expectancy REAL,
                    kelly_percentage REAL,
                    performance_score REAL,
                    strategy_health TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    alert_type TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    threshold_value REAL,
                    severity TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database performance inizializzato")
            
        except Exception as e:
            self.logger.error(f"Errore setup database performance: {e}")
    
    def calculate_all_metrics(self, period_days: int = 30) -> PerformanceMetrics:
        """Calcola tutte le metriche di performance"""
        try:
            # Ottieni dati trades
            trades_df = self.get_trades_data(period_days)
            
            if trades_df.empty:
                return self.get_empty_metrics()
            
            # Calcola metriche base
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['profit'] > 0])
            losing_trades = len(trades_df[trades_df['profit'] < 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calcola metriche di profitto
            profits = trades_df['profit'].values
            wins = profits[profits > 0]
            losses = profits[profits < 0]
            
            total_profit = np.sum(profits)
            gross_profit = np.sum(wins) if len(wins) > 0 else 0
            gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            average_win = np.mean(wins) if len(wins) > 0 else 0
            average_loss = np.mean(losses) if len(losses) > 0 else 0
            largest_win = np.max(wins) if len(wins) > 0 else 0
            largest_loss = np.min(losses) if len(losses) > 0 else 0
            
            # Calcola metriche di rischio
            equity_curve = self.calculate_equity_curve(trades_df)
            max_drawdown, current_drawdown = self.calculate_drawdown(equity_curve)
            recovery_factor = total_profit / max_drawdown if max_drawdown > 0 else 0
            
            # Calcola ratios
            sharpe_ratio = self.calculate_sharpe_ratio(profits)
            sortino_ratio = self.calculate_sortino_ratio(profits)
            calmar_ratio = (total_profit / len(profits) * 252) / max_drawdown if max_drawdown > 0 else 0
            
            # Calcola metriche di consistenza
            consecutive_stats = self.calculate_consecutive_stats(profits)
            
            # Calcola metriche temporali
            avg_duration, profit_per_day, trades_per_day = self.calculate_temporal_metrics(trades_df, period_days)
            
            # Calcola metriche di efficienza
            expectancy = self.calculate_expectancy(wins, losses, win_rate)
            kelly_percentage = self.calculate_kelly_percentage(wins, losses)
            optimal_f = self.calculate_optimal_f(profits)
            
            return PerformanceMetrics(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_profit=total_profit,
                gross_profit=gross_profit,
                gross_loss=gross_loss,
                profit_factor=profit_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                recovery_factor=recovery_factor,
                calmar_ratio=calmar_ratio,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                information_ratio=0.0,  # Richiede benchmark
                treynor_ratio=0.0,     # Richiede beta
                consecutive_wins=consecutive_stats['current_wins'],
                consecutive_losses=consecutive_stats['current_losses'],
                max_consecutive_wins=consecutive_stats['max_wins'],
                max_consecutive_losses=consecutive_stats['max_losses'],
                avg_trade_duration=avg_duration,
                profit_per_day=profit_per_day,
                trades_per_day=trades_per_day,
                expectancy=expectancy,
                kelly_percentage=kelly_percentage,
                optimal_f=optimal_f,
                calculation_time=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo metriche: {e}")
            return self.get_empty_metrics()
    
    def get_trades_data(self, period_days: int) -> pd.DataFrame:
        """Ottiene dati trades dal database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calcola data di inizio
            start_date = datetime.now() - timedelta(days=period_days)
            
            query = '''
                SELECT symbol, action, volume, price, profit, timestamp, status
                FROM trades 
                WHERE timestamp >= ? AND profit IS NOT NULL
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(start_date,))
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['profit'] = pd.to_numeric(df['profit'], errors='coerce')
                df = df.dropna(subset=['profit'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Errore nel recuperare dati trades: {e}")
            return pd.DataFrame()
    
    def calculate_equity_curve(self, trades_df: pd.DataFrame) -> List[float]:
        """Calcola curva equity"""
        try:
            if trades_df.empty:
                return [0.0]
            
            # Ordina per timestamp
            trades_df = trades_df.sort_values('timestamp')
            
            # Calcola equity cumulativa
            cumulative_profit = trades_df['profit'].cumsum()
            equity_curve = [0.0] + cumulative_profit.tolist()
            
            return equity_curve
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo equity curve: {e}")
            return [0.0]
    
    def calculate_drawdown(self, equity_curve: List[float]) -> Tuple[float, float]:
        """Calcola maximum drawdown e current drawdown"""
        try:
            if len(equity_curve) < 2:
                return 0.0, 0.0
            
            peak = equity_curve[0]
            max_drawdown = 0.0
            current_drawdown = 0.0
            
            for value in equity_curve:
                if value > peak:
                    peak = value
                
                drawdown = peak - value
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                
                # Current drawdown è l'ultimo drawdown
                current_drawdown = peak - equity_curve[-1]
            
            return max_drawdown, max(0, current_drawdown)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo drawdown: {e}")
            return 0.0, 0.0
    
    def calculate_sharpe_ratio(self, profits: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calcola Sharpe Ratio"""
        try:
            if len(profits) < 2:
                return 0.0
            
            # Calcola return medio e volatilità
            mean_return = np.mean(profits)
            std_return = np.std(profits)
            
            if std_return == 0:
                return 0.0
            
            # Annualizza (assumendo 252 trading days)
            annual_return = mean_return * 252
            annual_volatility = std_return * np.sqrt(252)
            
            sharpe = (annual_return - risk_free_rate) / annual_volatility
            return sharpe
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo Sharpe ratio: {e}")
            return 0.0
    
    def calculate_sortino_ratio(self, profits: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calcola Sortino Ratio"""
        try:
            if len(profits) < 2:
                return 0.0
            
            mean_return = np.mean(profits)
            negative_returns = profits[profits < 0]
            
            if len(negative_returns) == 0:
                return float('inf')
            
            downside_deviation = np.std(negative_returns)
            
            if downside_deviation == 0:
                return 0.0
            
            # Annualizza
            annual_return = mean_return * 252
            annual_downside_dev = downside_deviation * np.sqrt(252)
            
            sortino = (annual_return - risk_free_rate) / annual_downside_dev
            return sortino
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo Sortino ratio: {e}")
            return 0.0
    
    def calculate_consecutive_stats(self, profits: np.ndarray) -> Dict:
        """Calcola statistiche consecutive wins/losses"""
        try:
            if len(profits) == 0:
                return {'current_wins': 0, 'current_losses': 0, 'max_wins': 0, 'max_losses': 0}
            
            current_wins = 0
            current_losses = 0
            max_wins = 0
            max_losses = 0
            temp_wins = 0
            temp_losses = 0
            
            for profit in profits:
                if profit > 0:
                    temp_wins += 1
                    temp_losses = 0
                    max_wins = max(max_wins, temp_wins)
                elif profit < 0:
                    temp_losses += 1
                    temp_wins = 0
                    max_losses = max(max_losses, temp_losses)
            
            # Current streak
            for profit in reversed(profits):
                if profit > 0:
                    if current_losses == 0:
                        current_wins += 1
                    else:
                        break
                elif profit < 0:
                    if current_wins == 0:
                        current_losses += 1
                    else:
                        break
                else:
                    break
            
            return {
                'current_wins': current_wins,
                'current_losses': current_losses,
                'max_wins': max_wins,
                'max_losses': max_losses
            }
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo consecutive stats: {e}")
            return {'current_wins': 0, 'current_losses': 0, 'max_wins': 0, 'max_losses': 0}
    
    def calculate_temporal_metrics(self, trades_df: pd.DataFrame, period_days: int) -> Tuple[float, float, float]:
        """Calcola metriche temporali"""
        try:
            if trades_df.empty:
                return 0.0, 0.0, 0.0
            
            # Durata media trade (assumendo che ogni trade duri 1 ora in media)
            avg_duration = 1.0  # ore
            
            # Profitto per giorno
            total_profit = trades_df['profit'].sum()
            profit_per_day = total_profit / period_days if period_days > 0 else 0
            
            # Trades per giorno
            total_trades = len(trades_df)
            trades_per_day = total_trades / period_days if period_days > 0 else 0
            
            return avg_duration, profit_per_day, trades_per_day
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo metriche temporali: {e}")
            return 0.0, 0.0, 0.0
    
    def calculate_expectancy(self, wins: np.ndarray, losses: np.ndarray, win_rate: float) -> float:
        """Calcola expectancy della strategia"""
        try:
            if len(wins) == 0 and len(losses) == 0:
                return 0.0
            
            avg_win = np.mean(wins) if len(wins) > 0 else 0
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
            
            win_rate_decimal = win_rate / 100.0
            loss_rate = 1.0 - win_rate_decimal
            
            expectancy = (win_rate_decimal * avg_win) - (loss_rate * avg_loss)
            return expectancy
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo expectancy: {e}")
            return 0.0
    
    def calculate_kelly_percentage(self, wins: np.ndarray, losses: np.ndarray) -> float:
        """Calcola Kelly percentage per position sizing ottimale"""
        try:
            if len(wins) == 0 or len(losses) == 0:
                return 0.0
            
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            if avg_loss == 0:
                return 0.0
            
            win_rate = len(wins) / (len(wins) + len(losses))
            loss_rate = 1 - win_rate
            
            # Kelly formula: f = (bp - q) / b
            # dove b = avg_win/avg_loss, p = win_rate, q = loss_rate
            b = avg_win / avg_loss
            kelly = (b * win_rate - loss_rate) / b
            
            # Limita Kelly al 25% per sicurezza
            return min(kelly * 100, 25.0)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo Kelly percentage: {e}")
            return 0.0
    
    def calculate_optimal_f(self, profits: np.ndarray) -> float:
        """Calcola Optimal F per position sizing"""
        try:
            if len(profits) == 0:
                return 0.0
            
            # Trova la perdita massima
            max_loss = abs(min(profits)) if min(profits) < 0 else 1.0
            
            if max_loss == 0:
                return 0.0
            
            # Calcola TWR (Terminal Wealth Relative) per diversi valori di f
            best_f = 0.0
            best_twr = 0.0
            
            for f in np.arange(0.01, 0.5, 0.01):  # Test f da 1% a 50%
                twr = 1.0
                for profit in profits:
                    hpr = 1 + (f * profit / max_loss)  # Holding Period Return
                    if hpr > 0:
                        twr *= hpr
                    else:
                        twr = 0
                        break
                
                if twr > best_twr:
                    best_twr = twr
                    best_f = f
            
            return best_f * 100  # Restituisce in percentuale
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo Optimal F: {e}")
            return 0.0
    
    def get_empty_metrics(self) -> PerformanceMetrics:
        """Restituisce metriche vuote"""
        return PerformanceMetrics(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0,
            total_profit=0.0, gross_profit=0.0, gross_loss=0.0, profit_factor=0.0,
            average_win=0.0, average_loss=0.0, largest_win=0.0, largest_loss=0.0,
            max_drawdown=0.0, current_drawdown=0.0, recovery_factor=0.0, calmar_ratio=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, information_ratio=0.0, treynor_ratio=0.0,
            consecutive_wins=0, consecutive_losses=0, max_consecutive_wins=0, max_consecutive_losses=0,
            avg_trade_duration=0.0, profit_per_day=0.0, trades_per_day=0.0,
            expectancy=0.0, kelly_percentage=0.0, optimal_f=0.0,
            calculation_time=datetime.now()
        )
    
    def evaluate_strategy_health(self, metrics: PerformanceMetrics) -> Dict:
        """Valuta la salute della strategia"""
        try:
            score = 0
            max_score = 100
            health_issues = []
            recommendations = []
            
            # Valuta Win Rate (25 punti)
            if metrics.win_rate >= self.performance_thresholds['excellent_win_rate']:
                score += 25
            elif metrics.win_rate >= self.performance_thresholds['good_win_rate']:
                score += 20
            elif metrics.win_rate >= self.performance_thresholds['minimum_win_rate']:
                score += 15
            else:
                score += 5
                health_issues.append(f"Win rate basso: {metrics.win_rate:.1f}%")
                recommendations.append("Migliorare la precisione dei segnali di entrata")
            
            # Valuta Profit Factor (25 punti)
            if metrics.profit_factor >= self.performance_thresholds['excellent_profit_factor']:
                score += 25
            elif metrics.profit_factor >= self.performance_thresholds['good_profit_factor']:
                score += 20
            elif metrics.profit_factor >= self.performance_thresholds['minimum_profit_factor']:
                score += 15
            else:
                score += 5
                health_issues.append(f"Profit factor basso: {metrics.profit_factor:.2f}")
                recommendations.append("Ottimizzare take profit e stop loss")
            
            # Valuta Max Drawdown (25 punti)
            if metrics.max_drawdown <= self.performance_thresholds['warning_drawdown']:
                score += 25
            elif metrics.max_drawdown <= self.performance_thresholds['maximum_drawdown']:
                score += 20
            elif metrics.max_drawdown <= 30:
                score += 15
            else:
                score += 5
                health_issues.append(f"Drawdown eccessivo: {metrics.max_drawdown:.1f}%")
                recommendations.append("Implementare controlli di rischio più stringenti")
            
            # Valuta Sharpe Ratio (25 punti)
            if metrics.sharpe_ratio >= self.performance_thresholds['excellent_sharpe']:
                score += 25
            elif metrics.sharpe_ratio >= self.performance_thresholds['good_sharpe']:
                score += 20
            elif metrics.sharpe_ratio >= self.performance_thresholds['minimum_sharpe']:
                score += 15
            else:
                score += 5
                health_issues.append(f"Sharpe ratio basso: {metrics.sharpe_ratio:.2f}")
                recommendations.append("Migliorare il rapporto rischio/rendimento")
            
            # Determina health status
            if score >= 90:
                health_status = "EXCELLENT"
            elif score >= 75:
                health_status = "GOOD"
            elif score >= 60:
                health_status = "AVERAGE"
            elif score >= 40:
                health_status = "POOR"
            else:
                health_status = "CRITICAL"
            
            return {
                'health_score': score,
                'max_score': max_score,
                'health_status': health_status,
                'health_issues': health_issues,
                'recommendations': recommendations,
                'evaluation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Errore nella valutazione strategia: {e}")
            return {'health_score': 0, 'health_status': 'ERROR', 'health_issues': [str(e)]}
    
    def check_performance_alerts(self, metrics: PerformanceMetrics) -> List[Dict]:
        """Controlla e genera alert per performance degradate"""
        alerts = []
        
        try:
            # Alert per Win Rate basso
            if metrics.win_rate < self.performance_thresholds['minimum_win_rate']:
                alerts.append({
                    'type': 'WIN_RATE_LOW',
                    'metric': 'win_rate',
                    'current_value': metrics.win_rate,
                    'threshold': self.performance_thresholds['minimum_win_rate'],
                    'severity': 'HIGH' if metrics.win_rate < 40 else 'MEDIUM',
                    'message': f"Win rate sceso a {metrics.win_rate:.1f}% (soglia: {self.performance_thresholds['minimum_win_rate']:.1f}%)"
                })
            
            # Alert per Profit Factor basso
            if metrics.profit_factor < self.performance_thresholds['minimum_profit_factor']:
                alerts.append({
                    'type': 'PROFIT_FACTOR_LOW',
                    'metric': 'profit_factor',
                    'current_value': metrics.profit_factor,
                    'threshold': self.performance_thresholds['minimum_profit_factor'],
                    'severity': 'HIGH' if metrics.profit_factor < 1.0 else 'MEDIUM',
                    'message': f"Profit factor sceso a {metrics.profit_factor:.2f} (soglia: {self.performance_thresholds['minimum_profit_factor']:.2f})"
                })
            
            # Alert per Drawdown alto
            if metrics.max_drawdown > self.performance_thresholds['maximum_drawdown']:
                alerts.append({
                    'type': 'DRAWDOWN_HIGH',
                    'metric': 'max_drawdown',
                    'current_value': metrics.max_drawdown,
                    'threshold': self.performance_thresholds['maximum_drawdown'],
                    'severity': 'CRITICAL' if metrics.max_drawdown > 30 else 'HIGH',
                    'message': f"Max drawdown raggiunto {metrics.max_drawdown:.1f}% (soglia: {self.performance_thresholds['maximum_drawdown']:.1f}%)"
                })
            
            # Alert per consecutive losses
            if metrics.consecutive_losses > 5:
                alerts.append({
                    'type': 'CONSECUTIVE_LOSSES',
                    'metric': 'consecutive_losses',
                    'current_value': metrics.consecutive_losses,
                    'threshold': 5,
                    'severity': 'HIGH',
                    'message': f"Troppe perdite consecutive: {metrics.consecutive_losses}"
                })
            
            # Alert per expectancy negativa
            if metrics.expectancy < 0:
                alerts.append({
                    'type': 'NEGATIVE_EXPECTANCY',
                    'metric': 'expectancy',
                    'current_value': metrics.expectancy,
                    'threshold': 0,
                    'severity': 'CRITICAL',
                    'message': f"Expectancy negativa: {metrics.expectancy:.2f}"
                })
            
            # Salva alerts nel database
            if alerts:
                self.save_performance_alerts(alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Errore nel controllo alerts: {e}")
            return []
    
    def save_performance_snapshot(self, metrics: PerformanceMetrics, health_evaluation: Dict):
        """Salva snapshot delle performance"""
        try:
            conn = sqlite3.connect('performance_data.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_snapshots (
                    timestamp, total_trades, win_rate, profit_factor, total_profit,
                    max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio,
                    expectancy, kelly_percentage, performance_score, strategy_health
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), metrics.total_trades, metrics.win_rate,
                metrics.profit_factor, metrics.total_profit, metrics.max_drawdown,
                metrics.sharpe_ratio, metrics.sortino_ratio, metrics.calmar_ratio,
                metrics.expectancy, metrics.kelly_percentage,
                health_evaluation['health_score'], health_evaluation['health_status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore nel salvare snapshot: {e}")
    
    def save_performance_alerts(self, alerts: List[Dict]):
        """Salva alerts nel database"""
        try:
            conn = sqlite3.connect('performance_data.db')
            cursor = conn.cursor()
            
            for alert in alerts:
                cursor.execute('''
                    INSERT INTO performance_alerts (
                        timestamp, alert_type, metric_name, current_value,
                        threshold_value, severity, message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now(), alert['type'], alert['metric'],
                    alert['current_value'], alert['threshold'],
                    alert['severity'], alert['message']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore nel salvare alerts: {e}")
    
    def generate_performance_report(self, metrics: PerformanceMetrics, health_evaluation: Dict) -> str:
        """Genera report dettagliato delle performance"""
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           PERFORMANCE ANALYSIS REPORT                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ STRATEGY HEALTH: {health_evaluation['health_status']:>10} | Score: {health_evaluation['health_score']:>3}/100 ║
║ Analysis Time: {metrics.calculation_time.strftime('%Y-%m-%d %H:%M:%S'):>20}                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ TRADING STATISTICS                                                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Trades:           {metrics.total_trades:>8d}                           ║
║ Winning Trades:         {metrics.winning_trades:>8d}                         ║
║ Losing Trades:          {metrics.losing_trades:>8d}                          ║
║ Win Rate:               {metrics.win_rate:>8.1f}%                            ║
║                                                                              ║
║ PROFITABILITY METRICS                                                        ║
║ Total Profit:           {metrics.total_profit:>8.2f} €                       ║
║ Gross Profit:           {metrics.gross_profit:>8.2f} €                       ║
║ Gross Loss:             {metrics.gross_loss:>8.2f} €                         ║
║ Profit Factor:          {metrics.profit_factor:>8.2f}                        ║
║ Average Win:            {metrics.average_win:>8.2f} €                        ║
║ Average Loss:           {metrics.average_loss:>8.2f} €                       ║
║ Largest Win:            {metrics.largest_win:>8.2f} €                        ║
║ Largest Loss:           {metrics.largest_loss:>8.2f} €                       ║
║                                                                              ║
║ RISK METRICS                                                                 ║
║ Max Drawdown:           {metrics.max_drawdown:>8.2f}%                        ║
║ Current Drawdown:       {metrics.current_drawdown:>8.2f}%                    ║
║ Recovery Factor:        {metrics.recovery_factor:>8.2f}                      ║
║ Calmar Ratio:           {metrics.calmar_ratio:>8.2f}                         ║
║                                                                              ║
║ STATISTICAL RATIOS                                                           ║
║ Sharpe Ratio:           {metrics.sharpe_ratio:>8.2f}                         ║
║ Sortino Ratio:          {metrics.sortino_ratio:>8.2f}                        ║
║                                                                              ║
║ CONSISTENCY METRICS                                                          ║
║ Current Win Streak:     {metrics.consecutive_wins:>8d}                       ║
║ Current Loss Streak:    {metrics.consecutive_losses:>8d}                     ║
║ Max Win Streak:         {metrics.max_consecutive_wins:>8d}                   ║
║ Max Loss Streak:        {metrics.max_consecutive_losses:>8d}                 ║
║                                                                              ║
║ EFFICIENCY METRICS                                                           ║
║ Expectancy:             {metrics.expectancy:>8.2f} €                         ║
║ Kelly Percentage:       {metrics.kelly_percentage:>8.1f}%                    ║
║ Optimal F:              {metrics.optimal_f:>8.1f}%                           ║
║                                                                              ║
║ TEMPORAL METRICS                                                             ║
║ Profit per Day:         {metrics.profit_per_day:>8.2f} €                     ║
║ Trades per Day:         {metrics.trades_per_day:>8.1f}                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

HEALTH ISSUES:
"""
        
        if health_evaluation.get('health_issues'):
            for issue in health_evaluation['health_issues']:
                report += f"⚠️  {issue}\n"
        else:
            report += "✅ No critical issues detected\n"
        
        report += "\nRECOMMENDATIONS:\n"
        if health_evaluation.get('recommendations'):
            for rec in health_evaluation['recommendations']:
                report += f"💡 {rec}\n"
        else:
            report += "✅ Strategy performing well - continue current approach\n"
        
        return report
    
    def create_performance_dashboard(self, metrics: PerformanceMetrics, output_dir: str = "performance_charts"):
        """Crea dashboard visuale delle performance"""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            # Setup stile
            plt.style.use('seaborn-v0_8')
            fig = plt.figure(figsize=(20, 12))
            
            # Layout dashboard
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Gauge Win Rate
            ax1 = fig.add_subplot(gs[0, 0])
            self.create_gauge_chart(ax1, metrics.win_rate, "Win Rate", "%", [50, 70, 90])
            
            # 2. Gauge Profit Factor
            ax2 = fig.add_subplot(gs[0, 1])
            self.create_gauge_chart(ax2, metrics.profit_factor, "Profit Factor", "", [1.2, 1.5, 2.0])
            
            # 3. Gauge Sharpe Ratio
            ax3 = fig.add_subplot(gs[0, 2])
            self.create_gauge_chart(ax3, metrics.sharpe_ratio, "Sharpe Ratio", "", [1.0, 1.5, 2.0])
            
            # 4. Gauge Max Drawdown
            ax4 = fig.add_subplot(gs[0, 3])
            self.create_gauge_chart(ax4, metrics.max_drawdown, "Max Drawdown", "%", [10, 20, 30], reverse=True)
            
            # 5. Performance Summary Table
            ax5 = fig.add_subplot(gs[1, :2])
            self.create_metrics_table(ax5, metrics)
            
            # 6. Risk Analysis
            ax6 = fig.add_subplot(gs[1, 2:])
            self.create_risk_analysis_chart(ax6, metrics)
            
            # 7. Efficiency Metrics
            ax7 = fig.add_subplot(gs[2, :2])
            self.create_efficiency_chart(ax7, metrics)
            
            # 8. Recommendations
            ax8 = fig.add_subplot(gs[2, 2:])
            health_eval = self.evaluate_strategy_health(metrics)
            self.create_recommendations_text(ax8, health_eval)
            
            plt.suptitle('Trading Performance Dashboard', fontsize=20, fontweight='bold')
            plt.savefig(f'{output_dir}/performance_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Dashboard salvata in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione dashboard: {e}")
    
    def create_gauge_chart(self, ax, value, title, unit, thresholds, reverse=False):
        """Crea grafico gauge"""
        try:
            # Colori basati su thresholds
            if reverse:
                if value <= thresholds[0]:
                    color = 'green'
                elif value <= thresholds[1]:
                    color = 'yellow'
                else:
                    color = 'red'
            else:
                if value >= thresholds[2]:
                    color = 'green'
                elif value >= thresholds[1]:
                    color = 'yellow'
                else:
                    color = 'red'
            
            # Crea gauge semplificato
            ax.pie([value, 100-value], colors=[color, 'lightgray'], startangle=90, counterclock=False)
            ax.set_title(f'{title}\n{value:.1f}{unit}', fontsize=12, fontweight='bold')
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione gauge: {e}")
    
    def create_metrics_table(self, ax, metrics):
        """Crea tabella metriche"""
        try:
            ax.axis('off')
            
            data = [
                ['Total Trades', f'{metrics.total_trades:,}'],
                ['Win Rate', f'{metrics.win_rate:.1f}%'],
                ['Profit Factor', f'{metrics.profit_factor:.2f}'],
                ['Total Profit', f'{metrics.total_profit:.2f} €'],
                ['Max Drawdown', f'{metrics.max_drawdown:.2f}%'],
                ['Sharpe Ratio', f'{metrics.sharpe_ratio:.2f}'],
                ['Expectancy', f'{metrics.expectancy:.2f} €'],
                ['Kelly %', f'{metrics.kelly_percentage:.1f}%']
            ]
            
            table = ax.table(cellText=data, colLabels=['Metric', 'Value'],
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            ax.set_title('Key Metrics Summary', fontsize=14, fontweight='bold')
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione tabella: {e}")
    
    def create_risk_analysis_chart(self, ax, metrics):
        """Crea grafico analisi rischio"""
        try:
            categories = ['Win Rate', 'Profit Factor', 'Sharpe Ratio', 'Recovery Factor']
            values = [
                min(metrics.win_rate, 100),
                min(metrics.profit_factor * 20, 100),  # Scala per visualizzazione
                min(metrics.sharpe_ratio * 25, 100),   # Scala per visualizzazione
                min(metrics.recovery_factor * 10, 100) # Scala per visualizzazione
            ]
            
            bars = ax.barh(categories, values, color=['green' if v >= 60 else 'yellow' if v >= 40 else 'red' for v in values])
            ax.set_xlim(0, 100)
            ax.set_xlabel('Performance Score')
            ax.set_title('Risk Analysis', fontsize=14, fontweight='bold')
            
            # Aggiungi valori sulle barre
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + 2, bar.get_y() + bar.get_height()/2, f'{value:.1f}', 
                       va='center', fontweight='bold')
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione grafico rischio: {e}")
    
    def create_efficiency_chart(self, ax, metrics):
        """Crea grafico efficienza"""
        try:
            labels = ['Avg Win', 'Avg Loss', 'Largest Win', 'Largest Loss']
            values = [metrics.average_win, abs(metrics.average_loss), 
                     metrics.largest_win, abs(metrics.largest_loss)]
            colors = ['green', 'red', 'darkgreen', 'darkred']
            
            bars = ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_ylabel('Amount (€)')
            ax.set_title('Trade Efficiency Analysis', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Aggiungi valori sulle barre
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.2f}€', ha='center', va='bottom', fontweight='bold')
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione grafico efficienza: {e}")
    
    def create_recommendations_text(self, ax, health_eval):
        """Crea testo raccomandazioni"""
        try:
            ax.axis('off')
            
            text = f"STRATEGY HEALTH: {health_eval['health_status']}\n"
            text += f"Score: {health_eval['health_score']}/100\n\n"
            
            if health_eval.get('recommendations'):
                text += "RECOMMENDATIONS:\n"
                for i, rec in enumerate(health_eval['recommendations'][:3], 1):
                    text += f"{i}. {rec}\n"
            else:
                text += "✅ Strategy performing optimally\n"
                text += "Continue current approach"
            
            ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.set_title('Strategy Recommendations', fontsize=14, fontweight='bold')
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione raccomandazioni: {e}")
    
    def run_continuous_monitoring(self, interval_minutes: int = 30):
        """Esegue monitoraggio continuo delle performance"""
        import time
        
        self.logger.info(f"Avvio monitoraggio continuo ogni {interval_minutes} minuti")
        
        while True:
            try:
                # Calcola metriche
                metrics = self.calculate_all_metrics()
                
                # Valuta salute strategia
                health_eval = self.evaluate_strategy_health(metrics)
                
                # Controlla alerts
                alerts = self.check_performance_alerts(metrics)
                
                # Salva snapshot
                self.save_performance_snapshot(metrics, health_eval)
                
                # Log status
                self.logger.info(f"Performance check: Health={health_eval['health_status']}, "
                               f"Win Rate={metrics.win_rate:.1f}%, "
                               f"Profit Factor={metrics.profit_factor:.2f}")
                
                if alerts:
                    self.logger.warning(f"⚠️  {len(alerts)} performance alerts generated")
                
                # Aspetta prossimo check
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                self.logger.info("Monitoraggio interrotto dall'utente")
                break
            except Exception as e:
                self.logger.error(f"Errore nel monitoraggio: {e}")
                time.sleep(60)  # Aspetta 1 minuto prima di riprovare

def main():
    """Test del sistema di analisi performance"""
    print("📊 ANALIZZATORE DI PERFORMANCE AVANZATO")
    print("=" * 50)
    
    # Inizializza analyzer
    analyzer = PerformanceAnalyzer()
    
    # Calcola metriche per ultimi 30 giorni
    print("📈 Calcolando metriche performance...")
    metrics = analyzer.calculate_all_metrics(period_days=30)
    
    # Valuta salute strategia
    health_eval = analyzer.evaluate_strategy_health(metrics)
    
    # Controlla alerts
    alerts = analyzer.check_performance_alerts(metrics)
    
    # Genera report
    report = analyzer.generate_performance_report(metrics, health_eval)
    print(report)
    
    # Crea dashboard
    print("\n📊 Creando dashboard performance...")
    analyzer.create_performance_dashboard(metrics)
    
    # Mostra alerts se presenti
    if alerts:
        print(f"\n⚠️  PERFORMANCE ALERTS ({len(alerts)}):")
        for alert in alerts:
            print(f"   {alert['severity']}: {alert['message']}")
    else:
        print("\n✅ Nessun alert di performance")
    
    print("\n✅ Analisi performance completata!")

if __name__ == "__main__":
    main()
