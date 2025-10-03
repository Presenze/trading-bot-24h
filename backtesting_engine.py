#!/usr/bin/env python3
"""
SISTEMA DI BACKTESTING RIGOROSO
- Simulazione su dati storici reali
- Validazione su diverse condizioni di mercato
- Calcolo metriche di performance avanzate
- Test di robustezza della strategia
"""

import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import delle strategie esistenti
from global_market_strategy import GlobalMarketStrategy
from risk_management import AdvancedRiskManager
from config import TradingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Risultati del backtest"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    recovery_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int
    profit_per_month: float
    trades_per_month: float
    start_balance: float
    end_balance: float
    equity_curve: List[float]
    drawdown_curve: List[float]
    trade_history: List[Dict]

class BacktestingEngine:
    """
    Motore di backtesting avanzato per validazione strategie
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logger
        self.global_strategy = GlobalMarketStrategy()
        self.risk_manager = AdvancedRiskManager(config)
        
        # Parametri backtesting
        self.initial_balance = config.INITIAL_BALANCE
        self.commission_per_trade = 0.0  # Spread incluso nei dati
        self.slippage = 0.0001  # 1 pip di slippage
        
        # Database per risultati
        self.setup_database()
        
    def setup_database(self):
        """Setup database per risultati backtesting"""
        self.conn = sqlite3.connect('backtesting_results.db')
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                symbol TEXT,
                timeframe TEXT,
                start_date DATE,
                end_date DATE,
                initial_balance REAL,
                final_balance REAL,
                total_return REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                sharpe_ratio REAL,
                total_trades INTEGER,
                timestamp DATETIME
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id INTEGER,
                symbol TEXT,
                direction TEXT,
                entry_time DATETIME,
                exit_time DATETIME,
                entry_price REAL,
                exit_price REAL,
                volume REAL,
                profit REAL,
                commission REAL,
                duration_minutes INTEGER,
                FOREIGN KEY (backtest_id) REFERENCES backtest_runs (id)
            )
        ''')
        
        self.conn.commit()
        
    def get_historical_data(self, symbol: str, timeframe: int, 
                          start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Ottiene dati storici per il backtesting"""
        try:
            if not mt5.initialize():
                raise Exception("MT5 non inizializzato")
            
            # Calcola il numero di candele necessarie
            if timeframe == mt5.TIMEFRAME_M1:
                minutes_per_candle = 1
            elif timeframe == mt5.TIMEFRAME_M5:
                minutes_per_candle = 5
            elif timeframe == mt5.TIMEFRAME_M15:
                minutes_per_candle = 15
            elif timeframe == mt5.TIMEFRAME_H1:
                minutes_per_candle = 60
            elif timeframe == mt5.TIMEFRAME_H4:
                minutes_per_candle = 240
            elif timeframe == mt5.TIMEFRAME_D1:
                minutes_per_candle = 1440
            else:
                minutes_per_candle = 60
            
            total_minutes = int((end_date - start_date).total_seconds() / 60)
            count = int(total_minutes / minutes_per_candle) + 100  # Buffer
            
            # Ottieni dati da MT5
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                self.logger.error(f"Nessun dato storico per {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Filtra per il periodo richiesto
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            self.logger.info(f"Caricati {len(df)} candele per {symbol} dal {start_date} al {end_date}")
            return df
            
        except Exception as e:
            self.logger.error(f"Errore nel caricare dati storici: {e}")
            return pd.DataFrame()
    
    def run_backtest(self, symbol: str, timeframe: int, 
                    start_date: datetime, end_date: datetime,
                    strategy_name: str = "GlobalMarketStrategy") -> BacktestResult:
        """Esegue backtesting completo"""
        try:
            self.logger.info(f"Avvio backtesting per {symbol} dal {start_date} al {end_date}")
            
            # Carica dati storici
            df = self.get_historical_data(symbol, timeframe, start_date, end_date)
            if df.empty:
                raise Exception("Nessun dato storico disponibile")
            
            # Inizializza variabili
            balance = self.initial_balance
            equity_curve = [balance]
            drawdown_curve = [0.0]
            peak_balance = balance
            trades = []
            positions = []
            
            # Statistiche
            winning_trades = 0
            losing_trades = 0
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            # Simula trading su ogni candela
            for i in range(50, len(df)):  # Inizia da 50 per avere abbastanza dati per indicatori
                current_time = df.index[i]
                current_data = df.iloc[:i+1]  # Dati fino al momento corrente
                
                # Genera segnale usando la strategia
                signal = self.generate_backtest_signal(symbol, current_data)
                
                # Gestisci posizioni esistenti
                for pos in positions[:]:  # Copia la lista per modificarla
                    exit_signal = self.check_exit_conditions(pos, current_data.iloc[-1])
                    if exit_signal:
                        # Chiudi posizione
                        exit_price = current_data.iloc[-1]['close']
                        if pos['direction'] == 'BUY':
                            profit = (exit_price - pos['entry_price']) * pos['volume']
                        else:
                            profit = (pos['entry_price'] - exit_price) * pos['volume']
                        
                        # Applica commissioni e slippage
                        profit -= self.commission_per_trade
                        if pos['direction'] == 'BUY':
                            profit -= self.slippage * pos['volume']
                        else:
                            profit += self.slippage * pos['volume']
                        
                        balance += profit
                        
                        # Registra trade
                        trade = {
                            'symbol': symbol,
                            'direction': pos['direction'],
                            'entry_time': pos['entry_time'],
                            'exit_time': current_time,
                            'entry_price': pos['entry_price'],
                            'exit_price': exit_price,
                            'volume': pos['volume'],
                            'profit': profit,
                            'commission': self.commission_per_trade,
                            'duration_minutes': int((current_time - pos['entry_time']).total_seconds() / 60)
                        }
                        trades.append(trade)
                        
                        # Aggiorna statistiche
                        if profit > 0:
                            winning_trades += 1
                            consecutive_wins += 1
                            consecutive_losses = 0
                            max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                        else:
                            losing_trades += 1
                            consecutive_losses += 1
                            consecutive_wins = 0
                            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        
                        positions.remove(pos)
                
                # Apri nuova posizione se c'è segnale
                if signal and len(positions) < 5:  # Max 5 posizioni simultanee
                    # Calcola volume usando risk management
                    volume = self.calculate_position_size(balance, symbol, current_data.iloc[-1]['close'])
                    
                    if volume > 0:
                        entry_price = current_data.iloc[-1]['close']
                        
                        # Applica slippage
                        if signal['direction'] == 'BUY':
                            entry_price += self.slippage
                        else:
                            entry_price -= self.slippage
                        
                        position = {
                            'symbol': symbol,
                            'direction': signal['direction'],
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'volume': volume,
                            'stop_loss': self.calculate_stop_loss(entry_price, signal['direction']),
                            'take_profit': self.calculate_take_profit(entry_price, signal['direction'])
                        }
                        positions.append(position)
                
                # Aggiorna equity curve
                current_equity = balance
                for pos in positions:
                    current_price = current_data.iloc[-1]['close']
                    if pos['direction'] == 'BUY':
                        unrealized_pnl = (current_price - pos['entry_price']) * pos['volume']
                    else:
                        unrealized_pnl = (pos['entry_price'] - current_price) * pos['volume']
                    current_equity += unrealized_pnl
                
                equity_curve.append(current_equity)
                
                # Calcola drawdown
                if current_equity > peak_balance:
                    peak_balance = current_equity
                    drawdown = 0.0
                else:
                    drawdown = (peak_balance - current_equity) / peak_balance * 100
                drawdown_curve.append(drawdown)
            
            # Chiudi tutte le posizioni rimanenti
            final_price = df.iloc[-1]['close']
            for pos in positions:
                if pos['direction'] == 'BUY':
                    profit = (final_price - pos['entry_price']) * pos['volume']
                else:
                    profit = (pos['entry_price'] - final_price) * pos['volume']
                
                profit -= self.commission_per_trade
                balance += profit
                
                trade = {
                    'symbol': symbol,
                    'direction': pos['direction'],
                    'entry_time': pos['entry_time'],
                    'exit_time': df.index[-1],
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'volume': pos['volume'],
                    'profit': profit,
                    'commission': self.commission_per_trade,
                    'duration_minutes': int((df.index[-1] - pos['entry_time']).total_seconds() / 60)
                }
                trades.append(trade)
                
                if profit > 0:
                    winning_trades += 1
                else:
                    losing_trades += 1
            
            # Calcola metriche finali
            result = self.calculate_backtest_metrics(
                trades, equity_curve, drawdown_curve, 
                self.initial_balance, balance,
                winning_trades, losing_trades,
                max_consecutive_wins, max_consecutive_losses,
                start_date, end_date
            )
            
            # Salva risultati nel database
            self.save_backtest_results(result, symbol, timeframe, start_date, end_date, strategy_name)
            
            self.logger.info(f"Backtesting completato: {len(trades)} trades, Win Rate: {result.win_rate:.1f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"Errore nel backtesting: {e}")
            raise
    
    def generate_backtest_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Genera segnale per backtesting usando la strategia esistente"""
        try:
            # Usa la strategia globale esistente
            indicators = self.global_strategy.calculate_technical_indicators(df)
            if not indicators:
                return None
            
            # Logica semplificata per backtesting
            signal_strength = 0
            
            # RSI
            if indicators['rsi'] < 30:
                signal_strength += 0.3
            elif indicators['rsi'] > 70:
                signal_strength -= 0.3
            
            # MACD
            if indicators['macd'] > indicators['macd_signal']:
                signal_strength += 0.2
            else:
                signal_strength -= 0.2
            
            # Bollinger Bands
            if indicators['current_price'] < indicators['bb_lower']:
                signal_strength += 0.2
            elif indicators['current_price'] > indicators['bb_upper']:
                signal_strength -= 0.2
            
            # EMA
            if indicators['ema_fast'] > indicators['ema_slow']:
                signal_strength += 0.1
            else:
                signal_strength -= 0.1
            
            # Genera segnale
            if signal_strength > 0.3:
                return {'direction': 'BUY', 'strength': signal_strength}
            elif signal_strength < -0.3:
                return {'direction': 'SELL', 'strength': abs(signal_strength)}
            
            return None
            
        except Exception as e:
            self.logger.error(f"Errore nella generazione segnale backtesting: {e}")
            return None
    
    def check_exit_conditions(self, position: Dict, current_candle: pd.Series) -> bool:
        """Controlla condizioni di uscita per una posizione"""
        try:
            current_price = current_candle['close']
            
            # Stop Loss
            if position['direction'] == 'BUY':
                if current_price <= position['stop_loss']:
                    return True
                if current_price >= position['take_profit']:
                    return True
            else:
                if current_price >= position['stop_loss']:
                    return True
                if current_price <= position['take_profit']:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Errore nel controllo condizioni uscita: {e}")
            return False
    
    def calculate_position_size(self, balance: float, symbol: str, price: float) -> float:
        """Calcola dimensione posizione per backtesting"""
        try:
            # Usa 2% del balance per trade
            risk_amount = balance * 0.02
            
            # Calcola volume basato su stop loss di 50 pips
            stop_loss_pips = 50
            pip_value = 0.0001  # Per la maggior parte delle coppie forex
            
            volume = risk_amount / (stop_loss_pips * pip_value * price)
            
            # Limita volume
            volume = min(volume, balance * 0.1 / price)  # Max 10% del balance
            volume = max(volume, 0.01)  # Min 0.01 lotti
            
            return round(volume, 2)
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo position size: {e}")
            return 0.01
    
    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """Calcola stop loss"""
        stop_loss_pips = 50
        pip_value = 0.0001
        
        if direction == 'BUY':
            return entry_price - (stop_loss_pips * pip_value)
        else:
            return entry_price + (stop_loss_pips * pip_value)
    
    def calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """Calcola take profit"""
        take_profit_pips = 100
        pip_value = 0.0001
        
        if direction == 'BUY':
            return entry_price + (take_profit_pips * pip_value)
        else:
            return entry_price - (take_profit_pips * pip_value)
    
    def calculate_backtest_metrics(self, trades: List[Dict], equity_curve: List[float],
                                 drawdown_curve: List[float], start_balance: float,
                                 end_balance: float, winning_trades: int, losing_trades: int,
                                 max_consecutive_wins: int, max_consecutive_losses: int,
                                 start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calcola tutte le metriche di performance"""
        try:
            total_trades = len(trades)
            
            if total_trades == 0:
                return BacktestResult(
                    total_trades=0, winning_trades=0, losing_trades=0,
                    win_rate=0, profit_factor=0, total_return=0,
                    max_drawdown=0, sharpe_ratio=0, sortino_ratio=0,
                    calmar_ratio=0, recovery_factor=0, average_win=0,
                    average_loss=0, largest_win=0, largest_loss=0,
                    consecutive_wins=0, consecutive_losses=0,
                    profit_per_month=0, trades_per_month=0,
                    start_balance=start_balance, end_balance=end_balance,
                    equity_curve=equity_curve, drawdown_curve=drawdown_curve,
                    trade_history=trades
                )
            
            # Calcola metriche base
            profits = [trade['profit'] for trade in trades]
            wins = [p for p in profits if p > 0]
            losses = [p for p in profits if p < 0]
            
            win_rate = (winning_trades / total_trades) * 100
            total_return = ((end_balance - start_balance) / start_balance) * 100
            max_drawdown = max(drawdown_curve) if drawdown_curve else 0
            
            # Profit Factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Medie
            average_win = np.mean(wins) if wins else 0
            average_loss = np.mean(losses) if losses else 0
            largest_win = max(wins) if wins else 0
            largest_loss = min(losses) if losses else 0
            
            # Sharpe Ratio
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Sortino Ratio
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else 0
            sortino_ratio = np.mean(returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Calmar Ratio
            calmar_ratio = (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0
            
            # Recovery Factor
            recovery_factor = (total_return / 100) / (max_drawdown / 100) if max_drawdown > 0 else 0
            
            # Metriche temporali
            duration = end_date - start_date
            months = duration.days / 30.44
            profit_per_month = total_return / months if months > 0 else 0
            trades_per_month = total_trades / months if months > 0 else 0
            
            return BacktestResult(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                recovery_factor=recovery_factor,
                average_win=average_win,
                average_loss=average_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=max_consecutive_wins,
                consecutive_losses=max_consecutive_losses,
                profit_per_month=profit_per_month,
                trades_per_month=trades_per_month,
                start_balance=start_balance,
                end_balance=end_balance,
                equity_curve=equity_curve,
                drawdown_curve=drawdown_curve,
                trade_history=trades
            )
            
        except Exception as e:
            self.logger.error(f"Errore nel calcolo metriche: {e}")
            raise
    
    def save_backtest_results(self, result: BacktestResult, symbol: str, timeframe: int,
                            start_date: datetime, end_date: datetime, strategy_name: str):
        """Salva risultati backtesting nel database"""
        try:
            cursor = self.conn.cursor()
            
            # Salva run principale
            cursor.execute('''
                INSERT INTO backtest_runs (
                    strategy_name, symbol, timeframe, start_date, end_date,
                    initial_balance, final_balance, total_return, max_drawdown,
                    win_rate, profit_factor, sharpe_ratio, total_trades, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                strategy_name, symbol, f"TF_{timeframe}", start_date.date(), end_date.date(),
                result.start_balance, result.end_balance, result.total_return, result.max_drawdown,
                result.win_rate, result.profit_factor, result.sharpe_ratio, result.total_trades,
                datetime.now()
            ))
            
            backtest_id = cursor.lastrowid
            
            # Salva trades individuali
            for trade in result.trade_history:
                cursor.execute('''
                    INSERT INTO backtest_trades (
                        backtest_id, symbol, direction, entry_time, exit_time,
                        entry_price, exit_price, volume, profit, commission, duration_minutes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    backtest_id, trade['symbol'], trade['direction'],
                    trade['entry_time'], trade['exit_time'], trade['entry_price'],
                    trade['exit_price'], trade['volume'], trade['profit'],
                    trade['commission'], trade['duration_minutes']
                ))
            
            self.conn.commit()
            self.logger.info(f"Risultati backtesting salvati con ID: {backtest_id}")
            
        except Exception as e:
            self.logger.error(f"Errore nel salvare risultati: {e}")
    
    def generate_backtest_report(self, result: BacktestResult, symbol: str) -> str:
        """Genera report dettagliato del backtesting"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           BACKTEST REPORT - {symbol}                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ PERFORMANCE OVERVIEW                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Total Return:           {result.total_return:>8.2f}%                        ║
║ Start Balance:          {result.start_balance:>8.2f} €                      ║
║ End Balance:            {result.end_balance:>8.2f} €                        ║
║ Net Profit:             {result.end_balance - result.start_balance:>8.2f} € ║
║                                                                              ║
║ TRADE STATISTICS                                                             ║
║ Total Trades:           {result.total_trades:>8d}                           ║
║ Winning Trades:         {result.winning_trades:>8d}                         ║
║ Losing Trades:          {result.losing_trades:>8d}                          ║
║ Win Rate:               {result.win_rate:>8.1f}%                            ║
║                                                                              ║
║ RISK METRICS                                                                 ║
║ Profit Factor:          {result.profit_factor:>8.2f}                        ║
║ Max Drawdown:           {result.max_drawdown:>8.2f}%                        ║
║ Sharpe Ratio:           {result.sharpe_ratio:>8.2f}                         ║
║ Sortino Ratio:          {result.sortino_ratio:>8.2f}                        ║
║ Calmar Ratio:           {result.calmar_ratio:>8.2f}                         ║
║ Recovery Factor:        {result.recovery_factor:>8.2f}                      ║
║                                                                              ║
║ TRADE ANALYSIS                                                               ║
║ Average Win:            {result.average_win:>8.2f} €                        ║
║ Average Loss:           {result.average_loss:>8.2f} €                       ║
║ Largest Win:            {result.largest_win:>8.2f} €                        ║
║ Largest Loss:           {result.largest_loss:>8.2f} €                       ║
║ Max Consecutive Wins:   {result.consecutive_wins:>8d}                       ║
║ Max Consecutive Losses: {result.consecutive_losses:>8d}                     ║
║                                                                              ║
║ MONTHLY METRICS                                                              ║
║ Profit per Month:       {result.profit_per_month:>8.2f}%                    ║
║ Trades per Month:       {result.trades_per_month:>8.1f}                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

STRATEGY EVALUATION:
"""
        
        # Valutazione strategia
        if result.win_rate >= 60 and result.profit_factor >= 1.5 and result.max_drawdown <= 20:
            report += "✅ EXCELLENT STRATEGY - Recommended for live trading\n"
        elif result.win_rate >= 50 and result.profit_factor >= 1.2 and result.max_drawdown <= 30:
            report += "✅ GOOD STRATEGY - Consider optimization before live trading\n"
        elif result.win_rate >= 40 and result.profit_factor >= 1.0 and result.max_drawdown <= 40:
            report += "⚠️  AVERAGE STRATEGY - Needs significant optimization\n"
        else:
            report += "❌ POOR STRATEGY - Not recommended for live trading\n"
        
        return report
    
    def run_multi_symbol_backtest(self, symbols: List[str], timeframe: int,
                                start_date: datetime, end_date: datetime) -> Dict[str, BacktestResult]:
        """Esegue backtesting su multipli simboli"""
        results = {}
        
        self.logger.info(f"Avvio backtesting multi-simbolo su {len(symbols)} simboli")
        
        for symbol in symbols:
            try:
                self.logger.info(f"Backtesting {symbol}...")
                result = self.run_backtest(symbol, timeframe, start_date, end_date)
                results[symbol] = result
                
                # Stampa risultati parziali
                print(f"\n{symbol}: Return {result.total_return:.1f}%, Win Rate {result.win_rate:.1f}%, Max DD {result.max_drawdown:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Errore backtesting {symbol}: {e}")
                continue
        
        return results
    
    def create_performance_charts(self, result: BacktestResult, symbol: str, output_dir: str = "backtest_charts"):
        """Crea grafici di performance"""
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            # Setup stile
            plt.style.use('seaborn-v0_8')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Backtest Performance Analysis - {symbol}', fontsize=16, fontweight='bold')
            
            # 1. Equity Curve
            ax1.plot(result.equity_curve, color='blue', linewidth=2)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Balance (€)')
            ax1.grid(True, alpha=0.3)
            
            # 2. Drawdown
            ax2.fill_between(range(len(result.drawdown_curve)), result.drawdown_curve, 
                           color='red', alpha=0.3)
            ax2.plot(result.drawdown_curve, color='red', linewidth=1)
            ax2.set_title('Drawdown Curve')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            
            # 3. Trade Distribution
            profits = [trade['profit'] for trade in result.trade_history]
            ax3.hist(profits, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax3.set_title('Profit Distribution')
            ax3.set_xlabel('Profit (€)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)
            
            # 4. Monthly Returns
            if len(result.trade_history) > 0:
                monthly_profits = {}
                for trade in result.trade_history:
                    month = trade['exit_time'].strftime('%Y-%m')
                    if month not in monthly_profits:
                        monthly_profits[month] = 0
                    monthly_profits[month] += trade['profit']
                
                months = list(monthly_profits.keys())
                profits = list(monthly_profits.values())
                
                colors = ['green' if p >= 0 else 'red' for p in profits]
                ax4.bar(range(len(months)), profits, color=colors, alpha=0.7)
                ax4.set_title('Monthly Returns')
                ax4.set_ylabel('Profit (€)')
                ax4.set_xticks(range(len(months)))
                ax4.set_xticklabels(months, rotation=45)
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/backtest_{symbol}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Grafici salvati in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Errore nella creazione grafici: {e}")

def main():
    """Test del sistema di backtesting"""
    from config import TradingConfig
    import config_account
    
    # Configurazione
    config = TradingConfig(
        MT5_ACCOUNT_ID=config_account.MT5_ACCOUNT_ID,
        MT5_PASSWORD=config_account.MT5_PASSWORD,
        MT5_SERVER=config_account.MT5_SERVER,
        MT5_IS_DEMO=config_account.MT5_IS_DEMO,
        INITIAL_BALANCE=10000,
        RISK_PER_TRADE=0.02,
        MAX_DAILY_TRADES=10,
        COMMISSION_RATE=0.10,
        MIN_PROFIT_THRESHOLD=1.0
    )
    
    # Inizializza backtesting engine
    engine = BacktestingEngine(config)
    
    print("🔬 SISTEMA DI BACKTESTING RIGOROSO")
    print("=" * 50)
    
    # Definisci periodo di test
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # 3 mesi di dati
    
    # Test su simboli principali
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
    
    print(f"📊 Backtesting dal {start_date.date()} al {end_date.date()}")
    print(f"🎯 Simboli: {', '.join(symbols)}")
    print()
    
    # Esegui backtesting multi-simbolo
    results = engine.run_multi_symbol_backtest(symbols, mt5.TIMEFRAME_H1, start_date, end_date)
    
    # Stampa riassunto
    print("\n" + "="*80)
    print("RIASSUNTO BACKTESTING")
    print("="*80)
    
    total_return = 0
    total_trades = 0
    avg_win_rate = 0
    
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Return: {result.total_return:.2f}%")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.1f}%")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Total Trades: {result.total_trades}")
        
        total_return += result.total_return
        total_trades += result.total_trades
        avg_win_rate += result.win_rate
        
        # Genera report dettagliato
        report = engine.generate_backtest_report(result, symbol)
        print(report)
        
        # Crea grafici
        engine.create_performance_charts(result, symbol)
    
    # Statistiche aggregate
    if results:
        avg_return = total_return / len(results)
        avg_win_rate = avg_win_rate / len(results)
        
        print(f"\n📈 STATISTICHE AGGREGATE:")
        print(f"   Return Medio: {avg_return:.2f}%")
        print(f"   Win Rate Medio: {avg_win_rate:.1f}%")
        print(f"   Trades Totali: {total_trades}")
    
    print("\n✅ Backtesting completato!")

if __name__ == "__main__":
    main()
