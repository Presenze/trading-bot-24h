#!/usr/bin/env python3
"""
SISTEMA INTEGRATO 24/7 - TRADING AUTOMATICO COMPLETO
- Tutti i sistemi collegati e funzionanti
- Strategie adattive automatiche
- Gestione rischio avanzata
- Performance monitoring continuo
- Sicurezza enterprise
- Profitti automatici 24h/24h
"""

import asyncio
import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

# Import tutti i sistemi avanzati
from backtesting_engine import BacktestingEngine
from performance_analyzer import PerformanceAnalyzer
from adaptive_strategy_engine import AdaptiveStrategyEngine
from enhanced_risk_management import EnhancedRiskManager
from security_manager import SecurityManager
from main import AdvancedTradingStrategy, TradingWebInterface
from config import TradingConfig
import config_account

# Setup logging avanzato
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sistema_integrato_24h.log'),
        logging.StreamHandler()
    ]
)

class SistemaIntegrato24h:
    """
    Sistema di Trading AI Integrato 24/7
    Combina tutti i componenti avanzati per trading automatico continuo
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.start_time = datetime.now()
        
        # Configurazione
        self.config = TradingConfig(
            MT5_ACCOUNT_ID=config_account.MT5_ACCOUNT_ID,
            MT5_PASSWORD=config_account.MT5_PASSWORD,
            MT5_SERVER=config_account.MT5_SERVER,
            MT5_IS_DEMO=config_account.MT5_IS_DEMO,
            INITIAL_BALANCE=config_account.INITIAL_BALANCE,
            RISK_PER_TRADE=config_account.RISK_PER_TRADE,
            MAX_DAILY_TRADES=config_account.MAX_DAILY_TRADES,
            COMMISSION_RATE=config_account.COMMISSION_RATE,
            MIN_PROFIT_THRESHOLD=config_account.MIN_PROFIT_THRESHOLD
        )
        
        # Inizializza tutti i sistemi
        self.logger.info("🚀 Inizializzando Sistema Integrato 24/7...")
        
        # 1. Sistema di Sicurezza (primo per protezione)
        self.security_manager = SecurityManager({
            'encryption_password': 'trading_ai_secure_2024',
            'jwt_secret': 'sistema_integrato_jwt_secret'
        })
        
        # 2. Gestione Rischio Avanzata
        self.risk_manager = EnhancedRiskManager(self.config.INITIAL_BALANCE)
        
        # 3. Motore Strategia Adattiva
        self.adaptive_engine = AdaptiveStrategyEngine()
        
        # 4. Analizzatore Performance
        self.performance_analyzer = PerformanceAnalyzer()
        
        # 5. Sistema Trading Principale
        self.trading_strategy = AdvancedTradingStrategy(self.config)
        
        # 6. Interfaccia Web
        self.web_interface = TradingWebInterface(self.trading_strategy)
        
        # 7. Backtesting Engine (per validazione continua)
        self.backtesting_engine = BacktestingEngine(self.config)
        
        # Statistiche sistema
        self.stats = {
            'total_trades': 0,
            'profitable_trades': 0,
            'total_profit': 0.0,
            'uptime_hours': 0,
            'strategies_switched': 0,
            'risks_prevented': 0,
            'security_events': 0
        }
        
        self.logger.info("✅ Tutti i sistemi inizializzati con successo!")
    
    def start_sistema_completo(self):
        """Avvia il sistema completo 24/7"""
        try:
            self.running = True
            self.logger.info("🌟 AVVIO SISTEMA INTEGRATO 24/7")
            self.logger.info("=" * 60)
            
            # Avvia tutti i thread
            threads = []
            
            # 1. Thread Trading Principale
            trading_thread = threading.Thread(target=self.run_trading_loop, daemon=True)
            trading_thread.start()
            threads.append(trading_thread)
            
            # 2. Thread Strategia Adattiva
            adaptive_thread = threading.Thread(target=self.run_adaptive_loop, daemon=True)
            adaptive_thread.start()
            threads.append(adaptive_thread)
            
            # 3. Thread Performance Monitoring
            performance_thread = threading.Thread(target=self.run_performance_loop, daemon=True)
            performance_thread.start()
            threads.append(performance_thread)
            
            # 4. Thread Sicurezza
            security_thread = threading.Thread(target=self.run_security_loop, daemon=True)
            security_thread.start()
            threads.append(security_thread)
            
            # 5. Thread Web Interface
            web_thread = threading.Thread(target=self.run_web_interface, daemon=True)
            web_thread.start()
            threads.append(web_thread)
            
            # 6. Thread Statistiche
            stats_thread = threading.Thread(target=self.run_stats_loop, daemon=True)
            stats_thread.start()
            threads.append(stats_thread)
            
            self.logger.info("🚀 Tutti i thread avviati - Sistema 24/7 ATTIVO!")
            self.logger.info("🌐 Dashboard: http://localhost:5000")
            self.logger.info("📊 Monitoraggio: sistema_integrato_24h.log")
            
            # Loop principale di controllo
            self.run_main_control_loop()
            
        except Exception as e:
            self.logger.error(f"❌ Errore avvio sistema: {e}")
            self.stop_sistema()
    
    def run_trading_loop(self):
        """Loop trading principale con sistema integrato"""
        self.logger.info("💰 Trading Loop 24/7 avviato")
        
        while self.running:
            try:
                # Esegui ciclo trading con tutti i sistemi integrati
                self.execute_integrated_trading_cycle()
                
                # Pausa ottimizzata per performance
                time.sleep(10)  # 10 secondi tra i cicli
                
            except Exception as e:
                self.logger.error(f"❌ Errore trading loop: {e}")
                time.sleep(30)
    
    def execute_integrated_trading_cycle(self):
        """Esegue un ciclo di trading integrato completo"""
        try:
            # 1. Analisi mercato con strategia adattiva
            market_analysis = self.adaptive_engine.analyze_market_conditions("EURUSD")
            
            # 2. Selezione strategia ottimale
            optimal_strategy = self.adaptive_engine.select_optimal_strategy(market_analysis, "EURUSD")
            
            # 3. Controllo se cambiare strategia
            should_switch, reason = self.adaptive_engine.should_switch_strategy(
                self.adaptive_engine.current_strategy, optimal_strategy, market_analysis
            )
            
            if should_switch:
                self.adaptive_engine.switch_strategy(optimal_strategy, reason, market_analysis.condition)
                self.stats['strategies_switched'] += 1
                self.logger.info(f"🔄 Strategia cambiata: {optimal_strategy.value} - {reason}")
            
            # 4. Adatta parametri alle condizioni correnti
            adapted_params = self.adaptive_engine.adapt_strategy_parameters(market_analysis)
            
            # 5. Esegui trading con parametri adattati
            self.execute_adaptive_trading(adapted_params, market_analysis)
            
            # 6. Monitora e chiudi posizioni profittevoli
            self.monitor_and_close_profits()
            
        except Exception as e:
            self.logger.error(f"❌ Errore ciclo trading integrato: {e}")
    
    def execute_adaptive_trading(self, params, market_analysis):
        """Esegue trading con parametri adattivi"""
        try:
            # Ottieni segnali dal sistema globale
            signals = self.trading_strategy.global_market.scan_all_markets()
            
            for signal in signals[:10]:  # Top 10 segnali
                symbol = signal["symbol"]
                
                # Validazione rischio avanzata
                is_valid, message = self.risk_manager.validate_trade_request(
                    symbol, signal["direction"], params.position_size_multiplier,
                    signal.get("price", 1.0), 0, 0
                )
                
                if not is_valid:
                    self.stats['risks_prevented'] += 1
                    continue
                
                # Calcola position size dinamico
                position_size = self.risk_manager.calculate_dynamic_position_size(
                    symbol, signal.get("price", 1.0), 0, signal["confidence"]
                )
                
                # Calcola stop loss e take profit dinamici
                entry_price = signal.get("price", 1.0)
                stop_loss = self.risk_manager.calculate_dynamic_stop_loss(
                    symbol, entry_price, signal["direction"]
                )
                take_profit = self.risk_manager.calculate_dynamic_take_profit(
                    symbol, entry_price, stop_loss, signal["direction"]
                )
                
                # Esegui trade se tutto è valido
                if position_size > 0:
                    success = self.trading_strategy.execute_trade(
                        symbol, signal["direction"], position_size,
                        entry_price, stop_loss, take_profit
                    )
                    
                    if success:
                        self.stats['total_trades'] += 1
                        self.logger.info(f"✅ Trade eseguito: {symbol} {signal['direction']} - Size: {position_size}")
                
        except Exception as e:
            self.logger.error(f"❌ Errore trading adattivo: {e}")
    
    def monitor_and_close_profits(self):
        """Monitora e chiude posizioni profittevoli automaticamente"""
        try:
            # Usa il sistema di chiusura automatica esistente
            self.trading_strategy.auto_close_profitable_positions()
            self.trading_strategy.quick_profit_management()
            
        except Exception as e:
            self.logger.error(f"❌ Errore monitoraggio profitti: {e}")
    
    def run_adaptive_loop(self):
        """Loop strategia adattiva"""
        self.logger.info("🧠 Adaptive Strategy Loop avviato")
        
        while self.running:
            try:
                # Esegui ciclo adattivo ogni 5 minuti
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
                
                for symbol in symbols:
                    result = self.adaptive_engine.run_adaptive_cycle(symbol)
                    if result.get('strategy_switched'):
                        self.logger.info(f"🔄 {symbol}: {result['switch_reason']}")
                
                time.sleep(300)  # 5 minuti
                
            except Exception as e:
                self.logger.error(f"❌ Errore adaptive loop: {e}")
                time.sleep(60)
    
    def run_performance_loop(self):
        """Loop monitoraggio performance"""
        self.logger.info("📊 Performance Monitoring Loop avviato")
        
        while self.running:
            try:
                # Calcola metriche ogni 10 minuti
                metrics = self.performance_analyzer.calculate_all_metrics()
                
                # Controlla alert
                alerts = self.performance_analyzer.check_performance_alerts(metrics)
                
                if alerts:
                    for alert in alerts:
                        self.logger.warning(f"⚠️ PERFORMANCE ALERT: {alert['message']}")
                
                # Salva snapshot
                health_eval = self.performance_analyzer.evaluate_strategy_health(metrics)
                self.performance_analyzer.save_performance_snapshot(metrics, health_eval)
                
                # Aggiorna statistiche
                self.stats['profitable_trades'] = metrics.winning_trades
                self.stats['total_profit'] = metrics.total_profit
                
                time.sleep(600)  # 10 minuti
                
            except Exception as e:
                self.logger.error(f"❌ Errore performance loop: {e}")
                time.sleep(300)
    
    def run_security_loop(self):
        """Loop sicurezza"""
        self.logger.info("🔒 Security Monitoring Loop avviato")
        
        while self.running:
            try:
                # Controlli sicurezza ogni 2 minuti
                status = self.security_manager.get_security_status()
                
                if status.get('critical_events_1h', 0) > 0:
                    self.logger.warning(f"🚨 Eventi sicurezza critici: {status['critical_events_1h']}")
                    self.stats['security_events'] += status['critical_events_1h']
                
                # Backup automatico ogni ora
                if datetime.now().minute == 0:
                    self.security_manager.backup_security_data()
                
                time.sleep(120)  # 2 minuti
                
            except Exception as e:
                self.logger.error(f"❌ Errore security loop: {e}")
                time.sleep(60)
    
    def run_web_interface(self):
        """Avvia interfaccia web"""
        try:
            self.logger.info("🌐 Avviando Web Interface...")
            self.web_interface.run(host='0.0.0.0', port=5000)
        except Exception as e:
            self.logger.error(f"❌ Errore web interface: {e}")
    
    def run_stats_loop(self):
        """Loop statistiche sistema"""
        self.logger.info("📈 Stats Loop avviato")
        
        while self.running:
            try:
                # Aggiorna uptime
                uptime = datetime.now() - self.start_time
                self.stats['uptime_hours'] = uptime.total_seconds() / 3600
                
                # Log statistiche ogni ora
                if datetime.now().minute == 0:
                    self.log_system_stats()
                
                time.sleep(300)  # 5 minuti
                
            except Exception as e:
                self.logger.error(f"❌ Errore stats loop: {e}")
                time.sleep(60)
    
    def run_main_control_loop(self):
        """Loop principale di controllo sistema"""
        try:
            while self.running:
                # Controlli sistema ogni minuto
                self.check_system_health()
                time.sleep(60)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Interruzione manuale ricevuta")
            self.stop_sistema()
        except Exception as e:
            self.logger.error(f"❌ Errore control loop: {e}")
            self.stop_sistema()
    
    def check_system_health(self):
        """Controlla salute generale del sistema"""
        try:
            # Controlla connessione MT5
            if not self.trading_strategy.mt5_connected:
                self.logger.warning("⚠️ MT5 disconnesso - tentativo riconnessione...")
                self.trading_strategy.connect_mt5()
            
            # Controlla livelli di rischio
            risk_summary = self.risk_manager.get_risk_summary()
            if risk_summary.get('current_drawdown', 0) > 15:
                self.logger.warning(f"⚠️ Drawdown alto: {risk_summary['current_drawdown']:.1f}%")
            
            # Controlla threat level
            if self.security_manager.threat_level.value != 'none':
                self.logger.warning(f"🚨 Threat level: {self.security_manager.threat_level.value}")
            
        except Exception as e:
            self.logger.error(f"❌ Errore health check: {e}")
    
    def log_system_stats(self):
        """Log statistiche sistema"""
        try:
            self.logger.info("📊 STATISTICHE SISTEMA 24/7:")
            self.logger.info(f"   ⏰ Uptime: {self.stats['uptime_hours']:.1f} ore")
            self.logger.info(f"   💰 Trades Totali: {self.stats['total_trades']}")
            self.logger.info(f"   ✅ Trades Profittevoli: {self.stats['profitable_trades']}")
            self.logger.info(f"   💵 Profitto Totale: €{self.stats['total_profit']:.2f}")
            self.logger.info(f"   🔄 Strategie Cambiate: {self.stats['strategies_switched']}")
            self.logger.info(f"   🛡️ Rischi Prevenuti: {self.stats['risks_prevented']}")
            self.logger.info(f"   🔒 Eventi Sicurezza: {self.stats['security_events']}")
            
            # Salva statistiche
            with open('sistema_stats_24h.json', 'w') as f:
                json.dump({
                    **self.stats,
                    'timestamp': datetime.now().isoformat(),
                    'uptime_formatted': f"{self.stats['uptime_hours']:.1f} ore"
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"❌ Errore log stats: {e}")
    
    def stop_sistema(self):
        """Ferma il sistema completo"""
        try:
            self.logger.info("🛑 Fermando Sistema Integrato 24/7...")
            self.running = False
            
            # Salva statistiche finali
            self.log_system_stats()
            
            # Backup finale
            self.security_manager.backup_security_data()
            
            self.logger.info("✅ Sistema fermato correttamente")
            
        except Exception as e:
            self.logger.error(f"❌ Errore stop sistema: {e}")

def main():
    """Avvia il sistema integrato 24/7"""
    print("🌟 SISTEMA INTEGRATO 24/7 - TRADING AUTOMATICO COMPLETO")
    print("=" * 70)
    print("🚀 Avviando tutti i sistemi avanzati...")
    print("💰 Trading automatico continuo")
    print("🧠 Strategie adattive intelligenti")
    print("🛡️ Gestione rischio avanzata")
    print("📊 Monitoraggio performance continuo")
    print("🔒 Sicurezza enterprise")
    print("🌐 Dashboard web: http://localhost:5000")
    print("=" * 70)
    
    # Crea e avvia sistema
    sistema = SistemaIntegrato24h()
    sistema.start_sistema_completo()

if __name__ == "__main__":
    main()
