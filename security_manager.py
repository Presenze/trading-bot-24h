#!/usr/bin/env python3
"""
SISTEMA DI SICUREZZA AVANZATO PER TRADING BOT
- Protezione API con rate limiting e autenticazione
- Monitoraggio accessi e attività sospette
- Crittografia dati sensibili
- Sistema di backup e recovery
- Controlli di integrità continui
"""

import hashlib
import hmac
import time
import json
import logging
import sqlite3
import os
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import secrets
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import requests
import ipaddress
from functools import wraps
import jwt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Livelli di sicurezza"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatLevel(Enum):
    """Livelli di minaccia"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Evento di sicurezza"""
    timestamp: datetime
    event_type: str
    severity: ThreatLevel
    source_ip: str
    user_agent: str
    description: str
    action_taken: str
    additional_data: Dict[str, Any]

@dataclass
class APIAccess:
    """Accesso API"""
    api_key: str
    permissions: List[str]
    rate_limit: int
    requests_count: int
    last_request: datetime
    ip_whitelist: List[str]
    is_active: bool
    expires_at: Optional[datetime]

class SecurityManager:
    """
    Sistema di sicurezza completo per il trading bot
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        # Chiavi di crittografia
        self.encryption_key = None
        self.jwt_secret = None
        
        # Rate limiting
        self.rate_limits = {}
        self.request_counts = {}
        
        # IP whitelist/blacklist
        self.ip_whitelist = set()
        self.ip_blacklist = set()
        
        # API keys
        self.api_keys = {}
        
        # Security events
        self.security_events = []
        self.threat_level = ThreatLevel.NONE
        
        # Monitoring
        self.monitoring_active = True
        self.suspicious_activities = []
        
        # Setup
        self.setup_encryption()
        self.setup_database()
        self.load_security_config()
        self.start_monitoring()
        
        self.logger.info("Security Manager inizializzato")
    
    def setup_encryption(self):
        """Setup sistema di crittografia"""
        try:
            # Genera o carica chiave di crittografia
            key_file = "security_key.key"
            
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.encryption_key = f.read()
            else:
                # Genera nuova chiave
                password = self.config.get('encryption_password', 'default_password').encode()
                salt = os.urandom(16)
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                
                key = base64.urlsafe_b64encode(kdf.derive(password))
                self.encryption_key = key
                
                # Salva chiave
                with open(key_file, 'wb') as f:
                    f.write(key)
                
                # Salva salt
                with open("security_salt.key", 'wb') as f:
                    f.write(salt)
            
            # Setup JWT secret
            self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
            
            self.logger.info("Sistema di crittografia inizializzato")
            
        except Exception as e:
            self.logger.error(f"Errore setup crittografia: {e}")
    
    def setup_database(self):
        """Setup database per sicurezza"""
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            # Tabella eventi di sicurezza
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    event_type TEXT,
                    severity TEXT,
                    source_ip TEXT,
                    user_agent TEXT,
                    description TEXT,
                    action_taken TEXT,
                    additional_data TEXT
                )
            ''')
            
            # Tabella API keys
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    api_key TEXT UNIQUE,
                    name TEXT,
                    permissions TEXT,
                    rate_limit INTEGER,
                    ip_whitelist TEXT,
                    is_active BOOLEAN,
                    created_at DATETIME,
                    expires_at DATETIME,
                    last_used DATETIME
                )
            ''')
            
            # Tabella accessi
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    api_key TEXT,
                    endpoint TEXT,
                    method TEXT,
                    source_ip TEXT,
                    user_agent TEXT,
                    status_code INTEGER,
                    response_time REAL
                )
            ''')
            
            # Tabella configurazioni sicurezza
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT UNIQUE,
                    config_value TEXT,
                    updated_at DATETIME
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore setup database sicurezza: {e}")
    
    def load_security_config(self):
        """Carica configurazione sicurezza"""
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            # Carica IP whitelist
            cursor.execute("SELECT config_value FROM security_config WHERE config_key = 'ip_whitelist'")
            result = cursor.fetchone()
            if result:
                self.ip_whitelist = set(json.loads(result[0]))
            
            # Carica IP blacklist
            cursor.execute("SELECT config_value FROM security_config WHERE config_key = 'ip_blacklist'")
            result = cursor.fetchone()
            if result:
                self.ip_blacklist = set(json.loads(result[0]))
            
            # Carica API keys
            cursor.execute("SELECT * FROM api_keys WHERE is_active = 1")
            for row in cursor.fetchall():
                api_key = row[1]
                self.api_keys[api_key] = APIAccess(
                    api_key=api_key,
                    permissions=json.loads(row[3]),
                    rate_limit=row[4],
                    requests_count=0,
                    last_request=datetime.now(),
                    ip_whitelist=json.loads(row[5]) if row[5] else [],
                    is_active=bool(row[6]),
                    expires_at=datetime.fromisoformat(row[8]) if row[8] else None
                )
            
            conn.close()
            
            self.logger.info(f"Configurazione sicurezza caricata: {len(self.api_keys)} API keys attive")
            
        except Exception as e:
            self.logger.error(f"Errore caricamento config sicurezza: {e}")
    
    def encrypt_data(self, data: str) -> str:
        """Cripta dati sensibili"""
        try:
            if not self.encryption_key:
                return data
            
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Errore crittografia: {e}")
            return data
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decripta dati"""
        try:
            if not self.encryption_key:
                return encrypted_data
            
            fernet = Fernet(self.encryption_key)
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = fernet.decrypt(decoded_data)
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Errore decrittografia: {e}")
            return encrypted_data
    
    def generate_api_key(self, name: str, permissions: List[str], 
                        rate_limit: int = 100, ip_whitelist: List[str] = None,
                        expires_days: int = None) -> str:
        """Genera nuova API key"""
        try:
            # Genera API key sicura
            api_key = secrets.token_urlsafe(32)
            
            # Calcola scadenza
            expires_at = None
            if expires_days:
                expires_at = datetime.now() + timedelta(days=expires_days)
            
            # Salva nel database
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_keys (
                    api_key, name, permissions, rate_limit, ip_whitelist,
                    is_active, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                api_key, name, json.dumps(permissions), rate_limit,
                json.dumps(ip_whitelist or []), True, datetime.now(),
                expires_at
            ))
            
            conn.commit()
            conn.close()
            
            # Aggiungi alla cache
            self.api_keys[api_key] = APIAccess(
                api_key=api_key,
                permissions=permissions,
                rate_limit=rate_limit,
                requests_count=0,
                last_request=datetime.now(),
                ip_whitelist=ip_whitelist or [],
                is_active=True,
                expires_at=expires_at
            )
            
            self.logger.info(f"API key generata per {name}: {api_key[:8]}...")
            
            # Log evento sicurezza
            self.log_security_event(
                "API_KEY_GENERATED", ThreatLevel.NONE, "127.0.0.1", "System",
                f"API key generated for {name}", "API_KEY_CREATED"
            )
            
            return api_key
            
        except Exception as e:
            self.logger.error(f"Errore generazione API key: {e}")
            return ""
    
    def validate_api_key(self, api_key: str, required_permission: str = None,
                        source_ip: str = None) -> Tuple[bool, str]:
        """Valida API key"""
        try:
            # Controlla se API key esiste
            if api_key not in self.api_keys:
                self.log_security_event(
                    "INVALID_API_KEY", ThreatLevel.MEDIUM, source_ip or "unknown", "",
                    f"Invalid API key used: {api_key[:8]}...", "REQUEST_BLOCKED"
                )
                return False, "Invalid API key"
            
            api_access = self.api_keys[api_key]
            
            # Controlla se è attiva
            if not api_access.is_active:
                return False, "API key is disabled"
            
            # Controlla scadenza
            if api_access.expires_at and datetime.now() > api_access.expires_at:
                return False, "API key has expired"
            
            # Controlla IP whitelist
            if api_access.ip_whitelist and source_ip:
                if not self.is_ip_whitelisted(source_ip, api_access.ip_whitelist):
                    self.log_security_event(
                        "IP_NOT_WHITELISTED", ThreatLevel.HIGH, source_ip, "",
                        f"IP not in whitelist for API key: {api_key[:8]}...", "REQUEST_BLOCKED"
                    )
                    return False, "IP not whitelisted"
            
            # Controlla permessi
            if required_permission and required_permission not in api_access.permissions:
                return False, "Insufficient permissions"
            
            # Controlla rate limiting
            if not self.check_rate_limit(api_key):
                self.log_security_event(
                    "RATE_LIMIT_EXCEEDED", ThreatLevel.MEDIUM, source_ip or "unknown", "",
                    f"Rate limit exceeded for API key: {api_key[:8]}...", "REQUEST_THROTTLED"
                )
                return False, "Rate limit exceeded"
            
            # Aggiorna ultimo utilizzo
            api_access.last_request = datetime.now()
            self.update_api_key_usage(api_key)
            
            return True, "Valid API key"
            
        except Exception as e:
            self.logger.error(f"Errore validazione API key: {e}")
            return False, f"Validation error: {e}"
    
    def check_rate_limit(self, api_key: str) -> bool:
        """Controlla rate limiting"""
        try:
            if api_key not in self.api_keys:
                return False
            
            api_access = self.api_keys[api_key]
            current_time = time.time()
            
            # Inizializza se non esiste
            if api_key not in self.rate_limits:
                self.rate_limits[api_key] = []
            
            # Rimuovi richieste vecchie (oltre 1 ora)
            self.rate_limits[api_key] = [
                req_time for req_time in self.rate_limits[api_key]
                if current_time - req_time < 3600
            ]
            
            # Controlla limite
            if len(self.rate_limits[api_key]) >= api_access.rate_limit:
                return False
            
            # Aggiungi richiesta corrente
            self.rate_limits[api_key].append(current_time)
            api_access.requests_count += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Errore controllo rate limit: {e}")
            return False
    
    def is_ip_whitelisted(self, ip: str, whitelist: List[str]) -> bool:
        """Controlla se IP è nella whitelist"""
        try:
            ip_addr = ipaddress.ip_address(ip)
            
            for allowed in whitelist:
                try:
                    # Controlla se è un singolo IP
                    if ipaddress.ip_address(allowed) == ip_addr:
                        return True
                except ValueError:
                    # Potrebbe essere una subnet
                    try:
                        if ip_addr in ipaddress.ip_network(allowed, strict=False):
                            return True
                    except ValueError:
                        continue
            
            return False
            
        except Exception as e:
            self.logger.error(f"Errore controllo IP whitelist: {e}")
            return False
    
    def is_ip_blacklisted(self, ip: str) -> bool:
        """Controlla se IP è nella blacklist"""
        try:
            return ip in self.ip_blacklist
        except:
            return False
    
    def add_to_blacklist(self, ip: str, reason: str = "Suspicious activity"):
        """Aggiungi IP alla blacklist"""
        try:
            self.ip_blacklist.add(ip)
            
            # Salva nel database
            self.save_security_config('ip_blacklist', list(self.ip_blacklist))
            
            self.log_security_event(
                "IP_BLACKLISTED", ThreatLevel.HIGH, ip, "",
                f"IP added to blacklist: {reason}", "IP_BLOCKED"
            )
            
            self.logger.warning(f"IP {ip} aggiunto alla blacklist: {reason}")
            
        except Exception as e:
            self.logger.error(f"Errore aggiunta blacklist: {e}")
    
    def detect_suspicious_activity(self, request_data: Dict[str, Any]) -> bool:
        """Rileva attività sospette"""
        try:
            suspicious = False
            reasons = []
            
            source_ip = request_data.get('source_ip', '')
            user_agent = request_data.get('user_agent', '')
            endpoint = request_data.get('endpoint', '')
            
            # 1. Controlla IP blacklist
            if self.is_ip_blacklisted(source_ip):
                suspicious = True
                reasons.append("IP in blacklist")
            
            # 2. Controlla user agent sospetti
            suspicious_agents = ['bot', 'crawler', 'scanner', 'hack']
            if any(agent in user_agent.lower() for agent in suspicious_agents):
                suspicious = True
                reasons.append("Suspicious user agent")
            
            # 3. Controlla pattern di attacco
            attack_patterns = ['/admin', '/.env', '/config', 'SELECT', 'UNION', '<script>']
            if any(pattern in endpoint for pattern in attack_patterns):
                suspicious = True
                reasons.append("Attack pattern detected")
            
            # 4. Controlla frequenza richieste
            if self.check_request_frequency(source_ip):
                suspicious = True
                reasons.append("High request frequency")
            
            # 5. Controlla richieste fallite consecutive
            if self.check_failed_requests(source_ip):
                suspicious = True
                reasons.append("Multiple failed requests")
            
            if suspicious:
                self.log_security_event(
                    "SUSPICIOUS_ACTIVITY", ThreatLevel.HIGH, source_ip, user_agent,
                    f"Suspicious activity detected: {', '.join(reasons)}", "MONITORING_INCREASED"
                )
                
                # Considera blacklist automatica
                if len(reasons) >= 3:
                    self.add_to_blacklist(source_ip, "Multiple suspicious indicators")
            
            return suspicious
            
        except Exception as e:
            self.logger.error(f"Errore rilevamento attività sospette: {e}")
            return False
    
    def check_request_frequency(self, ip: str) -> bool:
        """Controlla frequenza richieste da IP"""
        try:
            current_time = time.time()
            
            # Ottieni richieste recenti da questo IP
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM access_log 
                WHERE source_ip = ? AND timestamp > datetime('now', '-1 minute')
            ''', (ip,))
            
            recent_requests = cursor.fetchone()[0]
            conn.close()
            
            # Soglia: più di 60 richieste al minuto
            return recent_requests > 60
            
        except Exception as e:
            self.logger.error(f"Errore controllo frequenza: {e}")
            return False
    
    def check_failed_requests(self, ip: str) -> bool:
        """Controlla richieste fallite consecutive"""
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM access_log 
                WHERE source_ip = ? AND status_code >= 400 
                AND timestamp > datetime('now', '-5 minutes')
            ''', (ip,))
            
            failed_requests = cursor.fetchone()[0]
            conn.close()
            
            # Soglia: più di 10 richieste fallite in 5 minuti
            return failed_requests > 10
            
        except Exception as e:
            self.logger.error(f"Errore controllo richieste fallite: {e}")
            return False
    
    def log_access(self, api_key: str, endpoint: str, method: str,
                  source_ip: str, user_agent: str, status_code: int, response_time: float):
        """Log accesso API"""
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO access_log (
                    timestamp, api_key, endpoint, method, source_ip,
                    user_agent, status_code, response_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), api_key, endpoint, method,
                source_ip, user_agent, status_code, response_time
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore log accesso: {e}")
    
    def log_security_event(self, event_type: str, severity: ThreatLevel,
                          source_ip: str, user_agent: str, description: str,
                          action_taken: str, additional_data: Dict = None):
        """Log evento di sicurezza"""
        try:
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                severity=severity,
                source_ip=source_ip,
                user_agent=user_agent,
                description=description,
                action_taken=action_taken,
                additional_data=additional_data or {}
            )
            
            self.security_events.append(event)
            
            # Salva nel database
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_events (
                    timestamp, event_type, severity, source_ip, user_agent,
                    description, action_taken, additional_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp, event.event_type, event.severity.value,
                event.source_ip, event.user_agent, event.description,
                event.action_taken, json.dumps(event.additional_data)
            ))
            
            conn.commit()
            conn.close()
            
            # Aggiorna threat level
            self.update_threat_level()
            
            # Log in base alla severità
            if severity == ThreatLevel.CRITICAL:
                self.logger.critical(f"SECURITY CRITICAL: {description}")
            elif severity == ThreatLevel.HIGH:
                self.logger.error(f"SECURITY HIGH: {description}")
            elif severity == ThreatLevel.MEDIUM:
                self.logger.warning(f"SECURITY MEDIUM: {description}")
            else:
                self.logger.info(f"SECURITY: {description}")
            
        except Exception as e:
            self.logger.error(f"Errore log security event: {e}")
    
    def update_threat_level(self):
        """Aggiorna livello di minaccia globale"""
        try:
            # Analizza eventi recenti (ultima ora)
            recent_events = [
                event for event in self.security_events
                if (datetime.now() - event.timestamp).seconds < 3600
            ]
            
            if not recent_events:
                self.threat_level = ThreatLevel.NONE
                return
            
            # Conta eventi per severità
            critical_count = sum(1 for e in recent_events if e.severity == ThreatLevel.CRITICAL)
            high_count = sum(1 for e in recent_events if e.severity == ThreatLevel.HIGH)
            medium_count = sum(1 for e in recent_events if e.severity == ThreatLevel.MEDIUM)
            
            # Determina threat level
            if critical_count > 0:
                new_level = ThreatLevel.CRITICAL
            elif high_count >= 5:
                new_level = ThreatLevel.HIGH
            elif high_count >= 2 or medium_count >= 10:
                new_level = ThreatLevel.MEDIUM
            elif medium_count >= 3:
                new_level = ThreatLevel.LOW
            else:
                new_level = ThreatLevel.NONE
            
            # Aggiorna se cambiato
            if new_level != self.threat_level:
                old_level = self.threat_level
                self.threat_level = new_level
                
                self.logger.warning(f"🚨 Threat level changed: {old_level.value} → {new_level.value}")
                
                # Attiva misure di sicurezza aggiuntive
                if new_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    self.activate_enhanced_security()
            
        except Exception as e:
            self.logger.error(f"Errore aggiornamento threat level: {e}")
    
    def activate_enhanced_security(self):
        """Attiva sicurezza potenziata"""
        try:
            self.logger.warning("🛡️ Attivando sicurezza potenziata...")
            
            # Riduci rate limits
            for api_key in self.api_keys:
                self.api_keys[api_key].rate_limit = max(10, self.api_keys[api_key].rate_limit // 2)
            
            # Aumenta monitoraggio
            self.monitoring_active = True
            
            # Notifica amministratori (se configurato)
            self.send_security_alert("Enhanced security activated due to high threat level")
            
        except Exception as e:
            self.logger.error(f"Errore attivazione sicurezza potenziata: {e}")
    
    def send_security_alert(self, message: str):
        """Invia alert di sicurezza"""
        try:
            # Implementazione per notifiche (email, webhook, etc.)
            self.logger.critical(f"🚨 SECURITY ALERT: {message}")
            
            # Qui potresti implementare:
            # - Invio email
            # - Webhook Discord/Slack
            # - SMS
            # - Push notification
            
        except Exception as e:
            self.logger.error(f"Errore invio security alert: {e}")
    
    def create_jwt_token(self, payload: Dict[str, Any], expires_hours: int = 24) -> str:
        """Crea JWT token"""
        try:
            # Aggiungi timestamp
            payload['iat'] = datetime.utcnow()
            payload['exp'] = datetime.utcnow() + timedelta(hours=expires_hours)
            
            # Crea token
            token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            return token
            
        except Exception as e:
            self.logger.error(f"Errore creazione JWT: {e}")
            return ""
    
    def verify_jwt_token(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """Verifica JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return True, payload
            
        except jwt.ExpiredSignatureError:
            return False, {"error": "Token expired"}
        except jwt.InvalidTokenError:
            return False, {"error": "Invalid token"}
        except Exception as e:
            self.logger.error(f"Errore verifica JWT: {e}")
            return False, {"error": str(e)}
    
    def backup_security_data(self) -> bool:
        """Backup dati di sicurezza"""
        try:
            backup_dir = "security_backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{backup_dir}/security_backup_{timestamp}.json"
            
            # Prepara dati per backup
            backup_data = {
                'timestamp': timestamp,
                'api_keys': {k: {
                    'permissions': v.permissions,
                    'rate_limit': v.rate_limit,
                    'ip_whitelist': v.ip_whitelist,
                    'is_active': v.is_active,
                    'expires_at': v.expires_at.isoformat() if v.expires_at else None
                } for k, v in self.api_keys.items()},
                'ip_whitelist': list(self.ip_whitelist),
                'ip_blacklist': list(self.ip_blacklist),
                'security_events': [{
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'severity': event.severity.value,
                    'source_ip': event.source_ip,
                    'description': event.description,
                    'action_taken': event.action_taken
                } for event in self.security_events[-100:]]  # Ultimi 100 eventi
            }
            
            # Cripta e salva
            encrypted_data = self.encrypt_data(json.dumps(backup_data))
            
            with open(backup_file, 'w') as f:
                f.write(encrypted_data)
            
            self.logger.info(f"Backup sicurezza salvato: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore backup sicurezza: {e}")
            return False
    
    def restore_security_data(self, backup_file: str) -> bool:
        """Ripristina dati di sicurezza"""
        try:
            if not os.path.exists(backup_file):
                self.logger.error(f"File backup non trovato: {backup_file}")
                return False
            
            # Leggi e decripta
            with open(backup_file, 'r') as f:
                encrypted_data = f.read()
            
            decrypted_data = self.decrypt_data(encrypted_data)
            backup_data = json.loads(decrypted_data)
            
            # Ripristina dati
            self.ip_whitelist = set(backup_data.get('ip_whitelist', []))
            self.ip_blacklist = set(backup_data.get('ip_blacklist', []))
            
            # Ripristina API keys (con validazione)
            for api_key, data in backup_data.get('api_keys', {}).items():
                self.api_keys[api_key] = APIAccess(
                    api_key=api_key,
                    permissions=data['permissions'],
                    rate_limit=data['rate_limit'],
                    requests_count=0,
                    last_request=datetime.now(),
                    ip_whitelist=data['ip_whitelist'],
                    is_active=data['is_active'],
                    expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None
                )
            
            self.logger.info(f"Dati sicurezza ripristinati da: {backup_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Errore ripristino sicurezza: {e}")
            return False
    
    def start_monitoring(self):
        """Avvia monitoraggio continuo"""
        try:
            def monitoring_loop():
                while self.monitoring_active:
                    try:
                        # Pulisci eventi vecchi
                        self.cleanup_old_events()
                        
                        # Controlla API keys scadute
                        self.check_expired_api_keys()
                        
                        # Backup periodico
                        if datetime.now().hour == 2 and datetime.now().minute < 5:  # 2 AM
                            self.backup_security_data()
                        
                        # Aspetta 5 minuti
                        time.sleep(300)
                        
                    except Exception as e:
                        self.logger.error(f"Errore monitoring loop: {e}")
                        time.sleep(60)
            
            # Avvia thread di monitoraggio
            monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
            monitoring_thread.start()
            
            self.logger.info("Monitoraggio sicurezza avviato")
            
        except Exception as e:
            self.logger.error(f"Errore avvio monitoraggio: {e}")
    
    def cleanup_old_events(self):
        """Pulisci eventi vecchi"""
        try:
            # Mantieni solo eventi degli ultimi 30 giorni
            cutoff_date = datetime.now() - timedelta(days=30)
            
            self.security_events = [
                event for event in self.security_events
                if event.timestamp > cutoff_date
            ]
            
            # Pulisci database
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM security_events WHERE timestamp < ?', (cutoff_date,))
            cursor.execute('DELETE FROM access_log WHERE timestamp < ?', (cutoff_date,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore pulizia eventi: {e}")
    
    def check_expired_api_keys(self):
        """Controlla API keys scadute"""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for api_key, access in self.api_keys.items():
                if access.expires_at and current_time > access.expires_at:
                    expired_keys.append(api_key)
            
            # Disattiva chiavi scadute
            for api_key in expired_keys:
                self.api_keys[api_key].is_active = False
                
                self.log_security_event(
                    "API_KEY_EXPIRED", ThreatLevel.LOW, "system", "",
                    f"API key expired: {api_key[:8]}...", "API_KEY_DISABLED"
                )
            
            if expired_keys:
                self.logger.info(f"Disattivate {len(expired_keys)} API keys scadute")
            
        except Exception as e:
            self.logger.error(f"Errore controllo API keys scadute: {e}")
    
    def save_security_config(self, key: str, value: Any):
        """Salva configurazione sicurezza"""
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO security_config (config_key, config_value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore salvataggio config: {e}")
    
    def update_api_key_usage(self, api_key: str):
        """Aggiorna utilizzo API key"""
        try:
            conn = sqlite3.connect('security.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE api_keys SET last_used = ? WHERE api_key = ?
            ''', (datetime.now(), api_key))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Errore aggiornamento utilizzo API key: {e}")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Ottieni status sicurezza"""
        try:
            recent_events = [
                event for event in self.security_events
                if (datetime.now() - event.timestamp).seconds < 3600
            ]
            
            return {
                'threat_level': self.threat_level.value,
                'active_api_keys': len([k for k, v in self.api_keys.items() if v.is_active]),
                'total_api_keys': len(self.api_keys),
                'ip_whitelist_size': len(self.ip_whitelist),
                'ip_blacklist_size': len(self.ip_blacklist),
                'recent_events': len(recent_events),
                'critical_events_1h': len([e for e in recent_events if e.severity == ThreatLevel.CRITICAL]),
                'high_events_1h': len([e for e in recent_events if e.severity == ThreatLevel.HIGH]),
                'monitoring_active': self.monitoring_active,
                'encryption_enabled': self.encryption_key is not None,
                'jwt_enabled': self.jwt_secret is not None
            }
            
        except Exception as e:
            self.logger.error(f"Errore status sicurezza: {e}")
            return {}
    
    def security_middleware(self, func):
        """Decorator per middleware sicurezza"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Estrai dati richiesta
                request_data = kwargs.get('request_data', {})
                api_key = request_data.get('api_key', '')
                source_ip = request_data.get('source_ip', '')
                
                # Controlla blacklist
                if self.is_ip_blacklisted(source_ip):
                    return {'error': 'IP blocked', 'status': 403}
                
                # Rileva attività sospette
                if self.detect_suspicious_activity(request_data):
                    return {'error': 'Suspicious activity detected', 'status': 429}
                
                # Valida API key
                is_valid, message = self.validate_api_key(api_key, source_ip=source_ip)
                if not is_valid:
                    return {'error': message, 'status': 401}
                
                # Esegui funzione
                start_time = time.time()
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                
                # Log accesso
                self.log_access(
                    api_key, request_data.get('endpoint', ''),
                    request_data.get('method', 'GET'), source_ip,
                    request_data.get('user_agent', ''), 200, response_time
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Errore security middleware: {e}")
                return {'error': 'Security error', 'status': 500}
        
        return wrapper

def main():
    """Test del sistema di sicurezza"""
    print("🔒 SISTEMA DI SICUREZZA AVANZATO")
    print("=" * 50)
    
    # Configurazione test
    config = {
        'encryption_password': 'test_password_123',
        'jwt_secret': 'test_jwt_secret_456'
    }
    
    # Inizializza security manager
    security_manager = SecurityManager(config)
    
    # Test generazione API key
    print("🔑 Generando API key di test...")
    api_key = security_manager.generate_api_key(
        name="Test API",
        permissions=["read", "write"],
        rate_limit=100,
        ip_whitelist=["127.0.0.1", "192.168.1.0/24"],
        expires_days=30
    )
    
    print(f"   API Key: {api_key[:16]}...")
    
    # Test validazione API key
    print("\n🔍 Testando validazione API key...")
    is_valid, message = security_manager.validate_api_key(api_key, source_ip="127.0.0.1")
    print(f"   Valida: {is_valid} - {message}")
    
    # Test crittografia
    print("\n🔐 Testando crittografia...")
    test_data = "Dati sensibili di trading"
    encrypted = security_manager.encrypt_data(test_data)
    decrypted = security_manager.decrypt_data(encrypted)
    
    print(f"   Originale: {test_data}")
    print(f"   Criptato: {encrypted[:32]}...")
    print(f"   Decriptato: {decrypted}")
    print(f"   Match: {test_data == decrypted}")
    
    # Test JWT
    print("\n🎫 Testando JWT tokens...")
    payload = {"user_id": "123", "permissions": ["read", "write"]}
    token = security_manager.create_jwt_token(payload)
    is_valid_jwt, decoded = security_manager.verify_jwt_token(token)
    
    print(f"   Token: {token[:32]}...")
    print(f"   Valido: {is_valid_jwt}")
    print(f"   Payload: {decoded}")
    
    # Test rilevamento attività sospette
    print("\n🕵️ Testando rilevamento attività sospette...")
    suspicious_request = {
        'source_ip': '192.168.1.100',
        'user_agent': 'HackBot/1.0',
        'endpoint': '/admin/config'
    }
    
    is_suspicious = security_manager.detect_suspicious_activity(suspicious_request)
    print(f"   Attività sospetta rilevata: {is_suspicious}")
    
    # Test backup
    print("\n💾 Testando backup sicurezza...")
    backup_success = security_manager.backup_security_data()
    print(f"   Backup completato: {backup_success}")
    
    # Mostra status sicurezza
    status = security_manager.get_security_status()
    print(f"\n📊 STATUS SICUREZZA:")
    print(f"   Threat Level: {status['threat_level']}")
    print(f"   API Keys Attive: {status['active_api_keys']}/{status['total_api_keys']}")
    print(f"   IP Whitelist: {status['ip_whitelist_size']} entries")
    print(f"   IP Blacklist: {status['ip_blacklist_size']} entries")
    print(f"   Eventi Recenti: {status['recent_events']}")
    print(f"   Crittografia: {'✅' if status['encryption_enabled'] else '❌'}")
    print(f"   JWT: {'✅' if status['jwt_enabled'] else '❌'}")
    
    print("\n✅ Test sistema sicurezza completato!")

if __name__ == "__main__":
    main()
