import logging
import structlog
import sys
from pathlib import Path
from datetime import datetime
import yaml
import os
from typing import Optional, Dict, Any


class HFTLogger:
    """High-performance structured logger for HFT system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.setup_logging()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Default config if file not found
            return {
                'logging': {
                    'level': 'INFO',
                    'log_to_file': True,
                    'log_rotation': 'daily',
                    'max_log_files': 30
                }
            }
    
    def setup_logging(self):
        """Setup structured logging with file rotation"""
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup Python logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=log_level,
        )
        
        # File handler if enabled
        if self.config.get('logging', {}).get('log_to_file', True):
            from logging.handlers import TimedRotatingFileHandler
            
            log_file = log_dir / f"hft_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=self.config.get('logging', {}).get('max_log_files', 30)
            )
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            logging.getLogger().addHandler(file_handler)
    
    def get_logger(self, name: str = None) -> structlog.BoundLogger:
        """Get a structured logger instance"""
        return structlog.get_logger(name or __name__)
    
    def log_trade(self, symbol: str, side: str, quantity: float, price: float, 
                  strategy: str, timestamp: datetime = None):
        """Log trade execution with structured data"""
        logger = self.get_logger("trade")
        logger.info(
            "Trade executed",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy=strategy,
            timestamp=timestamp or datetime.now(),
            event_type="trade_execution"
        )
    
    def log_signal(self, symbol: str, signal_type: str, strength: float, 
                   strategy: str, metadata: Dict = None):
        """Log trading signal generation"""
        logger = self.get_logger("signal")
        logger.info(
            "Signal generated",
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            strategy=strategy,
            metadata=metadata or {},
            event_type="signal_generation"
        )
    
    def log_risk_check(self, check_type: str, result: bool, details: Dict = None):
        """Log risk check results"""
        logger = self.get_logger("risk")
        level = "info" if result else "warning"
        getattr(logger, level)(
            "Risk check performed",
            check_type=check_type,
            passed=result,
            details=details or {},
            event_type="risk_check"
        )
    
    def log_performance(self, latency_ms: float, memory_mb: float, cpu_percent: float):
        """Log system performance metrics"""
        logger = self.get_logger("performance")
        logger.info(
            "Performance metrics",
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            event_type="performance_metrics"
        )
    
    def log_error(self, component: str, error: Exception, context: Dict = None):
        """Log errors with full context"""
        logger = self.get_logger("error")
        logger.error(
            "System error occurred",
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {},
            event_type="system_error",
            exc_info=True
        )


# Global logger instance
_hft_logger = None

def get_hft_logger() -> HFTLogger:
    """Get global HFT logger instance"""
    global _hft_logger
    if _hft_logger is None:
        _hft_logger = HFTLogger()
    return _hft_logger

def setup_logging(config_path: str = "config/config.yaml"):
    """Setup global logging configuration"""
    global _hft_logger
    _hft_logger = HFTLogger(config_path) 