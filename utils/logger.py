import os
import logging
import colorlog
import logging.handlers
from typing import Optional
from threading import Lock


class SingletonLogger:
    """
    单例日志工具类，封装了Python的logging模块
    
    功能特点：
    - 单例模式，确保全局只有一个日志实例
    - 同时支持控制台和文件输出
    - 支持按文件大小或时间轮换日志
    - 支持自定义日志格式
    - 支持不同日志级别
    """
    
    _instance = None
    _lock = Lock()  # 类级锁
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:  # 加锁
                if not cls._instance:  # 再次检查
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    
    def __init__(self, 
                 name: str = 'root',
                 log_file: Optional[str] = None,
                 level: str = 'INFO',
                 fmt: str = '%(asctime)s [%(threadName)s] [%(name)s] - %(levelname)s - %(message)s',
                 datefmt: str = '%Y-%m-%d %H:%M:%S',
                 max_bytes: int = 10*1024*1024,  # 10MB
                 backup_count: int = 5,
                 console: bool = True):
        """
        初始化日志记录器
        
        注意：单例模式下，只有第一次初始化时参数会生效
        """
        if self._initialized:
            return
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.upper())
        
        # 避免重复添加handler
        if not self.logger.handlers:
            formatter_file = logging.Formatter(fmt=fmt, datefmt=datefmt)
            formatter_console = colorlog.ColoredFormatter(
                fmt="%(log_color)s" + fmt,
                datefmt=datefmt,
                log_colors={
                    'DEBUG': 'white',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                }
            )
            
            # 控制台输出
            if console:
                # console_handler = colorlog.StreamHandler()
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter_console)
                self.logger.addHandler(console_handler)
            
            # 文件输出
            if log_file:
                # 确保日志目录存在
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                # 按大小轮换的文件处理器
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter_file)
                self.logger.addHandler(file_handler)
        
        self._initialized = True
    
        
    def info(self, msg: str, *args, **kwargs):
        """记录INFO级别日志"""
        self.logger.info(msg, *args, **kwargs)
    
    
    def debug(self, msg: str, *args, **kwargs):
        """记录DEBUG级别日志"""
        self.logger.debug(msg, *args, **kwargs)
    
    
    def warning(self, msg: str, *args, **kwargs):
        """记录WARNING级别日志"""
        self.logger.warning(msg, *args, **kwargs)
    
    
    def error(self, msg: str, *args, **kwargs):
        """记录ERROR级别日志"""
        self.logger.error(msg, *args, **kwargs)
    
    
    def critical(self, msg: str, *args, **kwargs):
        """记录CRITICAL级别日志"""
        self.logger.critical(msg, *args, **kwargs)
    
    
    def exception(self, msg: str, *args, **kwargs):
        """记录异常日志"""
        self.logger.exception(msg, *args, **kwargs)


# # 创建全局单例日志实例
# global_logger = SingletonLogger(
#     name='agent',
#     log_file='logs/agent.log',
#     level='INFO',
#     console=True
# )

