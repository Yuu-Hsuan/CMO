import logging
import os
import json
from datetime import datetime

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Logger:
    def __init__(self, log_dir=None):
        if log_dir is None:
            # 創建默認日誌目錄
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("logs", f"run_{current_time}")
        
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.stats = {}
        self.steps = {}  # 添加用於儲存步數的字典
        self.json_file = os.path.join(self.log_dir, "stats.json")  # 初始化 json_file
        
        # 設置文件處理器
        # self.file_handler = logging.FileHandler(os.path.join(log_dir, "agent.log"))
        # self.file_handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # self.file_handler.setFormatter(formatter)
        
        # 添加到logger
        # logger.addHandler(self.file_handler)
        
        # logger.info(f"Logger initialized. Logs will be saved to {log_dir}")
    
    def log_stat(self, key, value, step):
        """記錄統計數據"""
        if key not in self.stats:
            self.stats[key] = []
            self.steps[key] = []
            
        self.stats[key].append(value)
        self.steps[key].append(step)
        
        json_data = {}
        for stat_key in self.stats:
            json_data[stat_key] = self.stats[stat_key]
            json_data[f"{stat_key}_T"] = self.steps[stat_key]
        
        with open(self.json_file, "w") as f:
            json.dump(json_data, f, indent=4)
        
        logger.info(f"{key}: {value} (step: {step})")
    
    def close(self):
        """關閉logger"""
        json_data = {}
        for stat_key in self.stats:
            json_data[stat_key] = self.stats[stat_key]
            json_data[f"{stat_key}_T"] = self.steps[stat_key]
        
        with open(self.json_file, "w") as f:
            json.dump(json_data, f, indent=4)
                
        logger.removeHandler(self.file_handler)
        self.file_handler.close()