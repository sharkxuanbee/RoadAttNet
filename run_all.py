import os
import sys
import subprocess
import logging
from datetime import datetime

# 初始化日志功能，输出到终端和独立日志文件
log_file = f"run_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def run_script(script_name):
    logging.info(f"\n{'='*50}")
    logging.info(f"开始运行阶段: {script_name}")
    logging.info(f"{'='*50}\n")
    
    # 运行对应的 Python 脚本
    result = subprocess.run([sys.executable, script_name])
    
    # 检查是否成功执行
    if result.returncode != 0:
        logging.error(f"\n{'!'*50}")
        logging.error(f"[错误] {script_name} 运行失败，退出码: {result.returncode}")
        logging.error(f"{'!'*50}")
        sys.exit(result.returncode) # 如果失败则直接退出，不继续执行后面的步骤
    else:
        logging.info(f"\n{'-'*50}")
        logging.info(f"阶段完成: {script_name}")
        logging.info(f"{'-'*50}\n")

if __name__ == "__main__":
    logging.info(">>> 启动 RoadAttNet 完整流程: 特征提取 -> 模型训练 <<<\n")
    
    # 1. 首先运行特征提取
    run_script("batch_extract.py")
    
    # 2. 如果特征提取成功，则开始训练模型
    run_script("train.py")
    
    logging.info("恭喜，所有流程均已成功完成！")
