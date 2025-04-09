import os
from datetime import datetime


class SavePathConfig:
    def __init__(self):
        # 创建带时间戳的实验目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("experiments", f"exp_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

    def get_path(self, filename):
        return os.path.join(self.exp_dir, filename)
