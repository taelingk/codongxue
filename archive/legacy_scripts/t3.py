import sys
print(f"Python版本: {sys.version}")

# 检查 numpy 版本
import numpy as nppy
print(f"NumPy版本: {np.__version__}")

# 尝试导入 TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow版本: {tf.__version__}")
    print("TensorFlow 导入成功！")
except Exception as e:
    print(f"导入失败: {e}")