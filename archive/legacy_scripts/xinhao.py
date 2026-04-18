# -*- coding: gbk -*-
import numpy as np
import pandas as pd

fs = 100          # 采样率 Hz（按你系统实际采样率改）
dur = 60          # 信号时长 s
t = np.arange(0, dur, 1/fs)

# 通带 + 阻带 频率分量（你可以按需改）
pass_freqs = [0.5, 1, 2, 5, 10]
stop_freqs = [0.1, 0.2, 12, 15, 20]

x = np.zeros_like(t)

# 通带分量幅值设大一点，阻带分量设小一点（便于观察衰减）
for f in pass_freqs:
    x += 1.0 * np.sin(2*np.pi*f*t)

for f in stop_freqs:
    x += 0.3 * np.sin(2*np.pi*f*t)

# 加一点随机噪声（可选）
x += 0.05 * np.random.randn(len(t))

df = pd.DataFrame({"t": t, "x": x})
df.to_csv("ppg_test_multitone.csv", index=False, encoding="utf-8-sig")
print("Saved: ppg_test_multitone.csv")
