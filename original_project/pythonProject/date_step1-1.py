# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use("TkAgg")   # 或者 "Qt5Agg"
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import os
import re


# =========================
# 设置中文字体
# =========================
def setup_chinese_font():
    try:
        chinese_fonts = [
            'SimHei',
            'Microsoft YaHei',
            'SimSun',
            'KaiTi',
            'FangSong',
            'Arial Unicode MS',
            'DejaVu Sans'
        ]
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False
        print(f"使用字体: {plt.rcParams['font.sans-serif'][0]}")
    except Exception as e:
        print(f"字体设置警告: {e}")
        print("将使用默认字体，中文可能显示为方框")


def clean_file_path(path):
    """清理输入字符串，去除首尾引号和空格"""
    path = path.strip()
    path = re.sub(r'^[\'"]|[\'"]$', '', path)
    return path


def ask_file_path():
    path = input("请输入待处理数据的本地完整路径（例如：C:/Users/xxx/085240_红光红外信号.csv）: ")
    path = clean_file_path(path)

    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        print("请检查路径是否正确，或者文件是否被移动/删除")
        return ask_file_path()

    if not path.lower().endswith('.csv'):
        print("警告：文件不是CSV格式，但将继续尝试读取")

    return path


def invert_data(y):
    return -y


def resample_data(t, y, original_fs=225, target_fs=125):
    num_original = len(y)
    num_target = int(num_original * target_fs / original_fs)
    t_resampled = np.linspace(t.iloc[0], t.iloc[-1], num_target)
    y_resampled = np.interp(t_resampled, t, y)
    return t_resampled, y_resampled


def plot_data(t, y, title="全部数据", xlim=None):
    plt.figure(figsize=(12, 6))
    plt.plot(t, y, label='红光数据', color='red', linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel('时间 (秒)', fontsize=12)
    plt.ylabel('幅度', fontsize=12)
    if xlim:
        plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def extract_time_range(t, y, xlim):
    mask = (t >= xlim[0]) & (t <= xlim[1])
    return t[mask], y[mask]


def smart_baseline_correction(y, window_size=125):
    print("正在进行智能基线校正...")
    baseline = signal.medfilt(y, kernel_size=window_size)
    return y - baseline


def adaptive_bandpass_filter(y, fs, lowcut=0.5, highcut=10.0):
    print("正在进行自适应带通滤波...")
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    y_filtered = filtfilt(b, a, y)
    return y_filtered


def normalize_data(y):
    print("正在进行数据标准化...")
    y = np.asarray(y, dtype=float)
    denom = (np.max(y) - np.min(y))
    if denom == 0:
        return np.zeros_like(y)
    return (y - np.min(y)) / denom


def clean_output_path(path):
    path = path.strip()
    path = re.sub(r'^[\'"]|[\'"]$', '', path)
    return path


def ensure_dir_for_file(filepath):
    d = os.path.dirname(filepath)
    if d and (not os.path.exists(d)):
        os.makedirs(d)


# =========================
# 新增：频谱计算与绘图（并保存）
# =========================
def compute_fft_spectrum(y, fs):
    """
    返回：freqs(Hz), mag(线性幅值谱)
    使用单边 rfft；去直流；加 Hann 窗减小谱泄漏
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 4:
        return np.array([]), np.array([])

    y = y - np.mean(y)

    window = np.hanning(n)
    yw = y * window

    Y = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # 简单归一：方便前后对比（重点看频带变化）
    mag = np.abs(Y) * 2.0 / np.sum(window)
    return freqs, mag


def plot_spectrum(freqs, mag, title, lowcut=0.5, highcut=10.0, fmax=20.0, save_path=None):
    """
    画频谱，并标注0.5Hz与10Hz；如果save_path不为空则保存图片
    """
    if freqs.size == 0:
        print("数据太短，无法计算频谱")
        return

    mask = freqs <= fmax
    f = freqs[mask]
    m = mag[mask]

    plt.figure(figsize=(12, 6))
    plt.plot(f, m, linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel("频率 (Hz)", fontsize=12)
    plt.ylabel("幅值谱 (线性)", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 标注带通边界
    plt.axvline(lowcut, linestyle='--', linewidth=1)
    plt.axvline(highcut, linestyle='--', linewidth=1)
    ymax = np.max(m) if m.size else 1.0
    plt.text(lowcut, ymax * 0.9, f"{lowcut}Hz", rotation=90, va='top')
    plt.text(highcut, ymax * 0.9, f"{highcut}Hz", rotation=90, va='top')

    plt.tight_layout()

    # 保存（在show之前保存，避免窗口关闭影响）
    if save_path:
        try:
            ensure_dir_for_file(save_path)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[OK] 频谱图已保存: {save_path}")
        except Exception as e:
            print(f"[WARN] 频谱图保存失败: {e}")

    plt.show()


def main():
    setup_chinese_font()

    print("=" * 50)
    print("脉搏波数据处理程序")
    print("=" * 50)

    try:
        # 1. 读取文件
        file_path = ask_file_path()
        print(f"正在读取文件: {file_path}")

        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"成功使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"编码 {encoding} 读取失败: {e}")
                continue

        if df is None:
            print("无法读取文件，请检查文件格式或编码")
            return

        # 识别列名
        time_col = None
        reddata_col = None
        for col in df.columns:
            if '时间' in col or 'time' in col.lower():
                time_col = col
            if 'red' in col.lower() or '红光' in col:
                reddata_col = col

        if time_col is None or reddata_col is None:
            print("错误：无法识别必要的列名")
            print(f"文件中的列名: {list(df.columns)}")
            print("请确保文件包含'时间(秒)'和'Reddata'列")
            return

        print(f"识别到时间列: {time_col}")
        print(f"识别到红光数据列: {reddata_col}")

        t = df[time_col]
        y_red = df[reddata_col]

        print(f"数据读取成功！共 {len(y_red)} 个数据点")
        print(f"时间范围: {t.iloc[0]:.2f} - {t.iloc[-1]:.2f} 秒")

        # 2. 倒置
        print("正在进行数据倒置...")
        y_inverted = invert_data(y_red)

        # 3. 重采样 225->125
        print("正在进行重采样 (225Hz -> 125Hz)...")
        t_resampled, y_resampled = resample_data(t, y_inverted, 225, 125)
        print(f"重采样完成：{len(y_resampled)} 个数据点")

        # 4. 绘制全部数据
        plot_data(t_resampled, y_resampled, "重采样后的全部数据")

        # 5. 可选截取时间范围
        xlim_input = input("是否需要限制x轴范围？请输入起始和结束时间（如：0 10），或直接按回车跳过: ")

        xlim_start = None
        xlim_end = None

        if xlim_input:
            try:
                xlim_input = clean_file_path(xlim_input)
                xlim_start, xlim_end = map(float, xlim_input.split())
                print(f"选择的时间范围: {xlim_start} - {xlim_end} 秒")

                if xlim_start < t_resampled[0] or xlim_end > t_resampled[-1]:
                    print(f"警告：选择的时间范围超出数据范围 ({t_resampled[0]:.2f} - {t_resampled[-1]:.2f} 秒)")
                    print("将使用最接近的有效范围")
                    xlim_start = max(xlim_start, t_resampled[0])
                    xlim_end = min(xlim_end, t_resampled[-1])
                    print(f"调整后的时间范围: {xlim_start} - {xlim_end} 秒")

                t_selected, y_selected = extract_time_range(t_resampled, y_resampled, (xlim_start, xlim_end))

                if len(t_selected) == 0:
                    print("错误：选择的时间范围内没有数据，请重新选择")
                    return

                print(f"选定时间范围内的数据点数量: {len(t_selected)}")
                plot_data(t_selected, y_selected, f"选定时间范围的数据: {xlim_start} - {xlim_end} 秒")

                t_processed = t_selected
                y_to_process = y_selected

            except ValueError:
                print("输入格式错误，将使用全部数据范围进行处理")
                t_processed = t_resampled
                y_to_process = y_resampled
                xlim_start = None
                xlim_end = None
            except Exception as e:
                print(f"处理时间范围时出错: {e}")
                print("将使用全部数据范围进行处理")
                t_processed = t_resampled
                y_to_process = y_resampled
                xlim_start = None
                xlim_end = None
        else:
            print("使用全部数据范围进行处理")
            t_processed = t_resampled
            y_to_process = y_resampled

        # 6. 基线校正
        y_baseline_corrected = smart_baseline_correction(y_to_process)

        # —— 频谱保存到“默认路径”（与默认CSV同目录、同base_name）——
        # 默认CSV的命名仍沿用你的逻辑：base_name + 后缀
        base_name = os.path.splitext(file_path)[0]
        if xlim_start is not None and xlim_end is not None:
            tag = f"{xlim_start}_{xlim_end}s"
        else:
            tag = "full"

        spectrum_pre_path = f"{base_name}_spectrum_pre_{tag}.png"
        spectrum_post_path = f"{base_name}_spectrum_post_{tag}.png"

        # 6.1 滤波前频谱（基线校正后）
        fs = 125
        freqs_pre, mag_pre = compute_fft_spectrum(y_baseline_corrected, fs)
        plot_spectrum(
            freqs_pre, mag_pre,
            title="滤波前频谱图（基线校正后）",
            lowcut=0.5, highcut=10.0, fmax=20.0,
            save_path=spectrum_pre_path
        )

        # 7. 带通滤波 (0.5-10Hz)
        y_filtered = adaptive_bandpass_filter(y_baseline_corrected, fs)

        # 7.1 滤波后频谱
        freqs_post, mag_post = compute_fft_spectrum(y_filtered, fs)
        plot_spectrum(
            freqs_post, mag_post,
            title="滤波后频谱图（0.5–10 Hz 带通后）",
            lowcut=0.5, highcut=10.0, fmax=20.0,
            save_path=spectrum_post_path
        )

        # 8. 标准化
        y_normalized = normalize_data(y_filtered)

        # 9. 保存处理后的数据
        output_df = pd.DataFrame({
            '时间(秒)': t_processed,
            '处理后的Reddata': y_normalized
        })

        # 默认输出路径（你的原逻辑不改）
        if xlim_start is not None and xlim_end is not None:
            default_output = f"{base_name}_processed_{xlim_start}_{xlim_end}s.csv"
        else:
            default_output = f"{base_name}_processed_full.csv"

        output_path = input(f"请输入输出CSV文件的保存路径（直接回车使用默认路径: {default_output}）: ")
        output_path = clean_output_path(output_path)
        if not output_path:
            output_path = default_output

        ensure_dir_for_file(output_path)
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"处理后的数据已保存至: {output_path}")
        print(f"保存的数据点数量: {len(y_normalized)}")
        print(f"保存的时间范围: {t_processed[0]:.2f} - {t_processed[-1]:.2f} 秒")

        # 10. 绘制最终处理后的数据
        plot_data(t_processed, y_normalized, "最终处理后的数据")

        print("=" * 50)
        print("处理完成！")
        print("=" * 50)

        print("频谱图默认保存位置（与默认CSV同目录）:")
        print(f"  滤波前频谱图: {spectrum_pre_path}")
        print(f"  滤波后频谱图: {spectrum_post_path}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查文件格式和路径是否正确")


if __name__ == "__main__":
    main()
