import matplotlib
matplotlib.use("TkAgg")   # 或者 "Qt5Agg"
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import os
import matplotlib.font_manager as fm
import re


# 设置中文字体
def setup_chinese_font():
    try:
        # 尝试多种常见的中文字体
        chinese_fonts = [
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'SimSun',  # 宋体
            'KaiTi',  # 楷体
            'FangSong',  # 仿宋
            'Arial Unicode MS',  # Arial Unicode
            'DejaVu Sans'  # 备用字体
        ]

        # 设置字体
        plt.rcParams['font.sans-serif'] = chinese_fonts
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        # 检查是否成功设置
        test_font = plt.rcParams['font.sans-serif'][0]
        print(f"使用字体: {test_font}")

    except Exception as e:
        print(f"字体设置警告: {e}")
        print("将使用默认字体，中文可能显示为方框")


def clean_file_path(path):
    """清理文件路径，去除首尾的引号和空格"""
    # 去除首尾空格
    path = path.strip()
    # 去除首尾的引号（单引号和双引号）
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
    """提取指定时间范围内的数据"""
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
    return (y - np.min(y)) / (np.max(y) - np.min(y))


def clean_output_path(path):
    """清理输出文件路径，去除首尾的引号"""
    path = path.strip()
    path = re.sub(r'^[\'"]|[\'"]$', '', path)
    return path


def main():
    # 设置中文字体
    setup_chinese_font()

    print("=" * 50)
    print("脉搏波数据处理程序")
    print("=" * 50)

    try:
        # 1. 读取文件路径（自动处理引号）
        file_path = ask_file_path()
        print(f"正在读取文件: {file_path}")

        # 尝试多种编码方式读取CSV文件
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

        # 检查列名（不区分大小写）
        expected_columns = ['时间(秒)', 'Reddata']
        actual_columns = [col for col in df.columns]

        # 尝试找到匹配的列名
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

        # 2. 数据倒置
        print("正在进行数据倒置...")
        y_inverted = invert_data(y_red)

        # 3. 重采样：225Hz -> 125Hz
        print("正在进行重采样 (225Hz -> 125Hz)...")
        t_resampled, y_resampled = resample_data(t, y_inverted, 225, 125)
        print(f"重采样完成：{len(y_resampled)} 个数据点")

        # 4. 绘制全部数据
        plot_data(t_resampled, y_resampled, "重采样后的全部数据")

        # 5. 询问是否限制x轴范围
        xlim_input = input("是否需要限制x轴范围？请输入起始和结束时间（如：0 10），或直接按回车跳过: ")

        if xlim_input:
            try:
                # 清理输入（可能包含引号）
                xlim_input = clean_file_path(xlim_input)
                xlim_start, xlim_end = map(float, xlim_input.split())
                print(f"选择的时间范围: {xlim_start} - {xlim_end} 秒")

                # 检查时间范围是否有效
                if xlim_start < t_resampled[0] or xlim_end > t_resampled[-1]:
                    print(f"警告：选择的时间范围超出数据范围 ({t_resampled[0]:.2f} - {t_resampled[-1]:.2f} 秒)")
                    print("将使用最接近的有效范围")
                    xlim_start = max(xlim_start, t_resampled[0])
                    xlim_end = min(xlim_end, t_resampled[-1])
                    print(f"调整后的时间范围: {xlim_start} - {xlim_end} 秒")

                # 提取选定时间范围内的数据
                t_selected, y_selected = extract_time_range(t_resampled, y_resampled, (xlim_start, xlim_end))

                if len(t_selected) == 0:
                    print("错误：选择的时间范围内没有数据，请重新选择")
                    return

                print(f"选定时间范围内的数据点数量: {len(t_selected)}")

                # 绘制选定范围的数据
                plot_data(t_selected, y_selected, f"选定时间范围的数据: {xlim_start} - {xlim_end} 秒")

                # 更新为选定范围的数据
                t_processed = t_selected
                y_to_process = y_selected

            except ValueError:
                print("输入格式错误，请确保输入两个数字（如：0 10）")
                print("将使用全部数据范围进行处理")
                t_processed = t_resampled
                y_to_process = y_resampled
            except Exception as e:
                print(f"处理时间范围时出错: {e}")
                print("将使用全部数据范围进行处理")
                t_processed = t_resampled
                y_to_process = y_resampled
        else:
            print("使用全部数据范围进行处理")
            t_processed = t_resampled
            y_to_process = y_resampled

        # 6. 智能基线校正（只对选定范围的数据）
        y_baseline_corrected = smart_baseline_correction(y_to_process)

        # 7. 自适应带通滤波 (0.5 - 10 Hz)（只对选定范围的数据）
        fs = 125  # 重采样后的频率
        y_filtered = adaptive_bandpass_filter(y_baseline_corrected, fs)

        # 8. 标准化到 [0, 1]（只对选定范围的数据）
        y_normalized = normalize_data(y_filtered)

        # 9. 保存处理后的数据（只保存选定范围的数据）
        output_df = pd.DataFrame({
            '时间(秒)': t_processed,
            '处理后的Reddata': y_normalized
        })

        # 提供默认输出路径
        base_name = os.path.splitext(file_path)[0]
        if 'xlim_input' in locals() and xlim_input:
            default_output = f"{base_name}_processed_{xlim_start}_{xlim_end}s.csv"
        else:
            default_output = f"{base_name}_processed_full.csv"

        output_path = input(f"请输入输出CSV文件的保存路径（直接回车使用默认路径: {default_output}）: ")
        output_path = clean_output_path(output_path)

        if not output_path:
            output_path = default_output

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"处理后的数据已保存至: {output_path}")
        print(f"保存的数据点数量: {len(y_normalized)}")
        print(f"保存的时间范围: {t_processed[0]:.2f} - {t_processed[-1]:.2f} 秒")

        # 10. 绘制最终处理后的数据（只显示选定范围）
        plot_data(t_processed, y_normalized, "最终处理后的数据")

        print("=" * 50)
        print("处理完成！")
        print("=" * 50)

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查文件格式和路径是否正确")


if __name__ == "__main__":
    main()
