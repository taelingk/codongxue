import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque  # 关键修复：新增deque导入
from scipy.signal import find_peaks, butter, filtfilt, resample  # 补充缺失的信号处理导入
import onnxruntime as ort  # 补充ONNX Runtime导入
import joblib  # 补充标准化器加载导入
import serial  # 补充串口导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QRadioButton,
                             QButtonGroup, QPushButton, QMessageBox, QGroupBox,
                             QComboBox, QFileDialog, QDialog)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import matplotlib

# 全局设置matplotlib字体大小
matplotlib.rcParams.update({
    'font.size': 12,  # 全局字体大小
    'axes.titlesize': 14,  # 轴标题大小
    'axes.labelsize': 13,  # 轴标签大小
    'xtick.labelsize': 12,  # x轴刻度大小
    'ytick.labelsize': 12,  # y轴刻度大小
    'legend.fontsize': 12,  # 图例大小
})
matplotlib.use('Qt5Agg')
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# ================== 核心工具函数 ==================
def calculate_derivatives(signal):
    """计算一阶/二阶导数"""
    first_deriv = np.zeros_like(signal)
    second_deriv = np.zeros_like(signal)

    first_deriv[:, 1:-1] = (signal[:, 2:] - signal[:, :-2]) / 2
    first_deriv[:, 0] = signal[:, 1] - signal[:, 0]
    first_deriv[:, -1] = signal[:, -1] - signal[:, -2]

    second_deriv[:, 1:-1] = (first_deriv[:, 2:] - first_deriv[:, :-2]) / 2
    second_deriv[:, 0] = first_deriv[:, 1] - first_deriv[:, 0]
    second_deriv[:, -1] = first_deriv[:, -1] - first_deriv[:, -2]
    return first_deriv, second_deriv


def standardize_single_feature(self, data, scaler_name):
    """标准化单个特征（终极容错版）"""
    # 第一步：检查输入数据
    if data is None or len(data) == 0:
        logging.warning(f"{scaler_name}：输入数据为空，返回0")
        return np.zeros_like(data).astype(np.float32) if len(data) > 0 else np.array([0], np.float32)

    # 第二步：检查标准化器是否存在
    scaler = self.scalers.get(scaler_name)
    if scaler is None:
        logging.warning(f"{scaler_name}：标准化器未加载，返回原始数据")
        return data.astype(np.float32)

    # 第三步：安全执行标准化
    try:
        # 确保数据形状正确
        data_2d = data.reshape(-1, 1)
        standardized = scaler.transform(data_2d).flatten().astype(np.float32)
        logging.debug(f"{scaler_name}标准化：{data[0]:.4f} → {standardized[0]:.4f}")
        return standardized
    except Exception as e:
        logging.error(f"{scaler_name}标准化失败：{e}，返回原始数据")
        return data.astype(np.float32)


def resample_waveform(data, original_fs=225, target_fs=125):
    """
    波形降采样：从225Hz降到125Hz
    使用scipy.resample保证时间轴对齐，数据长度准确
    """
    if len(data) < original_fs:  # 至少需要1秒原始数据
        return np.zeros(target_fs)

    # 计算需要重采样的点数（保留全部数据，不截断）
    target_length = int(len(data) * target_fs / original_fs)
    if target_length == 0:
        return np.zeros(target_fs)

    # 重采样（频域方法，保持波形特征）
    resampled_data = resample(data, target_length)

    return resampled_data.astype(np.float32)


def generate_time_axis(data_length, fs, start_time=None):
    """
    生成波形的时间轴（横轴为实际时间，仅显示HH:MM:SS）
    :param data_length: 数据点数量
    :param fs: 采样率（Hz）
    :param start_time: 起始时间（datetime对象），默认当前时间往前推对应时长
    :return: 时间轴数组（datetime对象）、时间标签列表（如 "16:30:01"）
    """
    # 计算数据总时长（秒）
    total_seconds = data_length / fs
    # 确定起始时间（默认当前时间 - 总时长）
    if start_time is None:
        start_time = datetime.now() - timedelta(seconds=total_seconds)
    # 生成时间轴（每个采样点对应一个时间）
    time_axis = []
    time_labels = []
    for i in range(data_length):
        # 每个采样点的时间偏移（秒）
        offset = i / fs
        current_time = start_time + timedelta(seconds=offset)
        time_axis.append(current_time)
        # 仅显示时分秒，移除毫秒
        time_labels.append(current_time.strftime("%H:%M:%S"))
    return time_axis, time_labels


# ================== 串口接收线程（最终版） ==================
class SerialReceiveThread(QThread):
    data_received = pyqtSignal(np.ndarray)  # 发射225Hz原始处理后数据
    data_125hz = pyqtSignal(np.ndarray)  # 发射125Hz降采样后数据（供模型使用）
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(str)

    def __init__(self, port, baudrate=921600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.is_running = True
        self.paused = False

        # 核心参数
        self.FS_225 = 225  # 串口原始采样率
        self.FS_125 = 125  # 模型要求采样率
        self.LOWCUT = 0.5
        self.HIGHCUT = 15.0
        self.FILTER_ORDER = 2
        self.FILTER_MIN_LEN = 20

        self.serial_buffer = bytearray()
        self.red_data_raw = deque(maxlen=225 * 5)  # 缓存5秒的225Hz数据（1125个点）

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.01)
            self.status_updated.emit(f"串口已打开：{self.port} | 波特率：{self.baudrate}")

            while self.is_running and self.ser.is_open:
                if self.paused:
                    self.msleep(100)
                    continue

                # 读取串口数据
                if self.ser.in_waiting > 0:
                    raw_bytes = self.ser.read(min(self.ser.in_waiting, 1024))
                    self.serial_buffer.extend(raw_bytes)
                    if len(self.serial_buffer) > 550000:
                        self.serial_buffer = self.serial_buffer[-550000:]
                    self.parse_spo2_bytes_to_wave()  # 实时解析
                self.msleep(5)

        except Exception as e:
            self.error_occurred.emit(f"串口错误：{str(e)}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()
            self.status_updated.emit("串口已关闭")

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def butter_bandpass_filter(self, y):
        """带通滤波（225Hz数据）"""
        if len(y) < self.FILTER_MIN_LEN:
            return y
        nyquist = 0.5 * self.FS_225
        low = self.LOWCUT / nyquist
        high = self.HIGHCUT / nyquist
        b, a = butter(self.FILTER_ORDER, [low, high], btype='band')
        return filtfilt(b, a, y)

    def normalize_wave(self, wave_data):
        """归一化到[0,1]"""
        if len(wave_data) == 0 or np.max(wave_data) == np.min(wave_data):
            return wave_data
        return (wave_data - np.min(wave_data)) / (np.max(wave_data) - np.min(wave_data))

    def parse_spo2_bytes_to_wave(self):
        """解析串口字节为225Hz脉搏波，并降采样到125Hz"""
        if len(self.serial_buffer) < 1418:
            return

        data = np.array(self.serial_buffer.copy(), dtype=np.uint8)
        temp_red_wave = []
        k = 0

        while k < len(data) - 1418:
            if data[k] == 85 and data[k + 1] == 170:
                if k + 15 >= len(data):
                    break

                # 解析控制字节
                byte14 = data[k + 14]
                groundnum = int(bin(byte14)[2:].zfill(8)[:4], 2) + 1
                plusenum = int(bin(byte14)[2:].zfill(8)[4:], 2) + 1
                k += 16

                # 仅处理红光通道（i=1）
                for i in range(2):
                    if i != 1:
                        k += groundnum * 2 + plusenum * 2
                        continue
                    if k + groundnum * 2 + plusenum * 2 > len(data):
                        break

                    # 解析背景波和脉搏波
                    gw_data = data[k:k + groundnum * 2].tolist()
                    pw_data = data[k + groundnum * 2:k + groundnum * 2 + plusenum * 2].tolist()
                    k += groundnum * 2 + plusenum * 2

                    # 转换为16位数据
                    gw = np.array(gw_data, dtype=np.float32).reshape(-1, 2)
                    gw_16 = gw[:, 0] + 256 * gw[:, 1]
                    pw = np.array(pw_data, dtype=np.float32).reshape(-1, 2)
                    pw_16 = pw[:, 0] + 256 * pw[:, 1]

                    # 计算差分波
                    min_len = min(len(gw_16), len(pw_16))
                    if min_len > 0:
                        wave = pw_16[:min_len] - gw_16[:min_len]
                        temp_red_wave.extend(wave[:50])
            k += 2

        # 处理并发射数据
        if temp_red_wave:
            self.red_data_raw.extend(temp_red_wave)
            raw_wave = np.array(self.red_data_raw)

            # 1. 225Hz数据预处理（取反+滤波+归一化）
            wave_inverted = -raw_wave
            wave_filtered = self.butter_bandpass_filter(wave_inverted)
            wave_225hz = self.normalize_wave(wave_filtered)

            # 2. 降采样到125Hz（供模型使用）
            wave_125hz = resample_waveform(wave_225hz, self.FS_225, self.FS_125)

            # 3. 发射数据
            if len(wave_225hz) >= self.FILTER_MIN_LEN:
                self.data_received.emit(wave_225hz)
            if len(wave_125hz) >= 125:  # 至少1秒数据
                self.data_125hz.emit(wave_125hz)

    def stop(self):
        self.is_running = False
        self.wait()


# ================== 趋势图对话框 ==================
class TrendDialog(QDialog):
    def __init__(self, history_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("心功能参数趋势图")
        self.setGeometry(200, 200, 800, 500)

        # 设置UI字体
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        self.plot_trend(history_data)

    def plot_trend(self, history_data):
        if not history_data:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "暂无数据", ha='center', va='center', transform=ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        times = [item[0].strftime("%H:%M:%S") for item in history_data]
        co_values = [item[1] for item in history_data]
        sv_values = [item[2] for item in history_data]

        self.fig.clear()
        ax1 = self.fig.add_subplot(211)
        ax1.plot(times, co_values, 'b-o', label='心输出量(L/min)', linewidth=2)
        ax1.axhline(y=4, color='r', linestyle='--', alpha=0.6)
        ax1.axhline(y=8, color='r', linestyle='--', alpha=0.6)
        ax1.set_ylabel('心输出量', fontsize=13)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        ax2 = self.fig.add_subplot(212)
        ax2.plot(times, sv_values, 'g-o', label='每搏输出量(mL)', linewidth=2)
        ax2.axhline(y=60, color='r', linestyle='--', alpha=0.6)
        ax2.axhline(y=100, color='r', linestyle='--', alpha=0.6)
        ax2.set_xlabel('时间', fontsize=13)
        ax2.set_ylabel('每搏输出量', fontsize=13)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.xticks(rotation=45, fontsize=12)
        self.fig.tight_layout()
        self.canvas.draw()


# ================== 主窗口（最终版） ==================
class PulseWaveMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_config()
        self.init_ui()
        self.init_timer()

    def init_config(self):
        """初始化核心配置"""
        self.input_mode = "file"
        self.pulse_data_225hz = None  # 225Hz原始数据（显示用）
        self.pulse_data_125hz = None  # 125Hz降采样数据（模型用）
        self.sampling_rate_225 = 225
        self.sampling_rate_125 = 125
        self.window_size = 5  # 5秒窗口
        self.current_window_start = 0
        self.base_time = datetime.now()
        self.calculated_hr = 0
        self.is_scrolling = False

        # 串口相关
        self.serial_thread = None
        self.serial_data_buffer_225 = deque(maxlen=225 * 5)  # 缓存5秒的225Hz数据
        self.serial_data_buffer_125 = deque(maxlen=125 * 5)  # 缓存5秒的125Hz数据

        # 推理相关
        self.onnx_session = None
        self.model_loaded = False
        self.signal_scaler = None
        self.scalers = {}  # 标准化器缓存
        self.prediction_history = []  # 5秒均值历史
        self.one_second_predictions = []  # 1秒推理历史

        # 模型路径（必须修改为你的实际路径！）
        self.model_root = r"D:\心输出量项目\CNAP脉氧处理数据\脉氧+CNAP\tuili-嵌入模型3-1226 (2)\tuili-嵌入模型3-1226"
        self.model_path = os.path.join(self.model_root, "parallel_resnet_se_lstm_model.onnx")
        self.scalar_paths = {
            "signal": os.path.join(self.model_root, "signal_scaler.pkl"),
            "age": os.path.join(self.model_root, "age_scaler.pkl"),
            "weight": os.path.join(self.model_root, "weight_scaler.pkl"),
            "height": os.path.join(self.model_root, "height_scaler.pkl"),
            "bsa": os.path.join(self.model_root, "bsa_scaler.pkl"),
            "bmi": os.path.join(self.model_root, "bmi_scaler.pkl"),
            "hr": os.path.join(self.model_root, "hr_scaler.pkl")
        }

    def init_ui(self):
        """初始化界面（调大所有字体）"""
        self.setWindowTitle("心输出量监测系统（125Hz模型适配版）")
        self.setGeometry(100, 100, 1200, 800)  # 调大窗口尺寸

        # 设置全局字体
        font = QFont()
        font.setPointSize(12)
        self.setFont(font)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 1. 顶部控制区
        top_layout = QHBoxLayout()
        # 模式选择
        mode_label = QLabel("输入模式：")
        mode_label.setFont(QFont("Arial", 12))
        top_layout.addWidget(mode_label)

        self.mode_group = QButtonGroup()
        self.file_mode_btn = QRadioButton("文件输入")
        self.file_mode_btn.setFont(QFont("Arial", 12))
        self.serial_mode_btn = QRadioButton("串口输入")
        self.serial_mode_btn.setFont(QFont("Arial", 12))
        self.file_mode_btn.setChecked(True)
        self.file_mode_btn.toggled.connect(self.toggle_input_mode)
        self.mode_group.addButton(self.file_mode_btn)
        self.mode_group.addButton(self.serial_mode_btn)
        top_layout.addWidget(self.file_mode_btn)
        top_layout.addWidget(self.serial_mode_btn)

        # 文件控件
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择Excel/CSV数据文件")
        self.path_input.setFont(QFont("Arial", 12))
        top_layout.addWidget(self.path_input)

        self.browse_btn = QPushButton("浏览")
        self.browse_btn.setFont(QFont("Arial", 12))
        self.browse_btn.clicked.connect(self.browse_file)
        top_layout.addWidget(self.browse_btn)

        # 串口控件
        self.port_combo = QComboBox()
        self.port_combo.setFont(QFont("Arial", 12))
        try:
            import serial.tools.list_ports
            self.port_combo.addItems([p.device for p in serial.tools.list_ports.comports()])
        except:
            self.port_combo.addItems(["COM1", "COM2", "COM3"])
        top_layout.addWidget(self.port_combo)

        self.baud_combo = QComboBox()
        self.baud_combo.setFont(QFont("Arial", 12))
        self.baud_combo.addItems(["921600", "115200"])
        self.baud_combo.setCurrentText("921600")
        top_layout.addWidget(self.baud_combo)

        # 功能按钮
        self.load_btn = QPushButton("加载数据")
        self.load_btn.setFont(QFont("Arial", 12))
        self.load_btn.clicked.connect(self.load_data)
        top_layout.addWidget(self.load_btn)

        self.load_model_btn = QPushButton("加载ONNX模型")
        self.load_model_btn.setFont(QFont("Arial", 12))
        self.load_model_btn.clicked.connect(self.load_inference_model)
        top_layout.addWidget(self.load_model_btn)

        # 默认隐藏串口控件
        self.port_combo.hide()
        self.baud_combo.hide()
        main_layout.addLayout(top_layout)

        # 2. 中间参数/结果/控制区
        mid_layout = QHBoxLayout()
        # 2.1 生理参数输入
        param_group = QGroupBox("生理参数输入")
        param_group.setFont(QFont("Arial", 14))
        param_layout = QVBoxLayout(param_group)

        # 体重
        weight_label = QLabel("体重(kg)：")
        weight_label.setFont(QFont("Arial", 12))
        self.weight_input = QLineEdit()
        self.weight_input.setFont(QFont("Arial", 12))
        weight_layout = QHBoxLayout()
        weight_layout.addWidget(weight_label)
        weight_layout.addWidget(self.weight_input)
        param_layout.addLayout(weight_layout)

        # 年龄
        age_label = QLabel("年龄(岁)：")
        age_label.setFont(QFont("Arial", 12))
        self.age_input = QLineEdit()
        self.age_input.setFont(QFont("Arial", 12))
        age_layout = QHBoxLayout()
        age_layout.addWidget(age_label)
        age_layout.addWidget(self.age_input)
        param_layout.addLayout(age_layout)

        # 身高
        height_label = QLabel("身高(cm)：")
        height_label.setFont(QFont("Arial", 12))
        self.height_input = QLineEdit()
        self.height_input.setFont(QFont("Arial", 12))
        height_layout = QHBoxLayout()
        height_layout.addWidget(height_label)
        height_layout.addWidget(self.height_input)
        param_layout.addLayout(height_layout)

        # 性别
        gender_label = QLabel("性别：")
        gender_label.setFont(QFont("Arial", 12))
        self.male_btn = QRadioButton("男")
        self.male_btn.setFont(QFont("Arial", 12))
        self.female_btn = QRadioButton("女")
        self.female_btn.setFont(QFont("Arial", 12))
        self.male_btn.setChecked(True)
        gender_layout = QHBoxLayout()
        gender_layout.addWidget(gender_label)
        gender_layout.addWidget(self.male_btn)
        gender_layout.addWidget(self.female_btn)
        param_layout.addLayout(gender_layout)

        # BSA
        bsa_label = QLabel("体表面积(m²)：")
        bsa_label.setFont(QFont("Arial", 12))
        self.bsa_value = QLabel("--")
        self.bsa_value.setFont(QFont("Arial", 12))
        bsa_layout = QHBoxLayout()
        bsa_layout.addWidget(bsa_label)
        bsa_layout.addWidget(self.bsa_value)
        param_layout.addLayout(bsa_layout)

        # BMI
        bmi_label = QLabel("BMI：")
        bmi_label.setFont(QFont("Arial", 12))
        self.bmi_value = QLabel("--")
        self.bmi_value.setFont(QFont("Arial", 12))
        bmi_layout = QHBoxLayout()
        bmi_layout.addWidget(bmi_label)
        bmi_layout.addWidget(self.bmi_value)
        param_layout.addLayout(bmi_layout)

        self.calc_param_btn = QPushButton("计算BSA/BMI")
        self.calc_param_btn.setFont(QFont("Arial", 12))
        self.calc_param_btn.clicked.connect(self.calculate_bsa_bmi)
        param_layout.addWidget(self.calc_param_btn)
        mid_layout.addWidget(param_group)

        # 2.2 预测结果显示
        result_group = QGroupBox("心功能预测结果（125Hz模型）")
        result_group.setFont(QFont("Arial", 14))
        result_layout = QVBoxLayout(result_group)

        # 心输出量
        co_label = QLabel("心输出量(L/min)：")
        co_label.setFont(QFont("Arial", 12))
        self.co_value = QLabel("--")
        self.co_value.setFont(QFont("Arial", 16))  # 结果字体更大
        self.co_value.setStyleSheet("font-weight: bold;")
        co_layout = QHBoxLayout()
        co_layout.addWidget(co_label)
        co_layout.addWidget(self.co_value)
        result_layout.addLayout(co_layout)

        # 每搏输出量
        sv_label = QLabel("每搏输出量(mL)：")
        sv_label.setFont(QFont("Arial", 12))
        self.sv_value = QLabel("--")
        self.sv_value.setFont(QFont("Arial", 16))
        self.sv_value.setStyleSheet("font-weight: bold;")
        sv_layout = QHBoxLayout()
        sv_layout.addWidget(sv_label)
        sv_layout.addWidget(self.sv_value)
        result_layout.addLayout(sv_layout)

        # 心率
        hr_label = QLabel("心率(次/分)：")
        hr_label.setFont(QFont("Arial", 12))
        self.hr_value = QLabel("--")
        self.hr_value.setFont(QFont("Arial", 16))
        self.hr_value.setStyleSheet("font-weight: bold;")
        hr_layout = QHBoxLayout()
        hr_layout.addWidget(hr_label)
        hr_layout.addWidget(self.hr_value)
        result_layout.addLayout(hr_layout)

        mid_layout.addWidget(result_group)

        # 2.3 控制按钮
        control_group = QGroupBox("功能控制")
        control_group.setFont(QFont("Arial", 14))
        control_layout = QVBoxLayout(control_group)

        self.start_btn = QPushButton("开始监测")
        self.start_btn.setFont(QFont("Arial", 12))
        self.start_btn.clicked.connect(self.start_monitor)
        control_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setFont(QFont("Arial", 12))
        self.pause_btn.clicked.connect(self.pause_monitor)
        control_layout.addWidget(self.pause_btn)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.setFont(QFont("Arial", 12))
        self.reset_btn.clicked.connect(self.reset_monitor)
        control_layout.addWidget(self.reset_btn)

        self.save_btn = QPushButton("保存数据")
        self.save_btn.setFont(QFont("Arial", 12))
        self.save_btn.clicked.connect(self.save_data)
        control_layout.addWidget(self.save_btn)

        self.trend_btn = QPushButton("查看趋势")
        self.trend_btn.setFont(QFont("Arial", 12))
        self.trend_btn.clicked.connect(self.show_trend)
        control_layout.addWidget(self.trend_btn)

        mid_layout.addWidget(control_group)
        main_layout.addLayout(mid_layout)

        # 3. 波形显示区（调大尺寸）
        wave_title = QLabel("脉搏波波形（125Hz，最近5秒）")
        wave_title.setFont(QFont("Arial", 14))
        main_layout.addWidget(wave_title)

        self.fig = Figure(figsize=(12, 4), dpi=100)  # 调大图表尺寸
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(NavigationToolbar(self.canvas, self))
        main_layout.addWidget(self.canvas)

        # 4. 状态提示
        self.status_label = QLabel("状态：请先加载ONNX模型和输入数据 | 模型采样率：125Hz | 串口采样率：225Hz")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setStyleSheet("color: blue;")
        main_layout.addWidget(self.status_label)

        # 初始按钮状态
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.trend_btn.setEnabled(False)

    def _create_labeled_layout(self, label_text, *widgets):
        """创建带标签的水平布局"""
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFont(QFont("Arial", 12))
        layout.addWidget(label)
        for w in widgets:
            w.setFont(QFont("Arial", 12))
            layout.addWidget(w)
        return layout

    def init_timer(self):
        """初始化定时器"""
        self.scroll_timer = QTimer()
        self.scroll_timer.setInterval(1000)  # 1秒滚动一次
        self.scroll_timer.timeout.connect(self.scroll_window)

        self.predict_timer = QTimer()
        self.predict_timer.setInterval(1500)  # 1.5秒推理一次
        self.predict_timer.timeout.connect(self.predict_data)

        self.wave_timer = QTimer()
        self.wave_timer.setInterval(200)  # 提高波形刷新频率（200ms）
        self.wave_timer.timeout.connect(self.update_wave_display)

    # ================== 界面切换与数据加载 ==================
    def toggle_input_mode(self):
        """切换文件/串口输入模式"""
        if self.file_mode_btn.isChecked():
            self.input_mode = "file"
            self.path_input.show()
            self.browse_btn.show()
            self.port_combo.hide()
            self.baud_combo.hide()
            self.load_btn.setText("加载文件")
        else:
            self.input_mode = "serial"
            self.path_input.hide()
            self.browse_btn.hide()
            self.port_combo.show()
            self.baud_combo.show()
            self.load_btn.setText("连接串口")

    def browse_file(self):
        """浏览选择数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "Excel/CSV (*.xlsx *.csv)")
        if file_path:
            self.path_input.setText(file_path)

    def load_data(self):
        """加载文件/串口数据"""
        if self.input_mode == "file":
            self.load_from_file()
        else:
            self.load_from_serial()

    def load_from_file(self):
        """从文件加载脉搏波数据"""
        try:
            file_path = self.path_input.text()
            if not file_path:
                QMessageBox.warning(self, "警告", "请选择数据文件")
                return

            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)

            # 读取第一列作为225Hz原始数据
            self.pulse_data_225hz = df.iloc[:, 0].values.astype(float)
            # 降采样到125Hz（保留全部数据）
            self.pulse_data_125hz = resample_waveform(self.pulse_data_225hz, 225, 125)

            self.status_label.setText(
                f"已加载文件数据：{len(self.pulse_data_225hz)} 个采样点（225Hz）| 降采样后：{len(self.pulse_data_125hz)} 个点（125Hz）")
            self.calculate_heart_rate()
            self.update_wave_display()
            if self.model_loaded:
                self.start_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"文件加载失败：{str(e)}")

    def load_from_serial(self):
        """连接串口并接收实时数据"""
        try:
            # 停止之前的串口线程
            if self.serial_thread:
                self.serial_thread.stop()

            port = self.port_combo.currentText()
            baudrate = int(self.baud_combo.currentText())

            # 创建并启动串口线程
            self.serial_thread = SerialReceiveThread(port, baudrate)
            # 225Hz数据用于显示
            self.serial_thread.data_received.connect(self.on_serial_data_225hz)
            # 125Hz数据直接供模型使用
            self.serial_thread.data_125hz.connect(self.on_serial_data_125hz)
            self.serial_thread.error_occurred.connect(lambda e: QMessageBox.critical(self, "串口错误", e))
            self.serial_thread.status_updated.connect(self.status_label.setText)
            self.serial_thread.start()

            # 启动波形更新定时器
            self.wave_timer.start()
            if self.model_loaded:
                self.start_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"串口连接失败：{str(e)}")

    def on_serial_data_225hz(self, wave_data):
        """接收225Hz数据（显示用）"""
        self.serial_data_buffer_225.extend(wave_data)
        self.pulse_data_225hz = np.array(list(self.serial_data_buffer_225))
        # 实时计算心率（基于225Hz数据）
        self.calculate_heart_rate()
        self.hr_value.setText(f"{self.calculated_hr:.1f}")

    def on_serial_data_125hz(self, wave_data):
        """接收125Hz数据（模型用）- 核心！"""
        self.serial_data_buffer_125.extend(wave_data)
        # 确保缓存中始终是最新5秒的125Hz数据（625个点）
        if len(self.serial_data_buffer_125) > 125 * 5:
            self.serial_data_buffer_125 = deque(list(self.serial_data_buffer_125)[-125 * 5:])
        # 更新模型输入数据
        self.pulse_data_125hz = np.array(list(self.serial_data_buffer_125))

    # ================== 模型加载与推理（适配125Hz） ==================
    def load_inference_model(self):
        """加载ONNX模型和标量（新增空值校验+容错）"""
        try:
            # 初始化标准化器缓存（避免KeyError）
            self.scalers = {}
            # 检查模型文件
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"模型文件不存在：{self.model_path}")

            # 加载ONNX模型
            self.onnx_session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )

            # ========== 安全加载所有标准化器 ==========
            from sklearn.preprocessing import StandardScaler
            scaler_load_status = []
            for name, path in self.scalar_paths.items():
                try:
                    if not os.path.exists(path):
                        scaler_load_status.append(f"❌ {name}：文件不存在 {path}")
                        self.scalers[name] = None  # 明确赋值为None，方便后续判断
                        continue

                    # 加载并清理版本信息
                    scaler = joblib.load(path)
                    if '_sklearn_version' in scaler.__dict__:
                        scaler.__dict__.pop('_sklearn_version')

                    # 校验类型
                    if not isinstance(scaler, StandardScaler):
                        scaler_load_status.append(f"❌ {name}：类型错误（应为StandardScaler，实际{type(scaler)}）")
                        self.scalers[name] = None
                        continue

                    # 加载成功
                    self.scalers[name] = scaler
                    scaler_load_status.append(f"✅ {name}：加载成功")
                    logging.info(f"{name}标准化器加载成功：{path}")

                except Exception as e:
                    err_msg = f"❌ {name}：加载失败 {str(e)}"
                    scaler_load_status.append(err_msg)
                    self.scalers[name] = None
                    logging.error(err_msg)

            # 单独处理signal_scaler（兼容旧代码）
            self.signal_scaler = self.scalers.get("signal", None)

            # ========== 状态反馈 ==========
            success_count = sum([1 for s in self.scalers.values() if s is not None])
            total_count = len(self.scalar_paths)
            self.status_label.setText(f"ONNX模型加载成功 | 标准化器：{success_count}/{total_count}个可用")

            # 弹窗提示
            status_msg = "标准化器加载详情：\n" + "\n".join(scaler_load_status)
            if success_count == total_count:
                self.model_loaded = True
                QMessageBox.information(self, "成功", f"模型加载完成！\n\n{status_msg}")
            else:
                self.model_loaded = True  # 即使部分失败，仍允许运行（用原始数据）
                QMessageBox.warning(self, "注意",
                                    f"模型加载完成，但部分标准化器不可用！\n\n{status_msg}\n\n将使用原始数据推理")

            if self.pulse_data_225hz is not None:
                self.start_btn.setEnabled(True)

        except Exception as e:
            err_msg = f"模型加载失败：{str(e)}"
            logging.error(err_msg)
            QMessageBox.critical(self, "模型加载失败", err_msg)
            self.model_loaded = False
            # 重置标准化器缓存
            self.scalers = {}
            self.signal_scaler = None

    def prepare_model_inputs(self):
        """准备125Hz模型输入数据（增加标准化器空值校验）"""
        if self.pulse_data_125hz is None or len(self.pulse_data_125hz) < 125:
            self.status_label.setText("状态：125Hz数据不足（需至少125个点）")
            return None

        try:
            # 1. 取最新1秒的125Hz数据用于推理
            data_125hz = self.pulse_data_125hz[-125:]
            if len(data_125hz) < 125:
                data_125hz = np.pad(data_125hz, (0, 125 - len(data_125hz)), 'constant')

            # 2. 构建信号特征
            raw_signal = data_125hz.reshape(1, -1)
            # 处理异常值
            raw_signal = np.nan_to_num(raw_signal, nan=0.0, posinf=0.0, neginf=0.0)
            first_deriv, second_deriv = calculate_derivatives(raw_signal)
            X = np.stack([raw_signal, first_deriv, second_deriv], axis=-1).astype(np.float32)

            # 3. 安全标准化信号（核心修复：检查signal_scaler是否为None）
            X_reshaped = X.reshape(X.shape[0], -1)
            if self.signal_scaler is not None:
                try:
                    X_std = self.signal_scaler.transform(X_reshaped).reshape(X.shape).astype(np.float32)
                    logging.debug(
                        f"信号标准化成功 | 原始均值：{np.mean(X_reshaped):.4f} → 标准化后：{np.mean(X_std):.4f}")
                except Exception as e:
                    logging.error(f"信号标准化失败：{e}，使用原始数据")
                    X_std = X.astype(np.float32)
            else:
                logging.warning("信号标准化器未加载，使用原始信号数据")
                X_std = X.astype(np.float32)

            # 4. 读取生理参数（增加异常处理）
            try:
                weight = float(self.weight_input.text() or 0)
                age = int(self.age_input.text() or 0)
                height = float(self.height_input.text() or 0)
                gender = 1 if self.male_btn.isChecked() else 0
            except:
                logging.error("生理参数输入错误，使用默认值")
                weight, age, height, gender = 60, 30, 170, 1

            # 5. 计算BSA/BMI
            bsa, bmi = self.calculate_bsa_bmi(manual=False)

            # 6. 安全标准化生理参数（使用修复后的函数）
            age_std = standardize_single_feature(self, np.array([age]), "age")
            weight_std = standardize_single_feature(self, np.array([weight]), "weight")
            height_std = standardize_single_feature(self, np.array([height]), "height")
            bsa_std = standardize_single_feature(self, np.array([bsa]), "bsa")
            bmi_std = standardize_single_feature(self, np.array([bmi]), "bmi")
            hr_std = standardize_single_feature(self, np.array([self.calculated_hr]), "hr")
            gender_arr = np.array([gender], dtype=np.float32)

            # 7. 构建输入字典
            input_names = [inp.name for inp in self.onnx_session.get_inputs()]
            input_feed = {}
            if len(input_names) == 1:
                combined = np.concatenate([
                    X_std.flatten(), age_std, gender_arr, weight_std,
                    height_std, bsa_std, bmi_std, hr_std
                ]).reshape(1, -1)
                input_feed[input_names[0]] = combined
            else:
                input_list = [X_std, age_std, gender_arr, weight_std, height_std, bsa_std, bmi_std, hr_std]
                for i, name in enumerate(input_names):
                    if i < len(input_list):
                        input_feed[name] = input_list[i]

            return input_feed

        except Exception as e:
            logging.error(f"准备模型输入失败：{e}", exc_info=True)  # 输出完整堆栈信息
            self.status_label.setText(f"状态：准备输入失败 - {str(e)[:50]}")
            return None

    def predict_data(self):
        """执行125Hz模型推理"""
        if not self.model_loaded or not self.is_scrolling or self.calculated_hr <= 0:
            return

        # 准备125Hz模型输入
        input_feed = self.prepare_model_inputs()
        if input_feed is None:
            return

        try:
            # ONNX模型推理（125Hz数据）
            outputs = self.onnx_session.run(None, input_feed)
            pred_sv = float(outputs[0].flatten()[0])  # 每搏输出量
            pred_co = (pred_sv * self.calculated_hr) / 1000  # 心输出量

            # 限制医学合理范围
            pred_sv = np.clip(pred_sv, 30, 150)
            pred_co = np.clip(pred_co, 2, 15)

            # 保存1秒推理结果
            current_time = datetime.now()
            self.one_second_predictions.append((current_time, round(pred_co, 2), round(pred_sv, 1)))
            if len(self.one_second_predictions) > 100:
                self.one_second_predictions = self.one_second_predictions[-100:]

            # 每5个1秒结果计算一次均值并更新界面
            if len(self.one_second_predictions) % 5 == 0 and len(self.one_second_predictions) >= 5:
                recent = self.one_second_predictions[-5:]
                co_avg = round(np.mean([p[1] for p in recent]), 2)
                sv_avg = round(np.mean([p[2] for p in recent]), 1)
                # 更新界面
                self.co_value.setText(f"{co_avg}")
                self.sv_value.setText(f"{sv_avg}")
                # 异常值标红
                self.co_value.setStyleSheet("font-size: 16pt; font-weight: bold; color: red;" if not (
                            4 <= co_avg <= 8) else "font-size: 16pt; font-weight: bold;")
                self.sv_value.setStyleSheet("font-size: 16pt; font-weight: bold; color: red;" if not (
                            60 <= sv_avg <= 100) else "font-size: 16pt; font-weight: bold;")
                # 保存5秒均值历史
                self.prediction_history.append((current_time, co_avg, sv_avg))
                self.trend_btn.setEnabled(True)
                self.status_label.setText(
                    f"状态：推理中（125Hz模型）| 5秒均值 CO={co_avg} L/min, SV={sv_avg} mL | 心率={self.calculated_hr} 次/分")

        except Exception as e:
            logging.error(f"模型推理失败：{e}")
            self.status_label.setText(f"状态：推理错误（125Hz）- {str(e)}")

    # ================== 辅助功能 ==================
    def calculate_bsa_bmi(self, manual=True):
        """计算BSA和BMI"""
        try:
            weight = float(self.weight_input.text())
            height = float(self.height_input.text()) / 100  # 转米
            # BMI = 体重(kg) / 身高(m)²
            bmi = weight / (height ** 2)
            # BSA = 0.007184 × 体重^0.425 × 身高^0.725 (Du Bois公式)
            bsa = 0.007184 * (weight ** 0.425) * (height ** 0.725)
            # 更新显示
            self.bsa_value.setText(f"{bsa:.3f}")
            self.bmi_value.setText(f"{bmi:.2f}")
            if manual:
                QMessageBox.information(self, "计算完成", f"BSA: {bsa:.3f} m² | BMI: {bmi:.2f}")
            return bsa, bmi
        except:
            if manual:
                QMessageBox.warning(self, "输入错误", "请输入有效的体重/身高数值")
            return 0, 0

    def calculate_heart_rate(self):
        """基于225Hz数据计算心率（增加日志输出）"""
        if self.pulse_data_225hz is None or len(self.pulse_data_225hz) < self.sampling_rate_225 * 2:
            self.calculated_hr = 0
            return

        # 峰值检测（225Hz数据）
        peaks, _ = find_peaks(self.pulse_data_225hz, distance=self.sampling_rate_225 // 2, prominence=0.1)
        logging.info(f"检测到峰值数量：{len(peaks)} | 225Hz数据长度：{len(self.pulse_data_225hz)}")

        if len(peaks) >= 2:
            intervals = np.diff(peaks) / self.sampling_rate_225  # 峰间时间（秒）
            logging.info(f"峰间时间（秒）：{intervals} | 平均峰间时间：{np.mean(intervals):.2f}s")
            self.calculated_hr = int(60 / np.mean(intervals))  # 心率=60/平均峰间时间
        else:
            self.calculated_hr = 0
        self.hr_value.setText(f"{self.calculated_hr}")

    def update_wave_display(self):
        """显示5秒窗口的脉搏波波形（降采样到125Hz，横轴为实际时间HH:MM:SS，完整显示5秒）"""
        if self.pulse_data_225hz is None or len(self.pulse_data_225hz) == 0:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "暂无数据", ha='center', va='center', transform=self.ax.transAxes, fontsize=14)
            self.canvas.draw()
            return

        # ========== 1. 对225Hz原始数据降采样到125Hz（保留全部数据） ==========
        wave_data_225hz = self.pulse_data_225hz.copy()
        wave_data_125hz = resample_waveform(wave_data_225hz, original_fs=225, target_fs=125)

        # ========== 2. 截取最近5秒的125Hz数据（625个点） ==========
        target_points = 5 * self.sampling_rate_125  # 5秒×125Hz=625个点
        if len(wave_data_125hz) >= target_points:
            wave_display = wave_data_125hz[-target_points:]  # 取最新5秒数据
            # 起始时间：当前时间 - 5秒
            start_time = datetime.now() - timedelta(seconds=5)
        else:
            # 数据不足5秒时，补0到625个点
            wave_display = np.pad(wave_data_125hz, (0, target_points - len(wave_data_125hz)), 'constant')
            # 起始时间：当前时间 - 实际数据时长
            data_duration = len(wave_data_125hz) / self.sampling_rate_125
            start_time = datetime.now() - timedelta(seconds=data_duration)

        # ========== 3. 生成时间轴（完整5秒，仅显示HH:MM:SS） ==========
        time_axis, time_labels = generate_time_axis(
            data_length=len(wave_display),
            fs=self.sampling_rate_125,
            start_time=start_time
        )

        # ========== 4. 绘制完整的5秒波形 ==========
        self.ax.clear()
        # 绘制125Hz降采样后的波形（加粗线条）
        self.ax.plot(time_axis, wave_display, 'r-', linewidth=2, label='脉搏波（125Hz）')

        # ========== 5. 优化时间轴显示（完整展示5秒刻度，无毫秒） ==========
        self.ax.set_xlabel("时间（HH:MM:SS）", fontsize=13)
        self.ax.set_ylabel("归一化振幅", fontsize=13)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right', fontsize=12)

        # 调整x轴刻度：每1秒显示一个标签（5秒共5-6个标签），简洁易读
        tick_interval = self.sampling_rate_125  # 1秒=125个点
        tick_indices = list(range(0, len(time_labels), tick_interval))
        # 确保最后一个刻度也显示
        if tick_indices[-1] != len(time_labels) - 1:
            tick_indices.append(len(time_labels) - 1)
        tick_labels = [time_labels[i] for i in tick_indices]

        self.ax.set_xticks([time_axis[i] for i in tick_indices])
        self.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=12)

        # 调整布局，避免标签被遮挡
        self.fig.tight_layout()
        self.canvas.draw()

    def scroll_window(self):
        """滚动225Hz数据窗口（文件模式下生效）"""
        if self.pulse_data_225hz is None or self.input_mode == "serial":
            return  # 串口模式不滚动，始终显示最新5秒数据

        # 文件模式下，每次滚动1秒
        self.current_window_start += 1
        max_start = (len(self.pulse_data_225hz) / self.sampling_rate_225) - self.window_size
        if self.current_window_start > max_start:
            self.current_window_start = 0
        self.update_wave_display()

    # ================== 监测控制 ==================
    def start_monitor(self):
        """开始监测推理"""
        if not self.model_loaded:
            QMessageBox.warning(self, "警告", "请先加载ONNX模型")
            return
        if not (self.weight_input.text() and self.age_input.text() and self.height_input.text()):
            QMessageBox.warning(self, "警告", "请补全生理参数")
            return

        self.is_scrolling = True
        self.predict_timer.start()
        if self.input_mode == "file":
            self.scroll_timer.start()
        # 串口模式恢复数据接收
        if self.serial_thread:
            self.serial_thread.resume()

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.status_label.setText("状态：正在实时推理（125Hz模型）| 串口225Hz → 模型125Hz")

    def pause_monitor(self):
        """暂停监测"""
        self.is_scrolling = False
        self.scroll_timer.stop()
        self.predict_timer.stop()
        if self.serial_thread:
            self.serial_thread.pause()

        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.status_label.setText("状态：已暂停 | 模型采样率：125Hz | 串口采样率：225Hz")

    def reset_monitor(self):
        """重置系统"""
        self.is_scrolling = False
        self.scroll_timer.stop()
        self.predict_timer.stop()
        self.wave_timer.stop()

        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None

        self.pulse_data_225hz = None
        self.pulse_data_125hz = None
        self.serial_data_buffer_225.clear()
        self.serial_data_buffer_125.clear()
        self.current_window_start = 0
        self.prediction_history.clear()
        self.one_second_predictions.clear()

        self.co_value.setText("--")
        self.sv_value.setText("--")
        self.hr_value.setText("--")
        self.bsa_value.setText("--")
        self.bmi_value.setText("--")

        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.trend_btn.setEnabled(False)
        self.status_label.setText("状态：已重置 | 模型采样率：125Hz | 串口采样率：225Hz")

    def save_data(self):
        """保存预测数据"""
        if not self.prediction_history:
            QMessageBox.warning(self, "警告", "暂无数据可保存")
            return

        patient_id = "未知ID"
        # 准备数据
        five_second_data = pd.DataFrame({
            "数据类型": ["5秒均值"] * len(self.prediction_history),
            "患者ID": [patient_id] * len(self.prediction_history),
            "时间": [item[0].strftime("%H:%M:%S") for item in self.prediction_history],
            "心输出量(L/min)": [item[1] for item in self.prediction_history],
            "每搏输出量(mL)": [item[2] for item in self.prediction_history],
            "心率(次/分)": [self.calculated_hr] * len(self.prediction_history),
            "采样率": ["125Hz"] * len(self.prediction_history)
        })
        # 保存文件
        file_path, _ = QFileDialog.getSaveFileName(self, "保存数据", "心功能监测数据_125Hz.xlsx", "Excel (*.xlsx)")
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    five_second_data.to_excel(writer, sheet_name='5秒均值数据', index=False)
                QMessageBox.information(self, "成功", "数据保存完成（125Hz模型推理结果）")
            except Exception as e:
                QMessageBox.critical(self, "失败", f"保存错误：{str(e)}")

    def show_trend(self):
        """显示趋势图"""
        dialog = TrendDialog(self.prediction_history, self)
        dialog.exec_()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置应用程序字体
    font = QFont()
    font.setPointSize(12)
    app.setFont(font)

    window = PulseWaveMonitor()
    window.show()
    sys.exit(app.exec_())