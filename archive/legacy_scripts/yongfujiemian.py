# -*- coding: utf-8 -*-

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QRadioButton, QButtonGroup, QPushButton,
    QMessageBox, QComboBox, QFileDialog, QDialog, QFrame
)
from PyQt5.QtCore import Qt, QTimer

import matplotlib
from scipy.signal import butter, filtfilt, find_peaks
import onnxruntime as ort
import joblib

# ================== 日志配置 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# ================== Matplotlib + Qt 嵌入 ==================
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 全局设置Matplotlib支持中文
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ================== 推理模型核心工具函数 ==================
def read_csv_with_encoding(file_path):
    """兼容多种编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16']
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=',', on_bad_lines='skip')
            logging.info(f"Successfully read file using encoding: {encoding} (sep=',' engine='c')")
            return df
        except Exception:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=None, engine='python', on_bad_lines='skip')
                logging.info(f"Successfully read file using encoding: {encoding} (auto-sep engine='python')")
                return df
            except Exception as e:
                logging.warning(f"Failed to read with encoding {encoding}: {str(e)}")

    logging.error(f"All encodings failed for file: {file_path}")
    return None


def calculate_derivatives(signal):
    """计算信号的一阶和二阶导数（中心差分）"""
    if len(signal.shape) == 3:
        signal = signal.reshape(signal.shape[0], signal.shape[1])

    first_deriv = np.zeros_like(signal)
    second_deriv = np.zeros_like(signal)

    first_deriv[:, 1:-1] = (signal[:, 2:] - signal[:, :-2]) / 2
    first_deriv[:, 0] = signal[:, 1] - signal[:, 0]
    first_deriv[:, -1] = signal[:, -1] - signal[:, -2]

    second_deriv[:, 1:-1] = (first_deriv[:, 2:] - first_deriv[:, :-2]) / 2
    second_deriv[:, 0] = first_deriv[:, 1] - first_deriv[:, 0]
    second_deriv[:, -1] = first_deriv[:, -1] - first_deriv[:, -2]

    return first_deriv, second_deriv


def standardize_single_feature(data, scalar_path):
    """标准化单个特征（使用保存的 scaler.pkl）"""
    try:
        scaler = joblib.load(scalar_path)
        data_reshaped = data.reshape(-1, 1)
        return scaler.transform(data_reshaped).flatten()
    except Exception as e:
        logging.error(f"标准化失败: {e}")
        return data


# ================== 趋势图对话框 ==================
class TrendDialog(QDialog):
    """趋势图对话框"""

    def __init__(self, history_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("心功能参数趋势图")
        self.setGeometry(200, 200, 1600, 1000)

        self.history_data = history_data
        self.data_count = len(self.history_data)

        self.base_fig_width = 10
        self.base_fig_height = 6
        self.fig_width = max(self.base_fig_width, min(15, self.base_fig_width + (self.data_count - 10) * 0.4))

        self.fig = Figure(figsize=(self.fig_width, self.base_fig_height), dpi=100)
        self.canvas = FigureCanvas(self.fig)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, stretch=1)

        self.info_label = QLabel("点击图表查看详细数据...")
        layout.addWidget(self.info_label)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.plot_trend()

    def plot_trend(self):
        if not self.history_data:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "暂无预测数据，请先启动监测", ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#666666')
            ax.set_title("心功能参数趋势图（5秒均值）", fontsize=14, pad=20)
            self.canvas.draw()
            return

        times = [item[0] for item in self.history_data]
        co_values = [item[1] for item in self.history_data]
        sv_values = [item[2] for item in self.history_data]

        x_positions = range(len(times))
        time_labels = [t.strftime("%H:%M:%S") for t in times]

        self.fig.clear()

        ax1 = self.fig.add_subplot(211)
        ax1.plot(x_positions, co_values, '-o', linewidth=2, markersize=4,
                 label='心输出量(CO)', color='#1f77b4')
        ax1.set_ylabel('心输出量 (L/min)', fontsize=11, color='#333333')
        ax1.set_title("心功能参数趋势图（5秒均值）", fontsize=14, pad=20, color='#222222')
        ax1.grid(True, linestyle='--', alpha=0.7, color='#dddddd')
        ax1.axhline(y=4, color='#ff7f0e', linestyle='--', alpha=0.6, label='正常下限（4 L/min）')
        ax1.axhline(y=8, color='#ff7f0e', linestyle='--', alpha=0.6, label='正常上限（8 L/min）')
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax1.set_ylim(0, 12)
        ax1.spines['top'].set_visible(False)

        ax2 = self.fig.add_subplot(212)
        ax2.plot(x_positions, sv_values, '-o', linewidth=2, markersize=4,
                 label='每搏输出量(SV)', color='#2ca02c')
        ax2.set_xlabel('时间', fontsize=11, color='#333333')
        ax2.set_ylabel('每搏输出量 (mL)', fontsize=11, color='#333333')
        ax2.grid(True, linestyle='--', alpha=0.7, color='#dddddd')
        ax2.axhline(y=60, color='#ff7f0e', linestyle='--', alpha=0.6, label='正常下限（60 mL）')
        ax2.axhline(y=100, color='#ff7f0e', linestyle='--', alpha=0.6, label='正常上限（100 mL）')
        ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax2.set_ylim(20, 160)
        ax2.spines['top'].set_visible(False)

        if self.data_count <= 15:
            ax2.set_xticks(list(x_positions))
            ax2.set_xticklabels(time_labels, rotation=0, fontsize=9, color='#444444')
        else:
            step = max(1, self.data_count // 15)
            display_pos = list(x_positions)[::step]
            display_labels = time_labels[::step]
            ax2.set_xticks(display_pos)
            ax2.set_xticklabels(display_labels, rotation=45, fontsize=8, color='#444444')

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def on_click(self, event):
        if not self.history_data or event.inaxes is None or event.xdata is None:
            return

        x_idx = int(round(event.xdata))
        if 0 <= x_idx < len(self.history_data):
            time_obj, co_val, sv_val = self.history_data[x_idx]
            time_str = time_obj.strftime("%H:%M:%S")
            self.info_label.setText(
                f"📅 时间: {time_str} | ❤️ 心输出量: <b>{co_val}</b> L/min | 💓 每搏输出量: <b>{sv_val}</b> mL"
            )
            self.info_label.setStyleSheet("font-size: 10pt; color: #2c3e50;")


# ================== 主窗口 ==================
class PulseWaveMonitor(QMainWindow):
    """心输出量监测系统主窗口"""

    # ---------- 方案二：创建卡片组件 ----------
    def make_card(self, title: str):
        """
        返回：(card_frame, body_layout)
        card_frame: 外层卡片QFrame
        body_layout: 你往里面 addWidget/addLayout 的内容布局
        """
        card = QFrame()
        card.setObjectName("Card")

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 16)  # 卡片内边距
        card_layout.setSpacing(12)

        header = QLabel(title)
        header.setObjectName("CardHeader")

        body = QWidget()
        body.setObjectName("CardBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)

        card_layout.addWidget(header)
        card_layout.addWidget(body, stretch=1)

        return card, body_layout

    def apply_theme(self):
        qss = """
        /* 全局背景 */
        QMainWindow, QWidget {
            background: #F6F8FB;
            color: #1F2937;
            font-family: "Microsoft YaHei";
            font-size: 11pt;
        }

        /* 主标题 */
        QLabel#TitleLabel {
            font-size: 18pt;
            font-weight: 800;
            color: #0F172A;
        }

        /* 普通标签 */
        QLabel {
            background: transparent;
            color: #334155;
        }

        /* 输入框 */
        QLineEdit {
            background: #FFFFFF;
            border: 1px solid #D1D5DB;
            border-radius: 10px;
            padding: 8px 10px;
            selection-background-color: #93C5FD;
        }
        QLineEdit:focus {
            border: 1px solid #3B82F6;
        }

        /* 下拉框 */
        QComboBox {
            background: #FFFFFF;
            border: 1px solid #D1D5DB;
            border-radius: 10px;
            padding: 6px 10px;
        }
        QComboBox:focus {
            border: 1px solid #3B82F6;
        }

        /* 单选按钮 */
        QRadioButton {
            spacing: 8px;
            color: #334155;
        }

        /* 按钮：默认风格 */
        QPushButton {
            background: #F3F4F6;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 10px 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #EAECEF;
        }
        QPushButton:pressed {
            background: #E5E7EB;
        }
        QPushButton:disabled {
            background: #F3F4F6;
            color: #9CA3AF;
            border: 1px solid #E5E7EB;
        }

        /* 关键按钮：开始（蓝色主按钮） */
        QPushButton#PrimaryButton {
            background: #2563EB;
            color: #FFFFFF;
            border: none;
        }
        QPushButton#PrimaryButton:hover {
            background: #1D4ED8;
        }
        QPushButton#PrimaryButton:pressed {
            background: #1E40AF;
        }

        /* 危险按钮：重置（淡红） */
        QPushButton#DangerButton {
            background: #FEE2E2;
            color: #991B1B;
            border: 1px solid #FCA5A5;
        }
        QPushButton#DangerButton:hover {
            background: #FECACA;
        }

        /* 状态栏 Label */
        QLabel#StatusLabel {
            background: #EEF2FF;
            border: 1px solid #E0E7FF;
            border-radius: 12px;
            padding: 10px;
            color: #1E3A8A;
            font-weight: 600;
        }

        /* ========== 方案二：卡片样式 ========== */
        QFrame#Card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
        }

        QLabel#CardHeader {
            font-size: 13pt;
            font-weight: 800;
            color: #0F172A;
            padding-bottom: 8px;
            border-bottom: 1px solid #EEF2F7;
        }

        QWidget#CardBody {
            background: transparent;
        }

        /* 波形容器（可选） */
        QFrame#WaveCard {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
        }
        QLabel#WaveHeader {
            font-size: 12pt;
            font-weight: 800;
            color: #0F172A;
            padding-bottom: 8px;
            border-bottom: 1px solid #EEF2F7;
        }
        """
        self.setStyleSheet(qss)

    def __init__(self):
        super().__init__()

        # 输入相关
        self.input_mode = "file"
        self.serial_port = ""
        self.pulse_data = None

        # 数据参数
        self.sampling_rate = 125
        self.window_size = 5
        self.scroll_step = 1
        self.current_window_start = 0.0
        self.total_duration = 0.0

        # ✅ 固定时间基准：从 00:00:00 开始
        self.base_time = datetime(2000, 1, 1, 0, 0, 0)

        # 心率与历史
        self.calculated_hr = 0.0
        self.input_visible = False
        self.peaks_history = []

        # 预测历史：1秒推理 & 5秒均值
        self.one_second_predictions = []  # [(pred_time, co, sv)]
        self.prediction_history = []      # [(time, co_avg, sv_avg)]

        self.five_second_update_interval = 5.0
        self.last_five_second_update = 0.0  # 用 float 秒

        # ONNX模型相关
        self.onnx_session = None
        self.signal_scaler = None
        self.scalar_paths = {}
        self.model_loaded = False
        self.init_model_paths()

        self.init_ui()
        self.init_data()

    # ---------- 额外的辅助方法：ONNX 喂数据用 ----------
    def _to_float32(self, x):
        return np.asarray(x, dtype=np.float32)

    def _reshape_like_onnx(self, arr, onnx_input):
        """尽量把 arr reshape 成 ONNX 输入期望的形状（兼容 None 维度）。"""
        a = self._to_float32(arr)
        shp = onnx_input.shape
        if shp is None:
            return a

        if a.ndim == 1 and a.size == 1:
            if len(shp) == 2:
                return a.reshape(1, 1)
            if len(shp) == 1:
                return a.reshape(1,)
            return a.reshape(1, 1)

        if a.ndim == 3 and len(shp) == 3:
            return a
        if a.ndim == 3 and len(shp) == 2:
            return a.reshape(a.shape[0], -1)
        if a.ndim == 2:
            return a
        return a

    # ---------- 模型路径 ----------
    def init_model_paths(self):
        """初始化模型路径配置 - 按你实际路径修改"""
        model_root_dir = r"D:\心输出量项目\CNAP脉氧处理数据\脉氧+CNAP\tuili-嵌入模型3-1226 (2)\tuili-嵌入模型3-1226"

        self.model_config = {
            'model_path': os.path.join(model_root_dir, 'parallel_resnet_se_lstm_model.onnx'),
            'signal_scaler_path': os.path.join(model_root_dir, 'signal_scaler.pkl'),
            'scalar_dir': model_root_dir
        }

        self.scalar_paths = {
            'age': os.path.join(self.model_config['scalar_dir'], 'age_scaler.pkl'),
            'weight': os.path.join(self.model_config['scalar_dir'], 'weight_scaler.pkl'),
            'height': os.path.join(self.model_config['scalar_dir'], 'height_scaler.pkl'),
            'bsa': os.path.join(self.model_config['scalar_dir'], 'bsa_scaler.pkl'),
            'bmi': os.path.join(self.model_config['scalar_dir'], 'bmi_scaler.pkl'),
            'hr': os.path.join(self.model_config['scalar_dir'], 'hr_scaler.pkl')
        }

    def load_inference_model(self):
        """加载ONNX推理模型和标量"""
        try:
            if not os.path.exists(self.model_config['model_path']):
                raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_config['model_path']}")

            ort_options = ort.SessionOptions()
            ort_options.intra_op_num_threads = 4
            ort_options.inter_op_num_threads = 4
            ort_options.enable_cpu_mem_arena = True

            logging.info(f"Loading ONNX model from: {self.model_config['model_path']}")
            self.onnx_session = ort.InferenceSession(
                self.model_config['model_path'],
                sess_options=ort_options,
                providers=['CPUExecutionProvider']
            )

            input_names = [inp.name for inp in self.onnx_session.get_inputs()]
            output_names = [out.name for out in self.onnx_session.get_outputs()]
            logging.info(f"ONNX模型输入: {input_names}")
            logging.info(f"ONNX模型输出: {output_names}")

            if not os.path.exists(self.model_config['signal_scaler_path']):
                raise FileNotFoundError(f"信号标量文件不存在: {self.model_config['signal_scaler_path']}")

            logging.info(f"Loading signal scaler from: {self.model_config['signal_scaler_path']}")
            self.signal_scaler = joblib.load(self.model_config['signal_scaler_path'])

            missing_scalers = []
            for key, path in self.scalar_paths.items():
                if not os.path.exists(path):
                    missing_scalers.append(f"{key}: {path}")
            if missing_scalers:
                raise FileNotFoundError(f"缺失标量文件:\n{chr(10).join(missing_scalers)}")

            self.model_loaded = True
            self.model_status.setText("模型状态：已加载ResNet-LSTM-SE ONNX模型")
            self.model_status.setStyleSheet("color: green;")
            QMessageBox.information(self, "模型加载成功", "ResNet-LSTM-SE ONNX推理模型加载完成！")

            if self.pulse_data is not None:
                self.start_btn.setEnabled(True)

            return True

        except Exception as e:
            error_msg = f"ONNX模型加载失败: {str(e)}"
            logging.error(error_msg, exc_info=True)
            self.model_loaded = False
            self.onnx_session = None
            self.model_status.setText("模型状态：加载失败")
            self.model_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "模型加载失败", error_msg)
            return False

    # ---------- UI ----------
    def init_ui(self):
        self.setWindowTitle("心输出量监测系统（ONNX版）")
        self.setGeometry(100, 100, 1800, 1200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setSpacing(14)
        self.main_layout.setContentsMargins(18, 16, 18, 16)

        # 顶部控制区
        self.top_bar = QVBoxLayout()
        self.top_bar.setSpacing(10)

        self.input_control_layout = QHBoxLayout()
        self.input_btn = QPushButton("输入脉搏波波形")
        self.input_btn.clicked.connect(self.toggle_input_interface)
        self.input_control_layout.addWidget(self.input_btn, alignment=Qt.AlignLeft)

        self.load_model_btn = QPushButton("加载ONNX推理模型")
        self.load_model_btn.clicked.connect(self.load_inference_model)
        self.input_control_layout.addWidget(self.load_model_btn, alignment=Qt.AlignRight)

        self.input_control_layout.addStretch(1)
        self.top_bar.addLayout(self.input_control_layout)

        self.input_container = QWidget()
        self.input_container_layout = QVBoxLayout(self.input_container)
        self.input_container_layout.setContentsMargins(0, 0, 0, 0)
        self.input_container_layout.setSpacing(8)

        self.mode_layout = QHBoxLayout()
        self.mode_label = QLabel("输入模式:")
        self.mode_group = QButtonGroup()
        self.file_mode_btn = QRadioButton("文件输入")
        self.serial_mode_btn = QRadioButton("串口输入")
        self.file_mode_btn.setChecked(True)
        self.mode_group.addButton(self.file_mode_btn, 1)
        self.mode_group.addButton(self.serial_mode_btn, 2)
        self.file_mode_btn.toggled.connect(self.toggle_input_mode)

        self.mode_layout.addWidget(self.mode_label)
        self.mode_layout.addWidget(self.file_mode_btn)
        self.mode_layout.addWidget(self.serial_mode_btn)
        self.mode_layout.addStretch(1)
        self.mode_layout.setContentsMargins(0, 0, 0, 0)
        self.input_container_layout.addLayout(self.mode_layout)

        self.path_layout = QHBoxLayout()

        self.path_label = QLabel("数据文件路径:")
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("输入Excel/CSV文件路径或点击浏览选择")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_file)

        self.port_label = QLabel("串口选择:")
        self.port_combo = QComboBox()
        for i in range(1, 11):
            self.port_combo.addItem(f"COM{i}")

        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self.load_data)

        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_input, stretch=1)
        self.path_layout.addWidget(self.browse_btn)
        self.path_layout.addWidget(self.load_btn)
        self.path_layout.setContentsMargins(0, 0, 0, 0)
        self.input_container_layout.addLayout(self.path_layout)

        self.top_bar.addWidget(self.input_container)

        self.title_layout = QHBoxLayout()
        self.title_label = QLabel("心输出量监测系统（ResNet-LSTM-SE ONNX模型）")
        self.title_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        self.top_bar.addLayout(self.title_layout)

        self.id_layout = QHBoxLayout()
        self.id_label = QLabel("患者ID:")
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("输入患者ID")
        self.id_input.setFixedWidth(220)
        self.id_layout.addStretch(1)
        self.id_layout.addWidget(self.id_label)
        self.id_layout.addWidget(self.id_input)
        self.id_layout.addStretch(1)
        self.top_bar.addLayout(self.id_layout)

        self.main_layout.addLayout(self.top_bar)

        # ========== 中间三列：方案二卡片 ==========
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(14)

        # ---- 左卡：生理参数输入 ----
        self.param_card, param_body = self.make_card("生理参数输入")

        self.weight_layout = QHBoxLayout()
        self.weight_label = QLabel("体重 (kg):")
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("10-200")
        self.weight_layout.addWidget(self.weight_label)
        self.weight_layout.addWidget(self.weight_input)
        param_body.addLayout(self.weight_layout)

        self.age_layout = QHBoxLayout()
        self.age_label = QLabel("年龄 (岁):")
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("0-120")
        self.age_layout.addWidget(self.age_label)
        self.age_layout.addWidget(self.age_input)
        param_body.addLayout(self.age_layout)

        self.gender_layout = QHBoxLayout()
        self.gender_label = QLabel("性别:")
        self.gender_group = QButtonGroup()
        self.male_btn = QRadioButton("男")
        self.female_btn = QRadioButton("女")
        self.other_btn = QRadioButton("其他")
        self.male_btn.setChecked(True)
        self.gender_group.addButton(self.male_btn, 1)
        self.gender_group.addButton(self.female_btn, 2)
        self.gender_group.addButton(self.other_btn, 3)
        self.gender_layout.addWidget(self.gender_label)
        self.gender_layout.addWidget(self.male_btn)
        self.gender_layout.addWidget(self.female_btn)
        self.gender_layout.addWidget(self.other_btn)
        param_body.addLayout(self.gender_layout)

        self.height_layout = QHBoxLayout()
        self.height_label = QLabel("身高 (cm):")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("50-250")
        self.height_layout.addWidget(self.height_label)
        self.height_layout.addWidget(self.height_input)
        param_body.addLayout(self.height_layout)

        self.bsa_bmi_layout = QHBoxLayout()
        self.bsa_label = QLabel("体表面积 (m²):")
        self.bsa_value = QLabel("--")
        self.bmi_label = QLabel("BMI:")
        self.bmi_value = QLabel("--")
        self.bsa_bmi_layout.addWidget(self.bsa_label)
        self.bsa_bmi_layout.addWidget(self.bsa_value)
        self.bsa_bmi_layout.addStretch(1)
        self.bsa_bmi_layout.addWidget(self.bmi_label)
        self.bsa_bmi_layout.addWidget(self.bmi_value)
        param_body.addLayout(self.bsa_bmi_layout)

        self.hr_layout = QHBoxLayout()
        self.hr_label = QLabel("心率 (次/分):")
        self.hr_value = QLabel("--")
        self.hr_value.setStyleSheet("font-size: 12pt; font-weight: 700;")
        self.hr_layout.addWidget(self.hr_label)
        self.hr_layout.addWidget(self.hr_value)
        param_body.addLayout(self.hr_layout)

        self.normalize_label = QLabel("参数标准化状态: 未开始")
        self.normalize_label.setStyleSheet("color: gray;")
        param_body.addWidget(self.normalize_label)

        mid_layout.addWidget(self.param_card, stretch=1)

        # ---- 中卡：心功能预测结果 ----
        self.result_card, result_body = self.make_card("心功能预测结果")

        self.co_layout = QHBoxLayout()
        self.co_label = QLabel("心输出量 (L/min):")
        self.co_value = QLabel("--")
        self.co_value.setStyleSheet("font-size: 24pt; font-weight: 900;")
        self.co_layout.addWidget(self.co_label)
        self.co_layout.addStretch(1)
        self.co_layout.addWidget(self.co_value)
        result_body.addLayout(self.co_layout)

        self.co_range = QLabel("(正常范围: 4-8 L/min)")
        self.co_range.setStyleSheet("color: gray;")
        result_body.addWidget(self.co_range)

        self.sv_layout = QHBoxLayout()
        self.sv_label = QLabel("每搏输出量 (mL):")
        self.sv_value = QLabel("--")
        self.sv_value.setStyleSheet("font-size: 24pt; font-weight: 900;")
        self.sv_layout.addWidget(self.sv_label)
        self.sv_layout.addStretch(1)
        self.sv_layout.addWidget(self.sv_value)
        result_body.addLayout(self.sv_layout)

        self.sv_range = QLabel("(正常范围: 60-100 mL)")
        self.sv_range.setStyleSheet("color: gray;")
        result_body.addWidget(self.sv_range)

        self.update_status_label = QLabel("更新状态: 等待数据...")
        self.update_status_label.setStyleSheet("color: #666; font-size: 10pt;")
        result_body.addWidget(self.update_status_label)

        self.model_status = QLabel("模型状态: 未加载模型")
        self.model_status.setStyleSheet("color: #2563EB; font-weight: 700;")
        result_body.addWidget(self.model_status)

        result_body.addStretch(1)

        mid_layout.addWidget(self.result_card, stretch=1)

        # ---- 右卡：功能控制 ----
        self.control_card, control_body = self.make_card("功能控制")

        self.start_btn = QPushButton("开始预测")
        self.start_btn.clicked.connect(self.start_scrolling)

        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_scrolling)

        self.reset_btn = QPushButton("重置")
        self.reset_btn.clicked.connect(self.reset_scrolling)

        self.save_btn = QPushButton("保存数据")
        self.save_btn.clicked.connect(self.save_data)

        self.trend_btn = QPushButton("查看趋势")
        self.trend_btn.clicked.connect(self.show_trend)
        self.trend_btn.setEnabled(False)

        self.pause_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

        # 右侧按钮布局：仍保留居中效果（你也可以去掉 stretch 让按钮顶上）
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(14)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(self.pause_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(self.reset_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(self.save_btn, alignment=Qt.AlignCenter)
        btn_layout.addWidget(self.trend_btn, alignment=Qt.AlignCenter)
        btn_layout.addStretch(1)

        control_body.addLayout(btn_layout)
        mid_layout.addWidget(self.control_card, stretch=1)

        self.main_layout.addLayout(mid_layout)

        # ========== 波形卡片（也用方案二风格，统一美观） ==========
        wave_card = QFrame()
        wave_card.setObjectName("WaveCard")
        wave_layout = QVBoxLayout(wave_card)
        wave_layout.setContentsMargins(16, 14, 16, 16)
        wave_layout.setSpacing(12)

        wave_header = QLabel(f"脉搏波波形（采样率：{self.sampling_rate}Hz，窗口大小：{self.window_size}秒）")
        wave_header.setObjectName("WaveHeader")
        wave_layout.addWidget(wave_header)

        self.fig = Figure(figsize=(10, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("时间（时:分:秒）")
        self.ax.set_ylabel("振幅")
        self.canvas = FigureCanvas(self.fig)

        wave_layout.addWidget(self.canvas)
        self.main_layout.addWidget(wave_card, stretch=1)

        self.status_label = QLabel("状态：请先点击'加载ONNX推理模型'，然后加载脉搏波数据")
        self.main_layout.addWidget(self.status_label)

        # theme & objectName
        self.title_label.setObjectName("TitleLabel")
        self.status_label.setObjectName("StatusLabel")
        self.start_btn.setObjectName("PrimaryButton")
        self.reset_btn.setObjectName("DangerButton")

        self.hide_input_interface()
        self.apply_theme()

    # ---------- 定时器 ----------
    def init_data(self):
        self.is_scrolling = False

        self.scroll_timer = QTimer(self)
        self.scroll_timer.setInterval(self.scroll_step * 1000)
        self.scroll_timer.timeout.connect(self.scroll_window)

        self.predict_timer = QTimer(self)
        self.predict_timer.setInterval(1000)
        self.predict_timer.timeout.connect(self.update_one_second_prediction)

    # ---------- 输入界面显示隐藏 ----------
    def toggle_input_interface(self):
        if self.input_visible:
            self.hide_input_interface()
        else:
            self.show_input_interface()

    def show_input_interface(self):
        self.input_container.setVisible(True)
        self.input_visible = True
        self.input_btn.setText("隐藏输入选项")
        self.status_label.setText("状态：请选择输入模式并加载数据")

    def hide_input_interface(self):
        self.input_container.setVisible(False)
        self.input_visible = False
        self.input_btn.setText("输入脉搏波波形")

    # ---------- 切换文件/串口 ----------
    def toggle_input_mode(self):
        while self.path_layout.count():
            item = self.path_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.hide()
                self.path_layout.removeWidget(widget)

        if self.file_mode_btn.isChecked():
            self.input_mode = "file"
            self.path_layout.addWidget(self.path_label)
            self.path_layout.addWidget(self.path_input, stretch=1)
            self.path_layout.addWidget(self.browse_btn)
            self.path_layout.addWidget(self.load_btn)
            self.path_label.show()
            self.path_input.show()
            self.browse_btn.show()
            self.load_btn.setText("加载数据")
        else:
            self.input_mode = "serial"
            self.path_layout.addWidget(self.port_label)
            self.path_layout.addWidget(self.port_combo, stretch=1)
            self.path_layout.addWidget(self.load_btn)
            self.port_label.show()
            self.port_combo.show()
            self.load_btn.setText("连接串口")

        self.load_btn.show()

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择脉搏波数据", "", "数据文件 (*.xlsx *.xls *.csv)")
        if file_path:
            self.path_input.setText(file_path)

    def load_data(self):
        if self.input_mode == "file":
            self.load_from_file()
        else:
            self.load_from_serial()
        self.hide_input_interface()

    # ---------- 低通滤波 ----------
    def apply_filter(self, data):
        def butter_lowpass(cutoff, fs, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(_data, cutoff, fs, order=4):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = filtfilt(b, a, _data)
            return y

        return butter_lowpass_filter(data, 10, self.sampling_rate, order=4)

    # ---------- 文件读取 ----------
    def load_from_file(self):
        try:
            file_path = self.path_input.text()
            if not file_path:
                QMessageBox.warning(self, "路径为空", "请输入或选择文件路径")
                return

            if file_path.endswith('.csv'):
                df = read_csv_with_encoding(file_path)
                if df is None:
                    raise ValueError("CSV读取失败（编码/分隔符问题）")
            else:
                try:
                    df = pd.read_excel(file_path, header=0, converters={'Time': float})
                except Exception as e1:
                    logging.warning(f"方法1读取失败: {e1}")
                    try:
                        df = pd.read_excel(file_path, header=0, dtype=object)
                        if 'Time' in df.columns:
                            df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
                    except Exception as e2:
                        logging.warning(f"方法2读取失败: {e2}")
                        df = pd.read_excel(file_path, header=None, skiprows=1)
                        if df.shape[1] >= 2:
                            df.columns = ['Time', 'PLETH Signal']
                        else:
                            df.columns = ['Time']
                        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')

            logging.info(f"DataFrame columns: {df.columns.tolist()}")
            logging.info(f"DataFrame head:\n{df.head()}")

            if 'signal_1' in df.columns:
                signal_cols = [f'signal_{i}' for i in range(1, 126)]
                missing = [c for c in signal_cols if c not in df.columns]
                if missing:
                    raise ValueError(f"缺少列: {missing[:5]} ...")
                self.pulse_data = df[signal_cols].values.flatten()
            else:
                signal_cols = [col for col in df.columns if ('signal' in col.lower() or 'pleth' in col.lower())]
                if signal_cols:
                    signal_col = signal_cols[0]
                    logging.info(f"使用信号列: {signal_col}")
                    self.pulse_data = df[signal_col].values.astype(float)
                else:
                    if len(df.columns) >= 2:
                        self.pulse_data = df.iloc[:, 1].values.astype(float)
                    else:
                        raise ValueError("数据文件中未找到有效的信号列")

            if self.pulse_data is None or len(self.pulse_data) == 0:
                raise ValueError("加载的数据为空")

            self.pulse_data = self.apply_filter(self.pulse_data)
            self.process_loaded_data()

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"文件读取错误：{str(e)}")
            logging.error(f"文件加载错误: {e}", exc_info=True)
            self.pulse_data = None

    # ---------- 串口模拟 ----------
    def load_from_serial(self):
        self.serial_port = self.port_combo.currentText()
        try:
            duration = 60
            t = np.linspace(0, duration, int(duration * self.sampling_rate), endpoint=False)
            heart_rate = 75
            hr_freq = heart_rate / 60

            main_wave = (
                np.sin(2 * np.pi * hr_freq * t) * 30 +
                np.sin(4 * np.pi * hr_freq * t) * 10 +
                np.exp(-2 * np.pi * hr_freq * (t % 1) * 3) * 20
            )

            dicrotic_notch = 0.35
            dicrotic_wave = (
                np.sin(2 * np.pi * hr_freq * (t - dicrotic_notch)) * 12 +
                np.exp(-2 * np.pi * hr_freq * ((t - dicrotic_notch) % 1) * 4) * 8
            )

            baseline_drift = np.sin(2 * np.pi * 0.03 * t) * 5
            noise = np.random.normal(0, 1.5, len(t))
            raw_pulse = main_wave + dicrotic_wave + baseline_drift + noise + 80

            self.pulse_data = self.apply_filter(raw_pulse)
            self.process_loaded_data()

        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"错误：{str(e)}")
            self.pulse_data = None

    # ---------- 加载后初始化 ----------
    def process_loaded_data(self):
        if self.pulse_data is None or len(self.pulse_data) == 0:
            return

        self.base_time = datetime(2000, 1, 1, 0, 0, 0)

        self.total_duration = len(self.pulse_data) / self.sampling_rate
        self.current_window_start = 0.0

        self.peaks_history = []
        self.prediction_history = []
        self.one_second_predictions = []
        self.last_five_second_update = 0.0

        self.calculate_heart_rate()
        self.hr_value.setText(f"{self.calculated_hr:.1f}")

        self.trend_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)

        status_text = f"状态：已加载数据（总时长：{self.total_duration:.1f}秒，采样率：{self.sampling_rate}Hz，起始时间：00:00:00）"
        self.status_label.setText(status_text)

        self.plot_current_window()

        if not self.model_loaded:
            self.model_status.setText("模型状态：未加载ONNX模型，无法进行预测")
            self.model_status.setStyleSheet("color: red; font-weight:700;")
            self.start_btn.setEnabled(False)
        else:
            self.model_status.setText("模型状态：数据已加载，可开始预测")
            self.model_status.setStyleSheet("color: green; font-weight:700;")
            self.start_btn.setEnabled(True)

    # ---------- 谷值检测 ----------
    def detect_valley_points(self, data, sampling_rate=125):
        if len(data) < sampling_rate:
            return []
        window_size = int(sampling_rate * 0.2)
        if window_size % 2 == 0:
            window_size += 1
        baseline = np.convolve(data, np.ones(window_size) / window_size, mode='same')
        data_filtered = data - baseline
        valleys, _ = find_peaks(
            -data_filtered,
            distance=int(sampling_rate * 60 / 200),
            prominence=0.5
        )
        return valleys

    # ---------- 谷值对齐切片（返回片段+起点索引） ----------
    def get_valley_aligned_data(self, start_time, return_start_idx=False):
        fs = self.sampling_rate

        search_start = max(0, int((start_time - 2) * fs))
        search_end = min(len(self.pulse_data), int((start_time + 2) * fs))

        if search_end - search_start < fs:
            return (None, None) if return_start_idx else None

        search_data = self.pulse_data[search_start:search_end]
        valleys = self.detect_valley_points(search_data, fs)

        def fallback():
            start_idx = int(start_time * fs)
            end_idx = start_idx + fs
            if end_idx > len(self.pulse_data) or start_idx < 0:
                return (None, None) if return_start_idx else None
            seg = self.pulse_data[start_idx:end_idx]
            return (seg, start_idx) if return_start_idx else seg

        if len(valleys) == 0:
            return fallback()

        target_idx = int(start_time * fs) - search_start
        closest_valley = min(valleys, key=lambda x: abs(x - target_idx))
        valley_global_idx = search_start + closest_valley

        end_idx = valley_global_idx + fs
        if end_idx > len(self.pulse_data):
            return fallback()

        seg = self.pulse_data[valley_global_idx:end_idx]
        return (seg, valley_global_idx) if return_start_idx else seg

    # ---------- 心率 ----------
    def calculate_heart_rate(self):
        if self.pulse_data is None:
            self.calculated_hr = 0.0
            return

        window_length = int(15 * self.sampling_rate)
        current_pos = int(self.current_window_start * self.sampling_rate)
        start_idx = max(0, current_pos - window_length // 2)
        end_idx = min(len(self.pulse_data), current_pos + window_length // 2)
        data_window = self.pulse_data[start_idx:end_idx]

        if len(data_window) < self.sampling_rate * 2:
            self.calculated_hr = 0.0
            return

        window_size = int(self.sampling_rate * 0.2)
        if window_size % 2 == 0:
            window_size += 1
        baseline = np.convolve(data_window, np.ones(window_size) / window_size, mode='same')
        data_filtered = data_window - baseline

        data_mean = np.mean(data_filtered)
        data_std = np.std(data_filtered)
        threshold = data_mean + 1.2 * data_std

        peaks = []
        min_peak_interval = int(self.sampling_rate * 60 / 200)
        max_peak_interval = int(self.sampling_rate * 60 / 30)

        for i in range(1, len(data_filtered) - 1):
            if (data_filtered[i] > data_filtered[i - 1] and
                    data_filtered[i] >= data_filtered[i + 1] and
                    data_filtered[i] > threshold):
                if not peaks or (i - peaks[-1]) > min_peak_interval:
                    peaks.append(i)

        valid_peaks = []
        if peaks:
            valid_peaks.append(peaks[0])
            for i in range(1, len(peaks)):
                interval = peaks[i] - peaks[i - 1]
                if min_peak_interval < interval < max_peak_interval:
                    valid_peaks.append(peaks[i])
            peaks = valid_peaks

        self.peaks_history.extend([p + start_idx for p in peaks])
        if len(self.peaks_history) > 100:
            self.peaks_history = self.peaks_history[-100:]

        if len(self.peaks_history) >= 2:
            intervals = np.diff(self.peaks_history) / self.sampling_rate
            interval_mean = np.mean(intervals)
            interval_std = np.std(intervals)
            valid_intervals = intervals[np.abs(intervals - interval_mean) < 3 * interval_std]

            if len(valid_intervals) >= 2:
                avg_interval = np.mean(valid_intervals)
                self.calculated_hr = 60.0 / avg_interval if avg_interval > 0 else 0.0
            else:
                self.calculated_hr = 60.0 / interval_mean if interval_mean > 0 else 0.0

            self.calculated_hr = float(np.clip(self.calculated_hr, 30, 200))
        else:
            self.calculated_hr = 0.0

    # ---------- 波形绘制 ----------
    def plot_current_window(self):
        if self.pulse_data is None:
            return

        window_end_relative = self.current_window_start + self.window_size
        start_idx = int(self.current_window_start * self.sampling_rate)
        end_idx = int(window_end_relative * self.sampling_rate)
        end_idx = min(end_idx, len(self.pulse_data))
        window_pulse = self.pulse_data[start_idx:end_idx]

        if len(window_pulse) == 0:
            self.status_label.setText("状态：当前窗口无数据，请检查数据长度")
            return

        window_time_relative = np.linspace(
            self.current_window_start,
            self.current_window_start + (end_idx - start_idx) / self.sampling_rate,
            len(window_pulse)
        )

        self.ax.clear()
        self.ax.plot(window_time_relative, window_pulse, 'r-', linewidth=1.5)

        if len(window_pulse) >= self.sampling_rate:
            valleys = self.detect_valley_points(window_pulse, self.sampling_rate)
            if len(valleys) > 0:
                valley_times = window_time_relative[valleys]
                valley_values = window_pulse[valleys]
                self.ax.scatter(valley_times, valley_values, c='blue', s=50,
                                marker='v', label='舒张末期（谷值）')
                self.ax.legend()

        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_xlabel("时间（时:分:秒）")
        self.ax.set_ylabel("振幅")
        self.ax.set_xlim(self.current_window_start, window_end_relative)

        xticks = np.arange(np.ceil(self.current_window_start), window_end_relative, 1)
        xtick_labels = [(self.base_time + timedelta(seconds=float(t))).strftime("%H:%M:%S") for t in xticks]
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xtick_labels, rotation=0)

        y_min, y_max = float(np.min(window_pulse)), float(np.max(window_pulse))
        y_margin = (y_max - y_min) * 0.15 if y_max != y_min else 10
        self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

        self.canvas.draw()

    # ---------- 滚动 ----------
    def scroll_window(self):
        if self.pulse_data is None:
            return

        next_start = self.current_window_start + float(self.scroll_step)
        next_end = next_start + float(self.window_size)

        if next_start >= self.total_duration:
            self.status_label.setText(f"状态：已显示全部数据（总时长：{self.total_duration:.1f}秒）")
            self.pause_scrolling()
            return

        if next_end > self.total_duration:
            next_start = max(0.0, self.total_duration - float(self.window_size))
            next_end = next_start + float(self.window_size)

        self.current_window_start = float(next_start)
        self.plot_current_window()

        self.calculate_heart_rate()
        self.hr_value.setText(f"{self.calculated_hr:.1f}")

        start_actual = self.base_time + timedelta(seconds=float(self.current_window_start))
        end_actual = self.base_time + timedelta(seconds=float(next_end))
        self.status_label.setText(
            f"状态：预测中 当前窗口：{start_actual.strftime('%H:%M:%S')} - {end_actual.strftime('%H:%M:%S')}"
        )

    # ---------- 参数检查 ----------
    def check_params(self):
        try:
            if not (self.weight_input.text() and self.age_input.text() and self.height_input.text()):
                return False
            weight = float(self.weight_input.text())
            age = int(self.age_input.text())
            height = float(self.height_input.text())
            return (10 <= weight <= 200 and 0 <= age <= 120 and 50 <= height <= 250)
        except Exception:
            return False

    # ---------- 计算 BSA/BMI ----------
    def calculate_bsa_bmi(self):
        try:
            weight = float(self.weight_input.text())
            height_m = float(self.height_input.text()) / 100.0
            height_cm = height_m * 100.0

            bmi = weight / (height_m ** 2)
            self.bmi_value.setText(f"{bmi:.2f}")

            bsa = 0.007184 * (weight ** 0.425) * (height_cm ** 0.725)
            self.bsa_value.setText(f"{bsa:.3f}")

            return bsa, bmi
        except Exception:
            self.bsa_value.setText("--")
            self.bmi_value.setText("--")
            return 0.0, 0.0

    # ---------- 准备模型输入 ----------
    def prepare_model_inputs(self, pulse_data_1s=None):
        try:
            fs = self.sampling_rate

            if pulse_data_1s is None:
                start_idx = int(self.current_window_start * fs)
                end_idx = min(start_idx + fs, len(self.pulse_data))
                window_pulse = self.pulse_data[start_idx:end_idx]
            else:
                window_pulse = pulse_data_1s[:fs]

            if len(window_pulse) < fs:
                window_pulse = np.pad(window_pulse, (0, fs - len(window_pulse)), 'constant')
            else:
                window_pulse = window_pulse[:fs]

            raw_signal = window_pulse.reshape(1, -1)
            first_deriv, second_deriv = calculate_derivatives(raw_signal)
            X = np.stack([raw_signal, first_deriv, second_deriv], axis=-1)  # (1,125,3)

            weight = float(self.weight_input.text())
            age = int(self.age_input.text())
            height = float(self.height_input.text())
            gender = 1 if self.male_btn.isChecked() else 0 if self.female_btn.isChecked() else 2
            hr = float(self.calculated_hr)

            bsa, bmi = self.calculate_bsa_bmi()

            X_reshaped = X.reshape(X.shape[0], -1)
            X_std = self.signal_scaler.transform(X_reshaped).reshape(X.shape)

            age_std = standardize_single_feature(np.array([age]), self.scalar_paths['age'])
            weight_std = standardize_single_feature(np.array([weight]), self.scalar_paths['weight'])
            height_std = standardize_single_feature(np.array([height]), self.scalar_paths['height'])
            bsa_std = standardize_single_feature(np.array([bsa]), self.scalar_paths['bsa'])
            bmi_std = standardize_single_feature(np.array([bmi]), self.scalar_paths['bmi'])
            hr_std = standardize_single_feature(np.array([hr]), self.scalar_paths['hr'])

            X_std = X_std.astype(np.float32)
            age_std = age_std.astype(np.float32)
            gender_arr = np.array([gender], dtype=np.float32)
            weight_std = weight_std.astype(np.float32)
            height_std = height_std.astype(np.float32)
            bsa_std = bsa_std.astype(np.float32)
            bmi_std = bmi_std.astype(np.float32)
            hr_std = hr_std.astype(np.float32)

            return (X_std, age_std, gender_arr, weight_std, height_std, bsa_std, bmi_std, hr_std)

        except Exception as e:
            logging.error(f"准备ONNX模型输入失败: {e}", exc_info=True)
            return None

    # ---------- 推理 ----------
    def predict_co_sv(self, pulse_data_1s=None):
        if not self.model_loaded or self.onnx_session is None:
            return 0.0, 0.0

        try:
            inputs = self.prepare_model_inputs(pulse_data_1s)
            if inputs is None:
                return 0.0, 0.0

            X_std, age_std, gender_arr, weight_std, height_std, bsa_std, bmi_std, hr_std = inputs

            onnx_inputs = self.onnx_session.get_inputs()
            input_feed = {}

            if len(onnx_inputs) == 1:
                combined_input = np.concatenate([
                    X_std.reshape(1, -1),
                    age_std.reshape(1, -1),
                    gender_arr.reshape(1, -1),
                    weight_std.reshape(1, -1),
                    height_std.reshape(1, -1),
                    bsa_std.reshape(1, -1),
                    bmi_std.reshape(1, -1),
                    hr_std.reshape(1, -1)
                ], axis=1).astype(np.float32)
                input_feed[onnx_inputs[0].name] = self._reshape_like_onnx(combined_input, onnx_inputs[0])
            else:
                for inp in onnx_inputs:
                    n = inp.name.lower()
                    if any(k in n for k in ["signal", "pleth", "ppg", "wave", "x"]):
                        input_feed[inp.name] = self._reshape_like_onnx(X_std, inp)
                    elif "age" in n:
                        input_feed[inp.name] = self._reshape_like_onnx(age_std, inp)
                    elif any(k in n for k in ["gender", "sex"]):
                        input_feed[inp.name] = self._reshape_like_onnx(gender_arr, inp)
                    elif "weight" in n:
                        input_feed[inp.name] = self._reshape_like_onnx(weight_std, inp)
                    elif "height" in n:
                        input_feed[inp.name] = self._reshape_like_onnx(height_std, inp)
                    elif "bsa" in n:
                        input_feed[inp.name] = self._reshape_like_onnx(bsa_std, inp)
                    elif "bmi" in n:
                        input_feed[inp.name] = self._reshape_like_onnx(bmi_std, inp)
                    elif any(k in n for k in ["hr", "heartrate", "heart_rate"]):
                        input_feed[inp.name] = self._reshape_like_onnx(hr_std, inp)
                    else:
                        logging.warning(f"未识别的ONNX输入: {inp.name}，未喂入（可能导致推理失败）")

            outputs = self.onnx_session.run(None, input_feed)
            pred_sv = float(outputs[0].flatten()[0])
            pred_co = float((pred_sv * float(self.calculated_hr)) / 1000.0)

            pred_sv = float(np.clip(pred_sv, 30, 150))
            pred_co = float(np.clip(pred_co, 2, 10))
            return round(pred_co, 2), round(pred_sv, 1)

        except Exception as e:
            logging.error(f"ONNX模型预测失败: {e}", exc_info=True)
            return 0.0, 0.0

    # ---------- 开始/暂停/重置 ----------
    def start_scrolling(self):
        if self.pulse_data is None:
            QMessageBox.warning(self, "无数据", "请先加载数据！")
            return
        if not self.model_loaded:
            QMessageBox.critical(self, "模型未加载", "ONNX推理模型未加载成功，无法进行预测！")
            return
        if not self.check_params():
            QMessageBox.warning(self, "参数错误", "请补全并正确输入生理参数！")
            return

        self.calculate_bsa_bmi()

        self.current_window_start = 0.0
        self.prediction_history = []
        self.one_second_predictions = []
        self.last_five_second_update = 0.0

        self.is_scrolling = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)

        self.scroll_timer.start()
        self.predict_timer.start()

        self.normalize_label.setText("参数标准化状态: 已完成")
        self.model_status.setText("模型状态：ONNX模型运行中（每1秒推理一次，5秒均值更新）")
        self.model_status.setStyleSheet("color: green; font-weight:700;")

        self.update_status_label.setText("更新状态: 1秒级推理中...")

        start_actual = self.base_time + timedelta(seconds=float(self.current_window_start))
        end_actual = self.base_time + timedelta(seconds=float(self.window_size))
        self.status_label.setText(
            f"状态：预测中 当前窗口：{start_actual.strftime('%H:%M:%S')} - {end_actual.strftime('%H:%M:%S')}"
        )

        self.update_one_second_prediction()

    def pause_scrolling(self):
        self.is_scrolling = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.scroll_timer.stop()
        self.predict_timer.stop()

        start_actual = self.base_time + timedelta(seconds=float(self.current_window_start))
        end_actual = self.base_time + timedelta(seconds=float(self.current_window_start + self.window_size))
        self.status_label.setText(
            f"状态：已暂停（当前窗口：{start_actual.strftime('%H:%M:%S')} - {end_actual.strftime('%H:%M:%S')}）"
        )

        self.model_status.setText("模型状态：已暂停")
        self.model_status.setStyleSheet("color: #2563EB; font-weight:700;")
        self.update_status_label.setText("更新状态: 已暂停")

    def reset_scrolling(self):
        if self.pulse_data is not None:
            self.current_window_start = 0.0
            self.peaks_history = []
            self.prediction_history = []
            self.one_second_predictions = []
            self.last_five_second_update = 0.0

            self.plot_current_window()
            self.calculate_heart_rate()
            self.hr_value.setText(f"{self.calculated_hr:.1f}")

        self.co_value.setText("--")
        self.sv_value.setText("--")
        self.bsa_value.setText("--")
        self.bmi_value.setText("--")
        self.normalize_label.setText("参数标准化状态: 未开始")
        self.update_status_label.setText("更新状态: 等待数据...")

        self.status_label.setText("状态：已重置（起始时间：00:00:00）")

        if self.is_scrolling:
            self.pause_scrolling()

    # ---------- 1秒推理：时间戳对齐到片段起点 ----------
    def update_one_second_prediction(self):
        if (not self.is_scrolling) or (self.pulse_data is None) or (self.calculated_hr <= 0):
            return

        try:
            center_time_s = float(self.current_window_start) + (float(self.window_size) / 2.0)
            pulse_data_1s, start_idx = self.get_valley_aligned_data(center_time_s, return_start_idx=True)

            if pulse_data_1s is None or start_idx is None:
                self.update_status_label.setText("更新状态: 无法获取谷值对齐数据，跳过本次推理")
                return

            self.update_status_label.setText("更新状态: 已获取谷值对齐数据（1秒推理）")

            co, sv = self.predict_co_sv(pulse_data_1s)
            if co == 0 and sv == 0:
                self.model_status.setText("模型状态：1秒推理失败（模型未加载或输入错误）")
                self.model_status.setStyleSheet("color: red; font-weight:700;")
                return

            pred_time = self.base_time + timedelta(seconds=float(start_idx) / float(self.sampling_rate))

            self.one_second_predictions.append((pred_time, co, sv))
            if len(self.one_second_predictions) > 100:
                self.one_second_predictions = self.one_second_predictions[-100:]

            current_offset_s = float(start_idx) / float(self.sampling_rate)
            if (current_offset_s - float(self.last_five_second_update)) >= float(self.five_second_update_interval):
                self.calculate_five_second_average()
                self.last_five_second_update = float(current_offset_s)

        except Exception as e:
            self.model_status.setText(f"模型状态：1秒推理出错 - {str(e)}")
            self.model_status.setStyleSheet("color: red; font-weight:700;")
            logging.error(f"1秒推理更新失败: {e}", exc_info=True)

    # ---------- 5秒均值 ----------
    def calculate_five_second_average(self):
        if len(self.one_second_predictions) < 5:
            return

        recent_predictions = self.one_second_predictions[-5:]
        co_values = [p[1] for p in recent_predictions]
        sv_values = [p[2] for p in recent_predictions]

        co_avg = round(float(np.mean(co_values)), 2)
        sv_avg = round(float(np.mean(sv_values)), 1)

        self.co_value.setText(f"{co_avg}")
        self.sv_value.setText(f"{sv_avg}")

        self.co_value.setStyleSheet(
            "font-size: 24pt; font-weight: 900; color: red;"
            if not (4 <= co_avg <= 8) else "font-size: 24pt; font-weight: 900;"
        )
        self.sv_value.setStyleSheet(
            "font-size: 24pt; font-weight: 900; color: red;"
            if not (60 <= sv_avg <= 100) else "font-size: 24pt; font-weight: 900;"
        )

        current_time = recent_predictions[0][0]
        self.prediction_history.append((current_time, co_avg, sv_avg))

        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]

        self.model_status.setText(f"模型状态：5秒均值更新（CO:{co_avg} L/min, SV:{sv_avg} mL）")
        self.model_status.setStyleSheet("color: green; font-weight:700;")
        self.update_status_label.setText("更新状态: 5秒均值已更新（基于最近5个1秒推理值）")

    # ---------- 保存数据 ----------
    def save_data(self):
        if self.pulse_data is None or (not self.prediction_history and not self.one_second_predictions):
            QMessageBox.information(self, "提示", "暂无有效数据可保存！")
            return

        patient_id = self.id_input.text() or "未知ID"

        five_second_data = pd.DataFrame()
        if self.prediction_history:
            five_second_data = pd.DataFrame({
                "数据类型": ["5秒均值"] * len(self.prediction_history),
                "患者ID": [patient_id] * len(self.prediction_history),
                "时间": [item[0].strftime("%H:%M:%S") for item in self.prediction_history],
                "心输出量(L/min)": [item[1] for item in self.prediction_history],
                "每搏输出量(mL)": [item[2] for item in self.prediction_history],
                "心率(次/分)": [round(self.calculated_hr, 1)] * len(self.prediction_history)
            })

        one_second_data = pd.DataFrame()
        if self.one_second_predictions:
            one_second_data = pd.DataFrame({
                "数据类型": ["1秒推理值"] * len(self.one_second_predictions),
                "患者ID": [patient_id] * len(self.one_second_predictions),
                "时间": [item[0].strftime("%H:%M:%S") for item in self.one_second_predictions],
                "心输出量(L/min)": [item[1] for item in self.one_second_predictions],
                "每搏输出量(mL)": [item[2] for item in self.one_second_predictions],
                "心率(次/分)": [round(self.calculated_hr, 1)] * len(self.one_second_predictions)
            })

        save_df = pd.concat([five_second_data, one_second_data], ignore_index=True)

        save_df["体重(kg)"] = self.weight_input.text()
        save_df["年龄(岁)"] = self.age_input.text()
        save_df["身高(cm)"] = self.height_input.text()
        save_df["性别"] = "男" if self.male_btn.isChecked() else "女" if self.female_btn.isChecked() else "其他"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据", f"心功能监测_患者{patient_id}.xlsx", "Excel文件 (*.xlsx)"
        )
        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    save_df.to_excel(writer, sheet_name='所有数据', index=False)
                    if not five_second_data.empty:
                        five_second_data.to_excel(writer, sheet_name='5秒均值数据', index=False)
                    if not one_second_data.empty:
                        one_second_data.to_excel(writer, sheet_name='1秒推理数据', index=False)
                QMessageBox.information(self, "保存成功", f"数据已保存至：{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存错误：{str(e)}")

    # ---------- 趋势图 ----------
    def show_trend(self):
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "暂无5秒均值数据，请先开始监测！")
            return
        dialog = TrendDialog(self.prediction_history, self)
        dialog.exec_()


# ================== 程序入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PulseWaveMonitor()
    window.show()
    sys.exit(app.exec_())
