import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QRadioButton,
                             QButtonGroup, QPushButton, QMessageBox, QGroupBox,
                             QComboBox, QFileDialog, QDialog)
from PyQt5.QtCore import Qt, QTimer
import matplotlib
from scipy.signal import butter, filtfilt
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
import joblib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# 全局设置Matplotlib支持中文
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# ================== 推理模型核心函数 ==================
def read_csv_with_encoding(file_path):
    """兼容多种编码读取CSV文件"""
    encodings = ['utf-8', 'gbk', 'latin-1', 'utf-16']
    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=',',
                on_bad_lines='skip'
            )
            logging.info(f"Successfully read file using encoding: {encoding} (sep=',' engine='c')")
            return df
        except:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    sep=None,
                    engine='python',
                    on_bad_lines='skip'
                )
                logging.info(f"Successfully read file using encoding: {encoding} (auto-sep engine='python')")
                return df
            except Exception as e:
                logging.warning(f"Failed to read with encoding {encoding}: {str(e)}")
    logging.error(f"All encodings failed for file: {file_path}")
    return None


def calculate_derivatives(signal):
    """计算信号的一阶和二阶导数"""
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
    """标准化单个特征"""
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
        self.setGeometry(200, 200, 1000, 600)
        self.history_data = history_data
        self.data_count = len(self.history_data)

        self.base_fig_width = 10
        self.base_fig_height = 6
        # 限制最大宽度，避免窗口过宽
        self.fig_width = max(self.base_fig_width, min(15, self.base_fig_width + (self.data_count - 10) * 0.4))

        self.fig = Figure(figsize=(self.fig_width, self.base_fig_height), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.addWidget(self.toolbar)
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
            ax.set_title("心功能参数趋势图（每5秒更新一次）", fontsize=14, pad=20)
            self.canvas.draw()
            return

        times = [item[0] for item in self.history_data]
        co_values = [item[1] for item in self.history_data]
        sv_values = [item[2] for item in self.history_data]
        x_positions = range(len(times))
        time_labels = [t.strftime("%H:%M:%S") for t in times]

        self.fig.clear()

        ax1 = self.fig.add_subplot(211)
        ax1.plot(x_positions, co_values, 'b-o', linewidth=2, markersize=4,
                 label='心输出量(CO)', color='#1f77b4')
        ax1.set_ylabel('心输出量 (L/min)', fontsize=11, color='#333333')
        ax1.set_title("心功能参数趋势图（每5秒更新一次）", fontsize=14, pad=20, color='#222222')
        ax1.grid(True, linestyle='--', alpha=0.7, color='#dddddd')
        ax1.axhline(y=4, color='#ff7f0e', linestyle='--', alpha=0.6, label='正常下限（4 L/min）')
        ax1.axhline(y=8, color='#ff7f0e', linestyle='--', alpha=0.6, label='正常上限（8 L/min）')
        ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax1.set_ylim(0, 12)
        ax1.spines['top'].set_visible(False)

        ax2 = self.fig.add_subplot(212)
        ax2.plot(x_positions, sv_values, 'g-o', linewidth=2, markersize=4,
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
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(time_labels, rotation=0, fontsize=9, color='#444444')
        else:
            step = max(1, self.data_count // 15)
            display_pos = x_positions[::step]
            display_labels = time_labels[::step]
            ax2.set_xticks(display_pos)
            ax2.set_xticklabels(display_labels, rotation=45, fontsize=8, color='#444444')

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

    def on_click(self, event):
        if not self.history_data or event.inaxes is None:
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

    def __init__(self):
        super().__init__()
        self.input_mode = "file"
        self.excel_path = ""
        self.serial_port = ""
        self.pulse_data = None
        self.sampling_rate = 125
        self.window_size = 5
        self.scroll_step = 1
        self.current_window_start = 0
        self.total_duration = 0
        self.base_time = None
        self.is_first_load = True
        self.calculated_hr = 0
        self.input_visible = False
        self.peaks_history = []
        self.prediction_history = []

        # ONNX模型相关初始化
        self.onnx_session = None
        self.signal_scaler = None
        self.scalar_paths = {}
        self.model_loaded = False
        self.init_model_paths()

        self.init_ui()
        self.init_data()

    def init_model_paths(self):
        """初始化模型路径配置 - 更新为指定的Windows路径"""
        # 模型根目录
        model_root_dir = r"E:\Users\yangc\PycharmProjects\PythonProject3\tuili"

        self.model_config = {
            'model_path': os.path.join(model_root_dir, 'resnet_se_lstm_model.onnx'),  # ONNX模型文件
            'signal_scaler_path': os.path.join(model_root_dir, 'signal_scaler.pkl'),
            'scalar_dir': model_root_dir
        }

        # 初始化标量路径
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
            # 检查ONNX模型文件是否存在
            if not os.path.exists(self.model_config['model_path']):
                raise FileNotFoundError(f"ONNX模型文件不存在: {self.model_config['model_path']}")

            # 设置ONNX Runtime选项
            ort_options = ort.SessionOptions()
            ort_options.intra_op_num_threads = 4
            ort_options.inter_op_num_threads = 4
            ort_options.enable_cpu_mem_arena = True

            # 加载ONNX模型
            logging.info(f"Loading ONNX model from: {self.model_config['model_path']}")
            self.onnx_session = ort.InferenceSession(
                self.model_config['model_path'],
                sess_options=ort_options,
                providers=['CPUExecutionProvider']  # 使用CPU推理
            )

            # 获取输入输出信息
            input_names = [input.name for input in self.onnx_session.get_inputs()]
            output_names = [output.name for output in self.onnx_session.get_outputs()]
            logging.info(f"ONNX模型输入: {input_names}")
            logging.info(f"ONNX模型输出: {output_names}")

            # 检查信号标量文件是否存在
            if not os.path.exists(self.model_config['signal_scaler_path']):
                raise FileNotFoundError(f"信号标量文件不存在: {self.model_config['signal_scaler_path']}")

            # 加载信号标量
            logging.info(f"Loading signal scaler from: {self.model_config['signal_scaler_path']}")
            self.signal_scaler = joblib.load(self.model_config['signal_scaler_path'])

            # 检查所有标量文件
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
            return True

        except Exception as e:
            error_msg = f"ONNX模型加载失败: {str(e)}"
            logging.error(error_msg)
            self.model_loaded = False
            self.onnx_session = None
            self.model_status.setText("模型状态：加载失败")
            self.model_status.setStyleSheet("color: red;")
            QMessageBox.critical(self, "模型加载失败", error_msg)
            return False

    def init_ui(self):
        self.setWindowTitle("心输出量监测系统（ONNX版）")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)

        # 顶部控制区
        self.top_bar = QVBoxLayout()

        # 输入控制按钮
        self.input_control_layout = QHBoxLayout()
        self.input_btn = QPushButton("输入脉搏波波形")
        self.input_btn.clicked.connect(self.toggle_input_interface)
        self.input_control_layout.addWidget(self.input_btn, alignment=Qt.AlignLeft)

        # 加载模型按钮
        self.load_model_btn = QPushButton("加载ONNX推理模型")
        self.load_model_btn.clicked.connect(self.load_inference_model)
        self.input_control_layout.addWidget(self.load_model_btn, alignment=Qt.AlignRight)

        self.input_control_layout.addStretch(1)
        self.top_bar.addLayout(self.input_control_layout)

        # 输入选择容器
        self.input_container = QWidget()
        self.input_container_layout = QVBoxLayout(self.input_container)

        # 输入模式选择
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

        # 数据/串口路径选择
        self.path_layout = QHBoxLayout()

        # 文件模式控件
        self.path_label = QLabel("数据文件路径:")
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("输入Excel/CSV文件路径或点击浏览选择")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self.browse_file)

        # 串口模式控件
        self.port_label = QLabel("串口选择:")
        self.port_combo = QComboBox()
        for i in range(1, 11):
            self.port_combo.addItem(f"COM{i}")

        # 共用加载按钮
        self.load_btn = QPushButton("加载数据")
        self.load_btn.clicked.connect(self.load_data)

        # 默认显示文件模式控件
        self.path_layout.addWidget(self.path_label)
        self.path_layout.addWidget(self.path_input, stretch=1)
        self.path_layout.addWidget(self.browse_btn)
        self.path_layout.addWidget(self.load_btn)
        self.path_layout.setContentsMargins(0, 0, 0, 0)
        self.input_container_layout.addLayout(self.path_layout)

        self.top_bar.addWidget(self.input_container)

        # 标题单独一行
        self.title_layout = QHBoxLayout()
        self.title_label = QLabel("心输出量监测系统（ResNet-LSTM-SE ONNX模型）")
        self.title_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        self.title_layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        self.top_bar.addLayout(self.title_layout)

        # 患者ID单独一行
        self.id_layout = QHBoxLayout()
        self.id_label = QLabel("患者ID:")
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("输入患者ID")
        self.id_input.setFixedWidth(200)
        self.id_layout.addStretch(1)
        self.id_layout.addWidget(self.id_label)
        self.id_layout.addWidget(self.id_input)
        self.id_layout.addStretch(1)
        self.top_bar.addLayout(self.id_layout)

        self.main_layout.addLayout(self.top_bar)

        # 中间参数、结果与控制按钮区
        mid_layout = QHBoxLayout()

        # 左侧生理参数输入
        self.param_group = QGroupBox("生理参数输入")
        param_layout = QVBoxLayout()

        self.weight_layout = QHBoxLayout()
        self.weight_label = QLabel("体重 (kg):")
        self.weight_input = QLineEdit()
        self.weight_input.setPlaceholderText("10-200")
        self.weight_layout.addWidget(self.weight_label)
        self.weight_layout.addWidget(self.weight_input)
        param_layout.addLayout(self.weight_layout)

        self.age_layout = QHBoxLayout()
        self.age_label = QLabel("年龄 (岁):")
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("0-120")
        self.age_layout.addWidget(self.age_label)
        self.age_layout.addWidget(self.age_input)
        param_layout.addLayout(self.age_layout)

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
        param_layout.addLayout(self.gender_layout)

        self.height_layout = QHBoxLayout()
        self.height_label = QLabel("身高 (cm):")
        self.height_input = QLineEdit()
        self.height_input.setPlaceholderText("50-250")
        self.height_layout.addWidget(self.height_label)
        self.height_layout.addWidget(self.height_input)
        param_layout.addLayout(self.height_layout)

        # BSA和BMI计算显示
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
        param_layout.addLayout(self.bsa_bmi_layout)

        # 心率显示
        self.hr_layout = QHBoxLayout()
        self.hr_label = QLabel("心率 (次/分):")
        self.hr_value = QLabel("--")
        self.hr_value.setStyleSheet("font-size: 12pt;")
        self.hr_layout.addWidget(self.hr_label)
        self.hr_layout.addWidget(self.hr_value)
        param_layout.addLayout(self.hr_layout)

        self.normalize_label = QLabel("参数标准化状态: 未开始")
        self.normalize_label.setStyleSheet("color: gray;")
        param_layout.addWidget(self.normalize_label)
        self.param_group.setLayout(param_layout)
        mid_layout.addWidget(self.param_group, stretch=1)

        # 中间心功能预测结果
        self.result_group = QGroupBox("心功能预测结果")
        result_layout = QVBoxLayout()

        self.co_layout = QHBoxLayout()
        self.co_label = QLabel("心输出量 (L/min):")
        self.co_value = QLabel("--")
        self.co_value.setStyleSheet("font-size: 24pt; font-weight: bold;")
        self.co_range = QLabel("(正常范围: 4-8 L/min)")
        self.co_range.setStyleSheet("color: gray;")
        self.co_layout.addWidget(self.co_label)
        self.co_layout.addWidget(self.co_value)
        result_layout.addLayout(self.co_layout)
        result_layout.addWidget(self.co_range)

        self.sv_layout = QHBoxLayout()
        self.sv_label = QLabel("每搏输出量 (mL):")
        self.sv_value = QLabel("--")
        self.sv_value.setStyleSheet("font-size: 24pt; font-weight: bold;")
        self.sv_range = QLabel("(正常范围: 60-100 mL)")
        self.sv_range.setStyleSheet("color: gray;")
        self.sv_layout.addWidget(self.sv_label)
        self.sv_layout.addWidget(self.sv_value)
        result_layout.addLayout(self.sv_layout)
        result_layout.addWidget(self.sv_range)

        self.model_status = QLabel("模型状态: 未加载模型")
        self.model_status.setStyleSheet("color: blue;")
        result_layout.addWidget(self.model_status)
        self.result_group.setLayout(result_layout)
        mid_layout.addWidget(self.result_group, stretch=1)

        # 右侧控制按钮区
        self.control_group = QGroupBox("功能控制")
        control_layout = QVBoxLayout()

        self.control_btn_layout = QVBoxLayout()
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

        # 按钮垂直排列
        self.control_btn_layout.addStretch(1)
        self.control_btn_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        self.control_btn_layout.addSpacing(15)
        self.control_btn_layout.addWidget(self.pause_btn, alignment=Qt.AlignCenter)
        self.control_btn_layout.addSpacing(15)
        self.control_btn_layout.addWidget(self.reset_btn, alignment=Qt.AlignCenter)
        self.control_btn_layout.addSpacing(15)
        self.control_btn_layout.addWidget(self.save_btn, alignment=Qt.AlignCenter)
        self.control_btn_layout.addSpacing(15)
        self.control_btn_layout.addWidget(self.trend_btn, alignment=Qt.AlignCenter)
        self.control_btn_layout.addStretch(1)
        control_layout.addLayout(self.control_btn_layout)
        self.control_group.setLayout(control_layout)
        mid_layout.addWidget(self.control_group, stretch=1)

        self.main_layout.addLayout(mid_layout)

        # 底部波形显示区
        self.wave_group = QGroupBox(f"脉搏波波形（采样率：{self.sampling_rate}Hz，窗口大小：{self.window_size}秒）")
        wave_layout = QVBoxLayout()
        self.fig = Figure(figsize=(10, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("时间（时:分:秒）")
        self.ax.set_ylabel("振幅")
        self.canvas = FigureCanvas(self.fig)
        wave_layout.addWidget(self.canvas)
        wave_layout.addWidget(NavigationToolbar(self.canvas, self))  # 添加工具栏
        self.wave_group.setLayout(wave_layout)
        self.main_layout.addWidget(self.wave_group, stretch=1)

        # 状态提示区
        self.status_label = QLabel("状态：请先点击'加载ONNX推理模型'，然后加载脉搏波数据")
        self.status_label.setStyleSheet("font-size: 12pt; color: blue;")
        self.main_layout.addWidget(self.status_label)

        # 初始隐藏输入界面
        self.hide_input_interface()

    def calculate_bsa_bmi(self):
        """计算BSA和BMI"""
        try:
            weight = float(self.weight_input.text())
            height = float(self.height_input.text()) / 100  # 转换为米

            # 计算BMI: 体重(kg) / 身高²(m)
            bmi = weight / (height ** 2)
            self.bmi_value.setText(f"{bmi:.2f}")

            # 计算BSA (Du Bois公式): 0.007184 × 体重^0.425 × 身高^0.725
            bsa = 0.007184 * (weight ** 0.425) * (height * 100 ** 0.725)
            self.bsa_value.setText(f"{bsa:.3f}")

            return bsa, bmi
        except:
            self.bsa_value.setText("--")
            self.bmi_value.setText("--")
            return 0, 0

    def init_data(self):
        self.is_scrolling = False
        self.scroll_timer = QTimer(self)
        self.scroll_timer.setInterval(self.scroll_step * 1000)
        self.scroll_timer.timeout.connect(self.scroll_window)
        self.predict_timer = QTimer(self)
        self.predict_timer.setInterval(5000)
        self.predict_timer.timeout.connect(self.update_prediction)

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
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择脉搏波数据", "", "数据文件 (*.xlsx *.xls *.csv)"
        )
        if file_path:
            self.path_input.setText(file_path)

    def load_data(self):
        if self.input_mode == "file":
            self.load_from_file()
        else:
            self.load_from_serial()
        self.hide_input_interface()

    def load_from_file(self):
        """从Excel/CSV文件加载脉搏波数据"""
        try:
            file_path = self.path_input.text()
            if not file_path:
                QMessageBox.warning(self, "路径为空", "请输入或选择文件路径")
                return

            # 根据文件扩展名选择读取方式
            if file_path.endswith('.csv'):
                df = read_csv_with_encoding(file_path)
            else:
                df = pd.read_excel(file_path, header=0)

            # 假设脉搏波在第二列（索引为1），如果是125点/秒的数据，取signal列
            if 'signal_1' in df.columns:
                # 如果有signal_1到signal_125列，取这些列作为波形数据
                signal_cols = [f'signal_{i}' for i in range(1, 126)]
                self.pulse_data = df[signal_cols].values.flatten()
            else:
                # 否则取第二列
                self.pulse_data = df.iloc[:, 1].values.astype(float)

            self.pulse_data = self.apply_filter(self.pulse_data)
            self.process_loaded_data()

        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"文件读取错误：{str(e)}")
            logging.error(f"文件加载错误: {e}")
            self.pulse_data = None

    def apply_filter(self, data):
        """低通滤波处理脉搏波数据"""

        def butter_lowpass(cutoff, fs, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=4):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y

        filtered_data = butter_lowpass_filter(data, 10, self.sampling_rate, order=4)
        return filtered_data

    def load_from_serial(self):
        """模拟串口输入，生成生理特征的脉搏波数据"""
        self.serial_port = self.port_combo.currentText()

        try:
            duration = 60  # 生成60秒数据
            t = np.linspace(0, duration, duration * self.sampling_rate, endpoint=False)
            heart_rate = 75  # 心率75次/分
            hr_freq = heart_rate / 60  # 心率频率

            # 生成主波（包含收缩期特征）
            main_wave = np.sin(2 * np.pi * hr_freq * t) * 30 + \
                        np.sin(4 * np.pi * hr_freq * t) * 10 + \
                        np.exp(-2 * np.pi * hr_freq * t % 1 * 3) * 20

            # 生成重搏波（舒张期特征）
            dicrotic_notch = 0.35  # 重搏切迹位置
            dicrotic_wave = np.sin(2 * np.pi * hr_freq * (t - dicrotic_notch)) * 12 + \
                            np.exp(-2 * np.pi * hr_freq * (t - dicrotic_notch) % 1 * 4) * 8

            # 添加基线漂移和噪声
            baseline_drift = np.sin(2 * np.pi * 0.03 * t) * 5  # 呼吸性漂移
            noise = np.random.normal(0, 1.5, len(t))  # 高斯噪声
            raw_pulse = main_wave + dicrotic_wave + baseline_drift + noise + 80

            # 滤波处理
            self.pulse_data = self.apply_filter(raw_pulse)
            self.process_loaded_data()

        except Exception as e:
            QMessageBox.critical(self, "连接失败", f"错误：{str(e)}")
            self.pulse_data = None

    def process_loaded_data(self):
        """处理加载完成的数据，初始化参数"""
        if self.is_first_load:
            self.base_time = datetime.now()
            self.is_first_load = False
            self.status_label.setText(f"状态：基准时间已设置（{self.base_time.strftime('%H:%M:%S')}）")

        self.total_duration = len(self.pulse_data) / self.sampling_rate
        self.current_window_start = 0

        self.peaks_history = []
        self.prediction_history = []
        self.calculate_heart_rate()
        self.hr_value.setText(f"{self.calculated_hr:.1f}")

        self.trend_btn.setEnabled(True)

        if self.input_mode == "file":
            status_text = f"状态：已从文件加载数据（总时长：{self.total_duration:.1f}秒，采样率：{self.sampling_rate}Hz）"
        else:
            status_text = f"状态：已连接到 {self.serial_port}（总时长：{self.total_duration:.1f}秒）"

        self.status_label.setText(status_text)
        self.reset_btn.setEnabled(True)
        self.plot_current_window()

        # 检查模型是否已加载
        if not self.model_loaded:
            self.model_status.setText("模型状态：未加载ONNX模型，无法进行预测")
            self.model_status.setStyleSheet("color: red;")
            # 禁用开始预测按钮
            self.start_btn.setEnabled(False)
        else:
            self.model_status.setText("模型状态：数据已加载，可开始预测")
            self.start_btn.setEnabled(True)

    def calculate_heart_rate(self):
        """从脉搏波计算心率"""
        if self.pulse_data is None:
            self.calculated_hr = 0
            return

        # 取当前窗口前后15秒的数据计算心率
        window_length = int(15 * self.sampling_rate)
        current_pos = int(self.current_window_start * self.sampling_rate)
        start_idx = max(0, current_pos - window_length // 2)
        end_idx = min(len(self.pulse_data), current_pos + window_length // 2)
        data_window = self.pulse_data[start_idx:end_idx]

        if len(data_window) < self.sampling_rate * 2:  # 至少需要2秒数据
            self.calculated_hr = 0
            return

        # 去基线漂移
        window_size = int(self.sampling_rate * 0.2)
        if window_size % 2 == 0:
            window_size += 1
        baseline = np.convolve(data_window, np.ones(window_size) / window_size, mode='same')
        data_filtered = data_window - baseline

        # 阈值法检测峰值
        data_mean = np.mean(data_filtered)
        data_std = np.std(data_filtered)
        threshold = data_mean + 1.2 * data_std  # 阈值设为均值+1.2倍标准差

        peaks = []
        min_peak_interval = int(self.sampling_rate * 60 / 200)  # 最小峰值间隔（30次/分）
        max_peak_interval = int(self.sampling_rate * 60 / 30)  # 最大峰值间隔（200次/分）

        for i in range(1, len(data_filtered) - 1):
            if (data_filtered[i] > data_filtered[i - 1] and
                    data_filtered[i] >= data_filtered[i + 1] and
                    data_filtered[i] > threshold):
                if not peaks or (i - peaks[-1]) > min_peak_interval:
                    peaks.append(i)

        # 过滤异常峰值间隔
        valid_peaks = []
        if peaks:
            valid_peaks.append(peaks[0])
            for i in range(1, len(peaks)):
                interval = peaks[i] - peaks[i - 1]
                if min_peak_interval < interval < max_peak_interval:
                    valid_peaks.append(peaks[i])
            peaks = valid_peaks

        # 保存峰值历史
        self.peaks_history.extend([p + start_idx for p in peaks])
        if len(self.peaks_history) > 100:  # 限制历史长度
            self.peaks_history = self.peaks_history[-100:]

        # 计算心率
        if len(self.peaks_history) >= 2:
            intervals = np.diff(self.peaks_history) / self.sampling_rate  # 峰值间隔（秒）
            interval_mean = np.mean(intervals)
            interval_std = np.std(intervals)
            # 剔除3倍标准差外的异常值
            valid_intervals = intervals[np.abs(intervals - interval_mean) < 3 * interval_std]

            if len(valid_intervals) >= 2:
                avg_interval = np.mean(valid_intervals)
                self.calculated_hr = 60 / avg_interval if avg_interval > 0 else 0
            else:
                self.calculated_hr = 60 / interval_mean if interval_mean > 0 else 0
            self.calculated_hr = np.clip(self.calculated_hr, 30, 200)  # 限制心率范围
        else:
            self.calculated_hr = 0

    def plot_current_window(self):
        """绘制当前窗口的脉搏波"""
        if self.pulse_data is None or self.base_time is None:
            return

        window_end_relative = self.current_window_start + self.window_size
        start_idx = int(self.current_window_start * self.sampling_rate)
        end_idx = int(window_end_relative * self.sampling_rate)
        end_idx = min(end_idx, len(self.pulse_data))
        window_pulse = self.pulse_data[start_idx:end_idx]

        if len(window_pulse) == 0:
            self.status_label.setText("状态：当前窗口无数据，请检查数据长度")
            return

        # 计算相对时间
        window_time_relative = np.linspace(
            self.current_window_start,
            self.current_window_start + (end_idx - start_idx) / self.sampling_rate,
            len(window_pulse)
        )

        self.ax.clear()
        self.ax.plot(window_time_relative, window_pulse, 'r-', linewidth=1.5)
        self.ax.grid(True, linestyle='--', alpha=0.7)

        self.ax.set_xlabel("时间（时:分:秒）")
        self.ax.set_ylabel("振幅")
        self.ax.set_xlim(self.current_window_start, window_end_relative)

        # 设置时间标签
        xticks = np.arange(
            np.ceil(self.current_window_start),
            window_end_relative,
            1
        )
        xtick_labels = [
            (self.base_time + timedelta(seconds=t)).strftime("%H:%M:%S")
            for t in xticks
        ]
        self.ax.set_xticks(xticks)
        self.ax.set_xticklabels(xtick_labels, rotation=0)

        # 调整Y轴范围
        y_min, y_max = np.min(window_pulse), np.max(window_pulse)
        y_margin = (y_max - y_min) * 0.15 if y_max != y_min else 10
        self.ax.set_ylim(y_min - y_margin, y_max + y_margin)

        self.canvas.draw()

    def scroll_window(self):
        """滚动显示窗口"""
        if self.pulse_data is None:
            return

        next_start = self.current_window_start + self.scroll_step
        next_end = next_start + self.window_size

        # 检查是否到达数据末尾
        if next_start >= self.total_duration:
            self.status_label.setText(f"状态：已显示全部数据（总时长：{self.total_duration:.1f}秒）")
            self.pause_scrolling()
            return

        if next_end > self.total_duration:
            next_start = max(0, self.total_duration - self.window_size)

        self.current_window_start = next_start
        self.plot_current_window()

        # 更新心率
        self.calculate_heart_rate()
        self.hr_value.setText(f"{self.calculated_hr:.1f}")

        # 更新状态
        start_actual = self.base_time + timedelta(seconds=self.current_window_start)
        end_actual = self.base_time + timedelta(seconds=next_end)
        self.status_label.setText(
            f"状态：预测中 当前窗口：{start_actual.strftime('%H:%M:%S')} - {end_actual.strftime('%H:%M:%S')}"
        )

    def start_scrolling(self):
        """开始预测"""
        if self.pulse_data is None:
            QMessageBox.warning(self, "无数据", "请先加载数据！")
            return

        if not self.model_loaded:
            QMessageBox.critical(self, "模型未加载", "ONNX推理模型未加载成功，无法进行预测！")
            return

        if not self.check_params():
            QMessageBox.warning(self, "参数错误", "请补全并正确输入生理参数！")
            return

        # 计算BSA和BMI
        self.calculate_bsa_bmi()

        self.current_window_start = 0
        self.prediction_history = []

        self.is_scrolling = True
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.scroll_timer.start()
        self.predict_timer.start()
        self.normalize_label.setText("参数标准化状态: 已完成")

        self.model_status.setText("模型状态：ONNX模型运行中（每5秒更新预测）")

        start_actual = self.base_time + timedelta(seconds=self.current_window_start)
        end_actual = self.base_time + timedelta(seconds=self.window_size)
        self.status_label.setText(
            f"状态：预测中 当前窗口：{start_actual.strftime('%H:%M:%S')} - {end_actual.strftime('%H:%M:%S')}"
        )
        self.update_prediction()

    def pause_scrolling(self):
        """暂停预测"""
        self.is_scrolling = False
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.scroll_timer.stop()
        self.predict_timer.stop()

        start_actual = self.base_time + timedelta(seconds=self.current_window_start)
        end_actual = self.base_time + timedelta(seconds=self.current_window_start + self.window_size)
        self.status_label.setText(
            f"状态：已暂停（当前窗口：{start_actual.strftime('%H:%M:%S')} - {end_actual.strftime('%H:%M:%S')}"
        )
        self.model_status.setText("模型状态：已暂停")

    def reset_scrolling(self):
        """重置系统"""
        if self.pulse_data is not None:
            self.current_window_start = 0
            self.peaks_history = []
            self.prediction_history = []
            self.plot_current_window()
            self.calculate_heart_rate()
            self.hr_value.setText(f"{self.calculated_hr:.1f}")

        self.co_value.setText("--")
        self.sv_value.setText("--")
        self.bsa_value.setText("--")
        self.bmi_value.setText("--")
        self.normalize_label.setText("参数标准化状态: 未开始")

        if self.base_time:
            self.status_label.setText(f"状态：已重置（基准时间：{self.base_time.strftime('%H:%M:%S')}）")
        else:
            self.status_label.setText("状态：请点击'输入脉搏波波形'加载数据")

        if self.is_scrolling:
            self.pause_scrolling()

    def save_data(self):
        """保存预测数据到Excel"""
        if self.pulse_data is None or not self.prediction_history:
            QMessageBox.information(self, "提示", "暂无有效数据可保存！")
            return

        patient_id = self.id_input.text() or "未知ID"

        # 准备保存数据
        save_data = pd.DataFrame({
            "患者ID": [patient_id] * len(self.prediction_history),
            "时间": [item[0].strftime("%H:%M:%S") for item in self.prediction_history],
            "心输出量(L/min)": [item[1] for item in self.prediction_history],
            "每搏输出量(mL)": [item[2] for item in self.prediction_history],
            "心率(次/分)": [self.calculated_hr] * len(self.prediction_history),
            "体重(kg)": [self.weight_input.text()] * len(self.prediction_history),
            "年龄(岁)": [self.age_input.text()] * len(self.prediction_history),
            "身高(cm)": [self.height_input.text()] * len(self.prediction_history),
            "性别": ["男" if self.male_btn.isChecked() else "女" if self.female_btn.isChecked() else "其他"] * len(
                self.prediction_history)
        })

        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据", f"心功能监测_患者{patient_id}.xlsx", "Excel文件 (*.xlsx)"
        )
        if file_path:
            try:
                save_data.to_excel(file_path, index=False)
                QMessageBox.information(self, "保存成功", f"数据已保存至：{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"保存错误：{str(e)}")

    def show_trend(self):
        """显示趋势图"""
        if not self.prediction_history:
            QMessageBox.information(self, "提示", "暂无预测数据，请先开始监测！")
            return

        dialog = TrendDialog(self.prediction_history, self)
        dialog.exec_()

    def check_params(self):
        """检查生理参数是否有效"""
        try:
            # 检查是否为空
            if not (self.weight_input.text() and self.age_input.text() and self.height_input.text()):
                return False

            weight = float(self.weight_input.text())
            age = int(self.age_input.text())
            height = float(self.height_input.text())

            # 检查范围
            return (10 <= weight <= 200 and
                    0 <= age <= 120 and
                    50 <= height <= 250)
        except:
            return False

    def prepare_model_inputs(self):
        """准备ONNX模型输入数据"""
        try:
            # 获取当前窗口的波形数据
            start_idx = int(self.current_window_start * self.sampling_rate)
            end_idx = int((self.current_window_start + 1) * self.sampling_rate)  # 取1秒数据（125点）
            end_idx = min(end_idx, len(self.pulse_data))

            # 确保是125个点
            window_pulse = self.pulse_data[start_idx:end_idx]
            if len(window_pulse) < 125:
                # 补零或截断到125点
                if len(window_pulse) < 125:
                    window_pulse = np.pad(window_pulse, (0, 125 - len(window_pulse)), 'constant')
                else:
                    window_pulse = window_pulse[:125]

            # 重塑为模型输入格式
            raw_signal = window_pulse.reshape(1, -1)

            # 计算导数
            first_deriv, second_deriv = calculate_derivatives(raw_signal)

            # 构建特征张量
            X = np.stack([raw_signal, first_deriv, second_deriv], axis=-1)

            # 获取生理参数
            weight = float(self.weight_input.text())
            age = int(self.age_input.text())
            height = float(self.height_input.text())
            gender = 1 if self.male_btn.isChecked() else 0 if self.female_btn.isChecked() else 2
            hr = self.calculated_hr

            # 计算BSA和BMI
            bsa, bmi = self.calculate_bsa_bmi()

            # 标准化特征
            X_reshaped = X.reshape(X.shape[0], -1)
            X_std = self.signal_scaler.transform(X_reshaped).reshape(X.shape)

            age_std = standardize_single_feature(np.array([age]), self.scalar_paths['age'])
            weight_std = standardize_single_feature(np.array([weight]), self.scalar_paths['weight'])
            height_std = standardize_single_feature(np.array([height]), self.scalar_paths['height'])
            bsa_std = standardize_single_feature(np.array([bsa]), self.scalar_paths['bsa'])
            bmi_std = standardize_single_feature(np.array([bmi]), self.scalar_paths['bmi'])
            hr_std = standardize_single_feature(np.array([hr]), self.scalar_paths['hr'])

            # 转换为float32（ONNX常用格式）
            X_std = X_std.astype(np.float32)
            age_std = age_std.astype(np.float32)
            gender_arr = np.array([gender], dtype=np.float32)
            weight_std = weight_std.astype(np.float32)
            height_std = height_std.astype(np.float32)
            bsa_std = bsa_std.astype(np.float32)
            bmi_std = bmi_std.astype(np.float32)
            hr_std = hr_std.astype(np.float32)

            return (X_std, age_std, gender_arr, weight_std,
                    height_std, bsa_std, bmi_std, hr_std)

        except Exception as e:
            logging.error(f"准备ONNX模型输入失败: {e}")
            return None

    def predict_co_sv(self):
        """使用ONNX模型预测SV和CO"""
        # 仅当模型加载成功时才进行预测
        if not self.model_loaded or self.onnx_session is None:
            return 0, 0

        try:
            # 准备模型输入
            inputs = self.prepare_model_inputs()
            if inputs is None:
                return 0, 0

            X_std, age_std, gender_arr, weight_std, height_std, bsa_std, bmi_std, hr_std = inputs

            # 获取ONNX输入名称
            input_names = [input.name for input in self.onnx_session.get_inputs()]

            # 构建输入字典（适配不同的ONNX模型输入命名）
            input_feed = {}
            if len(input_names) == 1:
                # 如果只有一个输入，假设是合并的特征向量
                # 根据实际模型调整输入格式
                combined_input = np.concatenate([
                    X_std.flatten(), age_std, gender_arr, weight_std,
                    height_std, bsa_std, bmi_std, hr_std
                ]).reshape(1, -1).astype(np.float32)
                input_feed[input_names[0]] = combined_input
            else:
                # 多输入模型，按顺序赋值
                input_data_list = [X_std, age_std, gender_arr, weight_std,
                                   height_std, bsa_std, bmi_std, hr_std]

                # 确保输入数量匹配
                for i, name in enumerate(input_names):
                    if i < len(input_data_list):
                        input_feed[name] = input_data_list[i]
                    else:
                        logging.warning(f"ONNX模型输入{name}无对应数据")

            # ONNX模型推理
            outputs = self.onnx_session.run(None, input_feed)
            pred_sv = outputs[0].flatten()[0]

            # 计算CO
            pred_co = (pred_sv * self.calculated_hr) / 1000

            # 限制范围
            pred_sv = np.clip(pred_sv, 30, 150)
            pred_co = np.clip(pred_co, 2, 10)

            return round(pred_co, 2), round(pred_sv, 1)

        except Exception as e:
            logging.error(f"ONNX模型预测失败: {e}")
            return 0, 0

    def update_prediction(self):
        """更新预测结果"""
        if not self.is_scrolling or self.pulse_data is None or self.calculated_hr <= 0:
            return

        try:
            # 使用ONNX模型预测
            co, sv = self.predict_co_sv()

            # 如果预测值为0（模型未加载或预测失败），不更新显示
            if co == 0 and sv == 0:
                self.model_status.setText("模型状态：预测失败（ONNX模型未加载或输入数据错误）")
                self.model_status.setStyleSheet("color: red;")
                return

            # 更新显示
            self.co_value.setText(f"{co}")
            self.sv_value.setText(f"{sv}")

            # 异常值标红
            self.co_value.setStyleSheet(
                "font-size: 24pt; font-weight: bold; color: red;"
                if not (4 <= co <= 8) else "font-size: 24pt; font-weight: bold;"
            )
            self.sv_value.setStyleSheet(
                "font-size: 24pt; font-weight: bold; color: red;"
                if not (60 <= sv <= 100) else "font-size: 24pt; font-weight: bold;"
            )

            # 保存历史记录
            current_time = self.base_time + timedelta(seconds=self.current_window_start)
            self.prediction_history.append((current_time, co, sv))

            # 限制历史长度
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]

        except Exception as e:
            self.model_status.setText(f"模型状态：预测出错 - {str(e)}")
            self.model_status.setStyleSheet("color: red;")
            logging.error(f"预测更新失败: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PulseWaveMonitor()
    window.show()
    sys.exit(app.exec_())