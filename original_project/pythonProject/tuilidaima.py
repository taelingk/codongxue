# %%
import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import onnxruntime as ort
from sklearn.preprocessing import StandardScaler
import logging

# ================== 配置日志 ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


# ================== 自定义层定义（加载Keras模型必需） ==================
class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Block"""

    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = tf.keras.layers.GlobalAveragePooling1D()
        self.excitation1 = tf.keras.layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.excitation2 = tf.keras.layers.Dense(channels, activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation1(x)
        x = self.excitation2(x)
        x = tf.keras.layers.Reshape((1, -1))(x)
        return tf.keras.layers.Multiply()([inputs, x])

# ================== Model Definition with LSTM and SE ==================
# ================== Model Definition with LSTM and SE ==================
class ResidualSEBlock(tf.keras.layers.Layer):
    """Residual block with SE attention"""

    def __init__(self, filters, stride=1, use_1x1_conv=False, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters, 3, padding='same', strides=stride,
                                            kernel_regularizer=kernel_regularizer)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.conv2 = tf.keras.layers.Conv1D(filters, 3, padding='same', strides=1,
                                            kernel_regularizer=kernel_regularizer)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock()

        if use_1x1_conv:
            self.shortcut_conv = tf.keras.layers.Conv1D(filters, 1, strides=stride,
                                                        kernel_regularizer=kernel_regularizer)
            self.shortcut_bn = tf.keras.layers.BatchNormalization()
        self.use_1x1_conv = use_1x1_conv
        self.act2 = tf.keras.layers.Activation('relu')

    def call(self, inputs):
        x = self.act1(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)

        if self.use_1x1_conv:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
        else:
            shortcut = inputs
        return self.act2(x + shortcut)






# ================== 推理模型类（修复维度错误） ==================
class SVCOInferenceModel:
    def __init__(self, scaler_path, model_dir):
        """
        初始化推理模型（指定标准化器路径）
        :param scaler_path: signal_scaler.pkl的完整路径
        :param model_dir: 模型文件所在目录
        """
        self.scaler_path = scaler_path  # 单独指定标准化器路径
        self.model_dir = model_dir
        self.model_type = None
        self.model = None
        self.signal_scaler = None
        self._load_scaler()

    def _load_scaler(self):
        """加载指定路径的信号标准化器"""
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"信号标准化器不存在: {self.scaler_path}")
        self.signal_scaler = joblib.load(self.scaler_path)
        logging.info(f"✅ 信号标准化器加载完成（路径：{self.scaler_path}）")

    def load_keras_model(self):
        """加载Keras模型"""
        model_path = os.path.join(self.model_dir, "resnet18_lstm_se_model.keras")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Keras模型不存在: {model_path}")

        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={'SEBlock': SEBlock, 'ResidualSEBlock': ResidualSEBlock}
        )
        self.model_type = "keras"
        logging.info(f"✅ Keras模型加载完成（路径：{model_path}）")

    def load_onnx_model(self):
        """加载ONNX模型"""
        onnx_path = os.path.join(self.model_dir, "resnet18_lstm_se_model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX模型不存在: {onnx_path}")

        providers = ['CPUExecutionProvider']
        if ort.get_device() == "GPU":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.model = ort.InferenceSession(onnx_path, providers=providers)
        self.model_type = "onnx"
        logging.info(f"✅ ONNX模型加载完成（路径：{onnx_path}，运行设备: {providers[0]}）")

    def _calculate_derivatives(self, signal):
        """
        修复维度错误：适配1维信号输入
        :param signal: 1维数组 (125,)
        :return: first_deriv (125,), second_deriv (125,)
        """
        # 确保信号是1维数组
        signal = np.squeeze(signal)
        if len(signal.shape) != 1 or signal.shape[0] != 125:
            raise ValueError(f"信号维度错误！需为125维，当前shape: {signal.shape}")

        first_deriv = np.zeros_like(signal)
        second_deriv = np.zeros_like(signal)

        # 一阶导数（1维索引）
        first_deriv[1:-1] = (signal[2:] - signal[:-2]) / 2
        first_deriv[0] = signal[1] - signal[0]
        first_deriv[-1] = signal[-1] - signal[-2]

        # 二阶导数（1维索引）
        second_deriv[1:-1] = (first_deriv[2:] - first_deriv[:-2]) / 2
        second_deriv[0] = first_deriv[1] - first_deriv[0]
        second_deriv[-1] = first_deriv[-1] - first_deriv[-2]

        return first_deriv, second_deriv

    def preprocess_single_sample(self, raw_signal, clinical_features):
        """
        预处理单个样本（修复维度错误）
        :param raw_signal: 1维数组 (125,)
        :param clinical_features: 列表 [age, gender, weight, height, BSA, BMI, HR]
        """
        # 1. 生成3通道信号（125, 3）
        first_deriv, second_deriv = self._calculate_derivatives(raw_signal)
        signal_3d = np.stack([raw_signal, first_deriv, second_deriv], axis=-1)  # (125, 3)

        # 2. 信号标准化（使用指定路径的scaler）
        # scaler期望输入是 (n_samples, n_features)，所以先reshape为(1, 375)（125*3）
        signal_3d_flat = signal_3d.reshape(1, -1)  # (1, 375)
        signal_3d_std_flat = self.signal_scaler.transform(signal_3d_flat)
        signal_3d_std = signal_3d_std_flat.reshape(1, 125, 3)  # (1, 125, 3)

        # 3. 临床特征格式整理
        clinical_arr = np.array(clinical_features).reshape(7, 1)  # (7,1)

        # 4. 整理模型输入
        inputs = [
            signal_3d_std,  # 信号输入 (1, 125, 3)
            np.array([[clinical_arr[0][0]]]),  # age (1,1)
            np.array([[clinical_arr[1][0]]]),  # gender (1,1)
            np.array([[clinical_arr[2][0]]]),  # weight (1,1)
            np.array([[clinical_arr[3][0]]]),  # height (1,1)
            np.array([[clinical_arr[4][0]]]),  # bsa (1,1)
            np.array([[clinical_arr[5][0]]]),  # bmi (1,1)
            np.array([[clinical_arr[6][0]]])  # hr (1,1)
        ]

        return inputs

    def infer_single(self, raw_signal, clinical_features):
        """单样本推理"""
        if self.model is None:
            raise RuntimeError("模型未加载！请先调用 load_keras_model() 或 load_onnx_model()")

        # 预处理
        inputs = self.preprocess_single_sample(raw_signal, clinical_features)

        # 推理
        if self.model_type == "keras":
            pred_sv = self.model.predict(inputs, verbose=0)[0][0]
        elif self.model_type == "onnx":
            onnx_inputs = {
                "signal_input": inputs[0].astype(np.float32),
                "age_input": inputs[1].astype(np.float32),
                "gender_input": inputs[2].astype(np.float32),
                "weight_input": inputs[3].astype(np.float32),
                "height_input": inputs[4].astype(np.float32),
                "bsa_input": inputs[5].astype(np.float32),
                "bmi_input": inputs[6].astype(np.float32),
                "hr_input": inputs[7].astype(np.float32)
            }
            pred_sv = self.model.run(None, onnx_inputs)[0][0][0]

        # 计算CO
        hr = clinical_features[6]
        pred_co = (pred_sv * hr) / 1000

        return {
            "sv": round(float(pred_sv), 2),
            "co": round(float(pred_co), 2)
        }


# ================== 加载测试集并推理 ==================
def load_test_data(test_csv_path, x_test_path):
    """加载测试集数据（指定具体路径）"""
    # 1. 加载test_set.csv
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"测试集CSV不存在: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    logging.info(f"✅ 加载测试集CSV，共 {len(test_df)} 个样本")

    # 2. 加载X_test.npy
    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"测试集信号文件不存在: {x_test_path}")
    X_test = np.load(x_test_path)
    logging.info(f"✅ 加载测试集信号，形状: {X_test.shape}")

    # 3. 校验样本数量一致性
    if len(test_df) != len(X_test):
        raise ValueError(f"样本数量不匹配！CSV: {len(test_df)}, 信号: {len(X_test)}")

    # 4. 校验信号维度
    if X_test.shape[1:] != (125, 3):
        raise ValueError(f"信号维度错误！需为(125,3)，当前: {X_test.shape[1:]}")

    return test_df, X_test


def run_test_set_inference(scaler_path, model_dir, test_csv_path, x_test_path):
    """运行测试集推理并对比结果"""
    # 1. 初始化推理模型（指定标准化器路径）
    infer_model = SVCOInferenceModel(
        scaler_path=scaler_path,
        model_dir=model_dir
    )

    # 2. 加载模型（二选一）
    infer_model.load_keras_model()  # 优先测试Keras模型
    # infer_model.load_onnx_model()  # 可选：测试ONNX模型

    # 3. 加载测试集数据
    test_df, X_test = load_test_data(test_csv_path, x_test_path)

    # 4. 批量推理
    infer_sv_list = []
    infer_co_list = []
    logging.info("\n开始测试集推理...")

    for idx in range(len(test_df)):
        # 提取单个样本数据（1维原始信号）
        raw_signal = X_test[idx, :, 0]  # 原始信号（第0通道），shape=(125,)
        clinical_features = [
            test_df.iloc[idx]['age'],
            test_df.iloc[idx]['gender'],
            test_df.iloc[idx]['weight'],
            test_df.iloc[idx]['height'],
            test_df.iloc[idx]['BSA'],
            test_df.iloc[idx]['BMI'],
            test_df.iloc[idx]['HR']
        ]

        # 推理
        result = infer_model.infer_single(raw_signal, clinical_features)
        infer_sv_list.append(result['sv'])
        infer_co_list.append(result['co'])

        # 进度打印
        if (idx + 1) % 50 == 0:
            logging.info(f"进度: {idx + 1}/{len(test_df)} 样本已推理")

    # 5. 整理结果并对比
    test_df['infer_sv'] = infer_sv_list
    test_df['infer_co'] = infer_co_list

    # 计算推理结果与训练时预测值的差异
    test_df['sv_diff'] = abs(test_df['infer_sv'] - test_df['pred_sv'])
    test_df['co_diff'] = abs(test_df['infer_co'] - test_df['pred_co'])

    # 统计差异
    avg_sv_diff = test_df['sv_diff'].mean()
    avg_co_diff = test_df['co_diff'].mean()
    max_sv_diff = test_df['sv_diff'].max()
    max_co_diff = test_df['co_diff'].max()

    logging.info("\n================ 推理结果对比 ================")
    logging.info(f"训练时预测SV vs 推理模型SV：平均绝对差 = {avg_sv_diff:.2f} mL，最大绝对差 = {max_sv_diff:.2f} mL")
    logging.info(
        f"训练时预测CO vs 推理模型CO：平均绝对差 = {avg_co_diff:.2f} L/min，最大绝对差 = {max_co_diff:.2f} L/min")

    # 6. 保存推理结果
    result_save_path = os.path.join(model_dir, "test_set_infer_result.csv")
    test_df.to_csv(result_save_path, index=False)
    logging.info(f"\n✅ 推理结果已保存至：{result_save_path}")

    # 7. 打印前5个样本的对比示例
    logging.info("\n前5个样本对比示例：")
    compare_cols = ['true_sv', 'pred_sv', 'infer_sv', 'true_co', 'pred_co', 'infer_co']
    print(test_df[compare_cols].head())


# ================== 主函数（配置所有路径） ==================
if __name__ == "__main__":
    # ================== 核心路径配置（修正后） ==================
    # 1. 标准化器路径（你指定的路径）
    SCALER_PATH = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_LSTM_SE/20251221_210322/datasets/signal_scaler.pkl"

    # 2. 模型目录（包含keras/onnx模型文件）
    MODEL_DIR = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_LSTM_SE/20251221_210322"

    # 3. 测试集CSV路径（和信号同目录）
    TEST_CSV_PATH = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_LSTM_SE/20251221_210322/datasets/test_set.csv"

    # 4. 测试集信号文件路径（和CSV同目录）
    X_TEST_PATH = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_LSTM_SE/20251221_210322/datasets/X_test.npy"

    # ================== 执行推理 ==================
    try:
        run_test_set_inference(
            scaler_path=SCALER_PATH,
            model_dir=MODEL_DIR,
            test_csv_path=TEST_CSV_PATH,
            x_test_path=X_TEST_PATH
        )
    except Exception as e:
        logging.error(f"推理过程出错: {str(e)}")
        raise