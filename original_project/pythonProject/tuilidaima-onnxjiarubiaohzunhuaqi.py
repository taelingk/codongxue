# -*- coding: gb18030 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import onnxruntime as ort
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

# ================== 推理模型类（加入 SBP/DBP/PP；无外部标准化器） ==================
class SVCOInferenceModel:
    def __init__(self, model_dir):
        """
        初始化推理模型（不使用外部标准化器）
        :param model_dir: 模型文件所在目录
        """
        self.model_dir = model_dir
        self.model_type = None
        self.model = None

    def load_keras_model(self):
        """加载Keras模型（仅当Keras模型也支持 sbp/dbp/pp 输入时才可用）"""
        model_path = os.path.join(self.model_dir, "resnet_se_lstm_model.keras")
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
        onnx_path = os.path.join(self.model_dir, "resnet_se_lstm_model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX模型不存在: {onnx_path}")

        available = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        if ('CUDAExecutionProvider' in available) and (ort.get_device() == "GPU"):
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        self.model = ort.InferenceSession(onnx_path, providers=providers)
        self.model_type = "onnx"
        logging.info(f"✅ ONNX模型加载完成（路径：{onnx_path}，运行设备: {providers[0]}）")

    def _calculate_derivatives(self, signal_1d):
        """
        适配1维信号输入
        :param signal_1d: 1维数组 (125,)
        :return: first_deriv (125,), second_deriv (125,)
        """
        signal_1d = np.squeeze(signal_1d)
        if signal_1d.ndim != 1 or signal_1d.shape[0] != 125:
            raise ValueError(f"信号维度错误！需为(125,)，当前shape: {signal_1d.shape}")

        first_deriv = np.zeros_like(signal_1d)
        second_deriv = np.zeros_like(signal_1d)

        # 一阶导数
        first_deriv[1:-1] = (signal_1d[2:] - signal_1d[:-2]) / 2.0
        first_deriv[0] = signal_1d[1] - signal_1d[0]
        first_deriv[-1] = signal_1d[-1] - signal_1d[-2]

        # 二阶导数
        second_deriv[1:-1] = (first_deriv[2:] - first_deriv[:-2]) / 2.0
        second_deriv[0] = first_deriv[1] - first_deriv[0]
        second_deriv[-1] = first_deriv[-1] - first_deriv[-2]

        return first_deriv, second_deriv

    def preprocess_single_sample(self, raw_signal, clinical_features):
        """
        预处理单个样本（无外部标准化器）
        raw_signal 支持两种输入：
          - (125,)   ：自动计算一二阶导，拼成(125,3)
          - (125,3)  ：直接作为3通道输入（不再重复计算导数）
        :param clinical_features: 列表 [age, gender, weight, height, BSA, BMI, HR, SBP, DBP, PP]
        """
        raw_signal = np.asarray(raw_signal)

        # 1) 准备(125,3)信号
        if raw_signal.ndim == 2 and raw_signal.shape == (125, 3):
            signal_3d = raw_signal
        elif raw_signal.ndim == 1 and raw_signal.shape[0] == 125:
            first_deriv, second_deriv = self._calculate_derivatives(raw_signal)
            signal_3d = np.stack([raw_signal, first_deriv, second_deriv], axis=-1)  # (125,3)
        else:
            raise ValueError(f"raw_signal 维度必须是(125,)或(125,3)，当前: {raw_signal.shape}")

        # 2) 不做外部 scaler.transform
        signal_input = signal_3d.astype(np.float32).reshape(1, 125, 3)  # (1,125,3)

        # 3) 临床特征：现在是 10 个
        if clinical_features is None or len(clinical_features) != 10:
            raise ValueError("clinical_features 必须是长度为10的列表: [age, gender, weight, height, BSA, BMI, HR, SBP, DBP, PP]")

        cf = np.array(clinical_features, dtype=np.float32).reshape(10, 1)

        inputs_list = [
            signal_input,
            np.array([[cf[0, 0]]], dtype=np.float32),  # age
            np.array([[cf[1, 0]]], dtype=np.float32),  # gender
            np.array([[cf[2, 0]]], dtype=np.float32),  # weight
            np.array([[cf[3, 0]]], dtype=np.float32),  # height
            np.array([[cf[4, 0]]], dtype=np.float32),  # bsa
            np.array([[cf[5, 0]]], dtype=np.float32),  # bmi
            np.array([[cf[6, 0]]], dtype=np.float32),  # hr
            np.array([[cf[7, 0]]], dtype=np.float32),  # sbp
            np.array([[cf[8, 0]]], dtype=np.float32),  # dbp
            np.array([[cf[9, 0]]], dtype=np.float32),  # pp
        ]
        return inputs_list

    def infer_single(self, raw_signal, clinical_features):
        """单样本推理（加入 sbp/dbp/pp 输入）"""
        if self.model is None:
            raise RuntimeError("模型未加载！请先调用 load_keras_model() 或 load_onnx_model()")

        inputs = self.preprocess_single_sample(raw_signal, clinical_features)

        if self.model_type == "keras":
            # 注意：只有当Keras模型的输入也包含 sbp/dbp/pp 时，这里才不会报错
            pred_sv = self.model.predict(inputs, verbose=0)[0][0]

        elif self.model_type == "onnx":
            onnx_inputs = {
                "signal_input": inputs[0],
                "age_input": inputs[1],
                "gender_input": inputs[2],
                "weight_input": inputs[3],
                "height_input": inputs[4],
                "bsa_input": inputs[5],
                "bmi_input": inputs[6],
                "hr_input": inputs[7],
                "sbp_input": inputs[8],
                "dbp_input": inputs[9],
                "pp_input": inputs[10],
            }
            pred_sv = self.model.run(None, onnx_inputs)[0][0][0]
        else:
            raise RuntimeError(f"未知模型类型: {self.model_type}")

        # HR 在 clinical_features[6]
        hr = float(clinical_features[6])
        pred_co = (float(pred_sv) * hr) / 1000.0

        return {"sv": round(float(pred_sv), 2), "co": round(float(pred_co), 2)}

# ================== 加载测试集并推理 ==================
def load_test_data(test_csv_path, x_test_path):
    """加载测试集数据（指定具体路径）"""
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"测试集CSV不存在: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    logging.info(f"✅ 加载测试集CSV，共 {len(test_df)} 个样本")

    if not os.path.exists(x_test_path):
        raise FileNotFoundError(f"测试集信号文件不存在: {x_test_path}")
    X_test = np.load(x_test_path)
    logging.info(f"✅ 加载测试集信号，形状: {X_test.shape}")

    if len(test_df) != len(X_test):
        raise ValueError(f"样本数量不匹配！CSV: {len(test_df)}, 信号: {len(X_test)}")

    if X_test.shape[1:] != (125, 3):
        raise ValueError(f"信号维度错误！需为(N,125,3)，当前: {X_test.shape}")

    return test_df, X_test

def _get_col(row, names, required=True, default=None):
    """兼容不同列名写法"""
    for n in names:
        if n in row.index:
            return row[n]
    if required:
        raise KeyError(f"缺少列 {names}，当前CSV列：{list(row.index)}")
    return default

def run_test_set_inference(model_dir, test_csv_path, x_test_path):
    """运行测试集推理并对比结果（无外部标准化器；加入SBP/DBP/PP）"""
    infer_model = SVCOInferenceModel(model_dir=model_dir)

    # ✅ 默认使用 ONNX
    infer_model.load_onnx_model()
    # infer_model.load_keras_model()

    test_df, X_test = load_test_data(test_csv_path, x_test_path)

    infer_sv_list, infer_co_list = [], []
    logging.info("\n开始测试集推理...")

    for idx in range(len(test_df)):
        raw_signal = X_test[idx]  # shape=(125,3)
        row = test_df.iloc[idx]

        age = _get_col(row, ["age", "Age", "AGE"])
        gender = _get_col(row, ["gender", "Gender", "sex", "Sex", "SEX"])
        weight = _get_col(row, ["weight", "Weight", "WEIGHT"])
        height = _get_col(row, ["height", "Height", "HEIGHT"])
        bsa = _get_col(row, ["BSA", "bsa"])
        bmi = _get_col(row, ["BMI", "bmi"])
        hr = _get_col(row, ["HR", "hr"])

        sbp = _get_col(row, ["SBP", "sbp", "SBP_mmHg", "sbp_mmHg"])
        dbp = _get_col(row, ["DBP", "dbp", "DBP_mmHg", "dbp_mmHg"])
        pp = _get_col(row, ["PP", "pp", "PP_mmHg", "pp_mmHg"], required=False, default=None)
        if pp is None or (isinstance(pp, float) and np.isnan(pp)):
            pp = float(sbp) - float(dbp)

        clinical_features = [age, gender, weight, height, bsa, bmi, hr, sbp, dbp, pp]

        result = infer_model.infer_single(raw_signal, clinical_features)
        infer_sv_list.append(result['sv'])
        infer_co_list.append(result['co'])

        if (idx + 1) % 50 == 0:
            logging.info(f"进度: {idx + 1}/{len(test_df)} 样本已推理")

    test_df['infer_sv'] = infer_sv_list
    test_df['infer_co'] = infer_co_list

    # 对比差异（要求CSV有 pred_sv/pred_co）
    test_df['sv_diff'] = abs(test_df['infer_sv'] - test_df['pred_sv'])
    test_df['co_diff'] = abs(test_df['infer_co'] - test_df['pred_co'])

    avg_sv_diff = test_df['sv_diff'].mean()
    avg_co_diff = test_df['co_diff'].mean()
    max_sv_diff = test_df['sv_diff'].max()
    max_co_diff = test_df['co_diff'].max()

    logging.info("\n================ 推理结果对比 ================")
    logging.info(f"训练时预测SV vs 推理模型SV：平均绝对差 = {avg_sv_diff:.2f} mL，最大绝对差 = {max_sv_diff:.2f} mL")
    logging.info(f"训练时预测CO vs 推理模型CO：平均绝对差 = {avg_co_diff:.2f} L/min，最大绝对差 = {max_co_diff:.2f} L/min")

    result_save_path = os.path.join(model_dir, "test_set_infer_result.csv")
    test_df.to_csv(result_save_path, index=False)
    logging.info(f"\n✅ 推理结果已保存至：{result_save_path}")

    logging.info("\n前5个样本对比示例：")
    compare_cols = ['true_sv', 'pred_sv', 'infer_sv', 'true_co', 'pred_co', 'infer_co']
    print(test_df[compare_cols].head())

# ================== 主函数（配置所有路径） ==================
if __name__ == "__main__":
    MODEL_DIR = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_SE_LSTM/20260208_111846"
    TEST_CSV_PATH = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_SE_LSTM/20260208_111846/datasets/test_set.csv"
    X_TEST_PATH = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_SE_LSTM/20260208_111846/datasets/X_test.npy"

    try:
        run_test_set_inference(
            model_dir=MODEL_DIR,
            test_csv_path=TEST_CSV_PATH,
            x_test_path=X_TEST_PATH
        )
    except Exception as e:
        logging.error(f"推理过程出错: {str(e)}")
        raise