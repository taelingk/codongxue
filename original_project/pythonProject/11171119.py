#xunlianmoxing
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense, \
    Dropout, LSTM, Multiply, Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import tf2onnx  # 新增：导入tf2onnx库
import onnx  # 新增：导入onnx库

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('final_model_training.log'), logging.StreamHandler()]
)

# Set font
plt.rcParams["font.family"] = ["Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


# ================== Basic Utility Functions ==================
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def read_csv_with_encoding(file_path):
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
    """Calculate first and second derivatives of the signal"""
    if len(signal.shape) == 3:
        signal = signal.reshape(signal.shape[0], signal.shape[1])

    first_deriv = np.zeros_like(signal)
    second_deriv = np.zeros_like(signal)

    # First derivative
    first_deriv[:, 1:-1] = (signal[:, 2:] - signal[:, :-2]) / 2
    first_deriv[:, 0] = signal[:, 1] - signal[:, 0]
    first_deriv[:, -1] = signal[:, -1] - signal[:, -2]

    # Second derivative
    second_deriv[:, 1:-1] = (first_deriv[:, 2:] - first_deriv[:, :-2]) / 2
    second_deriv[:, 0] = first_deriv[:, 1] - first_deriv[:, 0]
    second_deriv[:, -1] = first_deriv[:, -1] - first_deriv[:, -2]

    return first_deriv, second_deriv


def augment_signal(signal, prob=0.3):
    """修复信号增强函数中的维度不匹配问题"""
    augmented = signal.copy()
    # 1. 随机时间偏移（±2个点）
    if np.random.random() < prob:
        shift = np.random.randint(-2, 3)
        augmented = np.roll(augmented, shift, axis=0)

        if shift > 0:
            augmented[:shift, :] = augmented[shift:shift + 1, :]
        elif shift < 0:
            augmented[shift:, :] = augmented[shift - 1:shift, :]

    # 2. 加轻微高斯噪声（噪声幅度为信号标准差的5%）
    if np.random.random() < prob:
        noise = np.random.normal(0, 0.05 * np.std(augmented, axis=0, keepdims=True), size=augmented.shape)
        augmented += noise

    return augmented


def extract_data(df):
    """Extract features and target values"""
    required_cols = ['time', 'age', 'gender', 'weight', 'height', 'BSA', 'BMI', 'HR', 'SV']
    signal_cols = [f'signal_{i}' for i in range(1, 126)]
    required_cols.extend(signal_cols)

    df = df[required_cols].copy()
    df = df.dropna(subset=['SV'])
    df = df.fillna(df.mean(numeric_only=True))

    time = df['time'].values
    indices = df.index.values

    age = df['age'].values
    gender = df['gender'].values
    weight = df['weight'].values
    height = df['height'].values
    bsa = df['BSA'].values
    bmi = df['BMI'].values
    hr = df['HR'].values

    raw_signal = df[signal_cols].values
    first_deriv, second_deriv = calculate_derivatives(raw_signal)

    # 3-channel input (num_samples, 125, 3)
    X = np.stack([raw_signal, first_deriv, second_deriv], axis=-1)

    y_sv = df['SV'].values
    y_co = (y_sv * hr) / 1000
    y = np.column_stack((y_sv, y_co))

    logging.info(f"Feature extraction completed - Input feature shape: {X.shape}")
    return X, y, age, gender, weight, height, bsa, bmi, hr, time, indices, df


def split_data_by_time_groups(X, y, age, gender, weight, height, bsa, bmi, hr, sv, time, indices,
                              val_ratio=0.2, test_ratio=0.2, random_state=42):
    """Split dataset by time groups"""
    unique_times = np.unique(time)
    n_groups = len(unique_times)
    logging.info(f"Detected {n_groups} time groups")

    if n_groups < 3:
        raise ValueError(f"Too few time groups ({n_groups} groups), at least 3 groups required")

    all_indices = np.arange(n_groups)
    total_test_groups = int(np.ceil(n_groups * test_ratio))
    total_val_groups = int(np.ceil((n_groups - total_test_groups) * val_ratio))

    if total_test_groups + total_val_groups >= n_groups:
        logging.warning("Insufficient groups, adjusting split ratio")
        total_test_groups = max(1, int(n_groups * 0.2))
        total_val_groups = max(1, int((n_groups - total_test_groups) * 0.2))

    from sklearn.model_selection import train_test_split
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=total_test_groups, random_state=random_state
    )
    train_indices, val_indices = train_test_split(
        train_val_indices, test_size=total_val_groups, random_state=random_state
    )

    train_times = unique_times[train_indices]
    val_times = unique_times[val_indices]
    test_times = unique_times[test_indices]

    train_mask = np.isin(time, train_times)
    val_mask = np.isin(time, val_times)
    test_mask = np.isin(time, test_times)

    def extract(mask):
        return (
            X[mask], y[mask], age[mask], gender[mask], weight[mask],
            height[mask], bsa[mask], bmi[mask], hr[mask], time[mask], indices[mask]
        )

    X_train, y_train, age_train, gender_train, weight_train, height_train, bsa_train, bmi_train, hr_train, time_train, indices_train = extract(
        train_mask)
    X_val, y_val, age_val, gender_val, weight_val, height_val, bsa_val, bmi_val, hr_val, time_val, indices_val = extract(
        val_mask)
    X_test, y_test, age_test, gender_test, weight_test, height_test, bsa_test, bmi_test, hr_test, time_test, indices_test = extract(
        test_mask)

    logging.info(
        f"Dataset split completed - Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples, Test set: {len(X_test)} samples")
    logging.info(
        f"Training set contains {len(train_indices)} time groups, Validation set contains {len(val_indices)} time groups, Test set contains {len(test_indices)} time groups")

    return (X_train, X_val, X_test, y_train, y_val, y_test,
            age_train, age_val, age_test, gender_train, gender_val, gender_test,
            weight_train, weight_val, weight_test, height_train, height_val, height_test,
            bsa_train, bsa_val, bsa_test, bmi_train, bmi_val, bmi_test,
            hr_train, hr_val, hr_test, time_train, time_val, time_test,
            indices_train, indices_val, indices_test)


def standardize_features(train, val, test, is_2d=False):
    """Standardize features"""
    scaler = StandardScaler()
    if is_2d:
        train_reshaped = train.reshape(train.shape[0], -1)
        scaler.fit(train_reshaped)
        return (
            scaler.transform(train_reshaped).reshape(train.shape),
            scaler.transform(val.reshape(val.shape[0], -1)).reshape(val.shape),
            scaler.transform(test.reshape(test.shape[0], -1)).reshape(test.shape),
            scaler
        )
    else:
        train_reshaped = train.reshape(-1, 1)
        scaler.fit(train_reshaped)
        return (
            scaler.transform(train_reshaped).flatten(),
            scaler.transform(val.reshape(-1, 1)).flatten(),
            scaler.transform(test.reshape(-1, 1)).flatten(),
            scaler
        )


# ================== 新增：生成预测结果的函数 ==================
def generate_predictions(model, X_std, age_std, gender_std, weight_std, height_std, bsa_std, bmi_std, hr_std,
                         hr_original):
    """
    生成SV（模型预测）和CO（基于SV预测值+原始HR计算）的预测结果
    """
    # 预测SV（模型输出）
    pred_sv = model.predict(
        [X_std, age_std, gender_std, weight_std, height_std, bsa_std, bmi_std, hr_std],
        verbose=0
    ).flatten()

    # 计算CO（CO = SV * HR / 1000，HR用原始值避免标准化误差）
    pred_co = (pred_sv * hr_original) / 1000

    return pred_sv, pred_co


# ================== 修改：保存数据集（新增预测结果列） ==================
def save_datasets_with_predictions(
        save_dir,
        # 原始数据集特征
        X_train, X_val, X_test, y_train, y_val, y_test,
        age_train, age_val, age_test, gender_train, gender_val, gender_test,
        weight_train, weight_val, weight_test, height_train, height_val, height_test,
        bsa_train, bsa_val, bsa_test, bmi_train, bmi_val, bmi_test,
        hr_train, hr_val, hr_test, time_train, time_val, time_test,
        indices_train, indices_val, indices_test,
        # 预测结果
        pred_sv_train=None, pred_co_train=None,
        pred_sv_val=None, pred_co_val=None,
        pred_sv_test=None, pred_co_test=None
):
    """
    保存训练/验证/测试集，包含原始特征+真实值+预测值
    """
    data_dir = os.path.join(save_dir, "datasets")
    os.makedirs(data_dir, exist_ok=True)

    # 训练集
    train_df = pd.DataFrame({
        # 原始临床特征
        'age': age_train, 'gender': gender_train, 'weight': weight_train,
        'height': height_train, 'BSA': bsa_train, 'BMI': bmi_train, 'HR': hr_train,
        # 时间与原始索引
        'time': time_train, 'original_index': indices_train,
        # 真实值（SV和CO）
        'true_sv': y_train[:, 0], 'true_co': y_train[:, 1]
    })
    # 添加训练集预测结果（如果提供）
    if pred_sv_train is not None and pred_co_train is not None:
        train_df['pred_sv'] = pred_sv_train
        train_df['pred_co'] = pred_co_train
    train_df.to_csv(os.path.join(data_dir, "train_set.csv"), index=False)

    # 验证集
    val_df = pd.DataFrame({
        'age': age_val, 'gender': gender_val, 'weight': weight_val,
        'height': height_val, 'BSA': bsa_val, 'BMI': bmi_val, 'HR': hr_val,
        'time': time_val, 'original_index': indices_val,
        'true_sv': y_val[:, 0], 'true_co': y_val[:, 1],
        # 验证集预测结果
        'pred_sv': pred_sv_val, 'pred_co': pred_co_val
    })
    val_df.to_csv(os.path.join(data_dir, "val_set.csv"), index=False)

    # 测试集
    test_df = pd.DataFrame({
        'age': age_test, 'gender': gender_test, 'weight': weight_test,
        'height': height_test, 'BSA': bsa_test, 'BMI': bmi_test, 'HR': hr_test,
        'time': time_test, 'original_index': indices_test,
        'true_sv': y_test[:, 0], 'true_co': y_test[:, 1],
        # 测试集预测结果
        'pred_sv': pred_sv_test, 'pred_co': pred_co_test
    })
    test_df.to_csv(os.path.join(data_dir, "test_set.csv"), index=False)

    # 保存原始信号特征
    np.save(os.path.join(data_dir, "X_train.npy"), X_train)
    np.save(os.path.join(data_dir, "X_val.npy"), X_val)
    np.save(os.path.join(data_dir, "X_test.npy"), X_test)

    logging.info(f"Datasets with predictions saved to: {data_dir}")


# ================== 新增：保存ONNX模型函数 ==================
def save_model_to_onnx(model, save_dir, model_name="resnet18_lstm_se_model.onnx"):
    """
    将TensorFlow模型转换并保存为ONNX格式
    """
    try:
        # 构建输入签名（匹配模型输入）
        input_signatures = [
            tf.TensorSpec((1, 125, 3), tf.float32, name="signal_input"),
            tf.TensorSpec((1, 1), tf.float32, name="age_input"),
            tf.TensorSpec((1, 1), tf.float32, name="gender_input"),
            tf.TensorSpec((1, 1), tf.float32, name="weight_input"),
            tf.TensorSpec((1, 1), tf.float32, name="height_input"),
            tf.TensorSpec((1, 1), tf.float32, name="bsa_input"),
            tf.TensorSpec((1, 1), tf.float32, name="bmi_input"),
            tf.TensorSpec((1, 1), tf.float32, name="hr_input")
        ]

        # 转换模型为ONNX格式
        onnx_model, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=input_signatures,
            opset=13,  # ONNX算子集版本
            output_path=os.path.join(save_dir, model_name)
        )

        # 验证ONNX模型
        onnx.checker.check_model(onnx_model)
        logging.info(f"ONNX模型已成功保存至：{os.path.join(save_dir, model_name)}")
        return True
    except Exception as e:
        logging.error(f"保存ONNX模型失败：{str(e)}")
        return False


# ================== SE Block Definition ==================
class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Block for channel attention"""

    def __init__(self, reduction_ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.squeeze = GlobalAveragePooling1D()
        self.excitation1 = Dense(channels // self.reduction_ratio, activation='relu')
        self.excitation2 = Dense(channels, activation='sigmoid')
        super().build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation1(x)
        x = self.excitation2(x)
        x = Reshape((1, -1))(x)
        return Multiply()([inputs, x])

    def get_config(self):
        config = super().get_config()
        config.update({
            'reduction_ratio': self.reduction_ratio,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ================== Model Definition with LSTM and SE ==================
class ResidualSEBlock(tf.keras.layers.Layer):
    """Residual block with SE attention mechanism"""

    def __init__(self, filters, stride=1, use_1x1_conv=False, kernel_regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.use_1x1_conv = use_1x1_conv
        self.kernel_regularizer = kernel_regularizer


        self.conv1 = Conv1D(filters, 3, padding='same', strides=stride, kernel_regularizer=kernel_regularizer)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv1D(filters, 3, padding='same', kernel_regularizer=kernel_regularizer)
        self.bn2 = BatchNormalization()
        self.se = SEBlock()

        if use_1x1_conv:
            self.shortcut_conv = Conv1D(filters, 1, strides=stride, kernel_regularizer=kernel_regularizer)
            self.shortcut_bn = BatchNormalization()
        self.use_1x1_conv = use_1x1_conv
        self.act2 = Activation('relu')

    def call(self, inputs):
        x = self.act1(self.bn1(self.conv1(inputs)))
        x = self.bn2(self.conv2(x))
        x = self.se(x)

        if self.use_1x1_conv:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
        else:
            shortcut = inputs
        return self.act2(x + shortcut)



#加上get_config
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'stride': self.stride,
            'use_1x1_conv': self.use_1x1_conv,
            'kernel_regularizer': tf.keras.regularizers.serialize(
                self.kernel_regularizer) if self.kernel_regularizer else None,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 反序列化 kernel_regularizer
        kernel_regularizer_config = config.pop('kernel_regularizer', None)
        if kernel_regularizer_config:
            config['kernel_regularizer'] = tf.keras.regularizers.deserialize(kernel_regularizer_config)
        return cls(**config)




#build


def build_resnet18_with_lstm_se(input_shape):
    """Build ResNet18 model with LSTM and SE modules"""
    best_hp = {
        "lr": 0.0008,
        "l2_reg": 4.114391549630821e-05,
        "dropout": 0.3,
        "dense_units": 128,
        "lstm_units": 64
    }

    regularizer = l2(best_hp["l2_reg"]) if best_hp["l2_reg"] > 0 else None

    inputs = Input(shape=input_shape, name="signal_input")

    # 先卷积提取局部特征
    x = Conv1D(64, 7, strides=2, padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 再用LSTM捕捉时序依赖
    x_lstm = LSTM(best_hp["lstm_units"], return_sequences=True, kernel_regularizer=regularizer)(x)

    # ResNet18 with SE blocks
    for _ in range(2):
        x = ResidualSEBlock(64, kernel_regularizer=regularizer)(x_lstm)

    x = ResidualSEBlock(128, stride=2, use_1x1_conv=True, kernel_regularizer=regularizer)(x)
    for _ in range(1):
        x = ResidualSEBlock(128, kernel_regularizer=regularizer)(x)

    x = ResidualSEBlock(256, stride=2, use_1x1_conv=True, kernel_regularizer=regularizer)(x)
    for _ in range(1):
        x = ResidualSEBlock(256, kernel_regularizer=regularizer)(x)

    x = ResidualSEBlock(512, stride=2, use_1x1_conv=True, kernel_regularizer=regularizer)(x)
    for _ in range(1):
        x = ResidualSEBlock(512, kernel_regularizer=regularizer)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(best_hp["dense_units"], activation='relu', kernel_regularizer=regularizer)(x)
    x = Dropout(best_hp["dropout"])(x)

    # Clinical feature inputs
    age_in = Input(shape=(1,), name="age_input")
    gender_in = Input(shape=(1,), name="gender_input")
    weight_in = Input(shape=(1,), name="weight_input")
    height_in = Input(shape=(1,), name="height_input")
    bsa_in = Input(shape=(1,), name="bsa_input")
    bmi_in = Input(shape=(1,), name="bmi_input")
    hr_in = Input(shape=(1,), name="hr_input")

    x = tf.keras.layers.Concatenate()([x, age_in, gender_in, weight_in, height_in, bsa_in, bmi_in, hr_in])
    outputs = Dense(1, name="sv_output")(x)

    model = Model(
        inputs=[inputs, age_in, gender_in, weight_in, height_in, bsa_in, bmi_in, hr_in],
        outputs=outputs
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=best_hp["lr"],
            clipvalue=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

    return model, best_hp


# ================== Model Evaluation ==================
def evaluate_final_model(model, X, y_true, hr, time_groups, dataset_name, save_dir):
    """Evaluate model and save results"""
    y_pred_sv = model.predict(X, verbose=0).flatten()

    if len(y_pred_sv) != len(y_true[:, 0]):
        min_length = min(len(y_pred_sv), len(y_true[:, 0]))
        y_pred_sv = y_pred_sv[:min_length]
        y_true = y_true[:min_length]
        hr = hr[:min_length]
        time_groups = time_groups[:min_length]
        logging.warning(f"Prediction and true value lengths do not match, truncated to {min_length} samples")

    y_pred_co = (y_pred_sv * hr) / 1000
    y_true_sv = y_true[:, 0]
    y_true_co = y_true[:, 1]

    def calculate_metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        corr, p_val = pearsonr(y_true, y_pred)

        logging.info(f"\n{name} evaluation metrics:")
        logging.info(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        logging.info(f"R²: {r2:.4f} | Correlation coefficient: {corr:.4f} (p={p_val:.4e})")

        return {
            "mse": mse, "rmse": rmse, "mae": mae,
            "r2": r2, "corr": corr, "p_value": p_val,
            "y_true": y_true, "y_pred": y_pred
        }

    sv_metrics = calculate_metrics(y_true_sv, y_pred_sv, f"{dataset_name} - SV")
    co_metrics = calculate_metrics(y_true_co, y_pred_co, f"{dataset_name} - CO (calculated)")

    # 误差分布分析
    plt.figure(figsize=(12, 5))
    for i, (item, title) in enumerate([
        (sv_metrics, f"{dataset_name} SV Error Distribution"),
        (co_metrics, f"{dataset_name} CO Error Distribution")
    ]):
        plt.subplot(1, 2, i + 1)
        error = item["y_pred"] - item["y_true"]
        plt.hist(error, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel("Prediction Error")
        plt.ylabel("Count")
        plt.title(title)
        plt.grid(ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name}_error_distribution.png", dpi=300)
    plt.close()

    # 预测值vs真实值
    plt.figure(figsize=(12, 5))
    for i, (item, title) in enumerate([
        (sv_metrics, f"{dataset_name} SV Predictions vs True Values (R²={sv_metrics['r2']:.2f})"),
        (co_metrics, f"{dataset_name} CO Predictions vs True Values (R²={co_metrics['r2']:.2f})")
    ]):
        plt.subplot(1, 2, i + 1)
        true_vals = item["y_true"]
        pred_vals = item["y_pred"]
        min_len = min(len(true_vals), len(pred_vals))

        plt.scatter(true_vals[:min_len], pred_vals[:min_len], alpha=0.6, s=30)
        min_val = min(true_vals[:min_len].min(), pred_vals[:min_len].min())
        max_val = max(true_vals[:min_len].max(), pred_vals[:min_len].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.title(title)
        plt.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{dataset_name}_pred_vs_true.png", dpi=300)
    plt.close()

    # 按时间组分析
    time_groups_unique = np.unique(time_groups)
    group_metrics = []
    for t in time_groups_unique:
        mask = time_groups == t
        if np.sum(mask) < 5:
            continue
        group_r2 = r2_score(y_true_sv[mask], y_pred_sv[mask])
        group_metrics.append({"time_group": t, "sample_count": np.sum(mask), "r2": group_r2})

    if group_metrics:
        pd.DataFrame(group_metrics).to_csv(f"{save_dir}/{dataset_name}_time_group_metrics.csv", index=False)
        logging.info(f"Saved time group metrics for {dataset_name} with {len(group_metrics)} groups")

    if dataset_name == "Test set":
        analyze_grouped_predictions(y_true_sv, y_pred_sv, y_true_co, y_pred_co, save_dir)

    return {"SV": sv_metrics, "CO": co_metrics}


def analyze_grouped_predictions(y_true_sv, y_pred_sv, y_true_co, y_pred_co, save_dir):
    """Grouped analysis for test set"""
    df_sv = pd.DataFrame({'true_value': y_true_sv, 'pred_value': y_pred_sv})
    df_co = pd.DataFrame({'true_value': y_true_co, 'pred_value': y_pred_co})

    grouped_sv = df_sv.groupby(df_sv['true_value'].round(1)).agg({'pred_value': 'mean'}).reset_index()
    grouped_co = df_co.groupby(df_co['true_value'].round(1)).agg({'pred_value': 'mean'}).reset_index()

    def calculate_grouped_metrics(grouped_df, name):
        mse = mean_squared_error(grouped_df['true_value'], grouped_df['pred_value'])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(grouped_df['true_value'], grouped_df['pred_value'])
        r2 = r2_score(grouped_df['true_value'], grouped_df['pred_value'])
        corr, p_val = pearsonr(grouped_df['true_value'], grouped_df['pred_value'])

        logging.info(f"\nGrouped {name} evaluation metrics:")
        logging.info(f"MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        logging.info(f"R²: {r2:.4f} | Correlation coefficient: {corr:.4f} (p={p_val:.4e})")

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "corr": corr, "p_value": p_val,
                "grouped_data": grouped_df}

    grouped_sv_metrics = calculate_grouped_metrics(grouped_sv, "SV")
    grouped_co_metrics = calculate_grouped_metrics(grouped_co, "CO")

    grouped_sv.to_csv(f"{save_dir}/test_grouped_sv.csv", index=False)
    grouped_co.to_csv(f"{save_dir}/test_grouped_co.csv", index=False)

    plt.figure(figsize=(12, 5))
    for i, (grouped_data, title, metric) in enumerate([
        (grouped_sv, f"Test Set Grouped SV Prediction Mean vs True Values (R²={grouped_sv_metrics['r2']:.2f})",
         grouped_sv_metrics),
        (grouped_co, f"Test Set Grouped CO Prediction Mean vs True Values (R²={grouped_co_metrics['r2']:.2f})",
         grouped_co_metrics)
    ]):
        plt.subplot(1, 2, i + 1)
        plt.scatter(grouped_data['true_value'], grouped_data['pred_value'], alpha=0.7, s=50, c='green')
        min_val = min(grouped_data['true_value'].min(), grouped_data['pred_value'].min())
        max_val = max(grouped_data['true_value'].max(), grouped_data['pred_value'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        plt.text(0.05, 0.95,
                 f"r = {metric['corr']:.3f}\nn = {len(grouped_data)}",
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.xlabel("True Value")
        plt.ylabel("Mean Predicted Value")
        plt.title(title)
        plt.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/test_grouped_pred_vs_true.png", dpi=300)
    plt.close()


# ================== Main Function ==================
def main():
    set_seed()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/resnet_LSTM_SE/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"All results will be saved to: {save_dir}")

    # 1. 读取数据
    file_path = "D:/心输出量项目/CNAP脉氧处理数据/脉氧+CNAP/总711-1112合并后的血压数据.csv"
    df = read_csv_with_encoding(file_path)
    if df is None or len(df) == 0:
        logging.error("Could not read valid data, program terminated")
        return

    # 2. 提取特征
    try:
        X, y, age, gender, weight, height, bsa, bmi, hr, time, indices, original_df = extract_data(df)
    except Exception as e:
        logging.error(f"Data extraction failed: {e}")
        return

    # 异常值处理
    sv_values = y[:, 0]
    q1, q3 = np.percentile(sv_values, [25, 75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    valid_mask = sv_values <= upper_bound

    X = X[valid_mask]
    y = y[valid_mask]
    age = age[valid_mask]
    gender = gender[valid_mask]
    weight = weight[valid_mask]
    height = height[valid_mask]
    bsa = bsa[valid_mask]
    bmi = bmi[valid_mask]
    hr = hr[valid_mask]
    time = time[valid_mask]
    indices = indices[valid_mask]
    original_df = original_df[valid_mask].copy()

    removed_count = len(valid_mask) - sum(valid_mask)
    logging.info(f"使用IQR法检测异常值，上界={upper_bound:.2f}，移除{removed_count}个样本，剩余有效样本: {len(X)}")

    if len(X) < 100:
        logging.error(f"Insufficient data volume after filtering (only {len(X)} samples), cannot train model")
        return

    # 3. 划分数据集
    try:
        split_result = split_data_by_time_groups(
            X, y, age, gender, weight, height, bsa, bmi, hr, y[:, 0], time, indices
        )
        (X_train, X_val, X_test, y_train, y_val, y_test,
         age_train, age_val, age_test, gender_train, gender_val, gender_test,
         weight_train, weight_val, weight_test, height_train, height_val, height_test,
         bsa_train, bsa_val, bsa_test, bmi_train, bmi_val, bmi_test,
         hr_train, hr_val, hr_test, time_train, time_val, time_test,
         indices_train, indices_val, indices_test) = split_result
    except Exception as e:
        logging.error(f"Dataset splitting failed: {e}")
        return

    # 训练集数据增强
    X_train_augmented = np.array([augment_signal(x) for x in X_train])
    logging.info(f"Applied data augmentation to training set: {X_train_augmented.shape}")

    # 4. 标准化特征
    logging.info("Standardizing features...")
    X_train_std, X_val_std, X_test_std, x_scaler = standardize_features(X_train_augmented, X_val, X_test, is_2d=True)

    age_train_std, age_val_std, age_test_std, _ = standardize_features(age_train, age_val, age_test)
    gender_train_std, gender_val_std, gender_test_std = gender_train, gender_val, gender_test
    weight_train_std, weight_val_std, weight_test_std, _ = standardize_features(weight_train, weight_val, weight_test)
    height_train_std, height_val_std, height_test_std, _ = standardize_features(height_train, height_val, height_test)
    bsa_train_std, bsa_val_std, bsa_test_std, _ = standardize_features(bsa_train, bsa_val, bsa_test)
    bmi_train_std, bmi_val_std, bmi_test_std, _ = standardize_features(bmi_train, bmi_val, bmi_test)
    hr_train_std, hr_val_std, hr_test_std, _ = standardize_features(hr_train, hr_val, hr_test)

    joblib.dump(x_scaler, f"{save_dir}/signal_scaler.pkl")
    logging.info("Feature scaler saved")

    # 5. 构建模型
    logging.info("Building ResNet18 model with LSTM and SE modules...")
    model, best_hp = build_resnet18_with_lstm_se(input_shape=(125, 3))
    model.summary(print_fn=lambda x: logging.info(x))

    with open(f"{save_dir}/best_hyperparameters.txt", "w") as f:
        for key, value in best_hp.items():
            f.write(f"{key}: {value}\n")

    # 6. 训练模型
    logging.info("Starting model training...")
    batch_size = 32 if len(X_train) < 1000 else 64
    logging.info(f"Using batch size: {batch_size} based on training set size")

    def cosine_annealing(epoch, lr):
        initial_lr = best_hp["lr"]
        epochs = 50
        return initial_lr * (1 + np.cos(np.pi * epoch / epochs)) / 2

    callbacks = [
        LearningRateScheduler(cosine_annealing, verbose=1)
    ]

    history = model.fit(
        [X_train_std, age_train_std, gender_train_std, weight_train_std,
         height_train_std, bsa_train_std, bmi_train_std, hr_train_std],
        y_train[:, 0],
        epochs=50,
        batch_size=batch_size,
        validation_data=(
            [X_val_std, age_val_std, gender_val_std, weight_val_std,
             height_val_std, bsa_val_std, bmi_val_std, hr_val_std],
            y_val[:, 0]
        ),
        callbacks=callbacks,
        verbose=1
    )

    pd.DataFrame(history.history).to_csv(f"{save_dir}/training_history.csv", index=False)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.legend()
    plt.grid(ls='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(ls='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300)
    plt.close()

    # 7. 生成预测结果
    logging.info("Generating predictions for all datasets...")
    # 训练集预测
    pred_sv_train, pred_co_train = generate_predictions(
        model, X_train_std, age_train_std, gender_train_std, weight_train_std,
        height_train_std, bsa_train_std, bmi_train_std, hr_train_std, hr_train
    )

    # 验证集预测
    pred_sv_val, pred_co_val = generate_predictions(
        model, X_val_std, age_val_std, gender_val_std, weight_val_std,
        height_val_std, bsa_val_std, bmi_val_std, hr_val_std, hr_val
    )

    # 测试集预测
    pred_sv_test, pred_co_test = generate_predictions(
        model, X_test_std, age_test_std, gender_test_std, weight_test_std,
        height_test_std, bsa_test_std, bmi_test_std, hr_test_std, hr_test
    )

    # 8. 保存包含预测结果的数据集
    save_datasets_with_predictions(
        save_dir,
        X_train, X_val, X_test, y_train, y_val, y_test,
        age_train, age_val, age_test, gender_train, gender_val, gender_test,
        weight_train, weight_val, weight_test, height_train, height_val, height_test,
        bsa_train, bsa_val, bsa_test, bmi_train, bmi_val, bmi_test,
        hr_train, hr_val, hr_test, time_train, time_val, time_test,
        indices_train, indices_val, indices_test,
        pred_sv_train, pred_co_train,
        pred_sv_val, pred_co_val,
        pred_sv_test, pred_co_test
    )

    # 9. 评估模型（修复NameError：将评估逻辑移到main函数内部，确保model变量可见）
    logging.info("\n===== Starting model evaluation =====")
    test_data = [X_test_std, age_test_std, gender_test_std, weight_test_std,
                 height_test_std, bsa_test_std, bmi_test_std, hr_test_std]
    val_data = [X_val_std, age_val_std, gender_val_std, weight_val_std,
                height_val_std, bsa_val_std, bmi_val_std, hr_val_std]
    train_data = [X_train_std, age_train_std, gender_train_std, weight_train_std,
                  height_train_std, bsa_train_std, bmi_train_std, hr_train_std]

    evaluate_final_model(model, train_data, y_train, hr_train, time_train, "Training set", save_dir)
    evaluate_final_model(model, val_data, y_val, hr_val, time_val, "Validation set", save_dir)
    test_metrics = evaluate_final_model(model, test_data, y_test, hr_test, time_test, "Test set", save_dir)

    # 10. 保存模型（包含ONNX格式）
    # 保存Keras原生模型
    model.save(f"{save_dir}/resnet18_lstm_se_model.keras")
    logging.info(f"ResNet18 with LSTM and SE model saved to: {save_dir}/resnet18_lstm_se_model")

    # 保存ONNX格式模型
    save_model_to_onnx(model, save_dir, "resnet18_lstm_se_model.onnx")

    logging.info("\nAll processes completed!")


if __name__ == "__main__":
    main()