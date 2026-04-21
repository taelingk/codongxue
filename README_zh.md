# SVCO 心输出量估计项目

本仓库是基于原始研究代码 `D:\乱七八糟\董雪毕设\pythonProject` 清理后整理的代码包。

## 项目简介

本项目主要关注从脉搏波 / 类似 PPG 的信号中估计每搏输出量（`SV`，Stroke Volume）和心输出量（`CO`，Cardiac Output）。代码库包含以下内容：

- 信号预处理脚本
- 数据切片生成及结构化数据集准备
- 基于 TensorFlow 的模型训练代码及 ONNX 格式导出
- ONNX 模型推理代码
- 基于 PyQt5 的桌面监控界面
- 实验用 Notebook 及旧版归档脚本

## 仓库结构

```text
.
|-- archive/
|   `-- legacy_scripts/        # 仅供对照与追溯保留的历史脚本
|-- data/
|   `-- sample/                # 少量示例数据
|-- docs/                      # 相关文档
|-- notebooks/                 # 实验分析与数据探索
|-- packaging/                 # 打包相关文件
`-- src/
    |-- cpp/
    |-- gui/
    |-- inference/
    |-- preprocessing/
    |-- tools/
    `-- training/
```

## 主要代码入口

- `src/preprocessing/data_step1.py`: 原始信号预处理
- `src/preprocessing/data_step2.py`: 波谷检测及切片生成
- `src/training/train_svco_model.py`: 模型训练及导出 ONNX 模型
- `src/inference/infer_svco_onnx.py`: 单样本/测试集 ONNX 推理
- `src/gui/svco_monitor_gui.py`: PyQt5 监控应用程序
- `packaging/svco.spec`: PyInstaller 打包配置文件

## 环境配置

推荐 Python 版本：`3.10`

安装项目依赖：

```bash
pip install -r requirements.txt
```

## 运行前注意事项

1. **未包含模型权重**：本仓库中未上传已训练好的模型权重文件。
2. **硬编码路径**：部分脚本可能仍残留原始开发环境（Windows）中的硬编码绝对路径，需要手动修改。
3. **命名与编码**：由于保留了原项目的历史痕迹，部分源文件可能使用了旧的命名习惯或混合的文本编码格式。
4. **归档脚本**：`archive/legacy_scripts/` 文件夹中的内容仅作存档之用，不建议直接运行。

## 建议阅读顺序

如果您是第一次接触该项目，建议按以下顺序阅读代码和文档：

1. `docs/repository-structure.md`
2. `src/preprocessing/data_step1.py`
3. `src/preprocessing/data_step2.py`
4. `src/training/train_svco_model.py`
5. `src/inference/infer_svco_onnx.py`
6. `src/gui/svco_monitor_gui.py`

## 数据与隐私说明

本仓库仅保留了一个脱敏后的小型示例 CSV 数据以供测试。未包含任何真实的临床数据、训练好的模型成品、打包好的可执行程序、IDE 元数据或相关的学位论文文档。
