# INVERTER_TRINAPOWER-MASTER 项目文档

## 项目概述
本项目是一个针对逆变器（INVERTER）数据的机器学习全流程工程，涵盖了从数据预处理、模型训练、模型推理到模型可解释性分析的完整链路。

---

## 目录结构与文件说明

### 1. `utils_tools/`
工具与辅助函数模块，提供项目通用的基础功能。

---

### 2. `data_preprocessing/`
数据预处理模块，负责原始数据的清洗、特征工程等工作。

| 文件名 | 功能描述 |
|--------|----------|
| `data_deal.py` | 核心数据处理脚本，负责数据清洗、格式转换、缺失值填充等基础操作 |
| `feature_engineering.py` | 特征工程脚本，用于从原始数据中提取、构造模型训练所需特征，特徵存在save_json中 |
| `Inactive_Data.py` | 处理非活跃或无效数据的脚本，用于过滤和清理低质量数据 |
| `load_predict_columns.py` | 加载并定义模型推理时需要使用的列名或特征列表 |
| `load_yaml.py` | 加载YAML配置文件的工具脚本，用于读取项目配置参数 |
| `main_deal.py` | 数据预处理主脚本，整合并执行完整的数据处理流程 |
| `reader.py` | 数据读取器，封装了从不同数据源（如数据库、文件）读取数据的统一接口 |

---

### 3. `model_inference/`
模型推理模块，用于加载训练好的模型并对新数据进行预测。

| 文件名 | 功能描述 |
|--------|----------|
| `model_predict.py` | 模型推理主脚本，实现模型加载、数据输入、预测执行及结果输出 |

---

### 4. `model_interpretation/`
模型可解释性模块，用于分析和可视化模型的决策过程。

| 文件名 | 功能描述 |
|--------|----------|
| `interpretation.py` | 模型可解释性分析脚本，通过SHAP值、特征重要性等方法解释模型预测结果 |

---

### 5. `model_storage/`
模型存储与管理模块，负责模型文件的保存、加载和版本管理。

| 文件名 | 功能描述 |
|--------|----------|
| `get_last_file.py` | 获取最新生成的模型文件或结果文件的工具脚本 |
| `storage.py` | 模型存储管理脚本，实现模型的保存、加载、版本控制和路径管理 |

---

### 6. `model_training/`
模型训练模块，定义模型结构并执行训练流程。

| 文件名 | 功能描述 |
|--------|----------|
| `model_definition.py` | 模型定义脚本，包含神经网络或其他机器学习模型的结构定义 |
| `model_training.py` | 模型训练主脚本，整合数据加载、模型编译、训练循环及评估等流程 |

---

## 核心流程

1.  **数据预处理**：运行 `data_preprocessing/main_deal.py`，通过 `data_deal.py` 和 `feature_engineering.py` 等脚本完成数据清洗与特征工程。
2.  **模型训练**：运行 `model_training/model_training.py`，使用定义好的模型结构（`model_definition.py`）对预处理后的数据进行训练。
3.  **模型存储**：训练完成的模型通过 `model_storage/storage.py` 进行保存和管理。
4.  **模型推理**：使用 `model_inference/model_predict.py` 加载最新模型（通过 `model_storage/get_last_file.py`），对新数据进行预测。
5.  **可解释性分析**：通过 `model_interpretation/interpretation.py` 对模型预测结果进行分析，提升模型透明度。

---

## 环境依赖

> 注：请根据实际使用补充版本信息

- Python 3.8+
- pandas
- numpy
- scikit-learn
- tensorflow / pytorch (根据模型定义选择)
- shap (用于模型可解释性)
- pyyaml (用于配置文件读取)
- jupyter notebook/lab (如用于交互式开发)

---

## 注意事项

1.  运行前请确保 `load_yaml.py` 正确加载了数据库连接、文件路径等配置。
2.  模型训练和推理前，需确认 `data_preprocessing` 模块已生成符合要求的特征数据。
3.  模型文件的存储路径由 `model_storage/storage.py` 管理，推理脚本需确保能正确访问到最新模型。
