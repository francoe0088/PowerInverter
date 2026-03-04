# INVERTER_TRINAPOWER-MASTER

**逆變器機器學習全流程工程專案**  
TrinaPower 逆變器運行端故障預測 + 製造端品質控制

## 專案概述

本專案針對逆變器（INVERTER）資料，建構完整的機器學習應用鏈路，涵蓋：

- 資料接入與預處理
- 特徵工程
- 模型訓練
- 模型儲存與版本管理
- 模型推理 / 預測
- 模型可解釋性分析（SHAP、特徵重要性等）

**核心目標**  
利用運行資料與製造資料，實現以下兩大方向：

- **一期**：運行端故障預測（提前預警即將故障的逆變器序號）
- **二期**：製造端品質控制（剔除高風險產品，輔助出貨檢驗）

最終達成降低運維成本、減少退換貨、提升系統可靠度與用戶體驗。

逆變器負責**交直流轉換、MPPT、保護、監控**，失效率相對較高，本專案從**運行端**與**製造端**雙向切入，降低整體故障率。

** 業務痛點與價值** 
現況痛點
-運維成本高、無有效預警
-退換貨週期長、用戶體驗差
-製造端缺乏關聯監控與預防手段

預期價值
-縮短維修時間、降低服務費用
-減少逆變器退換貨成本
-提升品牌力與智慧能源賣點

##技術驗證亮點
一期 - 運行端故障預測

資料：30,000 台逆變器、60 天、2000萬+ 記錄
模型：XGBoost / ExtraTrees / SVM 等
結果：準確率最高可達 92%（提前 1 天）
關鍵特徵：溫度、輸出功率、有功/無功功率、錯誤碼、電壓偏斜等

二期 - 製造端品質控制

資料：60 正常 + 61 異常逆變器，4 萬+ 製造記錄
特徵：FFT 頻域、標準差、自相關、偏度、峰度、老化測試時間等
卡控精度：80%+
--------------------------------------------------------
## 目錄結構
INVERTER_TRINAPOWER-MASTER/
├── data_preprocessing/           # 資料預處理
│   ├── data_deal.py              # 清洗、缺失值、格式轉換
│   ├── feature_engineering.py    # 特徵構造與提取
│   ├── Inactive_Data.py          # 過濾無效/非活躍資料
│   ├── load_predict_columns.py   # 定義推理所需欄位
│   ├── load_yaml.py              # 讀取 yaml 配置
│   ├── main_deal.py              # 資料處理主流程
│   └── reader.py                 # 統一資料讀取介面
├── model_training/               # 模型訓練
│   ├── model_definition.py       # 模型結構定義
│   └── model_training.py         # 訓練主腳本
├── model_storage/                # 模型儲存與版本管理
│   ├── get_last_file.py          # 取得最新模型/結果檔案
│   └── storage.py                # 模型儲存、載入、版本控制
├── model_inference/              # 模型推理
│   └── model_predict.py          # 推理主腳本
├── model_interpretation/         # 模型可解釋性
│   └── interpretation.py         # SHAP / 特徵重要性分析
├── utils_tools/                  # 通用工具函數
├── config/                       # 配置目錄（建議）
│   └── config.yaml
├── README.md
└── requirements.txt              # 依賴清單


## 主要模組功能一覽

| 模組                   | 主要檔案                  | 功能說明                              |
|------------------------|---------------------------|---------------------------------------|
| data_preprocessing     | main_deal.py             | 完整資料處理流程                      |
| data_preprocessing     | data_deal.py             | 基礎清洗、缺失值處理                  |
| data_preprocessing     | feature_engineering.py   | 特徵提取與構造（存 json）             |
| model_training         | model_training.py        | 訓練流程整合                          |
| model_training         | model_definition.py      | 模型架構定義（XGBoost/NN 等）         |
| model_storage          | storage.py               | 模型儲存、載入、版本管理              |
| model_inference        | model_predict.py         | 載入最新模型進行預測                  |
| model_interpretation   | interpretation.py        | SHAP 值、特徵重要性、可視化           |

## 完整執行流程

1. **資料預處理**  
python data_preprocessing/main_deal.py   python data_preprocessing/main_deal.py

2.模型訓練Bash
python model_training/model_training.py

3.模型儲存（訓練結束後由 storage.py 自動處理）
python model_inference/model_predict.py

4.模型推理Bash
python model_inference/model_predict.py

5.可解釋性分析Bash
python model_interpretation/interpretation.py

##環境依賴
Python 3.8+
pandas               >=1.5
numpy                >=1.23
scikit-learn         >=1.2
xgboost              （如使用 XGBoost）
tensorflow           或 pytorch（依模型選擇）
shap
pyyaml
jupyter              （可選，用於開發與分析）

建議建立 requirements.txt：
pip freeze > requirements.txt
# 或手動維護



