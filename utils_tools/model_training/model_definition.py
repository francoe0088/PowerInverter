import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
import polars as pl

class ModelEvaluator:
    def __init__(self):
        self.model = None

    def evaluate_model(self, model_name, X_train, X_test, y_train, y_test):
        model_functions = {
            "xgb_model": self.xgb_model,
        }

        if model_name not in model_functions:
            raise ValueError(f"Unsupported model name: {model_name}")

        # Call the function with the correct arguments
        model, pred = model_functions[model_name](X_train, X_test, y_train, y_test)
        
        return model, pred

    def xgb_model(self, X_train, X_test, y_train, y_test):
        print("Initializing XGBoost Model")
        column_names = X_train.columns
        df_columns = pl.DataFrame({"Column Names": column_names})

        # 将列名保存为 CSV 文件
        df_columns.write_csv('./output/save_json/columns.csv')

        # 如果是 Polars DataFrame，则转成 numpy
        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()

        if isinstance(X_test, pl.DataFrame):
            X_test = X_test.to_numpy()
            y_test = y_test.to_numpy()
        X_test_contig = np.ascontiguousarray(X_test)
        X_train_contig = np.ascontiguousarray(X_train)
        
        X_test_view = X_test_contig.view([('', X_test_contig.dtype)] * X_test_contig.shape[1])
        X_train_view = X_train_contig.view([('', X_train_contig.dtype)] * X_train_contig.shape[1])
        
        mask = np.intersect1d(X_test_view, X_train_view)

        print("0000",len(mask))
        # 合并训练、测试数据做交叉验证
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        # 计算样本权重（处理类别不平衡）
        sample_weights = compute_sample_weight(class_weight='balanced', y=y)

        # XGBoost 参数设置
        params = {
            'learning_rate': 0.08,
            'max_depth': 5,
            'min_child_weight': 5,
            'subsample': 0.9,
            'colsample_bytree': 0.6,
            'gamma': 1,
            'reg_alpha': 5,
            'reg_lambda': 10,
            'objective': 'multi:softmax',
            'eval_metric': 'mlogloss',  # 或者 'logloss'
            'num_class': 3,  # 这里有 3 个类别
            # 'use_label_encoder': False  # 如果版本较新，可以添加，避免不必要的 warning
        }

        # 初始化模型
        model = xgb.XGBClassifier(**params)

        # 5 折交叉验证
        #cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        #accuracies, recalls, f1_scores, confusion_matrices = [], [], [], []
#
        #for train_idx, val_idx in cv.split(X, y):
        #    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        #    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
#
        #    # 针对每一折重新计算样本权重
        #    sample_weights_fold = compute_sample_weight(
        #        class_weight='balanced',
        #        y=y_train_fold
        #    )
#
        #    # 训练模型（在 fit 时添加 eval_metric，更通用）
        #    model.fit(
        #        X_train_fold, 
        #        y_train_fold, 
        #        sample_weight=sample_weights_fold
        #     
        #    )
#
        #    # 预测
        #    y_pred_fold = model.predict(X_val_fold)
#
        #    # 计算各类指标
        #    acc = accuracy_score(y_val_fold, y_pred_fold)
        #    accuracies.append(acc)
#
        #    rec = recall_score(y_val_fold, y_pred_fold, average='weighted')
        #    recalls.append(rec)
#
        #    f1 = f1_score(y_val_fold, y_pred_fold, average='weighted')
        #    f1_scores.append(f1)
#
        #    cm = confusion_matrix(y_val_fold, y_pred_fold)
        #    confusion_matrices.append(cm)
#
        ## 输出交叉验证结果
        #print("Cross-validation scores for each fold:")
        #print(f"Accuracies: {accuracies}")
        #print(f"Recalls: {recalls}")
        #print(f"F1 Scores: {f1_scores}")
        #print(f"Average Accuracy: {np.mean(accuracies)}")
        #print(f"Average Recall: {np.mean(recalls)}")
        #print(f"Average F1 Score: {np.mean(f1_scores)}")
#
        #print("Confusion Matrices for each fold:")
        #for idx, cm in enumerate(confusion_matrices):
        #    print(f"Fold {idx + 1} Confusion Matrix:\n", cm)
#
        # -----------------------------------------
        # 交叉验证结束后，可再对整个训练集 X_train, y_train 做一次训练
        # 保证最终有一个 fit 完整训练集的模型
        # 然后对 X_test 做预测，返回 y_pred
        # -----------------------------------------
        final_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        #sample_weight=final_weights
        model.fit(X_train, y_train, sample_weight=final_weights)
        y_pred = model.predict(X_test)
        model_data = {
            'model': model,
            'feature_names': column_names
        }

        # 返回最终模型和对 X_test 的预测结果
        return model_data, y_pred

    def xgb_model1(self, X_train, X_test, y_train, y_test):
        print("Initializing XGBoost Model")

        # 如果是 Polars DataFrame，则转成 numpy
        if isinstance(X_train, pl.DataFrame):
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()

        if isinstance(X_test, pl.DataFrame):
            X_test = X_test.to_numpy()
            y_test = y_test.to_numpy()

        print(f"Training data type: {type(X_train)}")

        # 合并训练、测试数据做交叉验证
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)

        # XGBoost 参数设置
        params = {
            'learning_rate': 0.1,
            'n_estimators': 300,
            'max_depth': 5,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'gamma': 0,
            'reg_alpha': 4,     # L1 regularization
            'reg_lambda': 8,    # L2 regularization
            'objective': 'binary:logistic',  # 二分类
            'eval_metric': 'logloss',
            # 重点：根据正负样本数之比设置为3
            'scale_pos_weight':0.3
        }

        # 初始化模型
        model = xgb.XGBClassifier(**params)

        # 5 折交叉验证
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accuracies, recalls, f1_scores, confusion_matrices = [], [], [], []

        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # 训练模型（不再使用 sample_weight）
            model.fit(X_train_fold, y_train_fold)

            # 预测
            y_pred_fold = model.predict(X_val_fold)

            # 计算各类指标
            acc = accuracy_score(y_val_fold, y_pred_fold)
            accuracies.append(acc)

            rec = recall_score(y_val_fold, y_pred_fold, average='weighted')
            recalls.append(rec)

            f1 = f1_score(y_val_fold, y_pred_fold, average='weighted')
            f1_scores.append(f1)

            cm = confusion_matrix(y_val_fold, y_pred_fold)
            confusion_matrices.append(cm)

        # 输出交叉验证结果
        print("Cross-validation scores for each fold:")
        print(f"Accuracies: {accuracies}")
        print(f"Recalls: {recalls}")
        print(f"F1 Scores: {f1_scores}")
        print(f"Average Accuracy: {np.mean(accuracies)}")
        print(f"Average Recall: {np.mean(recalls)}")
        print(f"Average F1 Score: {np.mean(f1_scores)}")

        print("Confusion Matrices for each fold:")
        for idx, cm in enumerate(confusion_matrices):
            print(f"Fold {idx + 1} Confusion Matrix:\n", cm)

        # -----------------------------------------
        # 交叉验证结束后，再对完整的 X_train, y_train 训练一次
        # 以得到最终的模型并对 X_test 进行预测
        # -----------------------------------------
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 返回最终模型和对 X_test 的预测结果
        return model, y_pred
