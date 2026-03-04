import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#混淆矩阵
def Confusion(y_test, y_pred,days,test):
    # 设置字体为SimHei显示中文
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 用于正常显示负号
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['OK', 'NG'], 
            yticklabels=['OK', 'NG'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{test} Confusion Matrix ({days})days')
    #plt.savefig(f'./output/result_fenxi/confusion_matrix_training_{days}days.png')
    plt.show()
    return 
#获取特征重要性
def explain_feature(model, top_n=20):
    # 获取特征重要性
    importance = model.get_booster().get_score(importance_type='weight')
    # 将特征重要性按降序排序
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    # 计算总重要性
    total_importance = sum(score for _, score in sorted_importance)
    # 计算每个特征的重要性百分比
    importance_percentage = [(feature, score / total_importance * 100) for feature, score in sorted_importance]
    # 取前 N 个特征
    top_features = importance_percentage[:top_n]
    # 准备数据用于绘图
    features, percentages = zip(*top_features)
    # 绘制特征重要性图（百分比）
    plt.figure(figsize=(10, 8))
    plt.barh(features, percentages, color='skyblue')
    plt.xlabel('Feature Importance (%)')
    plt.ylabel('Features')
    plt.title(f'Top {top_n} Feature Importance (%)')
    plt.tight_layout()
    # 保存图像到本地文件
    plt.savefig('./output/result_fenxi/feature_importance_percentage_training.png')
    # 显示图像（可选）
    plt.show()