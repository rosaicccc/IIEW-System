# 核心库导入
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -------------------------- 1. 路径配置（无需改） --------------------------
features_folder = "D:/青少年压力预警系统/data/processed/individual"
subject_ids = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]  # 排除S12

# -------------------------- 2. 读取并过滤数据（核心：排除无效标签） --------------------------
all_dfs = []
for subject_id in subject_ids:
    file_path = os.path.join(features_folder, f"S{subject_id}_features.csv")
    df = pd.read_csv(file_path)
    # 过滤无效标签：只保留Label=1/2/3/4（排除0=未定义、5=忽略）
    df = df[df["Label"].isin([1,2,3,4])]
    all_dfs.append(df)
feature_df = pd.concat(all_dfs, ignore_index=True)
print("过滤后有效数据量：", len(feature_df))
print("过滤后Label类别：", feature_df["Label"].unique())  # 应输出[1,2,3,4]

# -------------------------- 3. 标签映射（压力vs非压力） --------------------------
# 映射规则：1/3/4→非压力（0），2→压力（1）
def map_stress_label(label):
    if label in [1,3,4]:  # 1=基线、3=愉悦、4=冥想 → 非压力
        return 0
    elif label == 2:      # 2=压力 → 压力
        return 1
    else:
        return None  # 已过滤，不会出现

feature_df["stress_label"] = feature_df["Label"].apply(map_stress_label)
y = feature_df["stress_label"].values
print("映射后标签（0=非压力，1=压力）：", np.unique(y))

# -------------------------- 4. 特征&分组数据（匹配你的列名） --------------------------
# 核心特征列（已匹配你的真实列名）
feature_cols = [
    "HR_Mean", "RMSSD", "LF_HF_Ratio", "EDA_Mean", 
    "Temp_Mean", "Acc_Magnitude_Mean", "Activity_Level"
]
X = feature_df[feature_cols].values

# 受试者ID列（匹配你的Subject_ID）
groups = feature_df["Subject_ID"].values

# -------------------------- 5. 随机森林训练（LOCO交叉验证） --------------------------
rf_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

logo = LeaveOneGroupOut()
all_preds = []
all_trues = []
print("\n开始15人LOCO交叉验证训练...")
for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_subject = groups[test_idx][0]
    print(f"训练：14人 → 测试：S{test_subject}")
    rf_model.fit(X_train, y_train)
    all_preds.extend(rf_model.predict(X_test))
    all_trues.extend(y_test)

# -------------------------- 6. 评估结果（压力预警核心指标） --------------------------
print("\n=== 压力预警模型训练结果 ===")
print(f"整体准确率：{accuracy_score(all_trues, all_preds):.3f}")
print("\n分类报告（核心看压力类的召回率）：")
print(classification_report(
    all_trues, all_preds,
    target_names=["非压力", "压力"],
    digits=3
))

# 混淆矩阵（直观看压力识别效果）
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(all_trues, all_preds),
    annot=True, fmt="d", cmap="Reds",
    xticklabels=["非压力", "压力"],
    yticklabels=["非压力", "压力"]
)
plt.title("压力vs非压力 混淆矩阵")
plt.savefig(os.path.join(features_folder, "stress_confusion_matrix.png"), dpi=300)
plt.show()

# 特征重要性（看哪些指标最能识别压力）
plt.figure(figsize=(10, 5))
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.bar(range(len(importances)), importances[sorted_idx], color="#ff7f0e")
plt.xticks(range(len(importances)), [feature_cols[i] for i in sorted_idx], rotation=45)
plt.title("压力识别 核心特征重要性")
plt.savefig(os.path.join(features_folder, "stress_feature_importance.png"), dpi=300)
plt.show()

# 保存压力预警模型
model_path = os.path.join(features_folder, "stress_warning_model.pkl")
joblib.dump(rf_model, model_path)
print(f"\n压力预警模型已保存到：{model_path}")
print("训练完成！模型可直接用于实时压力预警~")