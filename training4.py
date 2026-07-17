# 核心库导入
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import datetime

# -------------------------- 1. 路径配置 --------------------------
# 设置路径
features_folder = "D:/青少年压力预警系统/data/processed"

# 创建带时间戳的输出目录，防止冲突
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = "v2_9features"  # 版本标识，v2表示第二次训练，9features表示9个特征
output_dir = os.path.join(features_folder, f"model_results_{model_version}_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

print(f"📁 输出目录: {output_dir}")
print(f"🕒 时间戳: {timestamp}")
print(f"🔖 版本: {model_version}")

# -------------------------- 2. 定义训练集和测试集的受试者 --------------------------
print("\n" + "="*60)
print("📚 数据集划分方案")
print("="*60)

# 训练集：前13个受试者 (S2-S11, S13-S15)
train_subjects = ['S2', 'S3', 'S4', '5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15']

# 测试集：最后2个受试者 (S16, S17) - 用于最终测试和模拟
test_subjects = ['S16', 'S17']

print(f"训练集受试者 ({len(train_subjects)}个): {train_subjects}")
print(f"测试集受试者 ({len(test_subjects)}个): {test_subjects}")

# -------------------------- 3. 从个体文件加载数据 --------------------------
print("\n" + "="*60)
print("📥 加载个体数据文件")
print("="*60)

def load_subject_data(subject_id):
    """加载单个受试者的特征数据"""
    file_path = os.path.join(features_folder, "individual", f"{subject_id}_features.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['Subject_ID'] = subject_id  # 确保有Subject_ID列
        return df
    else:
        print(f"❌ {subject_id} 文件不存在: {file_path}")
        return None

# 加载训练集数据
train_dfs = []
for subject in train_subjects:
    df = load_subject_data(subject)
    if df is not None:
        train_dfs.append(df)
        print(f"✅ 已加载 {subject}: {len(df)} 个样本")

if train_dfs:
    train_df = pd.concat(train_dfs, ignore_index=True)
    print(f"训练集合并完成: {len(train_df)} 个样本")
else:
    print("❌ 无法加载训练集数据")
    exit()

# 加载测试集数据
test_dfs = []
for subject in test_subjects:
    df = load_subject_data(subject)
    if df is not None:
        test_dfs.append(df)
        print(f"✅ 已加载 {subject}: {len(df)} 个样本")

if test_dfs:
    test_df = pd.concat(test_dfs, ignore_index=True)
    print(f"测试集合并完成: {len(test_df)} 个样本")
else:
    print("⚠️ 无法加载测试集数据，将使用训练集的一部分作为测试集")
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])

# -------------------------- 4. 过滤无效标签（排除标签0和5） --------------------------
print("\n" + "="*60)
print("🔧 数据预处理：过滤无效标签")
print("="*60)

# 过滤无效标签：只保留Label=1/2/3/4（排除0=未定义、5=忽略）
train_df = train_df[train_df["Label"].isin([1,2,3,4])]
test_df = test_df[test_df["Label"].isin([1,2,3,4])]

print(f"训练集过滤后: {len(train_df)} 个样本")
print(f"测试集过滤后: {len(test_df)} 个样本")

# 标签映射：1/3/4→非压力（0），2→压力（1）
def map_stress_label(label):
    if label in [1,3,4]:  # 1=基线、3=愉悦、4=冥想 → 非压力
        return 0
    elif label == 2:      # 2=压力 → 压力
        return 1
    else:
        return None

# 应用标签映射
train_df["stress_label"] = train_df["Label"].apply(map_stress_label)
test_df["stress_label"] = test_df["Label"].apply(map_stress_label)

# 检查数据分布
print(f"\n训练集标签分布:")
print(f"  非压力(0): {len(train_df[train_df['stress_label']==0])} 个样本")
print(f"  压力(1): {len(train_df[train_df['stress_label']==1])} 个样本")

print(f"\n测试集标签分布:")
print(f"  非压力(0): {len(test_df[test_df['stress_label']==0])} 个样本")
print(f"  压力(1): {len(test_df[test_df['stress_label']==1])} 个样本")

# -------------------------- 5. 明确指定9个特征（根据您的要求） --------------------------
print("\n" + "="*60)
print("🎯 特征选择 - 使用9个重要特征")
print("="*60)

# 明确的9个特征列表
feature_cols = [
    # 心率相关特征 (3个)
    "HR_Mean", 
    "RMSSD", 
    "SDNN",
    
    # 温度特征 (4个)
    "Temp_Mean", 
    "Temp_Max", 
    "Temp_Min", 
    "Temp_Std",
    
    # 加速度特征 (2个)
    "Acc_Magnitude_Mean", 
    "Acc_Magnitude_Std"
]

# 检查哪些特征实际存在于数据集中
available_features = [f for f in feature_cols if f in train_df.columns]
missing_features = [f for f in feature_cols if f not in train_df.columns]

print(f"计划使用特征: {len(feature_cols)} 个")
print(f"实际可用特征: {len(available_features)} 个")
print(f"缺失特征: {len(missing_features)} 个")

if missing_features:
    print(f"缺失的特征列表: {missing_features}")
    print("⚠️ 将用0填充缺失的特征")
    
    # 用0填充缺失的特征
    for feature in missing_features:
        train_df[feature] = 0
        test_df[feature] = 0
        print(f"  - 已填充 {feature}")

# 特征中文名映射（用于报告）
feature_chinese_names = {
    # 心率相关特征
    "HR_Mean": "平均心率",
    "RMSSD": "心率变异性(RMSSD)",
    "SDNN": "心率变异性(SDNN)",
    
    # 温度特征
    "Temp_Mean": "平均体温",
    "Temp_Max": "体温最大值",
    "Temp_Min": "体温最小值",
    "Temp_Std": "体温标准差",
    
    # 加速度特征
    "Acc_Magnitude_Mean": "平均加速度幅度",
    "Acc_Magnitude_Std": "加速度幅度标准差"
}

print(f"\n最终使用的特征 ({len(feature_cols)}个):")
for i, feat in enumerate(feature_cols):
    chinese_name = feature_chinese_names.get(feat, feat)
    print(f"  {i+1:2d}. {feat:25s} ({chinese_name})")

# -------------------------- 6. LOOCV交叉验证（在训练集上） --------------------------
print("\n" + "="*60)
print("🔬 LOOCV交叉验证（在13个训练受试者上）")
print("="*60)

# 准备训练集数据
X_train = train_df[feature_cols].values
y_train = train_df["stress_label"].values
subject_ids_train = train_df["Subject_ID"].values

# 处理缺失值
if np.isnan(X_train).sum() > 0:
    print(f"处理训练集缺失值: {np.isnan(X_train).sum()} 个")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)

# 初始化LOOCV
logo = LeaveOneGroupOut()
train_subject_list = list(train_df["Subject_ID"].unique())

print(f"训练集包含 {len(train_subject_list)} 个受试者")
print(f"将进行 {logo.get_n_splits(groups=subject_ids_train)} 折交叉验证")

# 存储交叉验证结果
cv_results = []
cv_predictions = []
cv_true_labels = []

# 执行LOOCV
print("\n执行LOOCV交叉验证...")
for fold_idx, (train_index, val_index) in enumerate(logo.split(X_train, y_train, groups=subject_ids_train)):
    # 获取当前折的训练集和验证集
    X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
    
    # 获取当前验证集的受试者
    val_subject = subject_ids_train[val_index[0]] if len(val_index) > 0 else "Unknown"
    
    # 创建随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # 训练模型
    rf_model.fit(X_fold_train, y_fold_train)
    
    # 预测
    y_fold_pred = rf_model.predict(X_fold_val)
    y_fold_pred_proba = rf_model.predict_proba(X_fold_val)
    
    # 计算指标
    accuracy = accuracy_score(y_fold_val, y_fold_pred)
    
    # 计算精确率、召回率、F1分数
    tp = np.sum((y_fold_val == 1) & (y_fold_pred == 1))
    fp = np.sum((y_fold_val == 0) & (y_fold_pred == 1))
    fn = np.sum((y_fold_val == 1) & (y_fold_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 保存结果
    cv_results.append({
        'Fold': fold_idx + 1,
        'Subject_ID': val_subject,
        'Train_Subjects': len(np.unique(subject_ids_train[train_index])),
        'Train_Samples': len(X_fold_train),
        'Val_Samples': len(X_fold_val),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    })
    
    # 保存预测结果用于后续分析
    cv_predictions.extend(y_fold_pred)
    cv_true_labels.extend(y_fold_val)
    
    print(f"  折{fold_idx+1:2d}: 验证受试者={val_subject}, 准确率={accuracy:.3f}, 召回率={recall:.3f}")

# 转换交叉验证结果为DataFrame
cv_results_df = pd.DataFrame(cv_results)

print(f"\nLOOCV交叉验证汇总:")
print(f"  平均准确率: {cv_results_df['Accuracy'].mean():.3f} ± {cv_results_df['Accuracy'].std():.3f}")
print(f"  平均召回率: {cv_results_df['Recall'].mean():.3f} ± {cv_results_df['Recall'].std():.3f}")
print(f"  平均F1分数: {cv_results_df['F1_Score'].mean():.3f} ± {cv_results_df['F1_Score'].std():.3f}")

# -------------------------- 7. 使用全部训练集训练最终模型 --------------------------
print("\n" + "="*60)
print("🌲 训练最终随机森林模型（使用全部13个训练受试者）")
print("="*60)

# 创建最终模型
final_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

# 使用全部训练数据训练
final_rf_model.fit(X_train, y_train)
print("✅ 最终模型训练完成")

# 检查模型使用的特征数量
print(f"模型实际使用的特征数量: {final_rf_model.n_features_in_}")

# -------------------------- 8. 在测试集上评估模型 --------------------------
print("\n" + "="*60)
print("📊 在测试集上评估模型（S16, S17）")
print("="*60)

# 准备测试集数据
X_test = test_df[feature_cols].values
y_test = test_df["stress_label"].values
subject_ids_test = test_df["Subject_ID"].values

# 处理测试集缺失值
if np.isnan(X_test).sum() > 0:
    print(f"处理测试集缺失值: {np.isnan(X_test).sum()} 个")
    if 'imputer' in locals():
        X_test = imputer.transform(X_test)
    else:
        from sklearn.impute import SimpleImputer
        imputer_test = SimpleImputer(strategy='mean')
        X_test = imputer_test.fit_transform(X_test)

# 预测
y_pred = final_rf_model.predict(X_test)
y_pred_proba = final_rf_model.predict_proba(X_test)

# 整体评估
print(f"测试集整体准确率: {accuracy_score(y_test, y_pred):.3f}")
print("\n整体分类报告:")
print(classification_report(
    y_test, y_pred,
    target_names=["非压力", "压力"],
    digits=3
))

# 按测试受试者评估
print("\n按测试受试者评估:")
test_subject_results = []

for subject in test_subjects:
    # 获取该受试者的数据
    subject_mask = subject_ids_test == subject
    if np.sum(subject_mask) == 0:
        continue
        
    X_subject = X_test[subject_mask]
    y_subject = y_test[subject_mask]
    
    # 预测
    y_pred_subject = final_rf_model.predict(X_subject)
    
    # 计算指标
    accuracy = accuracy_score(y_subject, y_pred_subject)
    
    # 计算精确率、召回率、F1分数
    tp = np.sum((y_subject == 1) & (y_pred_subject == 1))
    fp = np.sum((y_subject == 0) & (y_pred_subject == 1))
    fn = np.sum((y_subject == 1) & (y_pred_subject == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    test_subject_results.append({
        'Subject_ID': subject,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'Samples': len(y_subject),
        'Stress_Samples': np.sum(y_subject == 1),
        'NonStress_Samples': np.sum(y_subject == 0)
    })
    
    print(f"  {subject}: 准确率={accuracy:.3f}, 召回率={recall:.3f}, 样本数={len(y_subject)}")

# 转换为DataFrame
test_subject_results_df = pd.DataFrame(test_subject_results)

print(f"\n测试集平均表现:")
print(f"  平均准确率: {test_subject_results_df['Accuracy'].mean():.3f}")
print(f"  平均召回率: {test_subject_results_df['Recall'].mean():.3f}")
print(f"  平均F1分数: {test_subject_results_df['F1_Score'].mean():.3f}")

# -------------------------- 9. 可视化结果 --------------------------
print("\n" + "="*60)
print("📈 可视化结果")
print("="*60)

# 1. LOOCV结果可视化
plt.figure(figsize=(12, 8))

# LOOCV准确率
plt.subplot(2, 2, 1)
plt.bar(range(len(cv_results_df)), cv_results_df['Accuracy'], color='skyblue')
plt.axhline(y=cv_results_df['Accuracy'].mean(), color='r', linestyle='--', label=f'平均: {cv_results_df["Accuracy"].mean():.3f}')
plt.xlabel('交叉验证折')
plt.ylabel('准确率')
plt.title(f'LOOCV各折准确率 (训练集) - {model_version}')
plt.legend()
plt.xticks(range(len(cv_results_df)), cv_results_df['Subject_ID'], rotation=45)

# LOOCV召回率
plt.subplot(2, 2, 2)
plt.bar(range(len(cv_results_df)), cv_results_df['Recall'], color='lightcoral')
plt.axhline(y=cv_results_df['Recall'].mean(), color='r', linestyle='--', label=f'平均: {cv_results_df["Recall"].mean():.3f}')
plt.xlabel('交叉验证折')
plt.ylabel('召回率')
plt.title(f'LOOCV各折召回率 (训练集) - {model_version}')
plt.legend()
plt.xticks(range(len(cv_results_df)), cv_results_df['Subject_ID'], rotation=45)

# 测试集表现
plt.subplot(2, 2, 3)
x = range(len(test_subject_results_df))
plt.bar([i - 0.2 for i in x], test_subject_results_df['Accuracy'], width=0.2, label='准确率', color='skyblue')
plt.bar(x, test_subject_results_df['Recall'], width=0.2, label='召回率', color='lightcoral')
plt.bar([i + 0.2 for i in x], test_subject_results_df['F1_Score'], width=0.2, label='F1分数', color='lightgreen')
plt.xlabel('测试受试者')
plt.ylabel('分数')
plt.title(f'测试集受试者表现 - {model_version}')
plt.legend()
plt.xticks(x, test_subject_results_df['Subject_ID'])

# 混淆矩阵
plt.subplot(2, 2, 4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["非压力", "压力"],
            yticklabels=["非压力", "压力"])
plt.title(f'测试集混淆矩阵 - {model_version}')
plt.ylabel('真实标签')
plt.xlabel('预测标签')

plt.tight_layout()
loocv_plot_path = os.path.join(output_dir, f"loocv_and_test_results_{model_version}.png")
plt.savefig(loocv_plot_path, dpi=300, bbox_inches='tight')
print(f"✅ LOOCV和测试结果图已保存: {loocv_plot_path}")
plt.show()

# 2. 特征重要性（带中文标签）
plt.figure(figsize=(12, 6))
importances = final_rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

# 获取所有特征的重要性
top_features = [feature_cols[i] for i in sorted_idx]
top_importances = importances[sorted_idx]
top_chinese_names = [feature_chinese_names.get(f, f) for f in top_features]

# 创建条形图
plt.bar(range(len(feature_cols)), top_importances, color="#2ca02c")
plt.xticks(range(len(feature_cols)), top_chinese_names, rotation=45, ha='right', fontsize=10)
plt.xlabel("特征（中文名）", fontsize=12)
plt.ylabel("重要性", fontsize=12)
plt.title(f"特征重要性排名 - {model_version} (共{len(feature_cols)}个特征）", fontsize=14, fontweight='bold')
plt.tight_layout()

# 在每个条形上添加数值和排名
for i, (importance, chinese_name) in enumerate(zip(top_importances, top_chinese_names)):
    plt.text(i, importance + 0.001, f'{i+1}: {importance:.3f}', ha='center', va='bottom', fontsize=9)

feature_importance_path = os.path.join(output_dir, f"feature_importance_{model_version}.png")
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
print(f"✅ 特征重要性图已保存: {feature_importance_path}")
plt.show()

# 3. 特征类别重要性分析
plt.figure(figsize=(10, 6))

# 按类别分组特征重要性
categories = {
    '心率相关': [f for f in feature_cols if 'HR' in f or 'SDNN' in f or 'RMSSD' in f],
    '温度相关': [f for f in feature_cols if 'Temp' in f],
    '加速度相关': [f for f in feature_cols if 'Acc' in f]
}

category_importance = {}
for category, features in categories.items():
    total_importance = 0
    for feature in features:
        if feature in feature_cols:
            idx = feature_cols.index(feature)
            total_importance += importances[idx]
    category_importance[category] = total_importance

# 创建饼图
plt.pie(category_importance.values(), labels=category_importance.keys(), 
        autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title(f'特征重要性按类别分布 - {model_version}', fontsize=14, fontweight='bold')
plt.axis('equal')

category_importance_path = os.path.join(output_dir, f"feature_category_importance_{model_version}.png")
plt.savefig(category_importance_path, dpi=300, bbox_inches='tight')
print(f"✅ 特征类别重要性图已保存: {category_importance_path}")
plt.show()

# -------------------------- 10. 保存结果和模型 --------------------------
print("\n" + "="*60)
print("💾 保存结果和模型")
print("="*60)

# 1. 保存LOOCV结果
cv_results_csv_path = os.path.join(output_dir, f"loocv_results_{model_version}.csv")
cv_results_df.to_csv(cv_results_csv_path, index=False)
print(f"✅ LOOCV结果已保存: {cv_results_csv_path}")

# 2. 保存测试集详细结果
detailed_results = pd.DataFrame({
    'Subject_ID': subject_ids_test,
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Probability_NonStress': y_pred_proba[:, 0],
    'Probability_Stress': y_pred_proba[:, 1]
})
detailed_results_path = os.path.join(output_dir, f"test_predictions_{model_version}.csv")
detailed_results.to_csv(detailed_results_path, index=False)
print(f"✅ 测试集详细预测结果已保存: {detailed_results_path}")

# 3. 保存测试受试者级别结果
test_subject_results_path = os.path.join(output_dir, f"test_subject_results_{model_version}.csv")
test_subject_results_df.to_csv(test_subject_results_path, index=False)
print(f"✅ 测试受试者级别结果已保存: {test_subject_results_path}")

# 4. 保存最终模型
final_model_path = os.path.join(output_dir, f"stress_warning_model_{model_version}.pkl")
joblib.dump(final_rf_model, final_model_path)
print(f"✅ 最终压力预警模型已保存: {final_model_path}")

# 5. 保存特征重要性详情（包含中文名）
importance_df = pd.DataFrame({
    'Feature_English': feature_cols,
    'Feature_Chinese': [feature_chinese_names.get(f, f) for f in feature_cols],
    'Importance': importances,
    'Rank': range(1, len(feature_cols) + 1)
}).sort_values('Importance', ascending=False)

importance_path = os.path.join(output_dir, f"feature_importance_{model_version}.csv")
importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig支持中文
print(f"✅ 特征重要性详情已保存: {importance_path}")

# 6. 保存配置信息
config = {
    'model_version': model_version,
    'timestamp': timestamp,
    'output_dir': output_dir,
    'feature_columns': feature_cols,
    'feature_columns_chinese': [feature_chinese_names.get(f, f) for f in feature_cols],
    'train_subjects': train_subjects,
    'test_subjects': test_subjects,
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'train_stress_ratio': np.sum(y_train == 1) / len(y_train),
    'test_stress_ratio': np.sum(y_test == 1) / len(y_test),
    'loocv_mean_accuracy': float(cv_results_df['Accuracy'].mean()),
    'loocv_std_accuracy': float(cv_results_df['Accuracy'].std()),
    'loocv_mean_recall': float(cv_results_df['Recall'].mean()),
    'loocv_std_recall': float(cv_results_df['Recall'].std()),
    'test_mean_accuracy': float(test_subject_results_df['Accuracy'].mean()),
    'test_mean_recall': float(test_subject_results_df['Recall'].mean()),
    'test_mean_f1': float(test_subject_results_df['F1_Score'].mean()),
    'top_features': importance_df.head(5).to_dict('records'),
    'model_parameters': {
        'n_estimators': 100,
        'max_depth': 8,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced'
    },
    'feature_categories': {
        'heart_rate_features': [f for f in feature_cols if 'HR' in f or 'SDNN' in f or 'RMSSD' in f],
        'temperature_features': [f for f in feature_cols if 'Temp' in f],
        'acceleration_features': [f for f in feature_cols if 'Acc' in f]
    }
}

import json
config_path = os.path.join(output_dir, f"model_config_{model_version}.json")
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print(f"✅ 模型配置已保存: {config_path}")

# 7. 保存训练摘要
summary_path = os.path.join(output_dir, f"training_summary_{model_version}.txt")
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(f"模型训练摘要 - {model_version}\n")
    f.write("="*60 + "\n")
    f.write(f"训练时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"输出目录: {output_dir}\n\n")
    
    f.write("数据集信息:\n")
    f.write(f"  训练集: {len(train_subjects)} 个受试者, {len(train_df)} 个样本\n")
    f.write(f"  测试集: {len(test_subjects)} 个受试者, {len(test_df)} 个样本\n\n")
    
    f.write("特征信息:\n")
    f.write(f"  使用特征总数: {len(feature_cols)} 个\n")
    f.write(f"  心率相关特征: {len(config['feature_categories']['heart_rate_features'])} 个\n")
    f.write(f"  温度相关特征: {len(config['feature_categories']['temperature_features'])} 个\n")
    f.write(f"  加速度相关特征: {len(config['feature_categories']['acceleration_features'])} 个\n\n")
    
    f.write("模型性能:\n")
    f.write(f"  LOOCV平均准确率: {cv_results_df['Accuracy'].mean():.3f} ± {cv_results_df['Accuracy'].std():.3f}\n")
    f.write(f"  LOOCV平均召回率: {cv_results_df['Recall'].mean():.3f} ± {cv_results_df['Recall'].std():.3f}\n")
    f.write(f"  测试集整体准确率: {accuracy_score(y_test, y_pred):.3f}\n")
    f.write(f"  测试集平均召回率: {test_subject_results_df['Recall'].mean():.3f}\n\n")
    
    f.write("Top 5 最重要的特征:\n")
    for i, row in importance_df.head(5).iterrows():
        f.write(f"  {i+1}. {row['Feature_Chinese']} ({row['Feature_English']}): {row['Importance']:.3f}\n")

print(f"✅ 训练摘要已保存: {summary_path}")

# -------------------------- 11. 总结报告 --------------------------
print("\n" + "="*60)
print("🎉 训练完成总结")
print("="*60)

print(f"\n📊 数据集划分:")
print(f"  训练集: {len(train_subjects)} 个受试者, {len(train_df)} 个样本")
print(f"  测试集: {len(test_subjects)} 个受试者, {len(test_df)} 个样本")

print(f"\n🎯 特征使用:")
print(f"  使用特征总数: {len(feature_cols)} 个")
print(f"  心率相关特征: {len(config['feature_categories']['heart_rate_features'])} 个")
print(f"  温度相关特征: {len(config['feature_categories']['temperature_features'])} 个")
print(f"  加速度相关特征: {len(config['feature_categories']['acceleration_features'])} 个")

print(f"\n📊 LOOCV交叉验证结果 (训练集):")
print(f"  平均准确率: {cv_results_df['Accuracy'].mean():.3f} ± {cv_results_df['Accuracy'].std():.3f}")
print(f"  平均召回率: {cv_results_df['Recall'].mean():.3f} ± {cv_results_df['Recall'].std():.3f}")
print(f"  平均F1分数: {cv_results_df['F1_Score'].mean():.3f} ± {cv_results_df['F1_Score'].std():.3f}")

print(f"\n📊 测试集结果 (最终评估):")
print(f"  整体准确率: {accuracy_score(y_test, y_pred):.3f}")
print(f"  平均召回率: {test_subject_results_df['Recall'].mean():.3f}")
print(f"  平均F1分数: {test_subject_results_df['F1_Score'].mean():.3f}")

print(f"\n💡 Top 5 最重要的特征:")
for i, row in importance_df.head(5).iterrows():
    print(f"  {i+1}. {row['Feature_Chinese']} ({row['Feature_English']}): {row['Importance']:.3f}")

print(f"\n📁 所有结果已保存到: {output_dir}")

print(f"\n✅ 模型训练完成！已准备好用于实时压力预警系统。")
print(f"   模型版本: {model_version}")
print(f"   时间戳: {timestamp}")
print(f"   最终模型: {final_model_path}")
print(f"   特征数量: {len(feature_cols)} 个（仅使用9个重要特征）")
print(f"   可用于模拟测试: 测试集包含 {len(test_subjects)} 个受试者 ({test_subjects})")