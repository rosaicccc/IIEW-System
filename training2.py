# 核心库导入
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# -------------------------- 1. 路径配置（改为使用训练集和测试集） --------------------------
# 设置路径
features_folder = "D:/青少年压力预警系统/data/processed"

# 训练集文件路径（前10个受试者）
train_file_path = os.path.join(features_folder, "train", "train_features.csv")

# 测试集文件路径（后5个受试者，用于实时检测）
test_file_path = os.path.join(features_folder, "test", "test_features.csv")

print("训练集文件路径:", train_file_path)
print("测试集文件路径:", test_file_path)

# 检查文件是否存在
if not os.path.exists(train_file_path):
    print(f"❌ 训练集文件不存在: {train_file_path}")
    print("请先运行批量数据处理脚本生成训练集数据")
    exit()
    
if not os.path.exists(test_file_path):
    print(f"⚠️  测试集文件不存在: {test_file_path}")
    print("测试模式将使用训练集的一部分作为测试集")

# -------------------------- 2. 读取训练集数据 --------------------------
print("\n" + "="*60)
print("📚 读取训练集数据（前10个受试者：S2-S11）")
print("="*60)

try:
    # 读取训练集
    train_df = pd.read_csv(train_file_path)
    print(f"✅ 成功读取训练集数据")
    print(f"训练集数据形状: {train_df.shape}")
    print(f"训练集受试者: {train_df['Subject_ID'].unique()}")
    print(f"训练集样本数: {len(train_df)}")
    
except Exception as e:
    print(f"❌ 读取训练集失败: {e}")
    print("尝试从个体文件重新构建训练集...")
    
    # 如果合并的训练集不存在，尝试从个体文件构建
    train_subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
    train_dfs = []
    
    for subject in train_subjects:
        file_path = os.path.join(features_folder, "individual", f"{subject}_features.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            train_dfs.append(df)
            print(f"   ✅ 读取 {subject}: {len(df)} 个样本")
        else:
            print(f"   ❌ {subject} 文件不存在")
    
    if train_dfs:
        train_df = pd.concat(train_dfs, ignore_index=True)
        print(f"训练集合并完成: {len(train_df)} 个样本")
    else:
        print("❌ 无法构建训练集")
        exit()

# -------------------------- 3. 读取测试集数据（用于实时检测） --------------------------
print("\n" + "="*60)
print("🔍 读取测试集数据（后5个受试者：S13-S17，用于实时检测）")
print("="*60)

try:
    if os.path.exists(test_file_path):
        test_df = pd.read_csv(test_file_path)
        print(f"✅ 成功读取测试集数据")
        print(f"测试集数据形状: {test_df.shape}")
        print(f"测试集受试者: {test_df['Subject_ID'].unique()}")
        print(f"测试集样本数: {len(test_df)}")
    else:
        print("⚠️  测试集文件不存在，将使用训练集的后20%作为测试集")
        # 从训练集中分割20%作为测试集
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])
        print(f"从训练集分割的测试集: {len(test_df)} 个样本")
        
except Exception as e:
    print(f"❌ 读取测试集失败: {e}")
    print("尝试从个体文件重新构建测试集...")
    
    # 如果合并的测试集不存在，尝试从个体文件构建
    test_subjects = ['S13', 'S14', 'S15', 'S16', 'S17']
    test_dfs = []
    
    for subject in test_subjects:
        file_path = os.path.join(features_folder, "individual", f"{subject}_features.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            test_dfs.append(df)
            print(f"   ✅ 读取 {subject}: {len(df)} 个样本")
        else:
            print(f"   ❌ {subject} 文件不存在")
    
    if test_dfs:
        test_df = pd.concat(test_dfs, ignore_index=True)
        print(f"测试集合并完成: {len(test_df)} 个样本")
    else:
        print("⚠️  无法构建测试集，将使用训练集的后20%作为测试集")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['Label'])

# -------------------------- 4. 过滤无效标签（核心：排除标签0和5） --------------------------
print("\n" + "="*60)
print("🔧 数据预处理：过滤无效标签")
print("="*60)

# 过滤训练集无效标签：只保留Label=1/2/3/4（排除0=未定义、5=忽略）
train_df = train_df[train_df["Label"].isin([1,2,3,4])]
print(f"训练集过滤后有效数据量：{len(train_df)}")
print(f"训练集Label类别：{sorted(train_df['Label'].unique())}")

# 过滤测试集无效标签
test_df = test_df[test_df["Label"].isin([1,2,3,4])]
print(f"测试集过滤后有效数据量：{len(test_df)}")
print(f"测试集Label类别：{sorted(test_df['Label'].unique())}")

# 检查是否有足够的压力样本
train_stress_count = len(train_df[train_df['Label'] == 2])
test_stress_count = len(test_df[test_df['Label'] == 2])
print(f"\n训练集压力样本(Label=2): {train_stress_count} ({train_stress_count/len(train_df)*100:.1f}%)")
print(f"测试集压力样本(Label=2): {test_stress_count} ({test_stress_count/len(test_df)*100:.1f}%)")

if train_stress_count < 10:
    print("⚠️  警告：训练集中压力样本过少，可能影响模型训练效果")

# -------------------------- 5. 标签映射（压力vs非压力） --------------------------
print("\n" + "="*60)
print("🏷️  标签映射：压力 vs 非压力")
print("="*60)

# 映射规则：1/3/4→非压力（0），2→压力（1）
def map_stress_label(label):
    if label in [1,3,4]:  # 1=基线、3=愉悦、4=冥想 → 非压力
        return 0
    elif label == 2:      # 2=压力 → 压力
        return 1
    else:
        return None  # 已过滤，不会出现

# 应用标签映射
train_df["stress_label"] = train_df["Label"].apply(map_stress_label)
test_df["stress_label"] = test_df["Label"].apply(map_stress_label)

# 检查映射结果
print(f"训练集标签分布:")
print(f"  非压力(0): {len(train_df[train_df['stress_label']==0])} 个样本")
print(f"  压力(1): {len(train_df[train_df['stress_label']==1])} 个样本")

print(f"\n测试集标签分布:")
print(f"  非压力(0): {len(test_df[test_df['stress_label']==0])} 个样本")
print(f"  压力(1): {len(test_df[test_df['stress_label']==1])} 个样本")

# 准备特征和标签
y_train = train_df["stress_label"].values
y_test = test_df["stress_label"].values

# -------------------------- 6. 特征选择（匹配你的列名） --------------------------
print("\n" + "="*60)
print("🎯 特征选择与准备")
print("="*60)

# 核心特征列（已匹配你的真实列名）
basic_features = [
    "HR_Mean", "RMSSD", "LF_HF_Ratio", "EDA_Mean", 
    "Temp_Mean", "Acc_Magnitude_Mean", "Activity_Level"
]

# 扩展特征列（如果有）
extended_features = [
    "HR_Std", "HR_Max", "HR_Min", "SDNN", "SDSD", "NN50", "pNN50", 
    "VLF_Power", "LF_Power", "HF_Power", "EDA_Max", "EDA_Min", "EDA_Std",
    "Temp_Max", "Temp_Min", "Temp_Std", "Acc_Magnitude_Std",
    "SCR_Count", "SCR_Amplitude_Sum", "SCR_RiseTime_Mean",
    "SCL_Mean", "SCR_Peaks_Per_Minute"
]

# 检查哪些特征实际存在于数据集中
available_basic_features = [f for f in basic_features if f in train_df.columns]
available_extended_features = [f for f in extended_features if f in train_df.columns]

print(f"基础特征可用: {len(available_basic_features)}/{len(basic_features)}")
print(f"扩展特征可用: {len(available_extended_features)}/{len(extended_features)}")

# 选择最终使用的特征
if len(available_extended_features) > 10:
    # 如果扩展特征较多，使用所有可用特征
    feature_cols = available_basic_features + available_extended_features
    print(f"使用所有可用特征: {len(feature_cols)} 个特征")
else:
    # 否则只使用基础特征
    feature_cols = available_basic_features
    print(f"使用基础特征: {len(feature_cols)} 个特征")

# 准备特征数据
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values

print(f"\n特征矩阵形状:")
print(f"  训练集 X: {X_train.shape}")
print(f"  测试集 X: {X_test.shape}")
print(f"  训练集 y: {y_train.shape}")
print(f"  测试集 y: {y_test.shape}")

# 检查是否有缺失值
print(f"\n缺失值检查:")
print(f"  训练集缺失值: {np.isnan(X_train).sum()} 个")
print(f"  测试集缺失值: {np.isnan(X_test).sum()} 个")

# 处理缺失值（用均值填充）
if np.isnan(X_train).sum() > 0 or np.isnan(X_test).sum() > 0:
    print("处理缺失值...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print("缺失值已用均值填充")

# -------------------------- 7. 随机森林训练（使用训练集训练，测试集验证） --------------------------
print("\n" + "="*60)
print("🌲 随机森林模型训练")
print("="*60)

# 设置随机森林参数
rf_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # 处理类别不平衡
)

print(f"训练集详细信息:")
print(f"  非压力样本: {len(y_train[y_train==0])}")
print(f"  压力样本: {len(y_train[y_train==1])}")
print(f"  样本比例(压力:非压力): {len(y_train[y_train==1])}:{len(y_train[y_train==0])}")

print(f"\n开始训练模型...")
rf_model.fit(X_train, y_train)
print("✅ 模型训练完成")

# -------------------------- 8. 模型评估（在测试集上） --------------------------
print("\n" + "="*60)
print("📊 模型评估（测试集，模拟实时检测）")
print("="*60)

# 在测试集上预测
y_pred = rf_model.predict(X_test)

# 评估结果
print(f"测试集准确率：{accuracy_score(y_test, y_pred):.3f}")
print("\n分类报告（核心看压力类的召回率）：")
print(classification_report(
    y_test, y_pred,
    target_names=["非压力", "压力"],
    digits=3
))

# -------------------------- 9. 详细分析（按受试者） --------------------------
print("\n" + "="*60)
print("👤 按受试者详细分析（测试集）")
print("="*60)

# 获取测试集的受试者ID
test_subjects = test_df["Subject_ID"].unique()
print(f"测试集包含 {len(test_subjects)} 个受试者")

# 为每个受试者单独评估
subject_results = []

for subject in test_subjects:
    # 获取该受试者的数据
    subject_mask = test_df["Subject_ID"] == subject
    X_subject = X_test[subject_mask]
    y_subject = y_test[subject_mask]
    
    if len(X_subject) > 0:
        # 预测
        y_pred_subject = rf_model.predict(X_subject)
        
        # 计算指标
        accuracy = accuracy_score(y_subject, y_pred_subject)
        
        # 计算混淆矩阵元素
        tp = np.sum((y_subject == 1) & (y_pred_subject == 1))
        fp = np.sum((y_subject == 0) & (y_pred_subject == 1))
        tn = np.sum((y_subject == 0) & (y_pred_subject == 0))
        fn = np.sum((y_subject == 1) & (y_pred_subject == 0))
        
        # 计算精确率、召回率、F1分数
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 统计样本信息
        stress_samples = np.sum(y_subject == 1)
        non_stress_samples = np.sum(y_subject == 0)
        
        subject_results.append({
            'Subject_ID': subject,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'Stress_Samples': stress_samples,
            'NonStress_Samples': non_stress_samples,
            'Total_Samples': len(y_subject)
        })
        
        print(f"  {subject}: 准确率={accuracy:.3f}, 压力样本={stress_samples}, 非压力样本={non_stress_samples}")

# 转换为DataFrame便于分析
subject_results_df = pd.DataFrame(subject_results)
print(f"\n测试集受试者平均表现:")
print(f"  平均准确率: {subject_results_df['Accuracy'].mean():.3f}")
print(f"  平均召回率: {subject_results_df['Recall'].mean():.3f}")
print(f"  平均F1分数: {subject_results_df['F1_Score'].mean():.3f}")

# -------------------------- 10. 可视化结果 --------------------------
print("\n" + "="*60)
print("📈 可视化结果")
print("="*60)

# 创建输出目录
output_dir = os.path.join(features_folder, "model_results")
os.makedirs(output_dir, exist_ok=True)

# 1. 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Reds",
    xticklabels=["非压力", "压力"],
    yticklabels=["非压力", "压力"]
)
plt.title("压力检测混淆矩阵 (测试集)", fontsize=14, fontweight='bold')
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.tight_layout()
conf_matrix_path = os.path.join(output_dir, "stress_confusion_matrix_test.png")
plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
print(f"✅ 混淆矩阵已保存: {conf_matrix_path}")
plt.show()

# 2. 特征重要性
plt.figure(figsize=(12, 6))
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.bar(range(len(importances)), importances[sorted_idx], color="#ff7f0e")
plt.xticks(range(len(importances)), [feature_cols[i] for i in sorted_idx], rotation=45, ha='right')
plt.title("压力检测特征重要性", fontsize=14, fontweight='bold')
plt.xlabel("特征", fontsize=12)
plt.ylabel("重要性", fontsize=12)
plt.tight_layout()
feature_importance_path = os.path.join(output_dir, "stress_feature_importance.png")
plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
print(f"✅ 特征重要性图已保存: {feature_importance_path}")
plt.show()

# 3. 受试者表现条形图
plt.figure(figsize=(10, 6))
x = range(len(subject_results_df))
width = 0.2

plt.bar([i - width for i in x], subject_results_df['Accuracy'], width, label='准确率', color='#2ca02c')
plt.bar(x, subject_results_df['Recall'], width, label='召回率', color='#d62728')
plt.bar([i + width for i in x], subject_results_df['F1_Score'], width, label='F1分数', color='#9467bd')

plt.xlabel('受试者', fontsize=12)
plt.ylabel('分数', fontsize=12)
plt.title('各受试者压力检测表现 (测试集)', fontsize=14, fontweight='bold')
plt.xticks(x, subject_results_df['Subject_ID'])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
subject_performance_path = os.path.join(output_dir, "subject_performance.png")
plt.savefig(subject_performance_path, dpi=300, bbox_inches='tight')
print(f"✅ 受试者表现图已保存: {subject_performance_path}")
plt.show()

# 4. 保存详细结果到CSV
results_df = pd.DataFrame({
    'Subject_ID': test_df['Subject_ID'],
    'True_Label': y_test,
    'Predicted_Label': y_pred,
    'Probability_NonStress': rf_model.predict_proba(X_test)[:, 0],
    'Probability_Stress': rf_model.predict_proba(X_test)[:, 1]
})

results_csv_path = os.path.join(output_dir, "detailed_predictions.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"✅ 详细预测结果已保存: {results_csv_path}")

# 保存受试者级别结果
subject_results_csv_path = os.path.join(output_dir, "subject_level_results.csv")
subject_results_df.to_csv(subject_results_csv_path, index=False)
print(f"✅ 受试者级别结果已保存: {subject_results_csv_path}")

# -------------------------- 11. 保存模型 --------------------------
print("\n" + "="*60)
print("💾 保存模型与配置")
print("="*60)

# 保存模型
model_path = os.path.join(output_dir, "stress_warning_model.pkl")
joblib.dump(rf_model, model_path)
print(f"✅ 压力预警模型已保存到: {model_path}")

# 保存特征列表
features_config = {
    'feature_columns': feature_cols,
    'train_subjects': list(train_df['Subject_ID'].unique()),
    'test_subjects': list(test_df['Subject_ID'].unique()),
    'train_samples': len(train_df),
    'test_samples': len(test_df),
    'train_stress_ratio': train_stress_count / len(train_df),
    'test_stress_ratio': test_stress_count / len(test_df),
    'model_accuracy': accuracy_score(y_test, y_pred)
}

import json
config_path = os.path.join(output_dir, "model_config.json")
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(features_config, f, indent=4, ensure_ascii=False)
print(f"✅ 模型配置已保存: {config_path}")

# 保存特征重要性详情
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

importance_csv_path = os.path.join(output_dir, "feature_importance.csv")
importance_df.to_csv(importance_csv_path, index=False)
print(f"✅ 特征重要性详情已保存: {importance_csv_path}")

# -------------------------- 12. 总结报告 --------------------------
print("\n" + "="*60)
print("🎉 训练完成总结")
print("="*60)

print(f"\n📊 训练集:")
print(f"  受试者: {len(train_df['Subject_ID'].unique())} 人")
print(f"  总样本: {len(train_df)} 个窗口")
print(f"  压力样本比例: {train_stress_count/len(train_df)*100:.1f}%")

print(f"\n📊 测试集（实时检测用）:")
print(f"  受试者: {len(test_df['Subject_ID'].unique())} 人")
print(f"  总样本: {len(test_df)} 个窗口")
print(f"  压力样本比例: {test_stress_count/len(test_df)*100:.1f}%")

print(f"\n🎯 模型表现:")
print(f"  测试集准确率: {accuracy_score(y_test, y_pred):.3f}")
print(f"  压力类召回率: {subject_results_df['Recall'].mean():.3f}")

print(f"\n💡 使用说明:")
print(f"  1. 实时检测: 使用保存的模型 {model_path}")
print(f"  2. 需要准备的特征: {len(feature_cols)} 个特征")
print(f"  3. 最重要的特征: {importance_df.iloc[0]['Feature']} (重要性: {importance_df.iloc[0]['Importance']:.3f})")

print(f"\n📁 所有结果已保存到: {output_dir}")

print("\n训练完成！模型已准备好用于实时压力预警~")