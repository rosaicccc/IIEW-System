import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
from datetime import datetime
import glob

# -------------------------- 1. 基础配置 --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 2. 路径配置 --------------------------
# 自动查找最新的模型文件夹
base_path = "C:/Users/Fan/Desktop/新建文件夹/data/processed"
model_folders = glob.glob(os.path.join(base_path, "model_results_*"))

if model_folders:
    # 按修改时间排序，获取最新的文件夹
    latest_folder = max(model_folders, key=os.path.getmtime)
    MODEL_PATH = os.path.join(latest_folder, "stress_warning_model_v2_9features.pkl")
    CONFIG_PATH = os.path.join(latest_folder, "model_config_v2_9features.json")
else:
    st.error("❌ 未找到模型文件夹，请先训练模型")
    MODEL_PATH = None
    CONFIG_PATH = None

DATA_FOLDER = "C:/Users/Fan/Desktop/新建文件夹/data/processed/individual"

# -------------------------- 3. 特征中文名映射（9个特征版本） --------------------------
def get_feature_chinese_names():
    """获取特征中文名映射 - 9个特征版本"""
    feature_mapping = {
        # 心率相关特征 (3个)
        "HR_Mean": "平均心率",
        "RMSSD": "心率变异性(RMSSD)",
        "SDNN": "心率变异性(SDNN)",
        
        # 温度特征 (4个)
        "Temp_Mean": "平均体温",
        "Temp_Max": "体温最大值",
        "Temp_Min": "体温最小值",
        "Temp_Std": "体温标准差",
        
        # 加速度特征 (2个)
        "Acc_Magnitude_Mean": "平均加速度幅度",
        "Acc_Magnitude_Std": "加速度幅度标准差"
    }
    return feature_mapping

# -------------------------- 4. 初始化Session State --------------------------
def init_session_state():
    """初始化会话状态"""
    defaults = {
        "start_sim": False,
        "current_sample": 0,
        "sim_complete": False,
        "recent_data": {"time": [], "hr": [], "subject": [], "stress_prob": [], "prediction": []},
        "stats": {"total_predictions": 0, "stress_detections": 0, "correct_predictions": 0, "by_subject": {}},
        "current_subject": None,
        "subject_index": 0,
        "subject_data": None,
        "subjects_data": {},
        "subject_keys": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -------------------------- 5. 加载模型和配置 --------------------------
@st.cache_resource
def load_model():
    """加载训练好的模型"""
    if MODEL_PATH is None or not os.path.exists(MODEL_PATH):
        st.error(f"❌ 模型文件不存在: {MODEL_PATH}")
        return None
    
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ 模型加载成功 - 包含 {model.n_features_in_} 个特征")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

def load_model_config():
    """加载模型配置信息"""
    if CONFIG_PATH is None or not os.path.exists(CONFIG_PATH):
        st.warning("⚠️ 配置文件不存在，将使用默认配置")
        return None
    
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
        st.success("✅ 模型配置加载成功")
        return config
    except Exception as e:
        st.error(f"❌ 配置加载失败: {str(e)}")
        return None

# -------------------------- 6. 加载测试数据 --------------------------
@st.cache_data
def load_test_data(feature_cols):
    """加载S16和S17两个受试者的测试数据"""
    test_subjects = ['S16', 'S17']
    all_data = []
    
    for subject in test_subjects:
        file_path = os.path.join(DATA_FOLDER, f"{subject}_features.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # 过滤无效标签
                df = df[df["Label"].isin([1, 2, 3, 4])]
                # 添加受试者列
                df["Subject_ID"] = subject
                # 创建压力标签：2=压力，其他=非压力
                df["stress_label"] = df["Label"].apply(lambda x: 1 if x == 2 else 0)
                
                # 确保包含所有需要的特征，如果缺少则用0填充
                for feature in feature_cols:
                    if feature not in df.columns:
                        df[feature] = 0
                        st.warning(f"⚠️ {subject} 缺少特征 {feature}，已用0填充")
                
                all_data.append(df)
                st.success(f"✅ 已加载 {subject}: {len(df)} 个样本")
            except Exception as e:
                st.error(f"❌ 加载 {subject} 失败: {str(e)}")
                continue
        else:
            st.error(f"❌ 文件不存在: {file_path}")
    
    if all_data:
        test_df = pd.concat(all_data, ignore_index=True)
        
        # 添加模拟时间戳
        time_series = pd.date_range(
            start=datetime.now().replace(hour=9, minute=0, second=0),
            periods=len(test_df),
            freq="10s"
        )
        test_df["Sim_Time"] = time_series
        
        st.success(f"✅ 测试集加载完成: 总共 {len(test_df)} 个样本")
        return test_df
    else:
        st.error("❌ 无法加载任何测试数据")
        return None

# -------------------------- 7. 特征重要性可视化 --------------------------
def display_feature_importance(model, feature_cols, feature_names_chinese, config=None):
    """显示特征重要性图表"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            '英文名': feature_cols,
            '中文名': [feature_names_chinese.get(f, f) for f in feature_cols],
            '重要性': importances
        }).sort_values('重要性', ascending=False)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 图1：所有特征水平条形图
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_cols)))
        bars1 = ax1.barh(range(len(feature_cols)), importance_df['重要性'], color=colors)
        ax1.set_yticks(range(len(feature_cols)))
        ax1.set_yticklabels(importance_df['中文名'])
        ax1.invert_yaxis()
        ax1.set_xlabel('重要性分数', fontsize=12)
        ax1.set_title(f'特征重要性排名（{len(feature_cols)}个特征）', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, importance in enumerate(importance_df['重要性']):
            ax1.text(importance + 0.001, i, f'{importance:.3f}', va='center', fontsize=10)
        
        # 图2：按类别重要性饼图
        categories = {
            '心率相关': [f for f in feature_cols if 'HR' in f or 'SDNN' in f or 'RMSSD' in f],
            '温度相关': [f for f in feature_cols if 'Temp' in f],
            '加速度相关': [f for f in feature_cols if 'Acc' in f]
        }
        
        category_importance = {}
        category_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (category, features) in enumerate(categories.items()):
            if features:
                indices = [i for i, f in enumerate(feature_cols) if f in features]
                total_importance = sum(importances[i] for i in indices if i < len(importances))
                category_importance[category] = total_importance
        
        if category_importance:
            categories_list = list(category_importance.keys())
            values = list(category_importance.values())
            
            # 过滤掉重要性为0的类别
            valid_indices = [i for i, v in enumerate(values) if v > 0]
            if valid_indices:
                categories_list = [categories_list[i] for i in valid_indices]
                values = [values[i] for i in valid_indices]
                colors = [category_colors[i] for i in valid_indices]
                
                wedges, texts, autotexts = ax2.pie(values, labels=categories_list, colors=colors, 
                                                   autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')
                ax2.set_title('特征重要性类别分布', fontsize=14, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, '无重要性数据', ha='center', va='center', fontsize=12)
                ax2.axis('off')
        else:
            ax2.text(0.5, 0.5, '无类别重要性数据', ha='center', va='center', fontsize=12)
            ax2.axis('off')
        
        plt.tight_layout()
        return fig, importance_df
    
    return None, None

# -------------------------- 8. 实时模拟控制 --------------------------
def simulation_controls():
    """模拟控制面板"""
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if not st.session_state["sim_complete"]:
            if not st.session_state["start_sim"]:
                if st.button("▶️ 开始模拟", type="primary", use_container_width=True):
                    st.session_state["start_sim"] = True
                    st.rerun()
            else:
                if st.button("⏸️ 暂停模拟", use_container_width=True):
                    st.session_state["start_sim"] = False
                    st.rerun()
        else:
            if st.button("🔄 重新开始", type="secondary", use_container_width=True):
                reset_simulation()
                st.rerun()
    
    with col2:
        if st.session_state["start_sim"] and not st.session_state["sim_complete"]:
            speed = st.select_slider("模拟速度", options=["慢速", "正常", "快速"], value="正常")
            return {"慢速": 1.0, "正常": 0.5, "快速": 0.2}[speed]
        return 0.5  # 默认速度
    
    with col3:
        if st.session_state["start_sim"]:
            total = len(st.session_state["subject_data"]) if st.session_state["subject_data"] is not None else 0
            if total > 0:
                progress = st.session_state["current_sample"] / total
                st.progress(progress, text=f"进度: {st.session_state['current_sample']}/{total}")

def reset_simulation():
    """重置模拟状态"""
    st.session_state["start_sim"] = False
    st.session_state["current_sample"] = 0
    st.session_state["sim_complete"] = False
    st.session_state["recent_data"] = {"time": [], "hr": [], "subject": [], "stress_prob": [], "prediction": []}
    st.session_state["stats"] = {"total_predictions": 0, "stress_detections": 0, "correct_predictions": 0, "by_subject": {}}
    st.session_state["current_subject"] = None
    st.session_state["subject_index"] = 0
    st.session_state["subject_data"] = None

# -------------------------- 9. 模拟数据显示 --------------------------
def display_simulation_status(test_df, model, feature_cols, delay):
    """显示模拟状态和数据"""
    if test_df is None or model is None:
        return
    
    # 如果是第一次运行，初始化数据
    if not st.session_state["subjects_data"]:
        subjects_data = {}
        for subject in ['S16', 'S17']:
            subject_df = test_df[test_df["Subject_ID"] == subject].copy()
            if len(subject_df) > 0:
                subjects_data[subject] = subject_df
        
        st.session_state["subjects_data"] = subjects_data
        st.session_state["subject_keys"] = list(subjects_data.keys())
        if st.session_state["subject_keys"]:
            st.session_state["current_subject"] = st.session_state["subject_keys"][0]
            st.session_state["subject_data"] = subjects_data[st.session_state["current_subject"]]
    
    current_data = st.session_state["subject_data"]
    if current_data is None or len(current_data) == 0:
        st.warning("⚠️ 当前受试者没有数据")
        return
    
    current_idx = st.session_state["current_sample"]
    
    if current_idx < len(current_data):
        # 获取当前样本
        sample = current_data.iloc[current_idx]
        current_time = sample["Sim_Time"].strftime("%H:%M:%S")
        current_subject = sample["Subject_ID"]
        true_label = sample["stress_label"]
        true_state = "🔴 压力" if true_label == 1 else "🟢 正常"
        
        # 准备特征数据
        sample_features = sample[feature_cols].values.reshape(1, -1)
        
        # 模型预测
        pred_label = model.predict(sample_features)[0]
        pred_prob = model.predict_proba(sample_features)[0][1]
        pred_state = "🔴 压力" if pred_label == 1 else "🟢 正常"
        pred_color = "#FF5252" if pred_label == 1 else "#4CAF50"
        
        # 更新统计
        update_statistics(current_subject, pred_label, true_label)
        
        # 更新实时数据
        update_recent_data(current_time, sample["HR_Mean"], current_subject, pred_prob, pred_label)
        
        # 显示当前状态
        display_current_status(current_subject, current_time, true_state, pred_state, pred_prob, pred_color)
        
        # 显示图表和特征
        display_charts_and_features(sample, sample_features, model, feature_cols)
        
        # 显示统计信息
        display_statistics()
        
        # 更新索引
        st.session_state["current_sample"] += 1
        
        # 延迟并重新运行
        time.sleep(delay)
        st.rerun()
    else:
        # 当前受试者完成，切换到下一个
        st.session_state["subject_index"] += 1
        if st.session_state["subject_index"] < len(st.session_state["subject_keys"]):
            st.session_state["current_subject"] = st.session_state["subject_keys"][st.session_state["subject_index"]]
            st.session_state["subject_data"] = st.session_state["subjects_data"][st.session_state["current_subject"]]
            st.session_state["current_sample"] = 0
            st.info(f"🎯 切换到受试者 {st.session_state['current_subject']}")
            time.sleep(1)
            st.rerun()
        else:
            # 所有受试者完成
            st.session_state["sim_complete"] = True
            st.session_state["start_sim"] = False
            st.success("🎉 模拟完成！")
            st.rerun()

def update_statistics(subject, pred_label, true_label):
    """更新统计信息"""
    stats = st.session_state["stats"]
    stats["total_predictions"] += 1
    
    if pred_label == 1:
        stats["stress_detections"] += 1
    
    if pred_label == true_label:
        stats["correct_predictions"] += 1
    
    if subject not in stats["by_subject"]:
        stats["by_subject"][subject] = {"total": 0, "stress": 0, "correct": 0}
    
    stats["by_subject"][subject]["total"] += 1
    if pred_label == 1:
        stats["by_subject"][subject]["stress"] += 1
    if pred_label == true_label:
        stats["by_subject"][subject]["correct"] += 1

def update_recent_data(time, hr, subject, stress_prob, prediction):
    """更新实时数据"""
    recent_data = st.session_state["recent_data"]
    recent_data["time"].append(time)
    recent_data["hr"].append(hr)
    recent_data["subject"].append(subject)
    recent_data["stress_prob"].append(stress_prob)
    recent_data["prediction"].append(prediction)
    
    # 保持最近20个点
    max_points = 20
    for key in recent_data:
        if len(recent_data[key]) > max_points:
            recent_data[key].pop(0)

def display_current_status(subject, time, true_state, pred_state, pred_prob, pred_color):
    """显示当前状态"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background-color:#2196F310; padding:15px; border-radius:10px; text-align:center;'>
            <h5 style='color:#2196F3; margin:0'>受试者</h5>
            <h3 style='color:#2196F3; margin:5px 0'>{subject}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background-color:#9C27B010; padding:15px; border-radius:10px; text-align:center;'>
            <h5 style='color:#9C27B0; margin:0'>当前时间</h5>
            <h3 style='color:#9C27B0; margin:5px 0'>{time}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background-color:#4CAF5010; padding:15px; border-radius:10px; text-align:center;'>
            <h5 style='color:#4CAF50; margin:0'>真实状态</h5>
            <h3 style='color:#4CAF50; margin:5px 0'>{true_state}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background-color:{pred_color}10; padding:15px; border-radius:10px; text-align:center;'>
            <h5 style='color:{pred_color}; margin:0'>预测状态</h5>
            <h3 style='color:{pred_color}; margin:5px 0'>{pred_state}</h3>
            <p style='margin:0; font-size:16px; font-weight:bold;'>压力概率: {pred_prob:.2%}</p>
        </div>
        """, unsafe_allow_html=True)

def display_charts_and_features(sample, sample_features, model, feature_cols):
    """显示图表和特征值"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 实时图表
        st.subheader("📈 实时生理指标监测")
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        
        recent_data = st.session_state["recent_data"]
        times = recent_data["time"]
        hrs = recent_data["hr"]
        stress_probs = recent_data["stress_prob"]
        
        if len(times) > 0:
            # 心率曲线
            ax1.plot(times, hrs, color="#FF4B4B", linewidth=2, marker="o", markersize=4)
            ax1.set_xlabel("时间", fontsize=10)
            ax1.set_ylabel("心率 (次/分)", fontsize=10, color="#FF4B4B")
            ax1.tick_params(axis='y', labelcolor="#FF4B4B")
            ax1.grid(True, alpha=0.3)
            ax1.set_title("实时心率变化", fontsize=12)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 压力概率
            ax2.plot(times, stress_probs, color="#FF9800", linewidth=2, marker="s", markersize=4)
            ax2.set_xlabel("时间", fontsize=10)
            ax2.set_ylabel("压力概率", fontsize=10, color="#FF9800")
            ax2.tick_params(axis='y', labelcolor="#FF9800")
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3)
            
            # 阈值线
            ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='阈值 (0.5)')
            ax2.legend(loc='upper right')
            ax2.set_title("压力概率变化", fontsize=12)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig1)
        else:
            st.info("等待数据...")
    
    with col2:
        # 关键特征值
        st.subheader("🔍 关键生理指标")
        
        feature_names_chinese = get_feature_chinese_names()
        
        # 显示所有9个特征
        for feat in feature_cols:
            if feat in feature_names_chinese:
                idx = feature_cols.index(feat)
                value = sample_features[0][idx]
                chinese_name = feature_names_chinese.get(feat, feat)
                
                if feat == "HR_Mean":
                    st.metric(f"❤️ {chinese_name}", f"{value:.1f} 次/分")
                elif "RMSSD" in feat or "SDNN" in feat:
                    st.metric(f"📊 {chinese_name}", f"{value:.2f} ms")
                elif "Temp" in feat:
                    st.metric(f"🌡️ {chinese_name}", f"{value:.2f} °C")
                elif "Acc" in feat:
                    st.metric(f"🏃 {chinese_name}", f"{value:.3f}")
        
        # 干预建议
        st.subheader("💡 健康建议")
        pred_prob = model.predict_proba(sample_features)[0][1]
        
        if pred_prob > 0.7:
            st.error("**⚠️ 高度压力状态**")
            st.markdown("""
            - 🧘 **立即休息**: 深呼吸5-10分钟
            - 💧 **补充水分**: 喝一杯温水
            - 🚶 **轻度活动**: 散步5分钟
            - 📱 **远离屏幕**: 休息10分钟
            """)
        elif pred_prob > 0.5:
            st.warning("**⚠️ 中度压力状态**")
            st.markdown("""
            - 🎵 **听轻音乐**: 放松5分钟
            - 🧠 **正念练习**: 专注当下呼吸
            - 💧 **喝水休息**: 短暂休息3分钟
            """)
        else:
            st.success("**✅ 状态良好**")
            st.markdown("保持当前状态，注意劳逸结合")

def display_statistics():
    """显示统计信息"""
    with st.expander("📊 实时统计", expanded=False):
        stats = st.session_state["stats"]
        total = stats["total_predictions"]
        
        if total > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总预测数", total)
            
            with col2:
                stress_rate = stats["stress_detections"] / total * 100
                st.metric("压力检测率", f"{stress_rate:.1f}%")
            
            with col3:
                accuracy = stats["correct_predictions"] / total * 100
                st.metric("准确率", f"{accuracy:.1f}%")
            
            # 当前受试者统计
            current_subject = st.session_state["current_subject"]
            if current_subject and current_subject in stats["by_subject"]:
                sub_stats = stats["by_subject"][current_subject]
                st.write(f"**当前受试者 {current_subject}:**")
                col4, col5 = st.columns(2)
                with col4:
                    st.write(f"样本数: {sub_stats['total']}")
                with col5:
                    if sub_stats['total'] > 0:
                        sub_accuracy = sub_stats['correct'] / sub_stats['total'] * 100
                        st.write(f"准确率: {sub_accuracy:.1f}%")
        else:
            st.info("暂无统计数据")

# -------------------------- 10. 模拟完成总结 --------------------------
def display_simulation_summary():
    """显示模拟完成总结"""
    st.divider()
    st.subheader("📋 模拟完成总结")
    
    stats = st.session_state["stats"]
    total = stats["total_predictions"]
    
    if total > 0:
        # 总体统计
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总样本数", total)
        
        with col2:
            stress_count = stats["stress_detections"]
            stress_rate = stress_count / total * 100
            st.metric("压力检测数", f"{stress_count} ({stress_rate:.1f}%)")
        
        with col3:
            accuracy = stats["correct_predictions"] / total * 100
            st.metric("总体准确率", f"{accuracy:.1f}%")
        
        # 受试者详细统计
        st.subheader("👥 各受试者表现")
        subject_stats = []
        
        for subject, data in stats["by_subject"].items():
            if data["total"] > 0:
                subject_acc = data["correct"] / data["total"] * 100
                subject_stress_rate = data["stress"] / data["total"] * 100
                subject_stats.append({
                    "受试者": subject,
                    "样本数": data["total"],
                    "压力检测数": data["stress"],
                    "压力检测率": f"{subject_stress_rate:.1f}%",
                    "准确率": f"{subject_acc:.1f}%"
                })
        
        if subject_stats:
            st.dataframe(pd.DataFrame(subject_stats), use_container_width=True)
        
        # 性能分析图表
        st.subheader("📈 性能分析")
        col4, col5 = st.columns(2)
        
        with col4:
            # 压力检测分布饼图
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            labels = ['压力检测', '非压力检测']
            sizes = [stats["stress_detections"], total - stats["stress_detections"]]
            colors = ['#FF6B6B', '#4ECDC4']
            
            if sum(sizes) > 0:
                wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')
                ax1.set_title('压力检测分布')
                st.pyplot(fig1)
        
        with col5:
            # 准确率柱状图
            subject_names = list(stats["by_subject"].keys())
            accuracies = []
            
            for subject in subject_names:
                data = stats["by_subject"][subject]
                if data["total"] > 0:
                    accuracies.append(data["correct"] / data["total"] * 100)
            
            if accuracies:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                bars = ax2.bar(subject_names, accuracies, color='#45B7D1')
                ax2.set_ylim([0, 100])
                ax2.set_ylabel('准确率 (%)')
                ax2.set_title('各受试者准确率')
                ax2.grid(True, alpha=0.3, axis='y')
                
                for i, v in enumerate(accuracies):
                    ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
                
                st.pyplot(fig2)
        
        # 重新开始按钮
        if st.button("🔄 重新开始模拟", type="primary", use_container_width=True):
            reset_simulation()
            st.rerun()

# -------------------------- 11. 主函数 --------------------------
def main():
    st.set_page_config(
        page_title="青少年压力实时检测系统",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 页面标题
    st.title("🧠 青少年压力实时检测系统")
    st.markdown("### 基于小米手环可采集的关键生理特征的智能压力预警系统")
    st.divider()
    
    # 初始化session state
    init_session_state()
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 显示模型文件夹信息
        if model_folders:
            latest_folder_name = os.path.basename(latest_folder)
            st.success(f"📁 检测到模型文件夹: {latest_folder_name}")
        
        # 模型信息
        st.subheader("📦 模型信息")
        model = load_model()
        config = load_model_config()
        
        if model and config:
            st.success(f"✅ 模型已加载")
            st.write(f"**特征数量:** {model.n_features_in_} 个")
            st.write(f"**模型版本:** {config.get('model_version', 'v2_9features')}")
            st.write(f"**训练时间:** {config.get('timestamp', '未知')}")
            st.write(f"**训练受试者:** {len(config.get('train_subjects', []))} 个")
            st.write(f"**测试受试者:** {len(config.get('test_subjects', []))} 个")
            
            # 显示LOOCV结果
            if 'loocv_mean_accuracy' in config:
                st.subheader("📊 模型性能")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("LOOCV准确率", f"{config['loocv_mean_accuracy']:.3f}")
                with col2:
                    test_acc = config.get('test_mean_accuracy', 0)
                    st.metric("测试集准确率", f"{test_acc:.3f}")
        
        st.divider()
        
        # 系统设置
        st.subheader("⚙️ 系统设置")
        show_details = st.checkbox("显示详细特征", value=False)
        
        st.divider()
        
        # 系统信息
        st.subheader("ℹ️ 系统信息")
        st.write(f"**版本:** 2.0.0 (9特征版)")
        st.write(f"**更新时间:** 2024-01-15")
        st.write(f"**数据来源:** WESAD数据集")
        st.write(f"**特征类型:** 9个关键时域特征")
        
        # 帮助
        with st.expander("❓ 使用帮助"):
            st.markdown("""
            1. **开始模拟**: 点击开始按钮启动实时检测
            2. **调整速度**: 使用滑块控制模拟速度
            3. **观察指标**: 关注心率和压力概率变化
            4. **查看建议**: 系统会根据压力状态提供干预建议
            5. **查看统计**: 模拟完成后查看详细统计
            """)
    
    # 主界面区域
    tab1, tab2, tab3 = st.tabs(["🏠 实时模拟", "📊 模型分析", "📋 数据概览"])
    
    with tab1:
        # 系统说明
        with st.expander("📖 系统说明", expanded=True):
            st.markdown("""
            **系统功能：**
            - 🧠 **智能检测**: 基于随机森林模型的压力状态识别
            - 📈 **实时监测**: 动态显示心率、体温等9个关键生理指标
            - ⚠️ **早期预警**: 压力概率超过阈值时自动预警
            - 💡 **智能干预**: 根据压力状态提供个性化建议
            
            **特征构成（9个关键特征）：**
            - ❤️ **心率相关** (3个): 平均心率、心率变异性(RMSSD)、心率变异性(SDNN)
            - 🌡️ **温度相关** (4个): 平均体温、体温最大值、体温最小值、体温标准差
            - 🏃 **加速度相关** (2个): 平均加速度幅度、加速度幅度标准差
            
            **测试数据集：**
            - **测试对象**: S16, S17 (2个独立受试者)
            - **数据来源**: WESAD腕部可穿戴设备数据
            - **样本窗口**: 60秒滑动窗口
            
            **技术特点：**
            - 使用13个受试者训练 (S2-S11, S13-S15)
            - LOOCV交叉验证确保模型泛化能力
            - 精选9个关键特征，提高模型可解释性
            - 实时可视化压力概率变化
            
            **使用方法：**
            1. 点击左侧"开始模拟"按钮
            2. 观察实时生理指标变化
            3. 查看压力状态和干预建议
            4. 模拟完成后查看详细统计
            """)
        
        if model is None:
            st.error("无法加载模型，请检查模型文件路径")
            return
        
        # 获取特征列
        if config and 'feature_columns' in config:
            feature_cols = config['feature_columns']
            st.success(f"✅ 从配置加载特征: {len(feature_cols)} 个")
        elif hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_.tolist()
            st.success(f"✅ 从模型加载特征: {len(feature_cols)} 个")
        else:
            # 9个特征列表
            feature_cols = [
                # 心率相关特征 (3个)
                "HR_Mean", "RMSSD", "SDNN",
                # 温度特征 (4个)
                "Temp_Mean", "Temp_Max", "Temp_Min", "Temp_Std",
                # 加速度特征 (2个)
                "Acc_Magnitude_Mean", "Acc_Magnitude_Std"
            ]
            st.warning(f"⚠️ 使用默认特征: {len(feature_cols)} 个")
        
        # 显示特征信息
        feature_names_chinese = get_feature_chinese_names()
        st.info(f"📊 **特征构成**: {len(feature_cols)} 个特征")
        st.write(f"- ❤️ 心率相关: {len([f for f in feature_cols if 'HR' in f or 'SDNN' in f or 'RMSSD' in f])} 个")
        st.write(f"- 🌡️ 温度相关: {len([f for f in feature_cols if 'Temp' in f])} 个")
        st.write(f"- 🏃 加速度相关: {len([f for f in feature_cols if 'Acc' in f])} 个")
        
        # 加载测试数据
        test_df = load_test_data(feature_cols)
        
        if test_df is None:
            st.error("无法加载测试数据，请检查数据文件")
            return
        
        st.success(f"✅ 系统准备就绪")
        
        # 模拟控制
        delay = simulation_controls()
        
        # 模拟运行
        if st.session_state["start_sim"]:
            display_simulation_status(test_df, model, feature_cols, delay)
        elif st.session_state["sim_complete"]:
            display_simulation_summary()
        else:
            st.info("👆 点击'开始模拟'按钮启动压力检测系统")
            
            # 显示受试者信息
            subjects_info = test_df["Subject_ID"].unique()
            st.write(f"**待测受试者:** {', '.join(subjects_info)}")
            
            # 显示样本分布
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("总样本数", len(test_df))
            with col_info2:
                stress_samples = test_df["stress_label"].sum()
                stress_percentage = stress_samples / len(test_df) * 100
                st.metric("压力样本比例", f"{stress_percentage:.1f}%")
    
    with tab2:
        st.header("📊 模型性能分析")
        
        if model is None or config is None:
            st.warning("请先加载模型和配置")
            return
        
        # 获取特征列
        if config and 'feature_columns' in config:
            feature_cols = config['feature_columns']
        elif hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_.tolist()
        else:
            feature_cols = [
                "HR_Mean", "RMSSD", "SDNN",
                "Temp_Mean", "Temp_Max", "Temp_Min", "Temp_Std",
                "Acc_Magnitude_Mean", "Acc_Magnitude_Std"
            ]
        
        feature_names_chinese = get_feature_chinese_names()
        
        # 特征重要性分析
        st.subheader("🎯 特征重要性分析")
        
        if hasattr(model, 'feature_importances_'):
            fig, importance_df = display_feature_importance(model, feature_cols, feature_names_chinese, config)
            if fig is not None:
                st.pyplot(fig)
                
                # 显示重要性表格
                with st.expander("📋 详细特征重要性", expanded=False):
                    st.dataframe(importance_df, use_container_width=True)
        else:
            st.info("无法获取特征重要性信息")
        
        # 模型性能指标
        st.subheader("📈 模型性能指标")
        
        if config:
            col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
            
            with col_metrics1:
                if 'loocv_mean_accuracy' in config:
                    st.metric("LOOCV准确率", f"{config['loocv_mean_accuracy']:.3f}")
            
            with col_metrics2:
                if 'test_mean_accuracy' in config:
                    st.metric("测试集准确率", f"{config['test_mean_accuracy']:.3f}")
            
            with col_metrics3:
                if 'test_mean_recall' in config:
                    st.metric("测试集召回率", f"{config['test_mean_recall']:.3f}")
            
            with col_metrics4:
                if 'test_mean_f1' in config:
                    st.metric("测试集F1分数", f"{config['test_mean_f1']:.3f}")
        
        # 特征类别分析
        st.subheader("🔍 特征类别分析")
        
        categories = {
            '心率相关': [f for f in feature_cols if 'HR' in f or 'SDNN' in f or 'RMSSD' in f],
            '温度相关': [f for f in feature_cols if 'Temp' in f],
            '加速度相关': [f for f in feature_cols if 'Acc' in f]
        }
        
        category_table = []
        for category, features in categories.items():
            if features:
                category_table.append({
                    '特征类别': category,
                    '特征数量': len(features),
                    '特征示例': ', '.join([feature_names_chinese.get(f, f) for f in features])
                })
        
        if category_table:
            st.table(pd.DataFrame(category_table))
        
        # 移除的特征说明
        st.subheader("🗑️ 已移除特征")
        removed_features = config.get('removed_features', [
            "EDA_Mean", "EDA_Max", "EDA_Min", "EDA_Std", "SCR_Count", "SCL_Mean",
            "Activity_Level", "LF_HF_Ratio", "VLF_Power", "LF_Power", "HF_Power"
        ])
        st.info(f"为简化模型和提高可解释性，已移除以下特征：{len(removed_features)} 个")
        st.write(f"包括皮肤电活动、频域特征等其他特征")
    
    with tab3:
        st.header("📋 数据概览与统计")
        
        if model is None:
            st.warning("请先加载模型")
            return
        
        # 获取特征列
        if config and 'feature_columns' in config:
            feature_cols = config['feature_columns']
        else:
            feature_cols = [
                "HR_Mean", "RMSSD", "SDNN",
                "Temp_Mean", "Temp_Max", "Temp_Min", "Temp_Std",
                "Acc_Magnitude_Mean", "Acc_Magnitude_Std"
            ]
        
        # 加载测试数据
        test_df = load_test_data(feature_cols)
        
        if test_df is None:
            st.warning("请先加载测试数据")
            return
        
        # 数据基本信息
        col_data1, col_data2, col_data3, col_data4 = st.columns(4)
        
        with col_data1:
            st.metric("总样本数", len(test_df))
        
        with col_data2:
            st.metric("特征数量", len(feature_cols))
        
        with col_data3:
            unique_subjects = test_df["Subject_ID"].unique()
            st.metric("受试者数量", len(unique_subjects))
        
        with col_data4:
            stress_count = test_df["stress_label"].sum()
            stress_percentage = stress_count / len(test_df) * 100
            st.metric("压力样本", f"{stress_percentage:.1f}%")
        
        # 受试者样本分布
        st.subheader("👥 受试者样本分布")
        
        subject_counts = test_df["Subject_ID"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(subject_counts.index, subject_counts.values, color='#4ECDC4')
        ax.set_xlabel('受试者')
        ax.set_ylabel('样本数')
        ax.set_title('各受试者样本数量')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(subject_counts.values):
            ax.text(i, v + 1, str(v), ha='center')
        
        st.pyplot(fig)
        
        # 特征分布
        st.subheader("📊 主要特征分布")
        
        # 选择要显示的特征
        selected_features = st.multiselect(
            "选择要查看的特征",
            options=feature_cols,
            default=feature_cols[:3] if len(feature_cols) >= 3 else feature_cols
        )
        
        if selected_features:
            fig2, axes = plt.subplots(len(selected_features), 1, figsize=(10, 4 * len(selected_features)))
            if len(selected_features) == 1:
                axes = [axes]
            
            for i, feat in enumerate(selected_features):
                ax = axes[i]
                ax.hist(test_df[feat].dropna(), bins=30, alpha=0.7, color='#FF6B6B')
                ax.set_xlabel(feat)
                ax.set_ylabel('频数')
                chinese_name = get_feature_chinese_names().get(feat, feat)
                ax.set_title(f'{chinese_name} ({feat}) 分布')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig2)
        
        # 特征相关性矩阵
        st.subheader("🔗 特征相关性分析")
        
        if len(feature_cols) > 1:
            corr_matrix = test_df[feature_cols].corr()
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            cax = ax3.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            fig3.colorbar(cax)
            
            # 设置坐标轴标签
            ax3.set_xticks(range(len(feature_cols)))
            ax3.set_yticks(range(len(feature_cols)))
            ax3.set_xticklabels([feature_names_chinese.get(f, f) for f in feature_cols], rotation=90)
            ax3.set_yticklabels([feature_names_chinese.get(f, f) for f in feature_cols])
            
            # 添加数值标签
            for i in range(len(feature_cols)):
                for j in range(len(feature_cols)):
                    text = f'{corr_matrix.iloc[i, j]:.2f}'
                    ax3.text(j, i, text, ha='center', va='center', color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
            
            ax3.set_title('特征相关性矩阵')
            st.pyplot(fig3)
        
        # 数据预览
        st.subheader("🔍 数据预览")
        
        preview_cols = ["Subject_ID", "Label", "stress_label"] + feature_cols[:5]
        preview_cols = [col for col in preview_cols if col in test_df.columns]
        
        st.dataframe(test_df[preview_cols].head(20), use_container_width=True)
        
        # 下载选项
        csv_data = test_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 下载完整数据 (CSV)",
            data=csv_data,
            file_name="压力检测测试数据_9特征版.csv",
            mime="text/csv"
        )

# -------------------------- 12. 启动系统 --------------------------
if __name__ == "__main__":
    main()