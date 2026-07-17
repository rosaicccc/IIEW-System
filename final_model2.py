import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# -------------------------- 1. 基础配置 --------------------------
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# -------------------------- 2. 路径配置 --------------------------
MODEL_PATH = "C:/Users/Fan/Desktop/青少年压力预警系统/data/processed/model_results/stress_warning_model.pkl"
TEST_DATA_PATH = "C:/Users/Fan/Desktop/青少年压力预警系统/data/processed/test/test_features.csv"

# -------------------------- 3. 模型使用的20个特征（根据你提供的列表） --------------------------
MODEL_FEATURES = [
    "HR_Mean",
    "RMSSD",
    "LF_HF_Ratio",
    "EDA_Mean",
    "Temp_Mean",
    "Acc_Magnitude_Mean",
    "Activity_Level",
    "SDNN",
    "VLF_Power",
    "LF_Power",
    "HF_Power",
    "EDA_Max",
    "EDA_Min",
    "EDA_Std",
    "Temp_Max",
    "Temp_Min",
    "Temp_Std",
    "Acc_Magnitude_Std",
    "SCR_Count",
    "SCL_Mean"
]

# -------------------------- 4. 初始化Session State --------------------------
def init_session_state():
    if "start_sim" not in st.session_state:
        st.session_state["start_sim"] = False
    if "current_sample" not in st.session_state:
        st.session_state["current_sample"] = 0
    if "sim_complete" not in st.session_state:
        st.session_state["sim_complete"] = False
    if "recent_data" not in st.session_state:
        st.session_state["recent_data"] = {
            "time": [], "hr": [], "subject": [], "stress_prob": [], "prediction": []
        }
    if "stats" not in st.session_state:
        st.session_state["stats"] = {
            "total_predictions": 0, "stress_detections": 0,
            "correct_predictions": 0, "by_subject": {}
        }

# -------------------------- 5. 加载模型 --------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ 模型文件不存在: {MODEL_PATH}")
        return None
    
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"✅ 模型加载成功，期望 {model.n_features_in_} 个特征")
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {str(e)}")
        return None

# -------------------------- 6. 加载测试集数据 --------------------------
@st.cache_data
def load_test_data():
    if not os.path.exists(TEST_DATA_PATH):
        st.error(f"❌ 测试集文件不存在: {TEST_DATA_PATH}")
        return None
    
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        # 检查测试集是否包含所有必需的特征
        missing_features = [f for f in MODEL_FEATURES if f not in test_df.columns]
        if missing_features:
            st.error(f"❌ 测试集缺少以下特征: {missing_features}")
            st.info("请确保测试集包含所有模型需要的特征")
            return None
        
        st.success(f"✅ 测试集加载成功: {len(test_df)} 个样本")
        
        # 生成模拟时间
        time_series = pd.date_range(
            start="2025-12-12 09:00:00",
            periods=len(test_df),
            freq="10s"
        )
        test_df["Sim_Time"] = time_series
        
        # 准备特征数据
        test_df["stress_label"] = test_df["Label"].apply(
            lambda x: 1 if x == 2 else 0
        )
        
        return test_df
    except Exception as e:
        st.error(f"❌ 测试集加载失败: {str(e)}")
        return None

# -------------------------- 7. 实时检测模拟函数 --------------------------
def realtime_simulation(test_df, model):
    if test_df is None or model is None:
        return
    
    st.subheader("🔴 实时压力检测模拟")
    
    # 提取特征数据
    X_test = test_df[MODEL_FEATURES].values
    y_test = test_df["stress_label"].values
    subject_ids = test_df["Subject_ID"].values
    hr_data = test_df["HR_Mean"].values
    total_samples = len(test_df)
    
    # 控制按钮
    col1, col2 = st.columns([1, 3])
    with col1:
        if not st.session_state["start_sim"] and not st.session_state["sim_complete"]:
            if st.button("▶️ 开始模拟", use_container_width=True, type="primary"):
                st.session_state["start_sim"] = True
                st.rerun()
        elif st.session_state["start_sim"] and not st.session_state["sim_complete"]:
            if st.button("⏸️ 暂停模拟", use_container_width=True):
                st.session_state["start_sim"] = False
                st.rerun()
        else:
            if st.button("🔄 重新开始", use_container_width=True):
                st.session_state["start_sim"] = False
                st.session_state["current_sample"] = 0
                st.session_state["sim_complete"] = False
                st.session_state["recent_data"] = {
                    "time": [], "hr": [], "subject": [], "stress_prob": [], "prediction": []
                }
                st.session_state["stats"] = {
                    "total_predictions": 0, "stress_detections": 0,
                    "correct_predictions": 0, "by_subject": {}
                }
                st.rerun()
    
    with col2:
        if st.session_state["start_sim"]:
            progress = st.session_state["current_sample"] / total_samples
            st.progress(progress, text=f"进度: {st.session_state['current_sample']}/{total_samples}")
    
    # 显示特征信息
    with st.expander("📋 模型特征信息", expanded=False):
        st.write(f"模型使用 {len(MODEL_FEATURES)} 个特征:")
        cols = st.columns(4)
        for i, feature in enumerate(MODEL_FEATURES):
            with cols[i % 4]:
                st.text(f"{i+1}. {feature}")
        
        # 显示特征重要性（如果有）
        if hasattr(model, 'feature_importances_'):
            st.write("特征重要性:")
            importance_df = pd.DataFrame({
                '特征': MODEL_FEATURES,
                '重要性': model.feature_importances_
            }).sort_values('重要性', ascending=False)
            st.dataframe(importance_df.head(10), use_container_width=True)
    
    # 模拟循环
    if st.session_state["start_sim"] and not st.session_state["sim_complete"]:
        i = st.session_state["current_sample"]
        
        if i < total_samples:
            try:
                # 获取当前样本
                current_features = X_test[i].reshape(1, -1)
                current_hr = hr_data[i]
                current_subject = f"S{subject_ids[i]}"
                current_time = test_df.iloc[i]["Sim_Time"].strftime("%H:%M:%S")
                true_label = y_test[i]
                true_state = "压力" if true_label == 1 else "正常"
                
                # 模型预测
                pred_label = model.predict(current_features)[0]
                pred_prob = model.predict_proba(current_features)[0][1]
                pred_state = "⚠️ 压力" if pred_label == 1 else "✅ 正常"
                pred_color = "#FF5252" if pred_label == 1 else "#4CAF50"
                
                # 更新统计数据
                st.session_state["stats"]["total_predictions"] += 1
                if pred_label == 1:
                    st.session_state["stats"]["stress_detections"] += 1
                if pred_label == true_label:
                    st.session_state["stats"]["correct_predictions"] += 1
                
                # 更新受试者统计
                if current_subject not in st.session_state["stats"]["by_subject"]:
                    st.session_state["stats"]["by_subject"][current_subject] = {
                        "total": 0, "stress": 0, "correct": 0
                    }
                st.session_state["stats"]["by_subject"][current_subject]["total"] += 1
                if pred_label == 1:
                    st.session_state["stats"]["by_subject"][current_subject]["stress"] += 1
                if pred_label == true_label:
                    st.session_state["stats"]["by_subject"][current_subject]["correct"] += 1
                
                # 更新实时数据
                st.session_state["recent_data"]["time"].append(current_time)
                st.session_state["recent_data"]["hr"].append(current_hr)
                st.session_state["recent_data"]["subject"].append(current_subject)
                st.session_state["recent_data"]["stress_prob"].append(pred_prob)
                st.session_state["recent_data"]["prediction"].append(pred_label)
                
                # 保持最近20个数据点
                max_points = 20
                for key in st.session_state["recent_data"]:
                    if len(st.session_state["recent_data"][key]) > max_points:
                        st.session_state["recent_data"][key].pop(0)
                
                # 显示当前状态
                st.markdown(f"""
                <div style='background-color:{pred_color}20; padding:15px; border-radius:10px; border-left:5px solid {pred_color}; margin:10px 0'>
                    <h3 style='color:{pred_color}; margin:0'>{pred_state}</h3>
                    <p style='margin:5px 0'>受试者: {current_subject} | 时间: {current_time}</p>
                    <p style='margin:5px 0'>压力概率: <strong>{pred_prob:.2%}</strong> | 真实状态: {true_state}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 创建两列布局
                col_left, col_right = st.columns([2, 1])
                
                with col_left:
                    # 实时心率图表
                    st.subheader("📈 实时心率监测")
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    
                    times = st.session_state["recent_data"]["time"]
                    hrs = st.session_state["recent_data"]["hr"]
                    stress_probs = st.session_state["recent_data"]["stress_prob"]
                    
                    # 绘制心率曲线
                    line = ax1.plot(times, hrs, color="#FF4B4B", linewidth=2, marker="o", markersize=4)[0]
                    ax1.set_xlabel("时间", fontsize=10)
                    ax1.set_ylabel("心率 (次/分)", fontsize=10, color="#FF4B4B")
                    ax1.tick_params(axis='y', labelcolor="#FF4B4B")
                    
                    # 添加第二个Y轴显示压力概率
                    ax2 = ax1.twinx()
                    ax2.plot(times, stress_probs, color="#FF9800", linewidth=1, linestyle="--", alpha=0.7)
                    ax2.set_ylabel("压力概率", fontsize=10, color="#FF9800")
                    ax2.tick_params(axis='y', labelcolor="#FF9800")
                    ax2.set_ylim([0, 1])
                    
                    # 添加压力阈值线
                    ax2.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='阈值 (0.5)')
                    
                    ax1.grid(alpha=0.3)
                    ax1.set_title("实时心率与压力概率", fontsize=12)
                    plt.xticks(rotation=45)
                    st.pyplot(fig1)
                
                with col_right:
                    # 核心指标
                    st.subheader("📊 核心指标")
                    
                    col_metric1, col_metric2 = st.columns(2)
                    with col_metric1:
                        st.metric("当前心率", f"{current_hr:.1f}")
                    with col_metric2:
                        st.metric("压力概率", f"{pred_prob:.2%}")
                    
                    # 显示其他重要特征
                    st.subheader("🔍 关键特征值")
                    
                    # 找到前5个最重要的特征（如果有）
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        top_indices = np.argsort(importance)[-5:][::-1]
                        
                        for idx in top_indices:
                            feature_name = MODEL_FEATURES[idx]
                            feature_value = current_features[0][idx]
                            st.text(f"{feature_name}: {feature_value:.2f}")
                    
                    # 显示决策信息
                    st.subheader("💡 决策分析")
                    if pred_label == 1:
                        st.warning("**检测到压力状态**")
                        st.markdown("""
                        - 🧘 建议进行深呼吸练习
                        - 🚶 起身活动5分钟
                        - 🎵 听舒缓音乐放松
                        """)
                    else:
                        st.success("**状态正常**")
                        st.markdown("保持良好的心理状态")
                
                # 显示统计信息
                with st.expander("📈 实时统计", expanded=False):
                    stats = st.session_state["stats"]
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("总预测数", stats["total_predictions"])
                    with col_stat2:
                        if stats["total_predictions"] > 0:
                            stress_rate = stats["stress_detections"] / stats["total_predictions"] * 100
                            st.metric("压力检测率", f"{stress_rate:.1f}%")
                        else:
                            st.metric("压力检测率", "0%")
                    with col_stat3:
                        if stats["total_predictions"] > 0:
                            accuracy = stats["correct_predictions"] / stats["total_predictions"] * 100
                            st.metric("准确率", f"{accuracy:.1f}%")
                        else:
                            st.metric("准确率", "0%")
                
                # 更新样本索引并设置延迟
                st.session_state["current_sample"] += 1
                time.sleep(0.5)  # 0.5秒延迟
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 处理样本 {i} 时出错: {str(e)}")
                st.session_state["start_sim"] = False
        else:
            # 模拟完成
            st.session_state["sim_complete"] = True
            st.session_state["start_sim"] = False
            st.success("🎉 模拟完成！")
            st.rerun()
    
    # 显示模拟完成后的总结
    if st.session_state["sim_complete"]:
        st.divider()
        st.subheader("📊 模拟完成总结")
        
        stats = st.session_state["stats"]
        total = stats["total_predictions"]
        
        if total > 0:
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("总样本数", total)
            with col_sum2:
                stress_count = stats["stress_detections"]
                stress_rate = stress_count / total * 100
                st.metric("压力检测数", f"{stress_count} ({stress_rate:.1f}%)")
            with col_sum3:
                accuracy = stats["correct_predictions"] / total * 100
                st.metric("总体准确率", f"{accuracy:.1f}%")
            
            # 受试者统计
            st.subheader("👥 各受试者表现")
            subject_stats = []
            for subject, data in stats["by_subject"].items():
                if data["total"] > 0:
                    subject_stats.append({
                        "受试者": subject,
                        "样本数": data["total"],
                        "压力检测数": data["stress"],
                        "压力检测率": f"{data['stress']/data['total']*100:.1f}%",
                        "准确率": f"{data['correct']/data['total']*100:.1f}%"
                    })
            
            if subject_stats:
                st.dataframe(pd.DataFrame(subject_stats), use_container_width=True)
        
        # 重新开始按钮
        if st.button("🔄 重新开始模拟", type="primary", use_container_width=True):
            st.session_state["start_sim"] = False
            st.session_state["current_sample"] = 0
            st.session_state["sim_complete"] = False
            st.session_state["recent_data"] = {
                "time": [], "hr": [], "subject": [], "stress_prob": [], "prediction": []
            }
            st.session_state["stats"] = {
                "total_predictions": 0, "stress_detections": 0,
                "correct_predictions": 0, "by_subject": {}
            }
            st.rerun()

# -------------------------- 8. 主函数 --------------------------
def main():
    st.set_page_config(
        page_title="青少年压力实时检测系统",
        page_icon="🎯",
        layout="wide"
    )
    
    # 页面标题
    st.title("🎯 青少年压力实时检测系统")
    st.markdown("### 基于测试集的实时压力检测模拟")
    st.divider()
    
    # 系统说明
    with st.expander("📖 系统说明（点击展开）", expanded=True):
        st.markdown(f"""
        **系统功能：**
        - 🧠 使用训练好的随机森林模型进行压力检测
        - 📊 基于测试集数据（后5个受试者：S13-S17）进行模拟
        - 📈 实时显示心率、压力概率等生理指标
        - ⚠️ 自动识别压力状态并提供干预建议
        
        **技术规格：**
        - 模型特征数量: **{len(MODEL_FEATURES)}** 个
        - 测试集样本: 约500-800个（60秒窗口）
        - 模拟频率: 每0.5秒处理一个样本
        - 检测阈值: 压力概率 > 50%
        
        **特征说明：**
        - 心率相关: HR_Mean, RMSSD, SDNN
        - 皮肤电活动: EDA_Mean, EDA_Max, EDA_Min, EDA_Std, SCR_Count, SCL_Mean
        - 温度相关: Temp_Mean, Temp_Max, Temp_Min, Temp_Std
        - 加速度: Acc_Magnitude_Mean, Acc_Magnitude_Std, Activity_Level
        - 频域特征: LF_HF_Ratio, VLF_Power, LF_Power, HF_Power
        
        **使用方法：**
        1. 点击"开始模拟"按钮启动实时检测
        2. 系统将按顺序处理测试集中的每个样本
        3. 观察实时心率和压力概率变化
        4. 压力状态超过阈值时系统会给出提示
        5. 模拟完成后可查看详细统计信息
        """)
    
    # 初始化session state
    init_session_state()
    
    # 加载模型
    st.write("正在加载模型...")
    model = load_model()
    
    if model is None:
        st.error("无法加载模型，请检查模型文件路径")
        return
    
    # 检查模型特征数量
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
        if expected_features != len(MODEL_FEATURES):
            st.warning(f"⚠️ 模型期望 {expected_features} 个特征，但配置了 {len(MODEL_FEATURES)} 个")
    
    # 加载测试集
    st.write("正在加载测试集...")
    test_df = load_test_data()
    
    if test_df is None:
        st.error("无法加载测试集，请检查数据文件路径")
        return
    
    # 显示测试集信息
    st.success(f"✅ 测试集加载完成: {len(test_df)} 个样本")
    with st.expander("📊 测试集统计信息", expanded=False):
        st.write(f"- 样本总数: {len(test_df)}")
        st.write(f"- 特征数量: {len(test_df.columns)}")
        st.write(f"- 受试者数量: {test_df['Subject_ID'].nunique()}")
        st.write(f"- 受试者列表: {sorted(test_df['Subject_ID'].unique().tolist())}")
        
        # 标签分布
        if 'Label' in test_df.columns:
            label_counts = test_df['Label'].value_counts().sort_index()
            label_names = {1: "基线", 2: "压力", 3: "愉悦", 4: "冥想"}
            st.write("- 标签分布:")
            for label, count in label_counts.items():
                if label in label_names:
                    percentage = count / len(test_df) * 100
                    st.write(f"  - {label_names[label]}({label}): {count} 个 ({percentage:.1f}%)")
    
    # 开始实时模拟
    st.divider()
    realtime_simulation(test_df, model)

# -------------------------- 9. 启动系统 --------------------------
if __name__ == "__main__":
    main()