import numpy as np
import pandas as pd
import neurokit2 as nk
import warnings
from scipy.signal import butter, filtfilt
from config import (
    INPUT_TIME_STEPS, OUTPUT_HOURS,
    HOURLY_SAMPLES, DOWN_SAMPLE_HOURS, DOWN_SAMPLE_SAMPLES,
    STEP_SIZE, TARGET_SR, FEATURE_DIM
)

# 低通滤波函数（保持不变）
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    elif normal_cutoff <= 0:
        normal_cutoff = 0.01
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def extract_downsample_features(downsample_df):
    """
    提取单个降采样窗口的特征（增加返回值维度校验，避免1维数组）
    """
    required_cols = ["BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z", "label"]
    missing_cols = [col for col in required_cols if col not in downsample_df.columns]
    if missing_cols or len(downsample_df) == 0:  # 增加空数据校验
        return np.zeros((FEATURE_DIM,))  # 明确返回1维数组（后续拼接为2维）

    features = []
    window_signal = downsample_df.copy()

    # --- 1. BVP特征（HRV_RMSSD、HRV_SDNN）---
    bvp_signal = window_signal["BVP"].values
    bvp_clean = butter_lowpass_filter(bvp_signal, cutoff=1.0, fs=TARGET_SR)
    try:
        peaks_dict = nk.ppg_findpeaks(bvp_clean, sampling_rate=TARGET_SR, method="kalidas2017")
        peaks = peaks_dict["PPG_Peaks"]
        if len(peaks) > 3:
            hrv = nk.hrv(peaks, sampling_rate=TARGET_SR, show=False, time_domain=True, frequency_domain=False)
            hrv_rmssd = hrv["HRV_RMSSD"].values[0] if "HRV_RMSSD" in hrv.columns and not np.isnan(hrv["HRV_RMSSD"].values[0]) else 0
            hrv_sdnn = hrv["HRV_SDNN"].values[0] if "HRV_SDNN" in hrv.columns and not np.isnan(hrv["HRV_SDNN"].values[0]) else 0
        else:
            hrv_rmssd = 0
            hrv_sdnn = 0
    except Exception as e:
        hrv_rmssd = 0
        hrv_sdnn = 0
    features.extend([hrv_rmssd, hrv_sdnn])

    # --- 2. EDA特征（SCR_Count、SCL_Mean）---
    eda_signal = window_signal["EDA"].values
    eda_clean = butter_lowpass_filter(eda_signal, cutoff=0.5, fs=TARGET_SR)
    try:
        eda_decomposed = nk.eda_phasic(eda_clean, sampling_rate=TARGET_SR, method="default")
        scr_peaks = nk.eda_findpeaks(eda_decomposed["EDA_Phasic"], sampling_rate=TARGET_SR)
        scr_count = len(scr_peaks["SCR_Peaks"]) if "SCR_Peaks" in scr_peaks else 0
        scl_mean = np.mean(eda_decomposed["EDA_Tonic"]) if "EDA_Tonic" in eda_decomposed.columns else np.mean(eda_clean)
    except Exception as e:
        scr_count = 0
        scl_mean = np.mean(eda_signal)
    features.extend([scr_count, scl_mean])

    # --- 3. TEMP特征（TEMP_Mean）---
    temp_signal = window_signal["TEMP"].values
    temp_mean = np.mean(temp_signal)
    features.append(temp_mean)

    # --- 4. ACC特征（ACC_Mean）---
    acc_x = window_signal["ACC_x"].values
    acc_y = window_signal["ACC_y"].values
    acc_z = window_signal["ACC_z"].values
    acc_magnitude = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    acc_mean = np.mean(acc_magnitude)
    features.append(acc_mean)

    # 数据清洗：明确转换为1维数组，避免维度丢失
    features_arr = np.array(features)[:FEATURE_DIM]
    features_arr = np.nan_to_num(features_arr, nan=0, posinf=0, neginf=0)
    return features_arr.reshape(-1,)  # 确保返回形状为(FEATURE_DIM,)

def extract_window_features(df):
    """
    核心修复：
    1. 增加降采样窗口数据完整性校验
    2. 确保数组转换后为2维（避免1维数组导致axis=1操作报错）
    3. 优化样本过滤逻辑，避免过滤后数据为空
    """
    # 计算窗口总需求（输入5天=24降采样步 + 输出48小时）
    total_input_samples = INPUT_TIME_STEPS * DOWN_SAMPLE_SAMPLES
    total_output_samples = OUTPUT_HOURS * HOURLY_SAMPLES
    total_window_samples = total_input_samples + total_output_samples

    # 最大起始索引（保证窗口连续性，步长1小时）
    max_start_idx = len(df) - total_window_samples
    if max_start_idx <= 0:
        raise ValueError(f"数据量不足，无法生成降采样窗口（需至少{total_window_samples}样本，当前仅{len(df)}样本）")

    X = []
    y = []
    start_indices = range(0, max_start_idx, STEP_SIZE * HOURLY_SAMPLES)

    for start_idx in start_indices:
        # ========== 1. 提取输入窗口：24个降采样时间步（增加数据完整性校验） ==========
        input_start = start_idx
        input_end = start_idx + total_input_samples
        # 校验输入窗口数据是否完整
        if input_end > len(df):
            continue
        input_window_df = df.iloc[input_start:input_end].copy()
        if len(input_window_df) != total_input_samples:
            continue

        # 拆分输入窗口为24个降采样步（增加每个步长的数据校验）
        downsample_input_features = []
        valid_downsample = True
        for step in range(INPUT_TIME_STEPS):
            step_start = step * DOWN_SAMPLE_SAMPLES
            step_end = step_start + DOWN_SAMPLE_SAMPLES
            # 校验单个降采样步的数据是否完整
            if step_end > len(input_window_df):
                valid_downsample = False
                break
            downsample_df = input_window_df.iloc[step_start:step_end].copy()
            if len(downsample_df) != DOWN_SAMPLE_SAMPLES:
                valid_downsample = False
                break

            # 提取该降采样步的特征
            downsample_feat = extract_downsample_features(downsample_df)
            # 校验特征是否有效（避免空特征）
            if len(downsample_feat) != FEATURE_DIM or np.sum(np.abs(downsample_feat)) == 0:
                valid_downsample = False
                break

            downsample_input_features.append(downsample_feat)

        # 若降采样特征无效，跳过当前样本
        if not valid_downsample or len(downsample_input_features) != INPUT_TIME_STEPS:
            continue

        # ========== 2. 提取输出窗口：48小时小时级标签（优化过滤，避免维度错误） ==========
        output_start = input_end
        output_end = output_start + total_output_samples
        # 校验输出窗口数据是否完整
        if output_end > len(df):
            continue
        output_window_df = df.iloc[output_start:output_end].copy()
        if len(output_window_df) != total_output_samples:
            continue

        # 拆分输出窗口为48个小时，生成标签
        hourly_output_labels = []
        valid_hourly = True
        for hour in range(OUTPUT_HOURS):
            hour_start = hour * HOURLY_SAMPLES
            hour_end = hour_start + HOURLY_SAMPLES
            # 校验单个小时的数据是否完整
            if hour_end > len(output_window_df):
                valid_hourly = False
                break
            hourly_label_df = output_window_df.iloc[hour_start:hour_end].copy()
            if len(hourly_label_df) != HOURLY_SAMPLES:
                valid_hourly = False
                break

            hourly_labels = hourly_label_df["label"].values
            if len(hourly_labels) == 0:
                valid_hourly = False
                break

            # 严格化标签判定（保留逻辑，增加数据校验）
            hour_stress_ratio = (hourly_labels == 1).mean()
            if len(hourly_output_labels) >= 1:
                prev_label = hourly_output_labels[-1]
                if prev_label == 1:
                    hour_stress = 1 if hour_stress_ratio >= 0.5 else 0
                else:
                    hour_stress = 1 if hour_stress_ratio >= 0.45 else 0
            else:
                hour_stress = 1 if hour_stress_ratio >= 0.45 else 0

            hourly_output_labels.append(hour_stress)

        # 若小时级标签无效，跳过当前样本
        if not valid_hourly or len(hourly_output_labels) != OUTPUT_HOURS:
            continue

        # ========== 3. 样本过滤（优化逻辑，避免过滤后数据为空） ==========
        hourly_output_labels_arr = np.array(hourly_output_labels, dtype=np.int32)
        # 校验数组维度（确保为1维）
        if hourly_output_labels_arr.ndim != 1:
            continue
        # 过滤全1/全0（保留逻辑，增加边界校验）
        sum_labels = hourly_output_labels_arr.sum()
        all_zeros = (sum_labels == 0)
        all_ones = (sum_labels == OUTPUT_HOURS)
        if all_zeros or all_ones:
            continue
        # 过滤波动过小（放宽条件，避免丢弃过多样本：波动值≥1即可）
        label_fluctuation = np.sum(np.abs(np.diff(hourly_output_labels_arr))) if len(hourly_output_labels_arr) > 1 else 1
        if label_fluctuation < 1:
            continue

        # ========== 4. 保存有效样本（核心：确保数组为2维，避免维度错误） ==========
        # 转换输入特征为2维数组（[INPUT_TIME_STEPS, FEATURE_DIM]）
        downsample_input_arr = np.array(downsample_input_features, dtype=np.float32)
        if downsample_input_arr.ndim != 2 or downsample_input_arr.shape != (INPUT_TIME_STEPS, FEATURE_DIM):
            continue
        # 转换输出标签为1维数组（[OUTPUT_HOURS,]）
        if hourly_output_labels_arr.ndim != 1 or hourly_output_labels_arr.shape != (OUTPUT_HOURS,):
            continue

        X.append(downsample_input_arr)
        y.append(hourly_output_labels_arr)

    # ========== 5. 最终数组转换（关键：确保返回2维/3维数组，避免axis=1报错） ==========
    # 校验样本列表是否为空
    if len(X) == 0 or len(y) == 0:
        raise ValueError("无有效样本生成（所有样本均不符合条件，可放宽标签判定阈值重试）")
    # 转换为numpy数组，明确维度
    X_arr = np.array(X, dtype=np.float32)  # 形状：[样本数, INPUT_TIME_STEPS, FEATURE_DIM]
    y_arr = np.array(y, dtype=np.int32)    # 形状：[样本数, OUTPUT_HOURS]
    # 校验最终数组维度
    if X_arr.ndim != 3 or y_arr.ndim != 2:
        raise ValueError(f"数组维度错误（X应为3维，当前{X_arr.ndim}维；y应为2维，当前{y_arr.ndim}维）")
    if X_arr.shape[1:] != (INPUT_TIME_STEPS, FEATURE_DIM) or y_arr.shape[1:] != (OUTPUT_HOURS,):
        raise ValueError(f"数组形状错误（X应为[,{INPUT_TIME_STEPS},{FEATURE_DIM}]，当前{X_arr.shape}；y应为[,{OUTPUT_HOURS}]，当前{y_arr.shape}）")

    return X_arr, y_arr

def extract_all_features(all_data):
    """
    批量提取所有受试者特征，增加错误捕获，避免单个受试者报错导致整体中断
    """
    from config import TRAIN_SUBJECTS, TEST_SUBJECTS

    # 训练集特征提取
    X_train_list = []
    y_train_list = []
    for subject_id in TRAIN_SUBJECTS:
        if subject_id not in all_data:
            print(f"❌ 受试者 {subject_id} 无数据，跳过")
            continue
        try:
            df = all_data[subject_id]
            X_subj, y_subj = extract_window_features(df)
            if len(X_subj) > 0 and len(y_subj) > 0 and X_subj.ndim == 3 and y_subj.ndim == 2:
                X_train_list.append(X_subj)
                y_train_list.append(y_subj)
                print(f"✅ 训练集受试者 {subject_id} 特征提取完成，共 {len(X_subj)} 个有效样本（含小时级差异）")
            else:
                print(f"⚠️  训练集受试者 {subject_id} 无有效特征样本，跳过")
        except Exception as e:
            print(f"❌ 训练集受试者 {subject_id} 特征提取失败：{str(e)[:100]}")
            continue

    # 测试集特征提取
    X_test_list = []
    y_test_list = []
    for subject_id in TEST_SUBJECTS:
        if subject_id not in all_data:
            print(f"❌ 受试者 {subject_id} 无数据，跳过")
            continue
        try:
            df = all_data[subject_id]
            X_subj, y_subj = extract_window_features(df)
            if len(X_subj) > 0 and len(y_subj) > 0 and X_subj.ndim == 3 and y_subj.ndim == 2:
                X_test_list.append(X_subj)
                y_test_list.append(y_subj)
                print(f"✅ 测试集受试者 {subject_id} 特征提取完成，共 {len(X_subj)} 个有效样本（含小时级差异）")
            else:
                print(f"⚠️  测试集受试者 {subject_id} 无有效特征样本，跳过")
        except Exception as e:
            print(f"❌ 测试集受试者 {subject_id} 特征提取失败：{str(e)[:100]}")
            continue

    # 合并数据（增加非空校验）
    if not X_train_list or not y_train_list:
        raise ValueError("训练集无有效特征样本，程序终止（可放宽标签判定阈值重试）")
    if not X_test_list or not y_test_list:
        raise ValueError("测试集无有效特征样本，程序终止（可放宽标签判定阈值重试）")

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # 输出汇总信息
    print(f"\n📊 特征提取汇总（降采样优化，含小时级差异）：")
    print(f"  - 训练集形状：{X_train.shape}（样本数×24降采样步×6特征）")
    print(f"  - 训练集标签形状：{y_train.shape}（样本数×48小时）")
    print(f"  - 测试集形状：{X_test.shape}（样本数×24降采样步×6特征）")
    print(f"  - 测试集标签形状：{y_test.shape}（样本数×48小时）")
    print(f"  - 训练集高压力小时数（整体）：{np.sum(y_train)}")
    print(f"  - 测试集高压力小时数（整体）：{np.sum(y_test)}")
    # 新增：统计单个样本的标签波动情况
    train_sample_fluctuation = [np.sum(np.abs(np.diff(y_train[i]))) for i in range(min(10, len(y_train)))]
    print(f"  - 训练集前10个样本标签波动值：{np.round(train_sample_fluctuation, 2)}（>0表示有小时级差异）")

    return X_train, y_train, X_test, y_test