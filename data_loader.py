import pickle
import pandas as pd
import numpy as np
from scipy.signal import resample
from config import WESAD_DATA_PATH, SUBJECTS, SAMPLING_RATES, TARGET_SR

def load_subject_data(subject_id):
    """
    加载单个受试者数据（4Hz采样，为降采样做准备，无额外耗时操作）
    """
    data_path = f"{WESAD_DATA_PATH}/{subject_id}/{subject_id}.pkl"
    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    except:
        with open(data_path, "rb") as f:
            data = pickle.load(f)

    wrist_signal = data["signal"]["wrist"]
    labels = np.squeeze(data["label"])
    target_sr = TARGET_SR
    target_length = len(labels) // (SAMPLING_RATES["BVP"] // target_sr)

    unified_signals = {}

    # 1. 处理BVP（64Hz → 4Hz，重采样）
    if "BVP" in wrist_signal:
        bvp = np.squeeze(wrist_signal["BVP"])
        bvp_resampled = resample(bvp, target_length)
        unified_signals["BVP"] = bvp_resampled

    # 2. 处理EDA（4Hz，截取到目标长度）
    if "EDA" in wrist_signal:
        eda = np.squeeze(wrist_signal["EDA"])
        eda_unified = np.zeros(target_length)
        eda_len = min(len(eda), target_length)
        eda_unified[:eda_len] = eda[:eda_len]
        unified_signals["EDA"] = eda_unified

    # 3. 处理TEMP（4Hz，截取到目标长度）
    if "TEMP" in wrist_signal:
        temp = np.squeeze(wrist_signal["TEMP"])
        temp_unified = np.zeros(target_length)
        temp_len = min(len(temp), target_length)
        temp_unified[:temp_len] = temp[:temp_len]
        unified_signals["TEMP"] = temp_unified

    # 4. 处理ACC（32Hz → 4Hz，重采样后拆分x/y/z）
    if "ACC" in wrist_signal:
        acc = np.squeeze(wrist_signal["ACC"])
        if len(acc.shape) == 1:
            acc = acc.reshape(-1, 3)
        acc_resampled = resample(acc, target_length)
        unified_signals["ACC_x"] = acc_resampled[:, 0]
        unified_signals["ACC_y"] = acc_resampled[:, 1]
        unified_signals["ACC_z"] = acc_resampled[:, 2]

    # 5. 处理标签（64Hz → 4Hz，降采样）
    labels_unified = labels[::(SAMPLING_RATES["BVP"] // target_sr)][:target_length]
    unified_signals["label"] = labels_unified.astype(int)

    # 创建DataFrame
    df = pd.DataFrame(unified_signals)
    df = df[df["label"].isin([0, 1])]
    df = df.reset_index(drop=True)

    # 验证数据有效性（满足降采样窗口需求）
    min_required_samples = (INPUT_DAYS * HOURS_PER_DAY + OUTPUT_HOURS) * HOURLY_SAMPLES
    if len(df) < min_required_samples:
        raise ValueError(f"数据量不足（总长度{len(df)}），无法满足降采样窗口需求（需至少{min_required_samples}样本）")

    return df

def load_all_subjects_data():
    """
    加载所有受试者数据，返回字典
    """
    from config import INPUT_DAYS, HOURS_PER_DAY, OUTPUT_HOURS, HOURLY_SAMPLES
    all_data = {}
    for subject in SUBJECTS:
        try:
            df = load_subject_data(subject)
            all_data[subject] = df
            print(f"✅ 成功加载受试者 {subject} 数据：")
            print(f"  - 数据长度：{len(df)} 条（4Hz采样）")
            print(f"  - 可拆分小时数：{len(df) // HOURLY_SAMPLES} 小时")
            print(f"  - 标签分布：{df['label'].value_counts().to_dict()}\n")
        except Exception as e:
            print(f"❌ 加载受试者 {subject} 失败：{str(e)}\n")
    return all_data

# 导入降采样配置参数
from config import INPUT_DAYS, HOURS_PER_DAY, OUTPUT_HOURS, HOURLY_SAMPLES

# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("📌 开始加载WESAD数据集（为降采样优化）")
    print("=" * 50)
    all_data = load_all_subjects_data()