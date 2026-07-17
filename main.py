import os
import numpy as np
import pandas as pd
import warnings
import absl.logging

# 禁用无关警告，提升运行效率
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU（轻量模型CPU训练足够高效，避免GPU初始化耗时）
warnings.filterwarnings("ignore")

# 创建输出目录
from config import OUTPUT_PATH
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# 导入核心函数
from data_loader import load_all_subjects_data
from feature_extractor import extract_all_features
from lstm_model import (
    standardize_data,
    train_lstm_model,
    save_48h_prediction_results
)

def main():
    print("=" * 80)
    print("📚 青少年压力智能预警系统 - 高效小时级预测（24降采样步→48小时，减少训练时间）")
    print("=" * 80)

    try:
        # ===================== 步骤1：加载WESAD数据集（高效，无冗余） =====================
        print("\n🔧 步骤1/5：加载WESAD数据集...")
        all_data = load_all_subjects_data()
        if not all_data:
            print("❌ 无有效数据加载，程序终止")
            return

        # ===================== 步骤2：提取降采样特征（高效，24步输入） =====================
        print("\n🔧 步骤2/5：提取降采样生理信号特征（24步→48小时输出）...")
        X_train, y_train, X_test, y_test = extract_all_features(all_data)

        # 保存高效特征数据
        np.save(f"{OUTPUT_PATH}/X_train_efficient.npy", X_train)
        np.save(f"{OUTPUT_PATH}/y_train_efficient.npy", y_train)
        np.save(f"{OUTPUT_PATH}/X_test_efficient.npy", X_test)
        np.save(f"{OUTPUT_PATH}/y_test_efficient.npy", y_test)
        print(f"\n✅ 降采样特征数据已保存到{OUTPUT_PATH}目录")

        # ===================== 步骤3：数据标准化（高效，适配3维输入） =====================
        print("\n🔧 步骤3/5：数据标准化（高效处理降采样数据）...")
        X_train_3d, X_test_3d, scaler = standardize_data(X_train, X_test)
        np.save(f"{OUTPUT_PATH}/scaler_mean_efficient.npy", scaler.mean_)
        np.save(f"{OUTPUT_PATH}/scaler_scale_efficient.npy", scaler.scale_)
        print(f"✅ 数据标准化完成，标准化器参数已保存")

        # ===================== 步骤4：训练轻量型LSTM模型（高效，减少训练时间） =====================
        print("\n🔧 步骤4/5：训练轻量型LSTM模型（高效收敛，小时级输出）...")
        model, history, y_pred, y_pred_prob = train_lstm_model(
            X_train_3d,
            y_train,
            X_test_3d,
            y_test,
            OUTPUT_PATH
        )

        # ===================== 步骤5：保存小时级详细结果（高效，无冗余） =====================
        print("\n🔧 步骤5/5：保存小时级详细预测结果...")
        save_48h_prediction_results(y_test, y_pred, y_pred_prob, OUTPUT_PATH)

        # ===================== 最终结果汇总 =====================
        print("\n" + "=" * 80)
        print(f"🎉 全流程执行完成！核心输出文件汇总（高效，小时级预测）")
        print("=" * 80)
        print(f"  1. 最优轻量型LSTM模型：{OUTPUT_PATH}/best_efficient_lstm_48h_model.keras")
        print(f"  2. 小时级全局指标：{OUTPUT_PATH}/efficient_hourly_global_metrics_48h.csv")
        print(f"  3. 小时级详细预测结果：{OUTPUT_PATH}/efficient_hourly_detailed_prediction_48h.csv")
        print(f"  4. 标准化器参数：{OUTPUT_PATH}/scaler_mean_efficient.npy & scaler_scale_efficient.npy")
        print(f"  5. 降采样特征数据：{OUTPUT_PATH}/X_train_efficient.npy & X_test_efficient.npy")

    except Exception as e:
        print(f"\n❌ 程序执行出错：{str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()