import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import warnings

warnings.filterwarnings("ignore")

# ===================== 1. 数据标准化（保持高效） =====================
def standardize_data(X_train, X_test):
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]

    X_train_reshaped = X_train.reshape(-1, n_features)
    X_test_reshaped = X_test.reshape(-1, n_features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train_3d = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    X_test_3d = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

    return X_train_3d, X_test_3d, scaler

# ===================== 2. 构建轻量型LSTM模型（核心修改：增加双向LSTM，捕捉小时级差异） =====================
def build_lstm_model(input_shape, output_units=48):
    """
    修改点：
    1. 改用双向LSTM，增强局部时序差异捕捉能力
    2. 增加输出层前的Dropout，减少过拟合（避免学习全1/全0）
    3. 保持轻量化，不增加过多训练时间
    """
    model = Sequential([
        Input(shape=input_shape),
        # 修改：双向LSTM，增强局部差异捕捉（仅增加少量计算量）
        Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1, kernel_regularizer=tf.keras.regularizers.l2(0.0005))),
        BatchNormalization(),
        # 保持简化全连接层
        Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
        Dropout(0.25),  # 轻微提高Dropout，减少同质化输出
        # 输出层：48个单元，sigmoid激活
        Dense(output_units, activation='sigmoid')
    ])

    # 优化器配置（保持高效收敛）
    optimizer = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    return model

# ===================== 3. 模型训练（核心修改：增加输出验证，统计全1/全0比例） =====================
def train_lstm_model(X_train, y_train, X_test, y_test, output_path):
    # 构建模型
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape, output_units=48)

    # 高效训练回调
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        f"{output_path}/best_efficient_lstm_48h_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    # 类别权重平衡
    y_train_flat = y_train.flatten()
    class_0_count = len(y_train_flat[y_train_flat == 0])
    class_1_count = len(y_train_flat[y_train_flat == 1])

    if class_1_count == 0:
        class_weight = {0: 1.0, 1: 1.0}
    else:
        class_weight = {
            0: 1.0,
            1: (class_0_count / class_1_count) * 1.5
        }
    print(f"\n⚖️  类别权重（高效训练）：平静={class_weight[0]:.2f}，高压力={class_weight[1]:.2f}")

    # 模型训练
    print("\n🚀 开始训练轻量型LSTM模型（24降采样步→48小时小时级输出，高效收敛）")
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[early_stop, model_checkpoint, reduce_lr],
        class_weight=class_weight,
        verbose=1
    )

    # 测试集评估
    print("\n📊 测试集模型评估（高效，小时级）")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"  - 测试集损失：{test_loss:.4f}")
    print(f"  - 测试集二分类准确率（小时级）：{test_acc:.4f}")

    # 预测结果
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # ===================== 核心修改：增加全1/全0样本统计（验证是否解决问题） =====================
    print("\n📊 输出样本同质化统计（关键：验证全1/全0问题）")
    test_all_zeros = 0
    test_all_ones = 0
    test_has_fluctuation = 0
    for i in range(len(y_pred)):
        pred_sample = y_pred[i]
        sum_pred = pred_sample.sum()
        if sum_pred == 0:
            test_all_zeros += 1
        elif sum_pred == 48:
            test_all_ones += 1
        else:
            test_has_fluctuation += 1

    print(f"  - 测试集全0输出样本数：{test_all_zeros}（{test_all_zeros/len(y_pred)*100:.2f}%）")
    print(f"  - 测试集全1输出样本数：{test_all_ones}（{test_all_ones/len(y_pred)*100:.2f}%）")
    print(f"  - 测试集有小时级差异的样本数：{test_has_fluctuation}（{test_has_fluctuation/len(y_pred)*100:.2f}%）")

    # 计算评估指标
    # 1. 样本级指标
    sample_accuracy = []
    sample_f1 = []
    for i in range(len(y_test)):
        if np.sum(y_test[i]) == 0 and np.sum(y_pred[i]) == 0:
            acc = 1.0
            f1 = 1.0
        else:
            acc = accuracy_score(y_test[i], y_pred[i])
            f1 = f1_score(y_test[i], y_pred[i], zero_division=0)
        sample_accuracy.append(acc)
        sample_f1.append(f1)

    print(f"\n📊 48小时预测样本级指标（平均）：")
    print(f"  - 平均样本准确率：{np.mean(sample_accuracy):.4f}")
    print(f"  - 平均样本F1分数：{np.mean(sample_f1):.4f}")

    # 2. 小时级全局指标
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    global_acc = accuracy_score(y_test_flat, y_pred_flat)
    global_f1 = f1_score(y_test_flat, y_pred_flat, zero_division=0)

    print(f"\n📊 小时级全局指标（所有样本汇总）：")
    print(f"  - 全局准确率：{global_acc:.4f}")
    print(f"  - 全局F1分数：{global_f1:.4f}")

    # 保存小时级指标
    hourly_metrics = {}
    for hour in range(48):
        y_test_hour = y_test[:, hour]
        y_pred_hour = y_pred[:, hour]
        acc = accuracy_score(y_test_hour, y_pred_hour)
        f1 = f1_score(y_test_hour, y_pred_hour, zero_division=0)
        hourly_metrics[hour+1] = {"accuracy": acc, "f1": f1}

    hourly_metrics_df = pd.DataFrame.from_dict(hourly_metrics, orient='index')
    hourly_metrics_df.index.name = "未来小时数"
    hourly_metrics_df.to_csv(f"{output_path}/efficient_hourly_global_metrics_48h.csv", index=True)
    print(f"\n💾 小时级全局指标已保存到：{output_path}/efficient_hourly_global_metrics_48h.csv")

    # 返回结果
    return model, history, y_pred, y_pred_prob

# ===================== 4. 保存小时级详细结果（增加单个样本示例，验证差异） =====================
def save_48h_prediction_results(y_test, y_pred, y_pred_prob, output_path):
    """
    增加：保存前5个样本的完整48小时标签，直观验证是否有差异
    """
    sample_num = min(100, len(y_test))
    prediction_data = []
    for sample_idx in range(sample_num):
        for hour in range(48):
            prediction_data.append({
                "样本编号": sample_idx,
                "未来小时数": hour+1,
                "真实压力状态": y_test[sample_idx, hour],
                "预测压力状态": y_pred[sample_idx, hour],
                "预测压力概率": y_pred_prob[sample_idx, hour]
            })

    # 保存详细结果
    pred_df = pd.DataFrame(prediction_data)
    pred_df.to_csv(f"{output_path}/efficient_hourly_detailed_prediction_48h.csv", index=False)

    # 额外：保存前5个样本的48小时标签汇总，直观验证
    sample_summary_data = []
    for sample_idx in range(min(5, len(y_test))):
        true_label_sum = y_test[sample_idx].sum()
        pred_label_sum = y_pred[sample_idx].sum()
        label_fluctuation = np.sum(np.abs(np.diff(y_pred[sample_idx])))
        sample_summary_data.append({
            "样本编号": sample_idx,
            "真实高压力小时数": true_label_sum,
            "预测高压力小时数": pred_label_sum,
            "预测标签波动值": label_fluctuation,
            "是否有小时级差异": "是" if label_fluctuation > 0 else "否"
        })

    sample_summary_df = pd.DataFrame(sample_summary_data)
    sample_summary_df.to_csv(f"{output_path}/sample_hourly_fluctuation_summary.csv", index=False)

    print(f"\n💾 小时级详细预测结果已保存到：{output_path}/efficient_hourly_detailed_prediction_48h.csv")
    print(f"💾 前5个样本波动汇总已保存到：{output_path}/sample_hourly_fluctuation_summary.csv")

# 导入配置参数
from config import EPOCHS, BATCH_SIZE, VAL_SPLIT, OUTPUT_PATH