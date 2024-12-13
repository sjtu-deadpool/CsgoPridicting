import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam

# 确保目录存在
os.makedirs("verification/gan", exist_ok=True)
os.makedirs("verification/gan/picture", exist_ok=True)
os.makedirs("verification/gan/对比曲线", exist_ok=True)

price_dir = "item_smooth_prices"
items = [
    "AK-47__Redline_(Battle-Scarred)",
    "AK-47__Blue_Laminate_(Field-Tested)",
    "AK-47__Asiimov_(Factory_New)",
    "AK-47__Nightwish_(Factory_New)"
]

data_complete_file = "data_complete.json"
records = []
with open(data_complete_file, "r", encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        record = json.loads(line)
        record_date = datetime.strptime(record["date"], "%Y-%m-%d")
        records.append([record_date])

df_data = pd.DataFrame(records, columns=["date"])
df_data = df_data.set_index("date").sort_index()

test_start = datetime(2024, 12, 1)
test_end = datetime(2024, 12, 31)
plot_start = datetime(2024, 11, 1)  # 从11月1日开始绘图
window_size = 30
noise_dim = 10  # 噪声维度
epochs = 100
batch_size = 32

def create_dataset(series, window_size=30):
    X, Y = [], []
    for i in range(len(series)-window_size):
        X.append(series[i:i+window_size])
        Y.append(series[i+window_size])
    return np.array(X), np.array(Y)

def build_generator(window_size, noise_dim):
    cond_input = Input(shape=(window_size,1), name="condition_input")
    cond_feat = LSTM(50)(cond_input)
    noise_input = Input(shape=(noise_dim,), name="noise_input")
    x = Concatenate()([cond_feat, noise_input])
    x = Dense(50)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='linear')(x)
    model = Model([cond_input, noise_input], out, name="Generator")
    return model

def build_discriminator(window_size):
    cond_input = Input(shape=(window_size,1), name="condition_input_d")
    cond_feat = LSTM(50)(cond_input)
    price_input = Input(shape=(1,), name="price_input_d")
    x = Concatenate()([cond_feat, price_input])
    x = Dense(50)(x)
    x = LeakyReLU(alpha=0.2)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model([cond_input, price_input], out, name="Discriminator")
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def d_loss_function(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return (real_loss + fake_loss) * 0.5

def g_loss_function(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

for item_name in items:
    item_path = os.path.join(price_dir, f"{item_name}.csv")
    if not os.path.exists(item_path):
        print(f"{item_path} not found, skipping {item_name}")
        continue

    item_df = pd.read_csv(item_path, parse_dates=["date"], index_col="date")
    if item_df.empty:
        print(f"No data for {item_name}, skipping CGAN forecast.")
        continue

    train_df = item_df[item_df.index < test_start]
    test_df = item_df[(item_df.index >= test_start) & (item_df.index <= test_end)]

    if test_df.empty or len(train_df) < window_size:
        print(f"Not enough train or test data for {item_name}, skipping.")
        continue

    series = train_df['price'].values.reshape(-1,1)
    scaler = MinMaxScaler((0,1))
    scaled_train = scaler.fit_transform(series)

    X_train, Y_train = create_dataset(scaled_train, window_size)
    if len(X_train) == 0:
        print(f"Not enough training data after windowing for {item_name}, skipping.")
        continue

    D = build_discriminator(window_size)
    G = build_generator(window_size, noise_dim)

    d_optimizer = Adam(0.0002, 0.5)
    g_optimizer = Adam(0.0002, 0.5)

    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).shuffle(len(X_train)).batch(batch_size)

    @tf.function
    def train_step(real_cond, real_price):
        noise = tf.random.normal([tf.shape(real_cond)[0], noise_dim])
        with tf.GradientTape() as tape_d, tf.GradientTape() as tape_g:
            fake_price = G([real_cond, noise], training=True)

            real_output = D([real_cond, real_price], training=True)
            fake_output = D([real_cond, fake_price], training=True)

            d_loss = d_loss_function(real_output, fake_output)
            g_loss = g_loss_function(fake_output)
        
        gradients_of_d = tape_d.gradient(d_loss, D.trainable_variables)
        d_optimizer.apply_gradients(zip(gradients_of_d, D.trainable_variables))

        gradients_of_g = tape_g.gradient(g_loss, G.trainable_variables)
        g_optimizer.apply_gradients(zip(gradients_of_g, G.trainable_variables))

        return d_loss, g_loss

    for epoch in range(epochs):
        d_loss_list = []
        g_loss_list = []
        for real_cond, real_price in dataset:
            d_loss, g_loss = train_step(real_cond, real_price)
            d_loss_list.append(d_loss.numpy())
            g_loss_list.append(g_loss.numpy())

        if (epoch+1) % 20 == 0:
            print(f"[{item_name}] Epoch {epoch+1}/{epochs} D_loss: {np.mean(d_loss_list):.4f}, G_loss: {np.mean(g_loss_list):.4f}")

    # 测试集预测
    test_series = test_df['price'].values.reshape(-1,1)
    scaled_test = scaler.transform(test_series)
    full_series = np.concatenate([scaled_train, scaled_test], axis=0)

    test_start_idx = len(scaled_train)
    real_values = test_df['price'].values
    test_dates = test_df.index

    # 初始cond为tf.tensor
    cond = tf.convert_to_tensor(full_series[test_start_idx - window_size:test_start_idx].reshape(1, window_size, 1), dtype=tf.float32)
    future_preds = []
    for _ in range(len(test_df)):
        noise = tf.random.normal([1, noise_dim])
        pred_price = G([cond, noise], training=False)
        pred_val = pred_price[0][0]
        future_preds.append(pred_val.numpy())
        # 使用tf操作更新cond
        cond = tf.concat([cond[:,1:,:], tf.reshape(pred_val,[1,1,1])], axis=1)

    future_preds = np.array(future_preds).reshape(-1,1)
    future_preds_inv = scaler.inverse_transform(future_preds)

    # 计算误差指标
    mse = mean_squared_error(real_values, future_preds_inv[:,0])
    mae = mean_absolute_error(real_values, future_preds_inv[:,0])
    rmse = np.sqrt(mse)

    print(f"{item_name} - December CGAN forecast metrics:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # 保存结果
    result_df = pd.DataFrame({
        "date": test_dates,
        "real_price": real_values,
        "predicted_price": future_preds_inv[:,0]
    })
    result_out_path = os.path.join("verification/gan", f"compare_{item_name}.csv")
    result_df.to_csv(result_out_path, index=False, date_format='%Y-%m-%d')
    print(f"✅ CGAN预测结果与真实值已保存: {result_out_path}")

    text_str = f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}"

    # 绘制图1：仅12月
    plt.figure(figsize=(10,5))
    plt.plot(test_dates, real_values, label='Real Price', color='blue')
    plt.plot(test_dates, future_preds_inv[:,0], label='Predicted Price (CGAN)', color='purple')
    plt.title(f"{item_name} December CGAN Forecast vs Real")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.text(test_dates.min(), max(real_values.max(), future_preds_inv[:,0].max())*0.9, text_str,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    picture_path = os.path.join("verification/gan/picture", f"{item_name}_verification_plot_december.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ CGAN 12月对比图已保存: {picture_path}")

    # 绘制图2：从11月开始
    plt.figure(figsize=(10,5))
    extended_plot_df = item_df.loc[(item_df.index >= plot_start) & (item_df.index <= test_end)]
    plt.plot(extended_plot_df.index, extended_plot_df['price'], label='Historical Price', color='blue')
    plt.plot(test_dates, future_preds_inv[:,0], label='Predicted Price (CGAN)', color='purple')
    plt.title(f"{item_name} From Nov to Dec CGAN Forecast vs Real")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.text(plot_start, max(extended_plot_df['price'].max(), future_preds_inv[:,0].max())*0.9, text_str,
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

    picture_path_nov = os.path.join("verification/gan/picture", f"{item_name}_verification_plot_from_nov.png")
    plt.savefig(picture_path_nov, dpi=100)
    plt.close()
    print(f"✅ CGAN从11月开始的对比图已保存: {picture_path_nov}")

    metrics_path = os.path.join("verification/gan/对比曲线", f"{item_name}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as mf:
        mf.write(f"MSE: {mse:.4f}\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}\n")
    print(f"✅ CGAN指标已保存: {metrics_path}")
