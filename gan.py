import os
import json
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, LSTM, Concatenate, LeakyReLU
from tensorflow.keras.optimizers import Adam

os.makedirs("consequence/cgan", exist_ok=True)
os.makedirs("consequence/cgan/picture", exist_ok=True)

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
        records.append(record_date)

df_data = pd.DataFrame(records, columns=["date"])
df_data = df_data.set_index("date").sort_index()

forecast_steps = 7
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
    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss

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

    series = item_df['price'].values.reshape(-1,1)
    scaler = MinMaxScaler((0,1))
    scaled_series = scaler.fit_transform(series)

    X, Y = create_dataset(scaled_series, window_size)
    if len(X) == 0:
        print(f"Not enough data for {item_name}, skipping CGAN.")
        continue

    D = build_discriminator(window_size)
    G = build_generator(window_size, noise_dim)

    d_optimizer = Adam(0.0002, 0.5)
    g_optimizer = Adam(0.0002, 0.5)

    dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(len(X)).batch(batch_size)

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

    # 开始训练
    for epoch in range(epochs):
        d_loss_list = []
        g_loss_list = []
        for real_cond, real_price in dataset:
            d_loss, g_loss = train_step(real_cond, real_price)
            d_loss_list.append(d_loss.numpy())
            g_loss_list.append(g_loss.numpy())

        if (epoch+1) % 20 == 0:
            print(f"[{item_name}] Epoch {epoch+1}/{epochs} D_loss: {np.mean(d_loss_list):.4f} G_loss: {np.mean(g_loss_list):.4f}")

    # 预测未来
    last_window = scaled_series[-window_size:]
    # 将cond转为tf.tensor
    cond = tf.convert_to_tensor(last_window.reshape(1, window_size, 1), dtype=tf.float32)
    future_preds = []
    for _ in range(forecast_steps):
        noise = tf.random.normal([1, noise_dim])
        pred_price = G([cond, noise], training=False)
        future_preds.append(pred_price[0][0].numpy())
        # 使用tf操作更新cond
        cond = tf.concat([cond[:,1:,:], tf.reshape(pred_price[0][0],[1,1,1])], axis=1)

    future_preds = np.array(future_preds).reshape(-1,1)
    future_preds_inv = scaler.inverse_transform(future_preds)
    future_dates = pd.date_range(start=item_df.index.max() + pd.Timedelta(days=1), periods=forecast_steps)
    forecast_df = pd.DataFrame(future_preds_inv, index=future_dates, columns=["forecast_price"])

    future_path = os.path.join("consequence/cgan", f"future_cgan_{item_name}.csv")
    forecast_df.to_csv(future_path, date_format='%Y-%m-%d')
    print(f"✅ CGAN未来预测已保存: {future_path}")

    complete_df = pd.concat([item_df[['price']], forecast_df], axis=1)
    complete_df['forecast_price'] = complete_df['forecast_price'].fillna('')
    complete_path = os.path.join("consequence/cgan", f"complete_cgan_{item_name}.csv")
    complete_df.to_csv(complete_path, date_format='%Y-%m-%d')
    print(f"✅ CGAN完整序列已保存: {complete_path}")

    plt.figure(figsize=(10,5))
    plt.plot(item_df.index, item_df['price'], label='Historical Price', color='blue')
    plt.plot(forecast_df.index, forecast_df['forecast_price'], label='CGAN Forecast', color='purple')
    plt.title(f"CGAN Price Forecast for {item_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    picture_dir = "consequence/cgan/picture"
    os.makedirs(picture_dir, exist_ok=True)
    picture_path = os.path.join(picture_dir, f"{item_name}_cgan_forecast_plot.png")
    plt.savefig(picture_path, dpi=100)
    plt.close()
    print(f"✅ CGAN预测图已保存: {picture_path}")
