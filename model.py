import tensorflow as tf
import numpy as np
import json
import os

# 建立 model 資料夾（如果不存在）
os.makedirs('./model', exist_ok=True)

# 載入 Fashion-MNIST 資料集
print("載入 Fashion-MNIST 資料集...")
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 正規化資料 (0-255 -> 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

print(f"訓練資料形狀: {train_images.shape}")
print(f"測試資料形狀: {test_images.shape}")

# 建立模型（只使用 Dense、ReLU、Softmax）
print("建立神經網路模型...")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 顯示模型架構
model.summary()

# 訓練模型
print("開始訓練模型...")
history = model.fit(
    train_images, train_labels,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# 評估模型
print("評估模型性能...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"測試準確率: {test_accuracy:.4f}")

# === 步驟 1: 儲存為 .h5 格式 ===
print("儲存模型為 .h5 格式...")
model.save('./model/fashion_mnist.h5')

# === 步驟 2: 轉換為 JSON 和 NPZ 格式 ===
print("轉換模型格式...")

# 2.1 儲存模型架構為 JSON
model_json = model.to_json()
with open('./model/fashion_mnist.json', 'w') as json_file:
    json_file.write(model_json)

# 2.2 提取權重並儲存為 NPZ 格式
weights_dict = {}
for i, layer in enumerate(model.layers):
    if len(layer.get_weights()) > 0:  # 只處理有權重的層
        layer_weights = layer.get_weights()
        # 儲存權重和偏差
        if len(layer_weights) >= 2:
            weights_dict[f'{layer.name}/kernel:0'] = layer_weights[0]
            weights_dict[f'{layer.name}/bias:0'] = layer_weights[1]

# 儲存權重為 NPZ 格式
np.savez('./model/fashion_mnist.npz', **weights_dict)

print("模型轉換完成！")
print("生成的檔案：")
print("- ./model/fashion_mnist.h5")
print("- ./model/fashion_mnist.json")
print("- ./model/fashion_mnist.npz")

# === 驗證轉換結果 ===
print("\n驗證轉換結果...")

# 載入 JSON 架構
with open('./model/fashion_mnist.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# 載入權重
loaded_weights = np.load('./model/fashion_mnist.npz')

print(f"JSON 架構檔案大小: {len(loaded_model_json)} 字元")
print(f"NPZ 權重檔案包含 {len(loaded_weights.files)} 個權重陣列")
print("權重檔案中的鍵值:")
for key in loaded_weights.files:
    print(f"  - {key}: {loaded_weights[key].shape}")

print("\n模型訓練和轉換完成！")
