import tensorflow as tf
import numpy as np
import json
import os
import gzip

# --- 編輯區 ---
DATA_PATH = './data/fashion'
MODEL_DIR = './model'
# ---------------

def load_local_fashion_mnist(path):
    # (此處程式碼與上一版相同，為簡潔省略)
    print(f"從本地路徑 '{path}' 載入 Fashion-MNIST 資料...")
    train_labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    train_images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    with gzip.open(train_labels_path, 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(train_images_path, 'rb') as imgpath:
        train_images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 28, 28)
    with gzip.open(test_labels_path, 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(test_images_path, 'rb') as imgpath:
        test_images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 28, 28)
    print("資料載入完成。")
    return (train_images, train_labels), (test_images, test_labels)

# 主程式
os.makedirs(MODEL_DIR, exist_ok=True)
(train_images, train_labels), (test_images, test_labels) = load_local_fashion_mnist(DATA_PATH)
train_images, test_images = train_images / 255.0, test_images / 255.0

print("\n建立神經網路模型...")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten'),
    tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
    tf.keras.layers.Dense(64, activation='relu', name='dense_2'),
    tf.keras.layers.Dense(10, activation='softmax', name='output')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\n開始訓練模型...")
model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_data=(test_images, test_labels), verbose=1)

# === 關鍵：儲存與轉換模型 ===
print("\n開始儲存與轉換模型...")

# 1. 儲存 H5 檔案 (備用)
model.save(os.path.join(MODEL_DIR, 'fashion_mnist.h5'))

# 2. 導出 JSON 架構檔
model_config = json.loads(model.to_json())['config']['layers']
formatted_arch = []
for layer_conf in model_config:
    layer_name = layer_conf['config']['name']
    layer_weights = []
    # 只有 Dense 層有權重，Flatten 層沒有
    if layer_conf['class_name'] == 'Dense':
        # 確保權重名稱與 nn_predict.py 的預期一致
        layer_weights.append(f'{layer_name}/kernel:0')
        layer_weights.append(f'{layer_name}/bias:0')
    
    formatted_arch.append({
        'name': layer_name,
        'type': layer_conf['class_name'],
        'config': layer_conf['config'],
        'weights': layer_weights
    })

json_path = os.path.join(MODEL_DIR, 'fashion_mnist.json')
with open(json_path, 'w') as f:
    json.dump(formatted_arch, f, indent=4)
print(f"-> 模型架構已儲存至: {json_path}")

# 3. 導出 NPZ 權重檔
weights_dict = {}
for layer in model.layers:
    if layer.get_weights():
        weights = layer.get_weights()
        # 使用與 JSON 中完全一致的名稱作為 key
        weights_dict[f'{layer.name}/kernel:0'] = weights[0]
        weights_dict[f'{layer.name}/bias:0'] = weights[1]

npz_path = os.path.join(MODEL_DIR, 'fashion_mnist.npz')
np.savez(npz_path, **weights_dict)
print(f"-> 模型權重已儲存至: {npz_path}")
print("\n=== 所有步驟執行完畢！ ===")
