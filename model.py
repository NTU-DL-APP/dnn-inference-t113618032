import tensorflow as tf
import numpy as np
import json
import os
import gzip

# ==============================================================================
# --- 您可以在這裡編輯您的路徑設定 ---
# ==============================================================================
# 指向您本地存放 Fashion-MNIST 原始檔案的資料夾
# 資料夾內應包含 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz' 等檔案
DATA_PATH = './data/fashion'

# 指定儲存訓練後模型的資料夾
MODEL_DIR = './model'
# ------------------------------------------------------------------------------


def load_local_fashion_mnist(path):
    """
    從本地端指定路徑載入 Fashion-MNIST 資料集的函式。
    此函式被整合進主腳本中，無需額外檔案。
    """
    print(f"從本地路徑 '{path}' 載入 Fashion-MNIST 資料...")
    
    # 定義訓練和測試資料的檔案路徑
    train_labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    train_images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')

    # 載入訓練資料
    with gzip.open(train_labels_path, 'rb') as lbpath:
        train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(train_images_path, 'rb') as imgpath:
        train_images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(train_labels), 28, 28)

    # 載入測試資料
    with gzip.open(test_labels_path, 'rb') as lbpath:
        test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(test_images_path, 'rb') as imgpath:
        test_images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(test_labels), 28, 28)

    print("資料載入完成。")
    return (train_images, train_labels), (test_images, test_labels)


# --- 主程式開始 ---

# 1. 準備環境與載入資料
os.makedirs(MODEL_DIR, exist_ok=True)
(train_images, train_labels), (test_images, test_labels) = load_local_fashion_mnist(DATA_PATH)

# 正規化圖像資料 (將像素值從 0-255 縮放到 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

print(f"訓練資料形狀: {train_images.shape}")
print(f"測試資料形狀: {test_images.shape}")

# 2. 建立符合作業規範的神經網路模型
print("\n建立神經網路模型...")
model = tf.keras.Sequential([
    # 將 28x28 圖像展平為 784 維向量
    tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_layer'),
    
    # 隱藏層 1：128個神經元，使用 ReLU 激活函數
    tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
    
    # 隱藏層 2：64個神經元，使用 ReLU 激活函數
    tf.keras.layers.Dense(64, activation='relu', name='dense_2'),
    
    # 輸出層：10個神經元（對應10個類別），使用 Softmax 激活函數
    tf.keras.layers.Dense(10, activation='softmax', name='output_layer')
])

# 3. 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 4. 訓練模型
print("\n開始訓練模型...")
history = model.fit(
    train_images,
    train_labels,
    epochs=20,  # 增加 epochs 以獲得更好準確率
    batch_size=128,
    validation_data=(test_images, test_labels),
    verbose=1
)

# 5. 評估模型
print("\n評估模型最終性能...")
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"模型在測試集上的最終準確率: {test_accuracy:.4f}")

# 6. 儲存與轉換模型
print("\n開始儲存與轉換模型...")
model_name_base = 'fashion_mnist'

# 步驟 6.1: 儲存為 .h5 格式
h5_path = os.path.join(MODEL_DIR, f'{model_name_base}.h5')
model.save(h5_path)
print(f"-> 模型已儲存為 .h5 格式: {h5_path}")

# 步驟 6.2: 轉換為作業指定的 .json (架構) 和 .npz (權重) 格式

# 儲存架構 (JSON)
json_path = os.path.join(MODEL_DIR, f'{model_name_base}.json')
model_config = json.loads(model.to_json())

# 根據 nn_predict.py 的需求，手動建構 JSON 格式
formatted_arch = []
for layer_config in model_config['config']['layers']:
    layer_name = layer_config['config']['name']
    layer_instance = model.get_layer(name=layer_name)
    layer_weights_info = []
    
    # 只有 Dense 層有權重，且權重名稱需符合推論腳本預期格式
    if isinstance(layer_instance, tf.keras.layers.Dense):
        layer_weights_info.append(f'{layer_name}/kernel:0')
        layer_weights_info.append(f'{layer_name}/bias:0')
    
    formatted_arch.append({
        'name': layer_name,
        'type': layer_config['class_name'],
        'config': layer_config['config'],
        'weights': layer_weights_info
    })

with open(json_path, 'w') as json_file:
    json.dump(formatted_arch, json_file, indent=4)
print(f"-> 模型架構已儲存為 JSON 格式: {json_path}")

# 儲存權重 (NPZ)
npz_path = os.path.join(MODEL_DIR, f'{model_name_base}.npz')
weights_dict = {}
for layer in model.layers:
    if layer.get_weights():  # 只處理有權重的層
        weights = layer.get_weights()
        weights_dict[f'{layer.name}/kernel:0'] = weights[0]
        weights_dict[f'{layer.name}/bias:0'] = weights[1]

np.savez(npz_path, **weights_dict)
print(f"-> 模型權重已儲存為 NPZ 格式: {npz_path}")

print("\n=== 所有步驟執行完畢！ ===")
