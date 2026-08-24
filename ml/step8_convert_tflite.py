import tensorflow as tf
import numpy as np
import os

DATA_DIR   = "data"
output_dir = os.path.join(DATA_DIR, "tflite_model")
os.makedirs(output_dir, exist_ok=True)

print("Building Keras equivalent of GRU model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 12)),
    tf.keras.layers.GRU(64, return_sequences=True, dropout=0.3),
    tf.keras.layers.GRU(64, return_sequences=False, dropout=0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()

print("\nLoading preprocessed data...")
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int32)
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy")).astype(np.float32)
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int32)

print("Training Keras model (3 epochs)...")
model.compile(
    optimizer = 'adam',
    loss      = 'sparse_categorical_crossentropy',
    metrics   = ['accuracy']
)
model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = 3,
    batch_size      = 2048,
    verbose         = 1
)

keras_path = os.path.join(output_dir, "gru_ids.keras")
model.save(keras_path)
print("Keras model saved to " + keras_path)

print("\nConverting to TFLite (float32)...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
tflite_model = converter.convert()

tflite_path = os.path.join(output_dir, "gru_ids_float32.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print("\n=== Conversion Complete ===")
print("Float32 TFLite : " + str(round(os.path.getsize(tflite_path) / 1024, 2)) + " KB")
print("INT8 skipped for GRU (Flex op limitation)")
print("Float32 fits within ESP32 flash (4MB available)")
