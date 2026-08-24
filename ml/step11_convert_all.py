import tensorflow as tf
import numpy as np
import os

DATA_DIR   = "data"
output_dir = os.path.join(DATA_DIR, "tflite_model")
os.makedirs(output_dir, exist_ok=True)

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int32)
X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy")).astype(np.float32)
y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int32)

def representative_dataset():
    for i in range(0, 500):
        yield [X_train[i:i+1]]

def convert_and_save(keras_model, name, use_int8=True):
    print("\n" + "="*50)
    print("Converting: " + name)
    print("="*50)

    
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    f32_path = os.path.join(output_dir, name + "_float32.tflite")
    with open(f32_path, "wb") as f:
        f.write(tflite_model)
    f32_kb = os.path.getsize(f32_path) / 1024
    print("  Float32 : " + str(round(f32_kb, 2)) + " KB")

    
    int8_kb = None
    if use_int8:
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_dataset
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type  = tf.float32
        converter_int8.inference_output_type = tf.float32
        tflite_int8 = converter_int8.convert()
        int8_path = os.path.join(output_dir, name + "_int8.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
        int8_kb = os.path.getsize(int8_path) / 1024
        print("  INT8    : " + str(round(int8_kb, 2)) + " KB")
        print("  Reduction: " + str(round((1 - int8_kb/f32_kb)*100, 1)) + "%")

    
    best_path = int8_path if use_int8 else f32_path
    with open(best_path, "rb") as f:
        data = f.read()

    c_array  = ", ".join([f"0x{b:02x}" for b in data])
    h_path   = os.path.join(output_dir, name + "_model.h")
    header   = f"""// Auto-generated: {name}
// Size: {len(data)} bytes ({round(len(data)/1024, 2)} KB)

#ifndef {name.upper()}_MODEL_H
#define {name.upper()}_MODEL_H

const unsigned int {name}_model_len = {len(data)};
const unsigned char {name}_model[] = {{
  {c_array}
}};

#endif
"""
    with open(h_path, "w") as f:
        f.write(header)
    print("  C header: " + h_path)
    return f32_kb, int8_kb



print("Building LSTM Keras model...")
lstm_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 12)),
    tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.3),
    tf.keras.layers.LSTM(64, return_sequences=False, dropout=0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])
lstm_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val),
               epochs=3, batch_size=2048, verbose=1)
lstm_f32, lstm_int8 = convert_and_save(lstm_model, "lstm", use_int8=True)



print("\nBuilding 1D-CNN Keras model...")
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 12)),
    tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
cnn_model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=3, batch_size=2048, verbose=1)
cnn_f32, cnn_int8 = convert_and_save(cnn_model, "cnn1d", use_int8=True)



print("\nBuilding MLP Keras model...")
mlp_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 12)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation='softmax')
])
mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
mlp_model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=3, batch_size=2048, verbose=1)
mlp_f32, mlp_int8 = convert_and_save(mlp_model, "mlp", use_int8=True)



print("\n" + "="*50)
print("CONVERSION SUMMARY")
print("="*50)
print(f"{'Model':<10} {'Float32':>10} {'INT8':>10}")
print(f"{'GRU':<10} {'187.48 KB':>10} {'N/A':>10}")
print(f"{'LSTM':<10} {str(round(lstm_f32,2))+' KB':>10} {str(round(lstm_int8,2))+' KB':>10}")
print(f"{'CNN1D':<10} {str(round(cnn_f32,2))+' KB':>10} {str(round(cnn_int8,2))+' KB':>10}")
print(f"{'MLP':<10} {str(round(mlp_f32,2))+' KB':>10} {str(round(mlp_int8,2))+' KB':>10}")
print("\nAll C headers saved to data/tflite_model/")
