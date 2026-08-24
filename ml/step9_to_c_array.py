import os

DATA_DIR   = "data"
tflite_path = os.path.join(DATA_DIR, "tflite_model", "gru_ids_float32.tflite")
output_path = os.path.join(DATA_DIR, "tflite_model", "gru_ids_model.h")

print("Reading TFLite model...")
with open(tflite_path, "rb") as f:
    data = f.read()

print("Converting to C array...")
c_array = ", ".join([f"0x{b:02x}" for b in data])
num_bytes = len(data)

header = f"""// Auto-generated from gru_ids_float32.tflite
// Model size: {num_bytes} bytes ({round(num_bytes/1024, 2)} KB)
// Generated for ESP32 deployment

#ifndef GRU_IDS_MODEL_H
#define GRU_IDS_MODEL_H

const unsigned int gru_ids_model_len = {num_bytes};
const unsigned char gru_ids_model[] = {{
  {c_array}
}};

#endif // GRU_IDS_MODEL_H
"""

with open(output_path, "w") as f:
    f.write(header)

print("=== C Header Complete ===")
print("  Output : " + output_path)
print("  Size   : " + str(round(num_bytes / 1024, 2)) + " KB")
print("  Bytes  : " + str(num_bytes))
print("\nNext step: copy gru_ids_model.h to your Arduino sketch folder")


