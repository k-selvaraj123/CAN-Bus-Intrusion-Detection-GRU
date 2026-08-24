import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

DATA_DIR = "data"

print("Loading windows...")
X = np.load(os.path.join(DATA_DIR, "X_windows.npy"))
y = np.load(os.path.join(DATA_DIR, "y_labels.npy"))
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")

print("\nClipping delta_time outliers...")
delta_idx = 10
flat = X[:, :, delta_idx].flatten()
cap = np.percentile(flat, 99)
X[:, :, delta_idx] = np.clip(X[:, :, delta_idx], 0, cap)
print(f"  delta_time clipped at 99th percentile: {cap:.6f}s")


print("\nNormalizing...")
N, T, F = X.shape
X_flat = X.reshape(-1, F)          # (N*T, F)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_flat).reshape(N, T, F)
print(f"  Scaled to [0, 1]  —  shape: {X_scaled.shape}")


print("\nSplitting dataset...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30,
    random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)


label_names = {0:"Normal", 1:"DoS", 2:"Fuzzy", 3:"Spoofing"}

def print_dist(name, y_arr):
    print(f"\n  {name} — {len(y_arr):,} samples")
    unique, counts = np.unique(y_arr, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    {label} ({label_names[label]}): {count:,}  ({count/len(y_arr)*100:.1f}%)")

print("\n=== Split Complete ===")
print_dist("Train", y_train)
print_dist("Val",   y_val)
print_dist("Test",  y_test)


print("\nSaving...")
np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
np.save(os.path.join(DATA_DIR, "X_val.npy"),   X_val)
np.save(os.path.join(DATA_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
np.save(os.path.join(DATA_DIR, "y_val.npy"),   y_val)
np.save(os.path.join(DATA_DIR, "y_test.npy"),  y_test)

import pickle
with open(os.path.join(DATA_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("  Saved X_train, X_val, X_test, y_train, y_val, y_test")
print("  Saved scaler.pkl  (needed later for ESP32 deployment)")