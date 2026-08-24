import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
WINDOW_SIZE = 30

print("Loading features...")
df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
print(f"  Loaded {len(df):,} rows")

feature_cols = ["CAN_ID_int", "DLC_int",
                "D0","D1","D2","D3","D4","D5","D6","D7",
                "delta_time", "id_freq"]

values = df[feature_cols].values   # shape: (N, 12)
labels = df["Label"].values        # shape: (N,)


print(f"Building windows (size={WINDOW_SIZE})...")

N = len(values)
num_windows = N - WINDOW_SIZE + 1

X = np.lib.stride_tricks.sliding_window_view(
    values, window_shape=(WINDOW_SIZE, values.shape[1])
).reshape(num_windows, WINDOW_SIZE, values.shape[1])

y = labels[WINDOW_SIZE - 1:]   # label = last row of each window

print(f"\n=== Windowing Complete ===")
print(f"  X shape     : {X.shape}  (windows, timesteps, features)")
print(f"  y shape     : {y.shape}")
print(f"\n  Label dist  :")
label_names = {0:"Normal", 1:"DoS", 2:"Fuzzy", 3:"Spoofing"}
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    pct = count / len(y) * 100
    print(f"    {label} ({label_names[label]}): {count:,}  ({pct:.1f}%)")


print(f"\nSaving...")
np.save(os.path.join(DATA_DIR, "X_windows.npy"), X)
np.save(os.path.join(DATA_DIR, "y_labels.npy"),  y)
print(f"  Saved X_windows.npy and y_labels.npy to data/")
