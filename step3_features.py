import pandas as pd
import numpy as np
import os

DATA_DIR = "data"

print("Loading combined dataset...")
df = pd.read_csv(os.path.join(DATA_DIR, "combined_labeled.csv"), low_memory=False)
print(f"  Loaded {len(df):,} rows")


def safe_hex(x):
    try:
        return int(str(x).strip(), 16)
    except (ValueError, TypeError):
        return 0


print("\nConverting hex to integers...")

df["CAN_ID_int"] = df["CAN_ID"].apply(safe_hex)
df["DLC_int"]    = pd.to_numeric(df["DLC"], errors='coerce').fillna(0).astype(int)

data_cols = ["D0","D1","D2","D3","D4","D5","D6","D7"]
for col in data_cols:
    df[col] = df[col].apply(safe_hex)


print("Computing delta time...")
df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors='coerce')
df = df.sort_values("Timestamp").reset_index(drop=True)
df["delta_time"] = df["Timestamp"].diff().fillna(0)
df["delta_time"] = df["delta_time"].clip(lower=0)


print("Computing CAN ID frequency...")
df["id_freq"] = (
    df.groupby("CAN_ID_int")["CAN_ID_int"]
    .transform(lambda x: x.expanding().count())
)

feature_cols = ["Timestamp", "CAN_ID_int", "DLC_int",
                "D0","D1","D2","D3","D4","D5","D6","D7",
                "delta_time", "id_freq", "Label"]

df_features = df[feature_cols].copy()


print(f"\n=== Feature Engineering Complete ===")
print(f"  Shape       : {df_features.shape}")
print(f"  Features    : {feature_cols[:-1]}")
print(f"\n  Sample stats:")
print(df_features[["CAN_ID_int","DLC_int","delta_time","id_freq"]].describe().round(4))
print(f"\n  Any nulls   : {df_features.isnull().sum().sum()}")
print(f"\n  Label dist  :")
label_names = {0:"Normal", 1:"DoS", 2:"Fuzzy", 3:"Spoofing"}
for label, count in df_features["Label"].value_counts().sort_index().items():
    print(f"    {label} ({label_names[label]}): {count:,}")


out_path = os.path.join(DATA_DIR, "features.csv")
df_features.to_csv(out_path, index=False)
print(f"\n  Saved to data/features.csv")