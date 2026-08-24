import pandas as pd
import os
import re

DATA_DIR = "data"

col_names = ["Timestamp", "CAN_ID", "DLC",
             "D0","D1","D2","D3","D4","D5","D6","D7", "Flag"]


def parse_normal_txt(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            try:
                timestamp = parts[1]
                can_id    = parts[3]
                dlc       = parts[6]
                data      = parts[7:15]
                flag      = "R"
                rows.append([timestamp, can_id, dlc] + data + [flag])
            except IndexError:
                continue
    return pd.DataFrame(rows, columns=col_names)

def load_csv(filepath):
    return pd.read_csv(filepath, header=None, names=col_names)


normal_df   = parse_normal_txt(os.path.join(DATA_DIR, "Normal_run_data.txt"))
dos_df      = load_csv(os.path.join(DATA_DIR, "DoS_dataset.csv"))
fuzzy_df    = load_csv(os.path.join(DATA_DIR, "Fuzzy_dataset.csv"))
rpm_df      = load_csv(os.path.join(DATA_DIR, "RPM_dataset.csv"))
gear_df     = load_csv(os.path.join(DATA_DIR, "gear_dataset.csv"))
spoofing_df = pd.concat([rpm_df, gear_df], ignore_index=True)



normal_df["Label"] = 0


dos_df      = dos_df[dos_df["Flag"] == "T"].copy();      dos_df["Label"]      = 1
fuzzy_df    = fuzzy_df[fuzzy_df["Flag"] == "T"].copy();  fuzzy_df["Label"]    = 2
spoofing_df = spoofing_df[spoofing_df["Flag"] == "T"].copy(); spoofing_df["Label"] = 3


combined_df = pd.concat(
    [normal_df, dos_df, fuzzy_df, spoofing_df],
    ignore_index=True
)

label_names = {0: "Normal", 1: "DoS", 2: "Fuzzy", 3: "Spoofing"}
print(f"\n=== Combined Dataset ===")
print(f"  Total rows : {combined_df.shape[0]}")
print(f"  Columns    : {list(combined_df.columns)}")
print(f"\n  Label distribution:")
for label, count in combined_df["Label"].value_counts().sort_index().items():
    pct = count / len(combined_df) * 100
    print(f"    {label} ({label_names[label]}): {count:>10,}  ({pct:.1f}%)")

print(f"\n  Sample row (Normal):\n{combined_df[combined_df['Label']==0].iloc[0]}")
print(f"\n  Sample row (DoS):\n{combined_df[combined_df['Label']==1].iloc[0]}")

combined_df.to_csv(os.path.join(DATA_DIR, "combined_labeled.csv"), index=False)
print(f"\n  Saved to data/combined_labeled.csv")