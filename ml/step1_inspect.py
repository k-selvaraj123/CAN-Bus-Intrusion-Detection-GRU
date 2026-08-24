import pandas as pd
import os

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

dataframes = {}


dataframes["Normal"] = parse_normal_txt(os.path.join(DATA_DIR, "Normal_run_data.txt"))


csv_files = {
    "DoS":           "DoS_dataset.csv",
    "Fuzzy":         "Fuzzy_dataset.csv",
    "Spoofing_RPM":  "RPM_dataset.csv",
    "Spoofing_Gear": "gear_dataset.csv",
}

for label, fname in csv_files.items():
    path = os.path.join(DATA_DIR, fname)
    dataframes[label] = pd.read_csv(path, header=None, names=col_names)


dataframes["Spoofing"] = pd.concat(
    [dataframes.pop("Spoofing_RPM"), dataframes.pop("Spoofing_Gear")],
    ignore_index=True
)


for label, df in dataframes.items():
    print(f"\n=== {label} ===")
    print(f"  Shape      : {df.shape}")
    print(f"  Flag counts:\n{df['Flag'].value_counts()}")
    print(f"  First row  :\n{df.iloc[0]}")