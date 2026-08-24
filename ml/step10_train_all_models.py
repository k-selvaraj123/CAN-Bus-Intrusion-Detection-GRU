import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import os

DATA_DIR    = "data"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 2048
EPOCHS      = 10
LR          = 0.001
NUM_CLASSES = 4
WINDOW      = 30
FEATURES    = 12

print("Using device: " + str(DEVICE))

#Load data
print("Loading data...")
X_train = torch.tensor(
    np.load(os.path.join(DATA_DIR, "X_train.npy")), dtype=torch.float32)
X_val   = torch.tensor(
    np.load(os.path.join(DATA_DIR, "X_val.npy")),   dtype=torch.float32)
X_test  = torch.tensor(
    np.load(os.path.join(DATA_DIR, "X_test.npy")),  dtype=torch.float32)
y_train = torch.tensor(
    np.load(os.path.join(DATA_DIR, "y_train.npy")), dtype=torch.long)
y_val   = torch.tensor(
    np.load(os.path.join(DATA_DIR, "y_val.npy")),   dtype=torch.long)
y_test  = torch.tensor(
    np.load(os.path.join(DATA_DIR, "y_test.npy")),  dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#Model definitions 
class LSTM_IDS(nn.Module):
    def __init__(self):
        super(LSTM_IDS, self).__init__()
        self.lstm = nn.LSTM(
            input_size  = FEATURES,
            hidden_size = 64,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.3
        )
        self.fc1     = nn.Linear(64, 32)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = out[:, -1, :]
        out    = self.fc1(out)
        out    = self.relu(out)
        out    = self.dropout(out)
        return self.fc2(out)


class CNN1D_IDS(nn.Module):
    def __init__(self):
        super(CNN1D_IDS, self).__init__()
        self.conv1   = nn.Conv1d(FEATURES, 64, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu    = nn.ReLU()
        self.pool    = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc1     = nn.Linear(128, 32)
        self.fc2     = nn.Linear(32, NUM_CLASSES)

    def forward(self, x):
        x = x.permute(0, 2, 1)       # (batch, features, timesteps)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class MLP_IDS(nn.Module):
    def __init__(self):
        super(MLP_IDS, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(WINDOW * FEATURES, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


#Train & evaluate 
def evaluate(model, loader, criterion):
    model.eval()
    correct  = 0
    total    = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb       = xb.to(DEVICE)
            yb       = yb.to(DEVICE)
            out      = model(xb)
            loss_sum += criterion(out, yb).item()
            correct  += (out.argmax(1) == yb).sum().item()
            total    += yb.size(0)
    return loss_sum / len(loader), correct / total


def train_model(name, model):
    print("\n" + "="*50)
    print("Training: " + name)
    print("="*50)
    print("Parameters: " + str(sum(p.numel() for p in model.parameters())))

    model     = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)

    best_val_acc = 0.0
    save_path    = os.path.join(DATA_DIR, name.lower() + "_model.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb   = xb.to(DEVICE)
            yb   = yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss        /= len(train_loader)
        val_loss, val_acc  = evaluate(model, val_loader, criterion)
        scheduler.step()

        print("Epoch " + str(epoch) + "/" + str(EPOCHS) +
              "  Train Loss: " + str(round(train_loss, 4)) +
              "  Val Loss: "   + str(round(val_loss,   4)) +
              "  Val Acc: "    + str(round(val_acc * 100, 2)) + "%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("  Best model saved")

    #Test evaluation
    model.load_state_dict(torch.load(
        save_path, map_location=DEVICE, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print("\nTest Accuracy : " + str(round(test_acc * 100, 2)) + "%")

    all_preds  = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            all_preds.extend(model(xb).argmax(1).cpu().numpy())
            all_labels.extend(yb.numpy())

    print(classification_report(
        all_labels, all_preds,
        target_names=["Normal", "DoS", "Fuzzy", "Spoofing"]
    ))
    return round(test_acc * 100, 2)



results = {}
results["LSTM"]   = train_model("lstm",  LSTM_IDS())
results["CNN1D"]  = train_model("cnn1d", CNN1D_IDS())
results["MLP"]    = train_model("mlp",   MLP_IDS())

print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
for name, acc in results.items():
    print(name + " Test Accuracy: " + str(acc) + "%")
