#!/usr/bin/env python3
"""
Train Trackster-Trackster (TS-TS) edge classifier. Input: 21-col features w/ deltas.
Usage: python training_ts_ts.py --input hgcal_edges.npz --output ts_ts_model.pth --epochs 100
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

import mplhep as hep

plt.style.use(hep.style.CMS)

class TsPreprocess(nn.Module):
    def __init__(self, num_features):  # 21
        super().__init__()
        self.register_buffer('scale_', torch.zeros(num_features))
        self.register_buffer('min_', torch.zeros(num_features))

    def forward(self, x):  # x: (batch, 21)
        # Logs
        x[:, 0] = torch.log(x[:, 0] + 1e-8)       # E1
        x[:, 4] = torch.log(x[:, 4] - 300 + 1e-8) # Z1 -300
        x[:, 7] = torch.log(x[:, 7] + 1e-8)       # E2
        x[:, 11] = torch.log(x[:, 11] - 300 + 1e-8) # Z2 -300

        # Special handling for times
        x[x[:, 5] == -99, 5] = 9.0    # time1 ==-99 →9
        x[x[:, 12] == -99, 12] = 9.0  # time2 ==-99 →9

        # timeErr1 & timeErr2
        for col in [6, 13]:  # timeErr1, timeErr2
            mask = x[:, col] != -1
            x[mask, col] = torch.sqrt(x[mask, col])
            x[~mask, col] = -0.001

        # Deltas clips
        x[:, 17] = torch.clamp(x[:, 17], -0.6, 0.6)  # deltaEta
        x[:, 18] = torch.clamp(x[:, 18], -0.6, 0.6)  # deltaPhi
        x[:, 19] = torch.sqrt(torch.clamp(x[:, 19]**2, 0, 1))  # deltaR
        x[:, 16] = torch.clamp(x[:, 16], -300, 300)  # deltaE

        # MinMax on ALL 21 columns
        x = (x - self.min_) * self.scale_
        return x

class TsEdgeMLP(nn.Module):
    def __init__(self, input_dim=21):
        super().__init__()
        self.preprocess = TsPreprocess(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.preprocess(x)
        return self.net(x)

class EdgeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def preprocess_numpy_ts(X):  # TS-TS transform for scaler fit
    x = torch.tensor(X, dtype=torch.float32)
    # Exact TsPreprocess.forward() replica
    x[:, 0] = torch.log(x[:, 0] + 1e-8)
    x[:, 4] = torch.log(x[:, 4] - 300 + 1e-8)
    x[:, 7] = torch.log(x[:, 7] + 1e-8)
    x[:, 11] = torch.log(x[:, 11] - 300 + 1e-8)

    x[x[:, 5] == -99, 5] = 9.0
    x[x[:, 12] == -99, 12] = 9.0

    for col in [6, 13]:
        mask = x[:, col] != -1
        x[mask, col] = torch.sqrt(x[mask, col])
        x[~mask, col] = -0.001

    x[:, 17] = torch.clamp(x[:, 17], -0.6, 0.6)
    x[:, 18] = torch.clamp(x[:, 18], -0.6, 0.6)
    x[:, 19] = torch.sqrt(torch.clamp(x[:, 19]**2, 0, 1))
    x[:, 16] = torch.clamp(x[:, 16], -300, 300)

    return x.cpu().numpy()

def plot_feature_subplots(df, feature_list, title, outdir):
    n_features = len(feature_list)
    ncols, nrows = 4, int(np.ceil(n_features / 4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    if nrows * ncols == 1: axes = [axes]
    else: axes = axes.flatten()

    for i, col in enumerate(feature_list):
        ax = axes[i]
        data0 = df[df["label"] == 0][col]
        data1 = df[df["label"] == 1][col]
        ax.hist(data0, bins=50, alpha=0.5, density=True, label='Negative')
        ax.hist(data1, bins=50, alpha=0.5, density=True, label='Positive')
        ax.set_title(col)
        ax.legend()

    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(Path(outdir) / f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close()

def train_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval(); total_loss, correct, total = 0, 0, 0
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze()
        loss = criterion(logits, y.squeeze())
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        total_loss += loss.item() * len(x)
        correct += (preds == y.squeeze().long()).sum().item()
        total += len(x)
        all_probs.append(probs.cpu())
        all_labels.append(y.squeeze().cpu())
    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return total_loss / total, correct / total, all_probs, all_labels

def main(args):
    # Load TS-TS data (21 cols w/ deltas)
    data = np.load(args.input)
    X_raw, y_raw = data['ts_features'], data['ts_labels']  # Shape (N, 21)

    columns = ["E1", "eta1", "sin_phi1", "cos_phi1", "Z1", "time1", "timeErr1",
               "E2", "eta2", "sin_phi2", "cos_phi2", "Z2", "time2", "timeErr2",
               "deltaTime", "samePid"]
    #["deltaE", "deltaEta", "deltaPhi", "deltaR", "deltaZ"]

    print(f"Loaded {X_raw.shape[0]} TS-TS edges ({X_raw.shape[1]} cols)")

    df = pd.DataFrame(X_raw, columns=columns)

    ts_delta_eta = df["eta1"] - df["eta2"]

    phi1 = np.arctan2(df["sin_phi1"], df["cos_phi1"])
    phi2 = np.arctan2(df["sin_phi2"], df["cos_phi2"])
    ts_delta_phi = phi1 - phi2
    ts_delta_phi = (ts_delta_phi + np.pi) % (2*np.pi) - np.pi
    ts_delta_R = np.sqrt(ts_delta_eta**2 + ts_delta_phi**2)

    df["deltaE"] = df["E1"] - df["E2"]
    df["deltaEta"] = ts_delta_eta
    df["deltaPhi"] = ts_delta_phi
    df["deltaR"] = ts_delta_R
    df["deltaZ"] = df["Z1"] - df["Z2"]
    df["label"] = y_raw

    # Balance
    df_0 = df[df["label"] == 0]
    df_1 = df[df["label"] == 1]
    n = min(len(df_0), len(df_1))
    df_balanced = pd.concat([df_0.sample(n=2*n, random_state=42),
                            df_1.sample(n=n, random_state=42)]).sample(frac=1, random_state=42)

    feature_cols = list(df.columns.values)
    feature_cols.remove('label')
    print(feature_cols)
    X_bal = df_balanced[feature_cols].values.astype(np.float32)
    y_bal = df_balanced["label"].values.astype(np.int64)

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_bal, y_bal, np.arange(len(y_bal)), test_size=0.1, random_state=42, stratify=y_bal)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # === Fit scaler on TRANSFORMED TRAIN ===
    print("Fitting scaler on post-transform train...")
    X_train_transformed = preprocess_numpy_ts(X_train)
    scaler = MinMaxScaler()
    scaler.fit(X_train_transformed)
    scale_params = scaler.scale_.astype(np.float32)
    min_params = scaler.min_.astype(np.float32)

    # Plots
    outdir = Path("plots"); outdir.mkdir(exist_ok=True)
    plot_feature_subplots(df_balanced, feature_cols, "TS-TS Pre-Transform", outdir)
    df_transformed = pd.DataFrame(X_train_transformed[:10000], columns=feature_cols)
    df_transformed["label"] = y_train[:10000]
    plot_feature_subplots(df_transformed, feature_cols, "TS-TS Post-Transform", outdir)

    # DataLoaders (RAW 21-col w/ deltas)
    train_loader = DataLoader(EdgeDataset(X_train, y_train), 2048, True)
    val_loader = DataLoader(EdgeDataset(X_val, y_val), 2048, False)
    test_loader = DataLoader(EdgeDataset(X_test, y_test), 2048, False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TsEdgeMLP().to(device)
    model.preprocess.scale_ = torch.tensor(scale_params).to(device)
    model.preprocess.min_ = torch.tensor(min_params).to(device)

    pos_weight = torch.tensor([(len(y_train)-y_train.sum())/y_train.sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    best_auc, train_losses, val_losses = 0, [], []
    model_dir = Path("models_ts_ts"); model_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_probs, val_labels = eval_epoch(model, val_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_probs)

        train_losses.append(train_loss); val_losses.append(val_loss)
        print(f"Epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f} acc={val_acc:.4f} AUC={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), model_dir / args.output)
            print(f"  → New best: {best_auc:.4f}")

    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(outdir / f"losses_ts.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Test + plots (same as Track-TS)
    test_loss, test_acc, test_probs, test_labels = eval_epoch(model, test_loader, criterion, device)
    test_auc = roc_auc_score(test_labels, test_probs)
    print(f"\nTest: AUC={test_auc:.4f}, Acc={test_acc:.4f}")

    # ROC
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Test AUC={test_auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(outdir / "roc_curve_ts.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Cost dist
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze()
        cost = -logits
        int_cost = (cost * 1000).round().int().cpu().numpy()

    cost_pos = int_cost[test_labels == 1]
    cost_neg = int_cost[test_labels == 0]
    plt.figure()
    plt.hist(cost_neg, bins=100, alpha=0.5, density=True, label='Negative')
    plt.hist(cost_pos, bins=100, alpha=0.5, density=True, label='Positive')
    plt.xlabel('MCF Cost'); plt.ylabel('Density'); plt.yscale('log'); plt.legend()
    plt.savefig(outdir / "mcf_cost_dist_ts.png", dpi=150, bbox_inches='tight')
    plt.close()

    #np.savez("splits_ts_ts.npz", train_idx=idx_train, val_idx=idx_val, test_idx=idx_test)
    #np.savez("scaler_ts_ts.npz", scale=scale_params, min=min_params)

    print(f"\n✓ Best model: {model_dir}/{args.output} (AUC={best_auc:.4f})")
    print(f"✓ Plots: {outdir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TS-TS model")
    parser.add_argument('--input', required=True, help="NPZ data (ts_features 21 cols)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', '-o', default='ts_ts_model.pth', help="Output model")
    args = parser.parse_args()
    main(args)

