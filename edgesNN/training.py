#!/usr/bin/env python3
"""
Train Track-TS edge classifier. Input: 18-col features (with deltas).
Scaler fit on post-transform train → baked in model.
Usage: python training.py --input hgcal_edges.npz --epochs 50 --output trk_ts_model.pth
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
from models import *

import mplhep as hep

plt.style.use(hep.style.CMS)

class EdgeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def preprocess_numpy(X):  # Exact model transform for scaler fitting
    """Numpy replica of Preprocess.forward()."""
    x = torch.tensor(X, dtype=torch.float32)

    # Logs
    x[:, 0] = torch.log(x[:, 0] + 1e-8)
    x[:, 1] = torch.log(x[:, 1] + 1e-8)
    x[:, 7] = torch.log(x[:, 7] + 1e-8)

    # Special
    x[x[:, 5] == 0, 5] = 9.0
    x[x[:, 6] == -1, 6] = 0.0
    x[x[:, 11] == -99, 11] = 9.0
    x[:, 11] = torch.clamp(x[:, 11], 9.0, 18.0)

    mask = x[:, 12] != -1
    x[mask, 12] = torch.sqrt(x[mask, 12])
    x[~mask, 12] = -0.001

    # Clips
    x[:, 13] = torch.clamp(x[:, 13], -0.3, 0.3)
    x[:, 14] = torch.clamp(x[:, 14], -1.0, 1.0)
    x[:, 15] = torch.clamp(x[:, 15], -0.5, 0.5)
    x[:, 16] = torch.clamp(x[:, 16], -0.5, 0.5)
    x[:, 17] = torch.clamp(x[:, 17], 0.0, 1.0)

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
    # Load 18-col data WITH DELTAS
    data = np.load(args.input)
    X_raw, y_raw = data['trk_features'], data['trk_labels']  # Shape (N, 18)

    columns = ["refPt", "refP", "refEta", "sin_refPhi", "cos_refPhi", "trk_time", "trk_timeErr",
               "tsEnergy", "tsEta", "sin_tsPhi", "cos_tsPhi", "tsTime", "tsTimeErr", "deltaTime"]
    #["deltaE", "deltaEta", "deltaPhi", "deltaR"]

    print(f"Loaded {X_raw.shape[0]} Track-TS edges ({X_raw.shape[1]} cols)")

    # DataFrame
    df = pd.DataFrame(X_raw, columns=columns)

    trk_delta_phi = np.arctan2(df["sin_refPhi"], df["cos_refPhi"]) - np.arctan2(df["sin_tsPhi"], df["cos_tsPhi"])
    trk_delta_phi = (trk_delta_phi + np.pi) % (2*np.pi) - np.pi
    trk_delta_eta = df["refEta"] - df["tsEta"]

    df["deltaE"] = (df["refP"] - df["tsEnergy"])/ df["refP"]
    df["deltaEta"] = trk_delta_eta
    df["deltaPhi"] = trk_delta_phi
    df["deltaR"] = np.sqrt(trk_delta_eta**2 + trk_delta_phi**2)
    df["label"] = y_raw

    # Balance (as notebook)
    df_0 = df[df["label"] == 0]
    df_1 = df[df["label"] == 1]
    n = min(len(df_0), len(df_1))
    df_balanced = pd.concat([df_0.sample(n=2*n, random_state=42), df_1.sample(n=n, random_state=42)]).sample(frac=1, random_state=42)

    feature_cols = list(df.columns.values)
    feature_cols.remove('label')
    print(feature_cols)
    X_bal = df_balanced[feature_cols].values.astype(np.float32)
    y_bal = df_balanced["label"].values.astype(np.int64)

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_bal, y_bal, np.arange(len(y_bal)), test_size=0.1, random_state=42, stratify=y_bal)
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # === CRITICAL: Fit scaler on TRANSFORMED TRAIN ===
    print("Fitting scaler on post-transform train...")
    X_train_transformed = preprocess_numpy(X_train)
    scaler = MinMaxScaler()
    scaler.fit(X_train_transformed)
    scale_params = scaler.scale_.astype(np.float32)
    min_params = scaler.min_.astype(np.float32)

    # Plot pre/post transform
    outdir = Path("plots"); outdir.mkdir(exist_ok=True)
    plot_feature_subplots(df_balanced, feature_cols, "Trk-Ts Pre-Transform", outdir)
    df_transformed = pd.DataFrame(X_train_transformed[:10000], columns=feature_cols)  # Sample
    df_transformed["label"] = y_train[:10000]
    plot_feature_subplots(df_transformed, feature_cols, "Trk-Ts Post-Transform", outdir)

    # DataLoaders (RAW 18-col input)
    train_loader = DataLoader(EdgeDataset(X_train, y_train), 2048, True)
    val_loader = DataLoader(EdgeDataset(X_val, y_val), 2048, False)
    test_loader = DataLoader(EdgeDataset(X_test, y_test), 2048, False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeMLP().to(device)
    #model = BipartiteEdgeConv(7, 6, 5).to(device)
    if (len(feature_cols) !=  7+6+5): print("ERROR IN MODEL CREATION")
    model.preprocess.scale_ = torch.tensor(scale_params).to(device)
    model.preprocess.min_ = torch.tensor(min_params).to(device)

    pos_weight = torch.tensor([(len(y_train)-y_train.sum())/y_train.sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    best_auc, train_losses, val_losses = 0, [], []
    model_dir = Path("models_trk_ts"); model_dir.mkdir(exist_ok=True)

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
    plt.savefig(outdir / f"losses_trk.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Final test
    test_loss, test_acc, test_probs, test_labels = eval_epoch(model, test_loader, criterion, device)
    test_auc = roc_auc_score(test_labels, test_probs)
    print(f"\nTest: AUC={test_auc:.4f}, Acc={test_acc:.4f}")

    # ROC + Cost plots
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Test AUC={test_auc:.4f}')
    plt.plot([0,1], [0,1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "roc_curve_trk.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Cost distribution
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32).to(device)).squeeze()
        cost = -logits  # MCF cost = -log(p)
        int_cost = (cost * 1000).round().int().cpu().numpy()

    cost_pos = int_cost[test_labels == 1]
    cost_neg = int_cost[test_labels == 0]
    plt.figure()
    plt.hist(cost_neg, bins=100, alpha=0.5, density=True, label='Negative')
    plt.hist(cost_pos, bins=100, alpha=0.5, density=True, label='Positive')
    plt.xlabel('MCF Cost'); plt.ylabel('Density'); plt.yscale('log')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(outdir / "mcf_cost_dist_trk.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save splits/scaler backup
    #np.savez("splits.npz", train_idx=idx_train, val_idx=idx_val, test_idx=idx_test)
    #np.savez("scaler.npz", scale=scale_params, min=min_params)

    print(f"\n✓ Best model: models/{args.output} (AUC={best_auc:.4f})")
    print(f"✓ Plots: {outdir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Track-TS model")
    parser.add_argument('--input', required=True, help="NPZ data (trk_features 18 cols)")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', '-o', default='trk_ts_model.pth', help="Output model")
    args = parser.parse_args()
    main(args)

