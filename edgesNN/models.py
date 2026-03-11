import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# NN trk
class Preprocess(nn.Module):
    def __init__(self, num_features):  # e.g., 18
        super().__init__()
        # Buffers: init with zeros, overwritten later with fitted values
        self.register_buffer('scale_', torch.zeros(num_features))
        self.register_buffer('min_', torch.zeros(num_features))
        # NO clip_limits buffer needed - use constants in forward

    def forward(self, x):  # x: (batch, 18)
        # Logs (safe with epsilon)
        x[:, 0] = torch.log(x[:, 0] + 1e-8)  # refPt
        x[:, 1] = torch.log(x[:, 1] + 1e-8)  # refP
        x[:, 7] = torch.log(x[:, 7] + 1e-8)  # tsEnergy

        # Special handling (masks are differentiable-safe for inference)
        x[x[:, 5] == 0, 5] = 9.0              # trk_time
        x[x[:, 6] == -1, 6] = 0.0             # trk_timeErr
        x[x[:, 11] == -99, 11] = 9.0          # tsTime
        x[:, 11] = torch.clamp(x[:, 11], min=9.0, max=18.0)

        # tsTimeErr special
        mask = x[:, 12] != -1
        x[mask, 12] = torch.sqrt(x[mask, 12])
        x[~mask, 12] = -0.001

        # Clips (hardcoded constants, as in your training code)
        x[:, 13] = torch.clamp(x[:, 13], -0.3, 0.3)   # deltaTime
        x[:, 14] = torch.clamp(x[:, 14], -1.0, 1.0)   # deltaE
        x[:, 15] = torch.clamp(x[:, 15], -0.5, 0.5)   # deltaEta
        x[:, 16] = torch.clamp(x[:, 16], -0.5, 0.5)   # deltaPhi
        x[:, 17] = torch.clamp(x[:, 17], 0.0, 1.0)    # deltaR (already sqrt-clipped in training)

        # MinMax: assumes all 18 cols are scaled (as per your scaler.fit_transform)
        x = (x - self.min_) * self.scale_

        return x

class EdgeMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.preprocess = Preprocess(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.preprocess(x)
        return self.net(x) # logits

# NN ts
class TsPreprocess(nn.Module):
    def __init__(self, num_features):  # 21
        super().__init__()
        self.register_buffer('scale_', torch.zeros(num_features))
        self.register_buffer('min_', torch.zeros(num_features))

    def forward(self, x):  # x: (batch, 21)
        # Column indices (match your df_ts order + deltas)
        # 0:E1,1:eta1,2:sin_phi1,3:cos_phi1,4:Z1,5:time1,6:timeErr1,
        # 7:E2,8:eta2,9:sin_phi2,10:cos_phi2,11:Z2,12:time2,13:timeErr2,
        # 14:deltaTime,15:samePid,
        # 16:deltaE,17:deltaEta,18:deltaPhi,19:deltaR,20:deltaZ

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

        # Deltas clips (already computed in input)
        x[:, 17] = torch.clamp(x[:, 17], -0.6, 0.6)  # deltaEta
        x[:, 18] = torch.clamp(x[:, 18], -0.6, 0.6)  # deltaPhi
        x[:, 19] = torch.sqrt(torch.clamp(x[:, 19]**2, 0, 1))  # deltaR → sqrt(clip(0,1))
        x[:, 16] = torch.clamp(x[:, 16], -300, 300)  # deltaE
        # deltaZ: no clip

        # MinMax on ALL 21 columns
        x = (x - self.min_) * self.scale_

        return x

class TsEdgeMLP(nn.Module):
    def __init__(self, input_dim):  # 21
        super().__init__()
        self.preprocess = TsPreprocess(input_dim)
        self.net = nn.Sequential(  # Your architecture
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.preprocess(x)
        return self.net(x)
