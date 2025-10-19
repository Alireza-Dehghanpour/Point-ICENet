import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------

# ---------------------------------------

class PointNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim=128, segmentation=False, **kwargs):
        super().__init__()
        
        self.segmentation = segmentation
        self.activation = nn.ReLU()

        self.fc_in = nn.Sequential(
            nn.Conv1d(in_channels + 3, 2 * hidden_dim, 1),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU()
        )

        mlp_layers = []
        for _ in range(10):
            mlp_layers.append(nn.Sequential(
                nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ))
        self.mlp_layers = nn.ModuleList(mlp_layers)

        self.fc_3 = nn.Sequential(
            nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        if segmentation:
            self.fc_out = nn.Sequential(
                nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(hidden_dim, out_channels, 1)
            )
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, out_channels)
            )

    def forward_spatial(self, data):
        return {}

    def forward(self, data, spatial_only=False, spectral_only=False, cat_in_last_layer=None):

        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:
            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value

        x = data["x"]
        pos = data["pos"]
        x = torch.cat([x, pos], dim=1)
        x = self.fc_in(x)

        for l in self.mlp_layers:
            x = l(x)
            x_pool = torch.max(x, dim=2, keepdim=True)[0]
            x = torch.cat([x, x_pool], dim=1)

        x = self.fc_3(x)

        if self.segmentation:
            x_pool = torch.max(x, dim=2, keepdim=True)[0]
            x = torch.cat([x, x_pool], dim=1)
            x = self.fc_out(x)
        else:
            x = torch.max(x, dim=2)[0]
            x = self.fc_out(x)

        return x


# ---------------------------------------

# ---------------------------------------

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim):
        super().__init__()

        self.fc_0 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.fc_1 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, 1),
            nn.BatchNorm1d(out_channels)
        )

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        nn.init.zeros_(self.fc_1[0].weight)  # init only conv, not BN

    def forward(self, x):
        x_short = self.shortcut(x)
        x = self.fc_0(x)
        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.activation(x + x_short)
        return x


# ---------------------------------------
# ---------------------------------------

class ResidualPointNet(nn.Module):

    def __init__(self, in_channels, out_channels, hidden_dim=128, segmentation=False, **kwargs):
        super().__init__()

        self.fc_in = nn.Sequential(
            nn.Conv1d(in_channels + 3, 2 * hidden_dim, 1),
            nn.BatchNorm1d(2 * hidden_dim),
            nn.ReLU()
        )

        # ?? Doubled number of residual blocks (from 5 to 10)
        self.blocks = nn.ModuleList([
            ResidualBlock(2 * hidden_dim, hidden_dim, hidden_dim) for _ in range(9)
        ])
        self.block_last = ResidualBlock(2 * hidden_dim, hidden_dim, hidden_dim)

        self.segmentation = segmentation

        if self.segmentation:
            self.fc_out = nn.Sequential(
                nn.Conv1d(2 * hidden_dim, hidden_dim, 1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Conv1d(hidden_dim, out_channels, 1)
            )
        else:
            self.fc_out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, out_channels)
            )

    def forward_spatial(self, data):
        return {}

    def forward(self, data, spatial_only=False, spectral_only=False, cat_in_last_layer=None):

        if spatial_only:
            return self.forward_spatial(data)

        if not spectral_only:
            spatial_data = self.forward_spatial(data)
            for key, value in spatial_data.items():
                data[key] = value

        x = data["x"]
        pos = data["pos"]
        x = torch.cat([x, pos], dim=1)

        x = self.fc_in(x)

        for block in self.blocks:
            x = block(x)
            x_pool = torch.max(x, dim=2, keepdim=True)[0]
            x = torch.cat([x, x_pool], dim=1)

        x = self.block_last(x)

        if self.segmentation:
            x_pool = torch.max(x, dim=2, keepdim=True)[0]
            x = torch.cat([x, x_pool], dim=1)
            x = self.fc_out(x)
        else:
            x = torch.max(x, dim=2)[0]
            x = self.fc_out(x)

        return x
