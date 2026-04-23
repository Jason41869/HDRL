import torch
import torch.nn as nn
import utils
import torch.nn.functional as F


class ConvE(nn.Module):
    def __init__(self, h_dim, out_channels, ker_sz):
        super().__init__()
        cfg = utils.get_global_config()
        self.cfg = cfg
        dataset = cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.h_dim = h_dim

        # 校验维度：标准 ConvE 使用 (k_h, 2*k_w) 布局
        assert h_dim == cfg.k_h * cfg.k_w, "h_dim must equal k_h * k_w"

        self.k_h = cfg.k_h
        self.k_w = cfg.k_w
        self.out_channels = out_channels
        self.ker_sz = ker_sz

        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm1d(h_dim)

        self.conv_drop = nn.Dropout(cfg.conv_drop)
        self.fc_drop = nn.Dropout(cfg.fc_drop)
        self.ent_drop = nn.Dropout(cfg.ent_drop_pred)

        # 卷积层
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=ker_sz,
            stride=1,
            padding=0,
            bias=False
        )

        # 计算卷积后展平维度 (使用正确布局: k_h x 2*k_w)
        conv_h = self.k_h - ker_sz + 1
        conv_w = 2 * self.k_w - ker_sz + 1
        self.flat_sz = out_channels * conv_h * conv_w

        self.fc = nn.Linear(self.flat_sz, h_dim, bias=False)

    def forward(self, head, rel, all_ent):
        # 1. 拼接 head + rel -> (bs, 2*h_dim)
        emb_cat = torch.cat([head, rel], dim=1)  # (bs, 2*h_dim)

        # 2. Reshape to 2D grid: (bs, 1, k_h, 2*k_w)
        emb_2d = emb_cat.view(-1, 1, self.k_h, 2 * self.k_w)

        # 3. 卷积块
        x = self.bn0(emb_2d)
        x = self.conv(x)  # (bs, out_channels, conv_h, conv_w)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_drop(x)

        # 4. 展平 + 全连接
        x = x.view(-1, self.flat_sz)  # (bs, flat_sz)
        x = self.fc(x)  # (bs, h_dim)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc_drop(x)

        # 5. 与实体嵌入点积 + sigmoid
        all_ent = self.ent_drop(all_ent)
        scores = torch.mm(x, all_ent.transpose(1, 0))  # (bs, n_ent)
        scores = torch.sigmoid(scores)
        return scores

