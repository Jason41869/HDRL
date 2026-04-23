#!/usr/bin/python3
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import utils
from utils import get_param
from decoder import ConvE
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn




class DSGNet(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        # get params
        self.cfg = utils.get_global_config()
        self.dataset = self.cfg.dataset
        self.device = self.cfg.device
        self.n_ent = utils.DATASET_STATISTICS[self.dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[self.dataset]['n_rel']
        self.ent_emb = get_param(self.n_ent, h_dim)
        self.max_n = self.cfg.max_n#最大语义空间数
        self.kg_n_layer_gate = nn.Parameter(torch.randn(self.max_n))#动态最优门控



        # relation embedding for aggregation
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.max_n)])

        # relation embedding for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * self.max_n, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)



        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)
        self.bce = nn.BCEWithLogitsLoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        #self.act = nn.Tanh()
        self.L = nn.Linear(h_dim, h_dim)
        self.S = nn.Linear(h_dim, h_dim)
        self.mea_func = lambda x, y: (x.mean(dim=0), y.mean(dim=0))
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.max_n)])
        # self.mlp1 = MLP1(h_dim, [200], h_dim)
        # self.mlp2 = MLP2(h_dim, [200], h_dim)

        #self.relu = nn.ReLU()




    def forward(self, h_id, r_id, kg):
        """
        matching computation between query (h, r) and answer t.
        :param h_id: head entity id, (bs, )
        :param r_id: relation id, (bs, )
        :param kg: aggregation graph
        :return: matching score, (bs, n_ent)
        """
        # aggregate embedding


        ent_emb, rel_emb, corr = self.aggragate_emb(kg)

        head = ent_emb[h_id].to(self.device, non_blocking=True)
        rel = rel_emb[r_id].to(self.device, non_blocking=True)
        # (bs, n_ent)
        score = self.predictor(head, rel, ent_emb)
        score = score

        return score, corr

    def loss(self, score, label, corr):
        # (bs, n_ent)

        loss = self.bce(score, label) + 0.1 * corr + 1e-4 * torch.norm(self.kg_n_layer_gate, 1)

        return loss

    # def fuse_semantic_gat(self, e_original, e_sem_list):
    #     """
    #     用GAT注意力计算语义空间融合权重
    #     :param e_original: 当前层原始实体嵌入，shape=(n_ent, h_dim)
    #     :param e_sem_list: 各语义空间聚合结果列表，len=n_sem，每个元素shape=(n_ent, h_dim)
    #     :return: 融合后的实体嵌入，shape=(n_ent, h_dim)
    #     """
    #     # 1. 初始化融合权重列表
    #     fuse_weights = []
    #
    #     # 2. 对每个语义空间计算GAT注意力分数
    #     for e_sem in e_sem_list:
    #         # 线性变换（GAT核心步骤）
    #         e_o_t = torch.matmul(e_original, self.W_fuse)  # 原始嵌入变换
    #         e_s_t = torch.matmul(e_sem, self.W_fuse)  # 语义空间嵌入变换
    #
    #         # 拼接特征（GAT注意力输入）
    #         attn_input = torch.cat([e_o_t, e_s_t], dim=1)  # shape=(n_ent, 2*h_dim)
    #
    #         # 计算注意力分数（LeakyReLU激活）
    #         score = self.LeakyReLU(torch.matmul(attn_input, self.a_fuse)).squeeze(-1)  # shape=(n_ent,)
    #         fuse_weights.append(score)
    #
    #     # 3. Softmax归一化融合权重（每个实体的语义空间权重和为1）
    #     fuse_weights = torch.stack(fuse_weights, dim=1)  # shape=(n_ent, n_sem)
    #     fuse_weights = torch.softmax(fuse_weights, dim=1)
    #
    #     # 4. 加权融合各语义空间嵌入（残差连接）
    #     e_fuse = e_original  # 保留原始嵌入的残差
    #     for i in range(len(e_sem_list)):
    #         e_fuse = e_fuse + fuse_weights[:, i].unsqueeze(1) * e_sem_list[i]  # 按权重加权
    #
    #     return e_fuse

    def get_effective_spaces(self):
        """
        基于门控分数选取有效语义空间
        :自适应选前top-k（k=有效分数>阈值的数量，比如阈值0.1）
        :return: 有效空间的索引、对应门控分数（归一化）
        """
        # 步骤1：对门控分数做softmax，得到每个空间的权重（归一化）
        gate_weights = F.softmax(self.kg_n_layer_gate, dim=0)  # (max_n,)

        # 步骤2：选取有效空间（自适应逻辑）
        # 方案1：自适应选前top-k（k=有效分数>阈值的数量，比如阈值0.1）
        effective_mask = gate_weights > 0.1
        effective_indices = torch.where(effective_mask)[0]
            # 兜底：至少选1个空间（避免无有效空间）

        if len(effective_indices) == 0:
            effective_indices = torch.topk(gate_weights, k=1).indices

        if len(effective_indices) > 3:
            effective_indices = effective_indices[:3]
        # 步骤3：提取有效空间的权重并归一化

        effective_gates = gate_weights[effective_indices]
        effective_gates = effective_gates / effective_gates.sum()  # 二次归一化

        return effective_indices, effective_gates

    def aggragate_emb(self, kg):
        ent_emb = self.ent_emb  # (n_ent, h_dim)
        common = self.S(ent_emb)  # (n_ent, h_dim)
        private = self.L(ent_emb)  # (n_ent, h_dim)

        effective_indices, effective_gates = self.get_effective_spaces()
        effective_n = len(effective_indices)
        corr_loss = 0.0
        e_sem_list = []

        for idx in effective_indices:
            comp_layer = self.comp_layers[idx]
            rel_emb_drop = self.rel_drop(self.rel_embs[idx])  # 已在 device 上

            # 聚合 common 和 private 信息
            comp_ent_common = comp_layer(kg, common, rel_emb_drop)
            comp_ent_private = comp_layer(kg, private, rel_emb_drop)

            ent_emb_sem = ent_emb + comp_ent_common + comp_ent_private
            e_sem_list.append(ent_emb_sem)

            # 相关性损失（detach 阻断梯度）
            corr_loss += compute_corr(comp_ent_common.detach(), comp_ent_private.detach())

        # 归一化相关性损失
        if effective_n > 0:
            corr_loss = corr_loss / effective_n

        # 加权融合实体嵌入
        e_sem_stack = torch.stack(e_sem_list, dim=1)  # (n_ent, effective_n, h_dim)
        gates = effective_gates.view(1, -1, 1)  # (1, effective_n, 1)
        ent_emb_fused = (e_sem_stack * gates).sum(dim=1)  # (n_ent, h_dim)

        # 加权融合关系嵌入（用于预测）
        rel_embs_stack = torch.stack([self.rel_embs[i] for i in effective_indices],
                                     dim=1)  # (n_rel*2, effective_n, h_dim)
        pred_rel_emb = (rel_embs_stack * gates).sum(dim=1)  # (n_rel*2, h_dim)

        ent_emb_fused = self.ent_drop(ent_emb_fused)
        pred_rel_emb = self.ent_drop(pred_rel_emb)

        return ent_emb_fused, pred_rel_emb, corr_loss


class CompLayer(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.cfg = utils.get_global_config()
        self.device = self.cfg.device
        dataset = self.cfg.dataset
        self.n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
        self.n_rel = utils.DATASET_STATISTICS[dataset]['n_rel']
        self.comp_op = self.cfg.comp_op
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.h_dim = h_dim
        self.topk = self.cfg.topk
        self.W = get_param(h_dim, h_dim)
        self.W_r = get_param(h_dim, h_dim)
        self.a = get_param(3 * h_dim, 1)
        assert self.comp_op in ['add', 'mul']
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.neigh_w = get_param(h_dim, h_dim).to(self.device)  # 480×480
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    def forward(self, kg, ent_emb, rel_emb):
        # ========== 第一步：全图计算采样权重（仅用 head=0 的参数） ==========
        with kg.local_scope():
            kg.ndata['emb'] = ent_emb
            if 'rel_id' in kg.edata:
                rel_id = kg.edata['rel_id']
            else:
                rel_id = torch.zeros(kg.num_edges(), dtype=torch.long, device=self.device)
            kg.edata['emb'] = rel_emb[rel_id]

            h_node = torch.matmul(ent_emb, self.W)
            h_rel = torch.matmul(rel_emb[rel_id], self.W_r)

            kg.ndata['h_src'] = h_node  # (n_ent, head_dim)
            kg.edata['r_emb'] = h_rel  # (n_edge, head_dim)

            def compute_score(edges):
                # 拼接 [src, dst, rel] -> (..., 3 * head_dim)
                feat = torch.cat([edges.src['h_src'], edges.dst['h_src'], edges.data['r_emb']], dim=1)
                return {'score_samp': self.LeakyReLU(torch.matmul(feat, self.a)).squeeze(-1)}

            kg.apply_edges(compute_score)
            # 采样：转到 CPU 做 topk，再移回 GPU
            sample_kg = dgl.sampling.select_topk(kg.cpu(), self.topk, 'score_samp', edge_dir='in').to(self.device)

        # ========== 第二步：注意力聚合 ==========

            with sample_kg.local_scope():
                # 获取子图中的全局节点 ID
                subgraph_nids = sample_kg.ndata.get('global_nid', sample_kg.nodes()).to(self.device)

                # 提取对应嵌入
                node_emb = ent_emb[subgraph_nids]  # (n_sub, h_dim)
                if 'rel_id' in sample_kg.edata:
                    edge_rel_ids = sample_kg.edata['rel_id']
                else:
                    edge_rel_ids = torch.zeros(sample_kg.num_edges(), dtype=torch.long, device=self.device)
                edge_rel_emb = rel_emb[edge_rel_ids]  # (n_edge_sub, h_dim)


                h_node_sub = torch.matmul(node_emb, self.W)
                h_rel_sub = torch.matmul(edge_rel_emb, self.W_r)

                sample_kg.ndata['h'] = h_node_sub
                sample_kg.edata['r'] = h_rel_sub

                # 计算注意力系数
                def attn_fn(edges):
                    alpha = torch.cat([edges.src['h'], edges.dst['h'], edges.data['r']], dim=1)
                    return {'e': self.LeakyReLU(torch.matmul(alpha, self.a)).squeeze(-1)}

                sample_kg.apply_edges(attn_fn)
                e = sample_kg.edata['e']
                sample_kg.edata['a'] = dgl.ops.edge_softmax(sample_kg, e)

                # 实体-关系组合 + 加权聚合
                if self.comp_op == 'add':
                    comp = sample_kg.ndata['h'][sample_kg.edges()[0]] + sample_kg.edata['r']
                else:  # mul
                    comp = sample_kg.ndata['h'][sample_kg.edges()[0]] * sample_kg.edata['r']

                comp_weighted = comp * sample_kg.edata['a'].unsqueeze(-1)
                sample_kg.edata['msg'] = comp_weighted
                sample_kg.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'neigh'))

                # scatter 到全图
                neigh_full = torch.zeros(self.n_ent, self.h_dim, device=self.device)
                neigh_full[subgraph_nids] = sample_kg.ndata['neigh']

        # ========== 第三步：投影+激活==========
        neigh_ent_emb = neigh_full @ self.neigh_w
        if self.bn is not None:
            neigh_ent_emb = self.bn(neigh_ent_emb)
        neigh_ent_emb = self.act(neigh_ent_emb)
        neigh_ent_emb = self.ent_drop(neigh_ent_emb)

        return neigh_ent_emb


def compute_corr(x1, x2):
    x1 = x1 - x1.mean(0, keepdim=True)
    x2 = x2 - x2.mean(0, keepdim=True)
    sigma1 = x1.pow(2).mean().sqrt()
    sigma2 = x2.pow(2).mean().sqrt()
    corr = (x1 * x2).mean().abs() / (sigma1 * sigma2 + 1e-8)  # 防除零
    return corr


# class MLP(nn.Module):
#     def __init__(self, input_d, structure, output_d, dropprob=0.0):
#         super(MLP, self).__init__()
#         self.net = nn.ModuleList()
#         self.dropout = torch.nn.Dropout(dropprob)
#         struc = [input_d] + structure + [output_d]
#
#         for i in range(len(struc) - 1):
#             self.net.append(nn.Linear(struc[i], struc[i + 1]))
#
#     def forward(self, x):
#         for i in range(len(self.net) - 1):
#             x = F.relu(self.net[i](x))
#             x = self.dropout(x)
#
#         # For the last layer
#         y = self.net[-1](x)
#
#         return y


# class MLP1(nn.Module):
#     def __init__(self, input_d, structure, output_d, dropprob=0.0):
#         super(MLP1, self).__init__()
#         self.net = nn.ModuleList()
#         self.dropout = torch.nn.Dropout(dropprob)
#         struc = [input_d] + structure + [output_d]
#
#         for i in range(len(struc) - 1):
#             self.net.append(nn.Linear(struc[i], struc[i + 1]))
#
#     def forward(self, x):
#         for i in range(len(self.net) - 1):
#             x = F.relu(self.net[i](x))
#             x = self.dropout(x)
#
#         # For the last layer
#         y = self.net[-1](x)
#
#         return y
#
#
# class MLP2(nn.Module):
#     def __init__(self, input_d, structure, output_d, dropprob=0.0):
#         super(MLP2, self).__init__()
#         self.net = nn.ModuleList()
#         self.dropout = torch.nn.Dropout(dropprob)
#         struc = [input_d] + structure + [output_d]
#
#         for i in range(len(struc) - 1):
#             self.net.append(nn.Linear(struc[i], struc[i + 1]))
#
#     def forward(self, x):
#         for i in range(len(self.net) - 1):
#             x = F.relu(self.net[i](x))
#             x = self.dropout(x)
#
#         # For the last layer
#         y = self.net[-1](x)
#
#         return y


# measurable functions \phi and \psi
# class Measure_F(nn.Module):
#     def __init__(self, view1_dim, view2_dim, phi_size, psi_size, latent_dim=1):
#         super(Measure_F, self).__init__()
#         self.phi = MLP(view1_dim, phi_size, latent_dim)
#         self.psi = MLP(view2_dim, psi_size, latent_dim)
#
#     def forward(self, x1, x2):
#         y1 = self.phi(x1)
#         y2 = self.psi(x2)
#         return y1, y2


