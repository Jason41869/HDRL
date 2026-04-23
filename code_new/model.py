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
        self.kg_n_layer = self.cfg.kg_layer


        # relation embedding for aggregation
        self.rel_embs = nn.ParameterList([get_param(self.n_rel * 2, h_dim) for _ in range(self.kg_n_layer)])

        # relation embedding for prediction
        if self.cfg.pred_rel_w:
            self.rel_w = get_param(h_dim * self.kg_n_layer, h_dim)
        else:
            self.pred_rel_emb = get_param(self.n_rel * 2, h_dim)

        self.predictor = ConvE(h_dim, out_channels=self.cfg.out_channel, ker_sz=self.cfg.ker_sz)
        self.bce = nn.BCELoss()
        self.ent_drop = nn.Dropout(self.cfg.ent_drop)
        self.rel_drop = nn.Dropout(self.cfg.rel_drop)
        self.com_drop = nn.Dropout(self.cfg.ent_drop+0.1)
        self.pri_drop = nn.Dropout(self.cfg.ent_drop+0.05)
        self.act = nn.Tanh()
        self.L = nn.Linear(h_dim, h_dim)
        self.S = nn.Linear(h_dim, h_dim)
        self.mea_func = lambda x, y: (x.mean(dim=0), y.mean(dim=0))
        self.comp_layers = nn.ModuleList([CompLayer(h_dim) for _ in range(self.kg_n_layer)])
        # self.mlp1 = MLP1(h_dim, [200], h_dim)
        # self.mlp2 = MLP2(h_dim, [200], h_dim)

        self.relu = nn.ReLU()
        self.W_fuse = get_param((h_dim, h_dim))
        self.a_fuse = get_param((2 * h_dim, 1))
        self.LeakyReLU = nn.LeakyReLU(0.2)
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

        head = ent_emb[h_id]
        rel = rel_emb[r_id]
        # (bs, n_ent)
        score = self.predictor(head, rel, ent_emb)

        return score, corr

    def loss(self, score, label, corr):
        # (bs, n_ent)
        loss = self.bce(score, label) + 0.001 * corr

        return loss

    def fuse_semantic_gat(self, e_original, e_sem_list):
        """
        用GAT注意力计算语义空间融合权重
        :param e_original: 当前层原始实体嵌入，shape=(n_ent, h_dim)
        :param e_sem_list: 各语义空间聚合结果列表，len=n_sem，每个元素shape=(n_ent, h_dim)
        :return: 融合后的实体嵌入，shape=(n_ent, h_dim)
        """
        # 1. 初始化融合权重列表
        fuse_weights = []

        # 2. 对每个语义空间计算GAT注意力分数
        for e_sem in e_sem_list:
            # 线性变换（GAT核心步骤）
            e_o_t = torch.matmul(e_original, self.W_fuse)  # 原始嵌入变换
            e_s_t = torch.matmul(e_sem, self.W_fuse)  # 语义空间嵌入变换

            # 拼接特征（GAT注意力输入）
            attn_input = torch.cat([e_o_t, e_s_t], dim=1)  # shape=(n_ent, 2*h_dim)

            # 计算注意力分数（LeakyReLU激活）
            score = self.LeakyReLU(torch.matmul(attn_input, self.a_fuse)).squeeze(-1)  # shape=(n_ent,)
            fuse_weights.append(score)

        # 3. Softmax归一化融合权重（每个实体的语义空间权重和为1）
        fuse_weights = torch.stack(fuse_weights, dim=1)  # shape=(n_ent, n_sem)
        fuse_weights = torch.softmax(fuse_weights, dim=1)

        # 4. 加权融合各语义空间嵌入（残差连接）
        e_fuse = e_original  # 保留原始嵌入的残差
        for i in range(len(e_sem_list)):
            e_fuse = e_fuse + fuse_weights[:, i].unsqueeze(1) * e_sem_list[i]  # 按权重加权

        return e_fuse

    def aggragate_emb(self, kg):
        """
        aggregate embedding.
        :param kg:
        :return:
        """
        ent_emb = self.ent_emb
        common = self.S(ent_emb)
        private = self.L(ent_emb)

        corr_loss = 0.0
        e_sem_list = []

        for comp_layer, rel_emb in zip(self.comp_layers, self.rel_embs):

            ent_emb_drop = self.ent_drop(ent_emb)
            rel_emb_drop = self.rel_drop(rel_emb)
            common = self.com_drop(common)
            private = self.pri_drop(private)

            comp_ent_common = comp_layer(kg, common, rel_emb_drop)
            comp_ent_private = comp_layer(kg, private, rel_emb_drop)
            ent_emb_sem = ent_emb_drop + comp_ent_common + comp_ent_private
            e_sem_list.append(ent_emb_sem)


            ent_emb = ent_emb_sem

            phi_c, phi_p = self.mea_func(comp_ent_common, comp_ent_private)
            corr_loss = corr_loss + compute_corr(phi_c, phi_p)


        ent_emb_fused = self.fuse_semantic_gat(ent_emb, e_sem_list)
        rel_embs_tensor_list = [param for param in self.rel_embs]
        pred_rel_emb = torch.stack(rel_embs_tensor_list, dim=1).mean(dim=1)

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
        self.n_heads = self.cfg.n_heads
        self.h_dim = h_dim
        self.head_dim = self.h_dim // self.n_heads  # 480//8=60
        self.topk = self.cfg.topk
        self.W = nn.ParameterList([get_param(h_dim, self.head_dim) for _ in range(self.n_heads)])
        self.W_r = nn.ParameterList([get_param(h_dim, self.head_dim) for _ in range(self.n_heads)])
        self.a = nn.ParameterList([get_param(3 * self.head_dim, 1) for _ in range(self.n_heads)])
        # self.W = get_param(h_dim, h_dim)  # 实体投影
        # self.W_r = get_param(h_dim, h_dim)  # 关系投影
        # self.a = get_param(3 * h_dim, 1)  # 注意力向量 [src || dst || rel]
        assert self.comp_op in ['add', 'mul']
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.neigh_w = get_param(h_dim, h_dim)  # 480×480
        self.act = nn.Tanh()
        if self.cfg.bn:
            self.bn = torch.nn.BatchNorm1d(h_dim)
        else:
            self.bn = None

    # def forward(self, kg, ent_emb, rel_emb):
    #     """
    #     简化为单头注意力聚合模式（移除子图采样，全图单头计算）
    #     :param kg: DGL图对象
    #     :param ent_emb: 实体嵌入 (n_ent, emb_dim)
    #     :param rel_emb: 关系嵌入 (n_rel, emb_dim)
    #     :return: 聚合后的实体嵌入 (n_ent, h_dim)
    #     """
    #     with kg.local_scope():
    #         # ========== 1. 初始化全图特征（移除子图采样相关逻辑） ==========
    #         kg.ndata['emb'] = ent_emb
    #         # 获取关系ID（兼容无rel_id的情况）
    #         rel_id = kg.edata.get('rel_id', torch.zeros(kg.num_edges(), dtype=torch.long, device=self.device))
    #         kg.edata['emb'] = rel_emb[rel_id]
    #
    #         # ========== 2. 线性变换（单头：直接映射到隐藏维度） ==========
    #         # 单头模式下，W/W_r直接映射到目标维度，无需拆分多头
    #         h_node = torch.matmul(ent_emb, self.W)  # (n_ent, h_dim)
    #         h_rel = torch.matmul(rel_emb[rel_id], self.W_r)  # (n_edge, h_dim)
    #         kg.ndata['h'] = h_node  # 实体线性变换后特征
    #         kg.edata['r'] = h_rel  # 关系线性变换后特征
    #
    #         # ========== 3. 单头注意力计算（全图直接计算，无采样） ==========
    #         def attn_fn(edges):
    #             """单头注意力分数计算：头+尾+关系拼接后过线性层"""
    #             # 拼接特征：src(头实体) + dst(尾实体) + rel(关系)
    #             feat = torch.cat([edges.src['h'], edges.dst['h'], edges.data['r']], dim=1)
    #             # 单头注意力分数（LeakyReLU + 线性层，squeeze为标量）
    #             e = self.LeakyReLU(torch.matmul(feat, self.a)).squeeze(-1)
    #             return {'e': e}
    #
    #         # 全图计算每条边的注意力分数
    #         kg.apply_edges(attn_fn)
    #         # 单头softmax归一化注意力权重（dgl内置edge_softmax，适配全图）
    #         kg.edata['a'] = dgl.ops.edge_softmax(kg, kg.edata['e'])
    #
    #         # ========== 4. 单头加权聚合（核心逻辑保留，简化组合方式） ==========
    #         # 头实体特征 + 关系特征（add/mul二选一，单头无需复杂组合）
    #         if self.comp_op == 'add':
    #             comp = kg.ndata['h'][kg.edges()[0]] + kg.edata['r']  # 头实体+关系
    #         else:
    #             comp = kg.ndata['h'][kg.edges()[0]] * kg.edata['r']  # 头实体×关系
    #
    #         # 注意力加权：每个边的组合特征 × 注意力权重（扩展维度匹配）
    #         comp_weighted = comp * kg.edata['a'].unsqueeze(-1)
    #         kg.edata['msg'] = comp_weighted
    #
    #         # 消息传递：聚合入边的加权特征到目标节点
    #         kg.update_all(
    #             fn.copy_e('msg', 'm'),  # 复制边的加权特征作为消息
    #             fn.sum('m', 'neigh')  # 对消息求和（单头聚合）
    #         )
    #
    #         # ========== 5. 投影+激活（和原逻辑一致） ==========
    #         neigh_ent_emb = kg.ndata['neigh'] @ self.neigh_w  # 聚合特征投影
    #         if self.bn is not None:
    #             neigh_ent_emb = self.bn(neigh_ent_emb)  # 批归一化
    #         neigh_ent_emb = self.act(neigh_ent_emb)  # 激活函数
    #         # 可选：如需Dropout，取消注释（单头模式可保留轻量Dropout）
    #         # neigh_ent_emb = self.ent_drop(neigh_ent_emb)
    #
    #     return neigh_ent_emb

    def forward(self, kg, ent_emb, rel_emb):
        # ========== 第一步：全图计算采样权重 ==========
        with kg.local_scope():
            kg.ndata['global_nid'] = kg.nodes().to(self.device)
            kg.ndata['emb'] = ent_emb.to(self.device)

            if 'rel_id' in kg.edata:
                rel_id = kg.edata['rel_id'].to(self.device)
            else:
                rel_id = torch.zeros(kg.number_of_edges(), dtype=torch.long).to(self.device)
            kg.edata['emb'] = rel_emb[rel_id].to(self.device)

            W0 = self.W[0]
            W_r0 = self.W_r[0]
            a0 = self.a[0]
            kg.ndata['emb_t_samp'] = torch.matmul(kg.ndata['emb'], W0)
            kg.edata['rel_t_samp'] = torch.matmul(kg.edata['emb'], W_r0)

            def cat_uv_edges_samp(edges):
                return {'u_v_cat': torch.cat([edges.src['emb_t_samp'], edges.dst['emb_t_samp']], dim=1)}
            kg.apply_edges(cat_uv_edges_samp)

            kg.edata['attn_in_samp'] = torch.cat([kg.edata['u_v_cat'], kg.edata['rel_t_samp']], dim=1)
            kg.edata['score_samp'] = self.LeakyReLU(torch.matmul(kg.edata['attn_in_samp'], a0).squeeze(-1))

            sample_kg = dgl.sampling.select_topk(kg.to('cpu'), self.topk, 'score_samp', edge_dir='in').to(self.device)

        # ========== 第二步：子图多头注意力计算 ==========
        head_outputs = []
        for head in range(self.n_heads):
            with sample_kg.local_scope():
                if 'global_nid' not in sample_kg.ndata:
                    sample_kg.ndata['global_nid'] = sample_kg.nodes().to(self.device)
                subgraph_nids = sample_kg.ndata['global_nid'].to(self.device)

                # 2. 子图特征初始化（原始emb用于线性变换，不参与组合）
                sample_kg.ndata['emb_ori'] = ent_emb[subgraph_nids].to(self.device)  # 重命名为emb_ori（480维）
                if 'rel_id' in sample_kg.edata:
                    sample_kg_rid = sample_kg.edata['rel_id'].to(self.device)
                else:
                    sample_kg_rid = torch.zeros(sample_kg.number_of_edges(), dtype=torch.long).to(self.device)
                sample_kg.edata['emb_ori'] = rel_emb[sample_kg_rid].to(self.device)  # 关系原始emb（480维）

                # 3. GAT线性变换（得到60维的emb_t/rel_t）
                W = self.W[head]
                W_r = self.W_r[head]
                a = self.a[head]
                sample_kg.ndata['emb_t'] = torch.matmul(sample_kg.ndata['emb_ori'], W)  # 60维
                sample_kg.edata['rel_t'] = torch.matmul(sample_kg.edata['emb_ori'], W_r)  # 60维

                # 4. 子图内拼接函数（用于注意力计算）
                def cat_uv_edges(edges):
                    return {'u_v_cat': torch.cat([edges.src['emb_t'], edges.dst['emb_t']], dim=1)}
                sample_kg.apply_edges(cat_uv_edges)

                # 5. 注意力计算
                sample_kg.edata['attn_in'] = torch.cat([sample_kg.edata['u_v_cat'], sample_kg.edata['rel_t']], dim=1)
                sample_kg.edata['score'] = self.LeakyReLU(torch.matmul(sample_kg.edata['attn_in'], a).squeeze(-1))
                sample_kg.edata['norm'] = dgl.ops.edge_softmax(sample_kg, sample_kg.edata['score'])

                # 6. 实体-关系组合（核心修改：用60维的emb_t/rel_t，而非480维的emb_ori）
                # 先把rel_t赋值为边的节点特征，再和emb_t组合
                sample_kg.edata['rel_t_node'] = sample_kg.edata['rel_t']  # 边的关系特征转为节点特征（60维）
                if self.comp_op == 'add':
                    # u_add_e: 源节点emb_t + 边的rel_t_node（都是60维）
                    sample_kg.apply_edges(fn.u_add_e('emb_t', 'rel_t_node', 'comp_emb'))
                else:
                    # u_mul_e: 源节点emb_t × 边的rel_t_node（都是60维）
                    sample_kg.apply_edges(fn.u_mul_e('emb_t', 'rel_t_node', 'comp_emb'))

                # 7. 加权聚合（comp_emb是60维，聚合后neigh也是60维）
                comp_emb = sample_kg.edata['comp_emb'] * sample_kg.edata['norm'].unsqueeze(-1)
                sample_kg.edata['comp_emb_non_inplace'] = comp_emb  # 新键存储
                sample_kg.update_all(fn.copy_e('comp_emb_non_inplace', 'm'), fn.sum('m', 'neigh'))

                # 8. 补全全量节点特征（此时neigh是60维，和neigh_ent_emb_head维度匹配）
                neigh_ent_emb_head = torch.zeros((self.n_ent, self.head_dim), device=self.device)
                neigh_ent_emb_head = neigh_ent_emb_head.scatter_(
                    0,  # 按第0维（实体ID）赋值
                    subgraph_nids.unsqueeze(1).expand(-1, self.head_dim),  # 扩展维度匹配
                    sample_kg.ndata['neigh']
                )
                head_outputs.append(neigh_ent_emb_head)

        # ========== 第三步：多头拼接+维度对齐 ==========
        neigh_ent_emb = torch.cat(head_outputs, dim=1)  # 14541×(8×60)=14541×480
        neigh_ent_emb = neigh_ent_emb.mm(self.neigh_w)  # 14541×480 × 480×480 → 正常

        if callable(self.bn):
            neigh_ent_emb = self.bn(neigh_ent_emb)
        neigh_ent_emb = self.act(neigh_ent_emb)

        return neigh_ent_emb


def compute_corr(x1, x2):
    # Subtract the mean
    x1_mean = torch.mean(x1, 0, True)
    x1 = x1 - x1_mean
    x2_mean = torch.mean(x2, 0, True)
    x2 = x2 - x2_mean

    # Compute the cross correlation
    sigma1 = torch.sqrt(torch.mean(x1.pow(2)))
    sigma2 = torch.sqrt(torch.mean(x2.pow(2)))
    corr = torch.abs(torch.mean(x1 * x2)) / (sigma1 * sigma2)

    return corr








