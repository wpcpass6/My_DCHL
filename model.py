import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroHyperConvLayer(nn.Module):
    """
    异构超图卷积层。

    该层用于处理“实体节点 + 语义节点（或用户节点）”组成的二部超图：
    - 节点到超边：先对 POI 聚合，再与超边所属实体节点融合
    - 超边到节点：将超边消息回传到 POI 侧
    """

    def __init__(self, emb_dim, device):
        super().__init__()
        self.poi_linear = nn.Linear(emb_dim, emb_dim, bias=False, device=device)
        self.edge_linear = nn.Linear(emb_dim, emb_dim, bias=False, device=device)
        self.fusion_linear = nn.Linear(2 * emb_dim, emb_dim, bias=False, device=device)

    def forward(self, poi_embs, edge_embs, hg_edge_to_poi, hg_poi_to_edge):
        # 1) 先将 POI 聚合到超边侧
        poi_msg = torch.sparse.mm(hg_poi_to_edge, self.poi_linear(poi_embs))

        # 2) 超边既接收 POI 聚合消息，也保留自身实体语义（user/region/category）
        edge_msg = self.edge_linear(edge_embs)
        fused_edge = self.fusion_linear(torch.cat([poi_msg, edge_msg], dim=1))

        # 3) 再将超边消息回传到 POI 侧
        propagated_poi = torch.sparse.mm(hg_edge_to_poi, fused_edge)
        return propagated_poi, fused_edge


class HeteroHyperConvNetwork(nn.Module):
    """堆叠多层异构超图卷积，并使用残差连接。"""

    def __init__(self, num_layers, emb_dim, dropout, device):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv = HeteroHyperConvLayer(emb_dim, device)

    def forward(self, poi_embs, edge_embs, hg_edge_to_poi, hg_poi_to_edge):
        final_poi_embs = [poi_embs]
        final_edge_embs = [edge_embs]
        for _ in range(self.num_layers):
            poi_embs, edge_embs = self.conv(poi_embs, edge_embs, hg_edge_to_poi, hg_poi_to_edge)
            poi_embs = F.dropout(poi_embs + final_poi_embs[-1], self.dropout)
            edge_embs = F.dropout(edge_embs + final_edge_embs[-1], self.dropout)
            final_poi_embs.append(poi_embs)
            final_edge_embs.append(edge_embs)
        poi_output = torch.mean(torch.stack(final_poi_embs), dim=0)
        edge_output = torch.mean(torch.stack(final_edge_embs), dim=0)
        return poi_output, edge_output


class DirectedHyperConvLayer(nn.Module):
    """有向 POI 转移卷积层"""

    def forward(self, poi_embs, hg_poi_src, hg_poi_tar):
        msg_tar = torch.sparse.mm(hg_poi_tar, poi_embs)
        msg_src = torch.sparse.mm(hg_poi_src, msg_tar)
        return msg_src


class DirectedHyperConvNetwork(nn.Module):
    """堆叠多层有向转移卷积。"""

    def __init__(self, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layer = DirectedHyperConvLayer()

    def forward(self, poi_embs, hg_poi_src, hg_poi_tar):
        final_poi_embs = [poi_embs]
        for _ in range(self.num_layers):
            poi_embs = self.layer(poi_embs, hg_poi_src, hg_poi_tar)
            poi_embs = F.dropout(poi_embs + final_poi_embs[-1], self.dropout)
            final_poi_embs.append(poi_embs)
        return torch.mean(torch.stack(final_poi_embs), dim=0)


class HDCHLB(nn.Module):
    """
    主模型。

    分支说明：
    - collaborative branch: User-POI 长期协同
    - transition branch: 有向 POI 转移
    - region branch: POI-Region 异构语义
    - category branch: POI-Category 异构语义
    """

    def __init__(self, num_users, num_pois, num_regions, num_categories, padding_idx, args, device):
        super().__init__()
        self.num_users = num_users
        self.num_pois = num_pois
        self.num_regions = num_regions
        self.num_categories = num_categories
        self.emb_dim = args.emb_dim
        self.device = device
        self.mask_rate_cat = args.mask_rate_cat
        self.lambda_cat = args.lambda_cat
        self.mask_rate_reg = args.mask_rate_reg
        self.lambda_reg = args.lambda_reg
        self.mask_alpha = args.mask_alpha

        # 四类节点的基础 embedding
        self.user_embedding = nn.Embedding(num_users, self.emb_dim)
        self.poi_embedding = nn.Embedding(num_pois + 1, self.emb_dim, padding_idx=padding_idx)
        self.region_embedding = nn.Embedding(num_regions, self.emb_dim)
        self.category_embedding = nn.Embedding(num_categories, self.emb_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.region_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)

        # Category mask token：当类别节点被遮蔽时，用可学习 token 代替原始类别嵌入
        self.mask_category_token = nn.Parameter(torch.rand(1, self.emb_dim, device=device))
        # Region mask token：当区域节点被遮蔽时，用可学习 token 代替原始区域嵌入
        self.mask_region_token = nn.Parameter(torch.rand(1, self.emb_dim, device=device))

        # 节点自门控：延续 DCHL 的 disentangled 输入风格，但现在对应四个结构分支
        self.w_gate_col = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_col = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_trans = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_trans = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_reg = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_reg = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        self.w_gate_cat = nn.Parameter(torch.FloatTensor(self.emb_dim, self.emb_dim))
        self.b_gate_cat = nn.Parameter(torch.FloatTensor(1, self.emb_dim))
        for weight in [
            self.w_gate_col, self.b_gate_col, self.w_gate_trans, self.b_gate_trans,
            self.w_gate_reg, self.b_gate_reg, self.w_gate_cat, self.b_gate_cat,
        ]:
            nn.init.xavier_normal_(weight.data)

        # 四个结构分支
        self.col_network = HeteroHyperConvNetwork(args.num_col_layers, args.emb_dim, args.dropout, device)
        self.reg_network = HeteroHyperConvNetwork(args.num_reg_layers, args.emb_dim, args.dropout, device)
        self.cat_network = HeteroHyperConvNetwork(args.num_cat_layers, args.emb_dim, args.dropout, device)
        self.trans_network = DirectedHyperConvNetwork(args.num_trans_layers, args.dropout)

        # 用户侧自适应融合门，维持 DCHL 风格
        self.col_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.trans_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.reg_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())
        self.cat_gate = nn.Sequential(nn.Linear(args.emb_dim, 1), nn.Sigmoid())

        # Category 重建解码器：将传播后的类别表示还原到原始类别嵌入空间
        self.category_decoder = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.ELU(),
            nn.Linear(args.emb_dim, args.emb_dim),
        )
        # Region 重建解码器：将传播后的区域表示还原到原始区域嵌入空间
        self.region_decoder = nn.Sequential(
            nn.Linear(args.emb_dim, args.emb_dim),
            nn.ELU(),
            nn.Linear(args.emb_dim, args.emb_dim),
        )

    @staticmethod
    def masked_mean_pooling(seq_embs, seq_mask):
        """对 prefix 序列做 masked mean pooling。"""
        mask = seq_mask.unsqueeze(-1).float()
        summed = torch.sum(seq_embs * mask, dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        return summed / denom

    def apply_category_mask(self, category_embs):
        """
        对类别节点做随机 mask。

        返回：
        1. 被替换后的类别嵌入
        2. 被 mask 的类别索引
        3. 原始未扰动类别嵌入（作为重建目标）
        """
        original_category_embs = category_embs.clone()
        if self.mask_rate_cat <= 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=category_embs.device)
            return category_embs, empty_idx, original_category_embs

        num_categories = category_embs.size(0)
        num_mask = max(1, int(self.mask_rate_cat * num_categories))
        # num_categories=5,perm = [3, 1, 4, 0, 2]随机打乱这些数字
        perm = torch.randperm(num_categories, device=category_embs.device)
        mask_idx = perm[:num_mask] # 取前num_mask个

        masked_category_embs = category_embs.clone()
        masked_category_embs[mask_idx] = 0.0
        masked_category_embs[mask_idx] += self.mask_category_token
        return masked_category_embs, mask_idx, original_category_embs

    def apply_region_mask(self, region_embs):
        """
        对区域节点做随机 mask。

        返回：
        1. 被替换后的区域嵌入
        2. 被 mask 的区域索引
        3. 原始未扰动区域嵌入（作为重建目标）
        """
        original_region_embs = region_embs.clone()
        if self.mask_rate_reg <= 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=region_embs.device)
            return region_embs, empty_idx, original_region_embs

        num_regions = region_embs.size(0)
        num_mask = max(1, int(self.mask_rate_reg * num_regions))
        perm = torch.randperm(num_regions, device=region_embs.device)
        mask_idx = perm[:num_mask]

        masked_region_embs = region_embs.clone()
        masked_region_embs[mask_idx] = 0.0
        masked_region_embs[mask_idx] += self.mask_region_token
        return masked_region_embs, mask_idx, original_region_embs

    def sce_loss(self, pred_embs, target_embs):
        """HygMap 风格的 SCE 重建损失。"""
        pred_embs = F.normalize(pred_embs, p=2, dim=-1)
        target_embs = F.normalize(target_embs, p=2, dim=-1)
        return (1 - (pred_embs * target_embs).sum(dim=-1)).pow(self.mask_alpha).mean()

    def forward(self, dataset, batch):
        # 1) 为不同分支生成独立的输入门控 POI 表示
        base_poi_embs = self.poi_embedding.weight[:-1]
        col_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_col + self.b_gate_col))
        trans_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_trans + self.b_gate_trans))
        reg_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_reg + self.b_gate_reg))
        cat_poi_embs = torch.multiply(base_poi_embs, torch.sigmoid(base_poi_embs @ self.w_gate_cat + self.b_gate_cat))

        # 2) 协同分支：User-POI 异构超图
        col_edge_embs = self.user_embedding.weight
        col_poi_out, col_user_out = self.col_network(col_poi_embs, col_edge_embs, dataset.HG_pu, dataset.HG_up)

        # 3) 区域分支：POI-Region 异构超图
        reg_edge_embs = self.region_embedding.weight
        reg_mask_idx = torch.empty(0, dtype=torch.long, device=self.device)
        original_reg_edge_embs = reg_edge_embs

        # 只在训练阶段启用 Region mask，测试阶段保持完整区域图结构
        if self.training and self.mask_rate_reg > 0:
            reg_edge_embs, reg_mask_idx, original_reg_edge_embs = self.apply_region_mask(reg_edge_embs)

        reg_poi_out, reg_region_out = self.reg_network(reg_poi_embs, reg_edge_embs, dataset.HG_pr, dataset.HG_rp)

        # 4) 类别分支：POI-Category 异构超图
        cat_edge_embs = self.category_embedding.weight
        cat_mask_idx = torch.empty(0, dtype=torch.long, device=self.device)
        original_cat_edge_embs = cat_edge_embs

        # 只在训练阶段启用 Category mask，测试阶段保持完整类别图结构
        if self.training and self.mask_rate_cat > 0:
            cat_edge_embs, cat_mask_idx, original_cat_edge_embs = self.apply_category_mask(cat_edge_embs)

        cat_poi_out, cat_category_out = self.cat_network(cat_poi_embs, cat_edge_embs, dataset.HG_pc, dataset.HG_cp)

        # 5) 转移分支：保持原 DCHL 风格
        trans_poi_out = self.trans_network(trans_poi_embs, dataset.HG_poi_src, dataset.HG_poi_tar)

        # 6) 基于当前 prefix 生成样本级表示，而不是只用静态 user embedding。
        batch_user_idx = batch["user_idx"]
        batch_prefix = batch["user_seq"]
        batch_prefix_mask = batch["user_seq_mask"]

        zero_pad = torch.zeros(1, self.emb_dim, device=self.device)
        col_poi_out_with_pad = torch.cat([col_poi_out, zero_pad], dim=0)
        reg_poi_out_with_pad = torch.cat([reg_poi_out, zero_pad], dim=0)
        cat_poi_out_with_pad = torch.cat([cat_poi_out, zero_pad], dim=0)
        trans_poi_out_with_pad = torch.cat([trans_poi_out, zero_pad], dim=0)

        col_prefix_embs = col_poi_out_with_pad[batch_prefix]
        reg_prefix_embs = reg_poi_out_with_pad[batch_prefix]
        cat_prefix_embs = cat_poi_out_with_pad[batch_prefix]
        trans_prefix_embs = trans_poi_out_with_pad[batch_prefix]

        col_prefix_user = self.masked_mean_pooling(col_prefix_embs, batch_prefix_mask)
        reg_prefix_user = self.masked_mean_pooling(reg_prefix_embs, batch_prefix_mask)
        cat_prefix_user = self.masked_mean_pooling(cat_prefix_embs, batch_prefix_mask)
        trans_prefix_user = self.masked_mean_pooling(trans_prefix_embs, batch_prefix_mask)

        col_batch_user = col_user_out[batch_user_idx] + col_prefix_user
        reg_batch_user = reg_prefix_user
        cat_batch_user = cat_prefix_user
        trans_batch_user = trans_prefix_user

        # 7) 归一化后做用户侧自适应融合
        norm_col_user = F.normalize(col_batch_user, p=2, dim=1)
        norm_reg_user = F.normalize(reg_batch_user, p=2, dim=1)
        norm_cat_user = F.normalize(cat_batch_user, p=2, dim=1)
        norm_trans_user = F.normalize(trans_batch_user, p=2, dim=1)

        col_coef = self.col_gate(norm_col_user)
        reg_coef = self.reg_gate(norm_reg_user)
        cat_coef = self.cat_gate(norm_cat_user)
        trans_coef = self.trans_gate(norm_trans_user)

        final_user_embs = (
            col_coef * norm_col_user + reg_coef * norm_reg_user + cat_coef * norm_cat_user + trans_coef * norm_trans_user
        )

        # 8) POI 侧先直接求和，后续如果需要可再扩展成更细的层次融合
        final_poi_embs = (
            F.normalize(col_poi_out, p=2, dim=1)
            + F.normalize(reg_poi_out, p=2, dim=1)
            + F.normalize(cat_poi_out, p=2, dim=1)
            + F.normalize(trans_poi_out, p=2, dim=1)
        )

        # 9) 计算下一 POI 的打分
        prediction = final_user_embs @ final_poi_embs.T

        # Category / Region mask 重建损失：直接监督异构节点表示，而不只让它们充当中间容器
        aux_loss = prediction.new_tensor(0.0)

        if self.training and reg_mask_idx.numel() > 0:
            reconstructed_reg = self.region_decoder(reg_region_out[reg_mask_idx])
            target_reg = original_reg_edge_embs[reg_mask_idx]
            aux_loss = aux_loss + self.lambda_reg * self.sce_loss(reconstructed_reg, target_reg)

        if self.training and cat_mask_idx.numel() > 0:
            reconstructed_cat = self.category_decoder(cat_category_out[cat_mask_idx])
            target_cat = original_cat_edge_embs[cat_mask_idx]
            aux_loss = aux_loss + self.lambda_cat * self.sce_loss(reconstructed_cat, target_cat)

        return prediction, aux_loss
