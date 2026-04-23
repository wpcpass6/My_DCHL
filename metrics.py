# coding=utf-8
"""H-DCHL-B 评估指标。"""

import numpy as np


def hit_k(y_pred, y_true, k):
    """计算单样本 Recall@K 命中。"""
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    return 1 if y_true in y_pred_indices else 0


def ndcg_k(y_pred, y_true, k):
    """计算单样本 NDCG@K。"""
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        position = y_pred_indices.index(y_true) + 1
        return 1 / np.log2(1 + position)
    return 0


def batch_performance(batch_y_pred, batch_y_true, k):
    """计算一个 batch 上的 Recall@K 与 NDCG@K。"""
    batch_size = batch_y_pred.size(0)
    batch_recall = 0
    batch_ndcg = 0
    for idx in range(batch_size):
        batch_recall += hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_ndcg += ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
    return batch_recall / batch_size, batch_ndcg / batch_size
