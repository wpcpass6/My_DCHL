# coding=utf-8
"""
H-DCHL-B 通用工具函数。

说明：
1. 尽量延续 DCHL 的稀疏矩阵实现风格，避免第一版就切到 PyG；
2. 这里重点提供异构超图构图、序列整理、文件读写等能力；
3. 所有函数都使用中文注释，便于后续继续扩展到掩码自监督版本。
"""

import json
import pickle
from math import radians, cos, sin, asin, sqrt

import numpy as np
import scipy.sparse as sp
import torch


_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def save_list_with_pkl(filename, list_obj):
    """将列表对象保存为 pickle 文件。"""
    with open(filename, "wb") as f:
        pickle.dump(list_obj, f)


def load_list_with_pkl(filename):
    """从 pickle 文件读取列表对象。"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_dict_to_pkl(filename, dict_obj):
    """将字典对象保存为 pickle 文件。"""
    with open(filename, "wb") as f:
        pickle.dump(dict_obj, f)


def load_dict_from_pkl(filename):
    """从 pickle 文件读取字典对象。"""
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_json(filename, obj):
    """使用标准库 json 保存配置，避免引入 yaml 依赖。"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def geohash_encode(latitude, longitude, precision=6):
    """
    使用纯 Python 实现 geohash 编码。

    这样做的好处是：
    1. 不依赖额外三方库；
    2. 可以在训练时灵活切换 region 粒度，而无需强依赖固定预处理产物。
    """
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    bits = [16, 8, 4, 2, 1]
    geohash_chars = []
    bit = 0
    ch = 0
    even = True

    while len(geohash_chars) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval[0] = mid
            else:
                lon_interval[1] = mid
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval[0] = mid
            else:
                lat_interval[1] = mid
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash_chars.append(_GEOHASH_BASE32[ch])
            bit = 0
            ch = 0

    return "".join(geohash_chars)


def build_poi_region_from_coos(pois_coos_dict, precision=6):
    """
    根据 POI 坐标动态构造 poi->region 映射。

    返回：
    - poi_region_dict: {poi_idx: region_idx}
    - num_regions: 区域总数
    - geohash_to_idx: geohash 字符串到区域索引的映射
    """
    geohash_values = []
    poi_geohash = {}
    for poi_idx, coos in pois_coos_dict.items():
        lat, lon = coos
        gh = geohash_encode(lat, lon, precision=precision)
        poi_geohash[poi_idx] = gh
        geohash_values.append(gh)

    unique_geohash = sorted(set(geohash_values))
    geohash_to_idx = {gh: idx for idx, gh in enumerate(unique_geohash)}
    poi_region_dict = {poi_idx: geohash_to_idx[gh] for poi_idx, gh in poi_geohash.items()}
    return poi_region_dict, len(unique_geohash), geohash_to_idx


def haversine_distance(lon1, lat1, lon2, lat2):
    """计算两点球面距离，单位为公里。"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r


def get_user_complete_traj(sessions_dict):
    """将用户的多个 session 拼接成完整轨迹。"""
    users_trajs_dict = {}
    users_trajs_lens_dict = {}
    for user_id, sessions in sessions_dict.items():
        traj = []
        for session in sessions:
            traj.extend(session)
        users_trajs_dict[user_id] = traj
        users_trajs_lens_dict[user_id] = len(traj)
    return users_trajs_dict, users_trajs_lens_dict


def get_user_reverse_traj(users_trajs_dict):
    """生成每个用户完整轨迹的逆序版本。"""
    return {user_id: traj[::-1] for user_id, traj in users_trajs_dict.items()}


def get_all_users_seqs(users_trajs_dict):
    """将所有用户完整轨迹转为 tensor 列表，便于后续 padding。"""
    return [torch.tensor(traj) for traj in users_trajs_dict.values()]


def normalized_adj(adj, is_symmetric=True):
    """对 scipy 稀疏邻接矩阵做归一化。"""
    rowsum = np.array(adj.sum(1))
    if is_symmetric:
        d_inv = np.power(rowsum + 1e-8, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        return d_mat_inv * adj * d_mat_inv
    d_inv = np.power(rowsum + 1e-8, -1.0).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    return d_mat_inv * adj


def transform_csr_matrix_to_tensor(csr_matrix):
    """将 scipy csr_matrix 转为 torch 稀疏张量。"""
    coo = csr_matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse_coo_tensor(i, v, torch.Size(coo.shape)).coalesce()


def get_hyper_deg(incidence_matrix):
    """
    计算超图节点度的倒数对角矩阵。

    输入 H 的形状为 [num_nodes, num_edges]，输出 D_v^{-1}。
    """
    rowsum = np.array(incidence_matrix.sum(1)).flatten()
    d_inv = np.zeros_like(rowsum, dtype=np.float64)
    np.divide(1.0, rowsum, out=d_inv, where=rowsum != 0)
    return sp.diags(d_inv)


def csr_matrix_drop_edge(csr_adj_matrix, keep_rate):
    """对 csr 稀疏矩阵随机删边，用于后续可能的结构增强。"""
    if keep_rate == 1.0:
        return csr_adj_matrix
    coo = csr_adj_matrix.tocoo()
    row = coo.row
    col = coo.col
    edge_num = row.shape[0]
    mask = np.floor(np.random.rand(edge_num) + keep_rate).astype(np.bool_)
    new_row = row[mask]
    new_col = col[mask]
    new_values = np.ones(new_row.shape[0], dtype=float)
    return sp.csr_matrix((new_values, (new_row, new_col)), shape=coo.shape)


def build_binary_incidence(num_rows, num_cols, pairs):
    """
    根据 (row, col) 二元组构造二值稀疏关联矩阵。

    这里用于统一构造：
    - POI-User
    - POI-Region
    - POI-Category
    等关系。
    """
    if not pairs:
        return sp.csr_matrix((num_rows, num_cols), dtype=float)
    rows = np.array([p[0] for p in pairs], dtype=np.int64)
    cols = np.array([p[1] for p in pairs], dtype=np.int64)
    vals = np.ones(len(pairs), dtype=float)
    return sp.csr_matrix((vals, (rows, cols)), shape=(num_rows, num_cols))


def gen_sparse_H_user(sessions_dict, num_pois, num_users):
    """构建 POI-User 关联矩阵 H_pu，表示用户长期访问历史。"""
    pairs = []
    for user_id, sessions in sessions_dict.items():
        visited = set()
        for session in sessions:
            for poi in session:
                visited.add(poi)
        for poi in visited:
            pairs.append((poi, user_id))
    return build_binary_incidence(num_pois, num_users, pairs)


def gen_sparse_H_poi_region(poi_region_dict, num_pois, num_regions):
    """构建 POI-Region 关联矩阵。"""
    pairs = [(poi_idx, region_idx) for poi_idx, region_idx in poi_region_dict.items()]
    return build_binary_incidence(num_pois, num_regions, pairs)


def gen_sparse_H_poi_category(poi_category_dict, num_pois, num_categories):
    """构建 POI-Category 关联矩阵。"""
    pairs = [(poi_idx, cat_idx) for poi_idx, cat_idx in poi_category_dict.items()]
    return build_binary_incidence(num_pois, num_categories, pairs)


def gen_sparse_directed_H_poi(users_trajs_dict, num_pois):
    """
    构建有向 POI 转移矩阵。
    行表示源 POI，列表示目标 POI。
    为了延续 DCHL 风格，这里仍采用“全后续点均可视作目标”的全局转移建模方式。
    """
    H = np.zeros((num_pois, num_pois), dtype=float)
    for _, traj in users_trajs_dict.items():
        for src_idx in range(len(traj) - 1):
            for tar_idx in range(src_idx + 1, len(traj)):
                src_poi = traj[src_idx]
                tar_poi = traj[tar_idx]
                H[src_poi, tar_poi] = 1.0
    return sp.csr_matrix(H)


def gen_sparse_directed_H_poi_from_sessions(user_sessions_dict, num_pois):
    """
    基于 session 内部构建有向 POI 转移矩阵。

    只在单个 session 内建立“当前位置 -> 后续位置”的边，避免跨 session 信息泄漏。
    """
    H = np.zeros((num_pois, num_pois), dtype=float)
    for _, sessions in user_sessions_dict.items():
        for session in sessions:
            for src_idx in range(len(session) - 1):
                for tar_idx in range(src_idx + 1, len(session)):
                    src_poi = session[src_idx]
                    tar_poi = session[tar_idx]
                    H[src_poi, tar_poi] = 1.0
    return sp.csr_matrix(H)


def load_meta(meta_path):
    """读取预处理阶段输出的 meta 信息。"""
    return load_dict_from_pkl(meta_path)
