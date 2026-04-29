import argparse
import datetime as dt
import os
from collections import defaultdict
from utils import save_dict_to_pkl, save_list_with_pkl

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"

# 固定权重转移图中的两个衰减系数。
# - step_gap 会先按 session 内理论最大步距归一化到 (0, 1]；
# - time_gap 会按 delta_hour / 24 归一化到 [0, 1] 左右。
TRANS_STEP_DECAY = 3
TRANS_TIME_DECAY = 0


def geohash_encode(latitude, longitude, precision=6):
    """使用纯 Python 实现 geohash 编码，避免额外依赖。"""
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


def parse_time(timestr):
    """解析 TSMC2014 原始时间字段。"""
    return dt.datetime.strptime(timestr, "%a %b %d %H:%M:%S %z %Y")


def load_raw_events(raw_path):
    """读取原始事件，并按用户聚合。"""
    user_events = defaultdict(list)
    poi_users = defaultdict(set)
    poi_coos_raw = {}
    poi_cat_raw = {}

    with open(raw_path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            user_id = parts[0]
            poi_id = parts[1]
            cat_id = parts[2]
            lat = float(parts[4])
            lon = float(parts[5])
            ts = parse_time(parts[7])

            user_events[user_id].append((ts, poi_id, lat, lon, cat_id))
            poi_users[poi_id].add(user_id)
            poi_coos_raw[poi_id] = (lat, lon)
            poi_cat_raw[poi_id] = cat_id

    return user_events, poi_users, poi_coos_raw, poi_cat_raw


def build_sessions(user_events, keep_poi_set, session_gap_hours=24, min_session_len=3):
    """按 24 小时时间窗切分 session，并过滤过短 session。"""
    split_delta = dt.timedelta(hours=session_gap_hours)
    user_sessions = {}
    for user_id, events in user_events.items():
        events_sorted = sorted(events, key=lambda x: x[0])
        filtered = [e for e in events_sorted if e[1] in keep_poi_set]
        if not filtered:
            continue

        sessions = []
        current = [filtered[0]]
        for i in range(1, len(filtered)):
            if filtered[i][0] - filtered[i - 1][0] > split_delta:
                sessions.append(current)
                current = [filtered[i]]
            else:
                current.append(filtered[i])
        sessions.append(current)

        valid_sessions = [s for s in sessions if len(s) >= min_session_len]
        if valid_sessions:
            user_sessions[user_id] = valid_sessions
    return user_sessions


def split_users_sessions(user_sessions_raw, train_ratio=0.8, min_user_sessions=3):
    """按用户时间顺序切 8/2 切分训练集测试集。"""
    valid_users = []
    train_user_sessions_raw = {}
    test_user_sessions_raw = {}
    total_train_sessions = 0
    total_test_sessions = 0

    for uid, sessions in user_sessions_raw.items():
        if len(sessions) < min_user_sessions:
            continue

        train_cut = max(1, int(len(sessions) * train_ratio))
        if train_cut >= len(sessions):
            train_cut = len(sessions) - 1
        train_raw = sessions[:train_cut]
        test_raw = sessions[train_cut:]

        if len(train_raw) < 1 or len(test_raw) < 1:
            continue
        valid_users.append(uid)
        train_user_sessions_raw[uid] = train_raw
        test_user_sessions_raw[uid] = test_raw
        total_train_sessions += len(train_raw)
        total_test_sessions += len(test_raw)

    return valid_users, train_user_sessions_raw, test_user_sessions_raw, total_train_sessions, total_test_sessions


def build_entity_mappings(valid_users, user_sessions_raw, poi_coos_raw, poi_cat_raw, geohash_precision=6):
    """统一 remap 用户、POI、类别和区域索引。"""
    poi_set = set()
    for uid in valid_users:
        for session in user_sessions_raw[uid]:
            for _, poi_id, _, _, _ in session:
                poi_set.add(poi_id)
    poi_list = sorted(list(poi_set))

    user2idx = {u: i for i, u in enumerate(valid_users)}
    poi2idx = {p: i for i, p in enumerate(poi_list)}

    cat_values = sorted({poi_cat_raw[p] for p in poi_list})
    cat2idx = {c: i for i, c in enumerate(cat_values)}

    geohash_values = sorted({geohash_encode(poi_coos_raw[p][0], poi_coos_raw[p][1], precision=geohash_precision) for p in poi_list})
    geohash2idx = {g: i for i, g in enumerate(geohash_values)}

    poi_coos_idx, poi_cat_idx, poi_region_idx = {}, {}, {}
    for raw_poi, poi_idx in poi2idx.items():
        lat, lon = poi_coos_raw[raw_poi]
        poi_coos_idx[poi_idx] = [lat, lon]
        poi_cat_idx[poi_idx] = cat2idx[poi_cat_raw[raw_poi]]
        poi_region_idx[poi_idx] = geohash2idx[geohash_encode(lat, lon, precision=geohash_precision)]

    return user2idx, poi2idx, poi_coos_idx, poi_cat_idx, poi_region_idx, len(cat_values), len(geohash_values)


def remap_sessions_for_users(user_ids, user_sessions_raw, user2idx, poi2idx):
    """将指定用户集合的 session 重映射为整数索引。"""
    remapped = {}
    for raw_user in user_ids:
        user_idx = user2idx[raw_user]
        remapped[user_idx] = [[poi2idx[e[1]] for e in session] for session in user_sessions_raw[raw_user]]
    return remapped


def remap_sessions_with_time_for_users(user_ids, user_sessions_raw, user2idx, poi2idx):
    """
    将训练/测试 session 重映射为“带时间信息”的整数序列。
    输出结构：
    {
        user_idx: [
            [(poi_idx, timestamp_seconds), ...],
            ...
        ]
    }

    其中 timestamp_seconds 是 Unix 时间戳（秒）。之所以提前转成数值，是为了：
    - 后续构图阶段直接做减法即可得到时间差；
    - 避免在 dataset 阶段重复处理 datetime 对象。
    """
    remapped = {}
    for raw_user in user_ids:
        user_idx = user2idx[raw_user]
        timed_sessions = []
        for session in user_sessions_raw[raw_user]:
            timed_session = []
            for timestamp, poi_id, _, _, _ in session:
                poi_idx = poi2idx[poi_id]
                timed_session.append((poi_idx, timestamp.timestamp()))
            timed_sessions.append(timed_session)
        remapped[user_idx] = timed_sessions
    return remapped


def build_prefix_samples(remapped_sessions, poi_cat_idx, poi_region_idx):
    """训练阶段：滑窗，将每个 session 展开为 prefix -> next POI。"""
    samples = []
    for user_idx, sessions in remapped_sessions.items():
        for session_idx, session in enumerate(sessions):
            for prefix_end in range(1, len(session)):
                label_poi = session[prefix_end]
                samples.append({
                    "user_idx": user_idx,
                    "session_idx": session_idx,
                    "prefix_pois": session[:prefix_end],
                    "label_poi": label_poi,
                    "label_category": poi_cat_idx[label_poi],
                    "label_region": poi_region_idx[label_poi],
                })
    return samples


def build_train_last_step_samples(remapped_sessions, poi_cat_idx, poi_region_idx):
    """训练阶段：留一，每个 session 仅保留最后一步样本。"""
    samples = []
    for user_idx, sessions in remapped_sessions.items():
        for session_idx, session in enumerate(sessions):
            if len(session) < 2:
                continue
            label_poi = session[-1]
            samples.append({
                "user_idx": user_idx,
                "session_idx": session_idx,
                "prefix_pois": session[:-1],
                "label_poi": label_poi,
                "label_category": poi_cat_idx[label_poi],
                "label_region": poi_region_idx[label_poi],
            })
    return samples


def build_last_step_samples(remapped_sessions, poi_cat_idx, poi_region_idx):
    """测试阶段：每个 session 仅保留最后一步样本。"""
    samples = []
    for user_idx, sessions in remapped_sessions.items():
        for session_idx, session in enumerate(sessions):
            if len(session) < 2:
                continue
            label_poi = session[-1]
            samples.append({
                "user_idx": user_idx,
                "session_idx": session_idx,
                "prefix_pois": session[:-1],
                "label_poi": label_poi,
                "label_category": poi_cat_idx[label_poi],
                "label_region": poi_region_idx[label_poi],
            })
    return samples


def collect_train_pois(remapped_sessions):
    """仅收集训练 session 中出现过的 POI，用于构造训练集局部语义边。"""
    train_poi_set = set()
    for _, sessions in remapped_sessions.items():
        for session in sessions:
            for poi in session:
                train_poi_set.add(poi)
    return train_poi_set


def filter_poi_semantic_mapping_by_train_pois(mapping_dict, train_poi_set):
    """仅保留训练集 POI 的语义边，语义索引值保持全局编号不变。"""
    return {poi: semantic_idx for poi, semantic_idx in mapping_dict.items() if poi in train_poi_set}


def summarize_user_events(user_events, poi_users, poi_cat_raw):
    """统计原始事件规模，便于填写数据集规模表。"""
    raw_users = len(user_events)
    raw_checkins = sum(len(events) for events in user_events.values())
    raw_pois = len(poi_users)
    raw_categories = len(set(poi_cat_raw.values()))
    return raw_users, raw_checkins, raw_pois, raw_categories


def summarize_sessions(user_sessions_dict):
    """统计会话相关信息：用户数、会话总数、签到总数、平均会话长度。"""
    user_count = len(user_sessions_dict)
    session_count = sum(len(sessions) for sessions in user_sessions_dict.values())
    checkin_count = sum(len(session) for sessions in user_sessions_dict.values() for session in sessions)
    avg_sessions_per_user = (session_count / user_count) if user_count > 0 else 0.0
    avg_session_len = (checkin_count / session_count) if session_count > 0 else 0.0
    return user_count, session_count, checkin_count, avg_sessions_per_user, avg_session_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="datasets/dataset_TSMC2014_TKY.txt")
    parser.add_argument("--output_dir", type=str, default="datasets/TKY")
    parser.add_argument("--min_poi_users", type=int, default=5)
    parser.add_argument("--min_session_len", type=int, default=3)
    parser.add_argument("--min_user_sessions", type=int, default=3)
    parser.add_argument("--session_gap_hours", type=int, default=24)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--geohash_precision", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] 读取原始数据...")
    user_events, poi_users, poi_coos_raw, poi_cat_raw = load_raw_events(args.raw_path)

    raw_users, raw_checkins, raw_pois, _ = summarize_user_events(user_events, poi_users, poi_cat_raw)
    print("[FIG1-A][过滤前原始] 用户数=%d，POI数=%d，签到记录数=%d" % (raw_users, raw_pois, raw_checkins))

    print("[2/5] 过滤低频 POI 并切分 session...")
    keep_poi_set = {poi for poi, users in poi_users.items() if len(users) >= args.min_poi_users}
    user_sessions_raw = build_sessions(
        user_events,
        keep_poi_set,
        session_gap_hours=args.session_gap_hours,
        min_session_len=args.min_session_len,
    )

    print("[3/5] 用户内按时间划分训练集测试集...")
    valid_users, train_user_sessions_raw, test_user_sessions_raw, total_train_sessions, total_test_sessions = split_users_sessions(
        user_sessions_raw,
        train_ratio=args.train_ratio,
        min_user_sessions=args.min_user_sessions,
    )

    final_user_sessions_raw = {uid: user_sessions_raw[uid] for uid in valid_users}
    final_users, final_sessions, final_checkins, _, _ = summarize_sessions(final_user_sessions_raw)
    final_pois = len({event[1] for sessions in final_user_sessions_raw.values() for session in sessions for event in session})

    user2idx, poi2idx, poi_coos_idx, poi_cat_idx, poi_region_idx, num_categories, num_regions = build_entity_mappings(
        valid_users,
        user_sessions_raw,
        poi_coos_raw,
        poi_cat_raw,
        geohash_precision=args.geohash_precision,
    )

    print("[节点规模] 类别节点数量=%d，区域节点数量=%d" % (num_categories, num_regions))

    train_user_sessions = remap_sessions_for_users(valid_users, train_user_sessions_raw, user2idx, poi2idx)
    test_user_sessions = remap_sessions_for_users(valid_users, test_user_sessions_raw, user2idx, poi2idx)

    # 方案A额外保存“带时间的训练 session”。
    # 这份文件不直接用于样本训练，而是专门供 dataset.py 在构造固定加权转移图时读取：
    # - 步距来自 session 内的位置差；
    # - 时间差来自这里保存的 timestamp。
    train_user_sessions_with_time = remap_sessions_with_time_for_users(
        valid_users,
        train_user_sessions_raw,
        user2idx,
        poi2idx,
    )

    # ===== 与 preprocess.py 的不同部分 1：只保留训练集 POI 的类别边/区域边 =====
    # 说明：这里仍然保留全局 num_categories 和 num_regions，只裁剪 POI->语义节点的边。
    train_poi_set = collect_train_pois(train_user_sessions)
    local_poi_cat_idx = filter_poi_semantic_mapping_by_train_pois(poi_cat_idx, train_poi_set)
    local_poi_region_idx = filter_poi_semantic_mapping_by_train_pois(poi_region_idx, train_poi_set)

    print("[4/5] 生成训练与测试样本...")
    # 训练集滑窗方式
    # train_samples = build_prefix_samples(train_user_sessions, poi_cat_idx, poi_region_idx)

    # 训练集留一方式
    train_samples = build_train_last_step_samples(train_user_sessions, poi_cat_idx, poi_region_idx)
    test_samples = build_last_step_samples(test_user_sessions, poi_cat_idx, poi_region_idx)

    print(
        "[过滤后] 用户数=%d，POI数=%d，签到记录数=%d，Session数量=%d，训练样本数=%d，测试样本数=%d"
        % (final_users, final_pois, final_checkins, final_sessions, len(train_samples), len(test_samples))
    )

    meta = {
        "num_users": len(valid_users),
        "num_pois": len(poi2idx),
        "padding_idx": len(poi2idx),
        "num_categories": num_categories,
        "num_regions": num_regions,
        "train_ratio": args.train_ratio,
        "session_gap_hours": args.session_gap_hours,
        "min_session_len": args.min_session_len,
        "min_user_sessions": args.min_user_sessions,
        "geohash_precision": args.geohash_precision,
        "num_train_sessions": total_train_sessions,
        "num_test_sessions": total_test_sessions,
        "num_train_samples": len(train_samples),
        "num_test_samples": len(test_samples),
        "train_protocol": "sliding_window",
        "test_protocol": "last_step_only",
        # ===== 与 preprocess.py 的不同部分 2：明确记录语义图构造口径 =====
        "semantic_graph_protocol": "train_poi_only_edges",
        # 方案A固定权重转移图的两个超参数直接记录到 meta，方便 dataset.py 读取并复现。
        "trans_step_decay": TRANS_STEP_DECAY,
        "trans_time_decay": TRANS_TIME_DECAY,
    }

    print("[5/5] 保存文件...")
    save_list_with_pkl(os.path.join(args.output_dir, "train_samples.pkl"), train_samples)
    save_list_with_pkl(os.path.join(args.output_dir, "test_samples.pkl"), test_samples)
    save_dict_to_pkl(os.path.join(args.output_dir, "train_user_sessions.pkl"), train_user_sessions)
    save_dict_to_pkl(os.path.join(args.output_dir, "test_user_sessions.pkl"), test_user_sessions)
    # 这份文件只给方案A的固定加权转移图使用，不影响协同图和样本构造流程。
    save_dict_to_pkl(os.path.join(args.output_dir, "train_user_sessions_with_time.pkl"), train_user_sessions_with_time)
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_coos.pkl"), poi_coos_idx)

    # ===== 与 preprocess.py 的不同部分 3：输出同名文件，但内容改为训练集局部语义边 =====
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_category.pkl"), local_poi_cat_idx)
    save_dict_to_pkl(os.path.join(args.output_dir, "poi_region.pkl"), local_poi_region_idx)

    save_dict_to_pkl(os.path.join(args.output_dir, "meta.pkl"), meta)
    print("预处理完成：", meta)


if __name__ == "__main__":
    main()
