'''
统计测试集中的 POI 与训练集的重合情况，以及测试样本中 label_poi 的未见比例
'''
import os
import pickle


BASE_DIR = r"E:\code\lwCode\My_DCHL\datasets\TKY"
TRAIN_SESSIONS_PATH = os.path.join(BASE_DIR, "train_user_sessions.pkl")
TEST_SESSIONS_PATH = os.path.join(BASE_DIR, "test_user_sessions.pkl")
TEST_SAMPLES_PATH = os.path.join(BASE_DIR, "test_samples.pkl")


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collect_pois_from_sessions(user_sessions_dict):
    poi_set = set()
    for sessions in user_sessions_dict.values():
        for session in sessions:
            poi_set.update(session)
    return poi_set


def main():
    train_user_sessions = load_pkl(TRAIN_SESSIONS_PATH)
    test_user_sessions = load_pkl(TEST_SESSIONS_PATH)
    test_samples = load_pkl(TEST_SAMPLES_PATH)

    train_pois = collect_pois_from_sessions(train_user_sessions)
    test_pois = collect_pois_from_sessions(test_user_sessions)

    overlap_pois = train_pois & test_pois
    unseen_test_pois = test_pois - train_pois

    test_labels = [sample["label_poi"] for sample in test_samples]
    unseen_test_labels = [poi for poi in test_labels if poi not in train_pois]

    print("=== POI 集合统计 ===")
    print(f"训练集唯一 POI 数: {len(train_pois)}")
    print(f"测试集唯一 POI 数: {len(test_pois)}")
    print(f"训练/测试重合 POI 数: {len(overlap_pois)}")
    print(f"测试集中未在训练集出现过的唯一 POI 数: {len(unseen_test_pois)}")
    if len(test_pois) > 0:
        print(f"测试集唯一 POI 未见比例: {len(unseen_test_pois) / len(test_pois):.4f}")

    print("\n=== 测试标签统计 ===")
    print(f"测试样本数: {len(test_labels)}")
    print(f"测试样本中 label_poi 未在训练集出现过的数量: {len(unseen_test_labels)}")
    if len(test_labels) > 0:
        print(f"测试样本 label_poi 未见比例: {len(unseen_test_labels) / len(test_labels):.4f}")

    if unseen_test_pois:
        preview = sorted(list(unseen_test_pois))[:20]
        print("\n前 20 个未见测试 POI 索引:", preview)


if __name__ == "__main__":
    main()
