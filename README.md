# My_DCHL

基于多分支超图建模的下一 POI 推荐项目。当前模型融合了 4 个结构分支：

- `User-POI` 长期协同分支
- `POI-POI` 有向转移分支
- `POI-Region` 异构语义分支
- `POI-Category` 异构语义分支

模型主入口是 `train.py`，核心模型定义在 `model.py`，数据预处理在 `preprocess.py` / `local_preprocess.py`。

## 1. 项目整体框架

### 1.1 数据流

1. 原始签到数据读取：`preprocess.py`
2. 按时间切 session，并按用户时间顺序切分 train/test
3. 生成训练/测试样本 `train_samples.pkl`、`test_samples.pkl`
4. 保存训练 session，用于后续构图
5. `dataset.py` 从训练 session 构造 4 类图结构
6. `train.py` 加载样本和图结构训练 `HDCHLB`

### 1.2 四个分支

- 协同分支：`User-POI` 二部超图，建模长期偏好
- 转移分支：`POI-POI` 有向图，建模 session 内转移关系
- 区域分支：`POI-Region` 二部超图，建模空间语义
- 类别分支：`POI-Category` 二部超图，建模类别语义

### 1.3 当前前向逻辑

- 4 个分支各自传播 POI 表示
- 使用当前 prefix 序列做 masked mean pooling，得到样本级动态用户表示
- 用户侧对 4 个分支做门控融合
- POI 侧对 4 个分支表示直接求和
- 用用户表示和 POI 表示点积，预测下一个 POI

## 2. 目录说明

- `train.py`：训练入口
- `model.py`：模型定义
- `dataset.py`：样本读取与图构造
- `preprocess.py`：标准预处理脚本
- `local_preprocess.py`：局部 CR 构图版本的预处理脚本
- `utils.py`：稀疏图、序列和存取工具
- `metrics.py`：Recall / NDCG 评估
- `count_numpoi.py`：统计训练/测试 POI 重叠情况

## 3. 数据协议切换：滑窗 vs 留一

当前仓库里，样本构造函数已经都写好了：

- `build_prefix_samples(...)`：滑窗，生成多个 `prefix -> next POI` 样本
- `build_train_last_step_samples(...)`：训练留一，每个 session 只保留最后一步
- `build_last_step_samples(...)`：测试留一，每个 session 只保留最后一步

### 3.1 当前默认状态

当前 `preprocess.py` 和 `local_preprocess.py` 的默认配置是：

- 训练集：滑窗
- 测试集：留一

```

### 3.2 推荐切换方式：训练滑窗 + 测试留一

如果要切换到更常见的 next-POI 协议，建议使用：

- 训练集：滑窗
- 测试集：留一

在 `preprocess.py` 中，把：

```python
# train_samples = build_prefix_samples(train_user_sessions, poi_cat_idx, poi_region_idx)
train_samples = build_train_last_step_samples(train_user_sessions, poi_cat_idx, poi_region_idx)
test_samples = build_last_step_samples(test_user_sessions, poi_cat_idx, poi_region_idx)
```

改为：

```python
train_samples = build_prefix_samples(train_user_sessions, poi_cat_idx, poi_region_idx)
test_samples = build_last_step_samples(test_user_sessions, poi_cat_idx, poi_region_idx)
```

`local_preprocess.py` 也做同样修改。

### 3.3 如果想切换成全留一

保持当前写法即可：

```python
train_samples = build_train_last_step_samples(train_user_sessions, poi_cat_idx, poi_region_idx)
test_samples = build_last_step_samples(test_user_sessions, poi_cat_idx, poi_region_idx)
```

### 3.4 切换后别忘了同步 `meta`

`meta` 里会记录协议：

```python
"train_protocol": "last_step_only",
"test_protocol": "last_step_only",
```

切换样本协议后，建议把它同步改掉，避免后续实验混淆。比如训练滑窗、测试留一时可以写成：

```python
"train_protocol": "sliding_window",
"test_protocol": "last_step_only",
```

## 4. CR 构图方式切换

这里的 CR 指的是 `Category` 和 `Region` 两个语义分支的构图方式。

当前仓库里有两种方式：

### 4.1 全局 CR 构图

```bash
python preprocess.py --raw_path datasets/dataset_TSMC2014_TKY.txt --output_dir datasets/TKY
```

### 4.2 局部 CR 构图

```bash
python local_preprocess.py --raw_path datasets/dataset_TSMC2014_TKY.txt --output_dir datasets/TKY_local
```

## 5. 训练与构图参数

`train.py` 中几个常用参数：

- `--keep_rate`：协同图 `User-POI` 的随机保边率
- `--keep_rate_poi`：转移图 `POI-POI` 的随机保边率
- `--num_col_layers`：协同分支层数
- `--num_reg_layers`：区域分支层数
- `--num_cat_layers`：类别分支层数
- `--num_trans_layers`：转移分支层数
- `--mask_rate_cat` / `--mask_rate_reg`：语义分支 mask 比例
- `--lambda_cat` / `--lambda_reg`：语义重建损失权重

示例：

```bash
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --deviceID 0
```
