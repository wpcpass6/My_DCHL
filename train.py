# coding=utf-8
"""
H-DCHL-B 训练入口。

当前版本采用论文常见的 best test 汇报方式，先验证纯结构增益是否有效。
"""

import argparse
import datetime
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import HDCHLBDataset, collate_fn
from metrics import batch_performance
from model import HDCHLB
from utils import load_meta, save_json


torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def build_logger(save_dir):
    """创建文件与控制台双通道日志。"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "log_training.txt"),
        filemode="w+",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def evaluate(model, dataset_obj, dataloader, criterion, device, ks_list, log_interval):
    """在测试集上评估多项排名指标。"""
    model.eval()
    total_loss = 0.0
    recall_array = np.zeros((len(dataloader), len(ks_list)))
    ndcg_array = np.zeros((len(dataloader), len(ks_list)))

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            predictions, aux_loss = model(dataset_obj, batch)
            loss_rec = criterion(predictions, batch["label"].to(device))
            loss = loss_rec + aux_loss
            total_loss += float(loss.item())
            if idx % log_interval == 0 or idx == len(dataloader) - 1:
                logging.info(
                    "Test. Batch %d/%d loss_rec=%.4f aux_loss=%.4f loss=%.4f",
                    idx,
                    len(dataloader),
                    loss_rec.item(),
                    float(aux_loss),
                    float(loss),
                )

            for k in ks_list:
                recall, ndcg = batch_performance(predictions.detach().cpu(), batch["label"].detach().cpu(), k)
                col_idx = ks_list.index(k)
                recall_array[idx, col_idx] = recall
                ndcg_array[idx, col_idx] = ndcg

    metrics = {}
    for k in ks_list:
        col_idx = ks_list.index(k)
        metrics[f"Rec{k}"] = float(np.mean(recall_array[:, col_idx]))
        metrics[f"NDCG{k}"] = float(np.mean(ndcg_array[:, col_idx]))
    return total_loss / max(len(dataloader), 1), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets/TKY")
    parser.add_argument("--meta_path", type=str, default="datasets/TKY/meta.pkl")
    parser.add_argument("--poi_region_path", type=str, default=None,
                        help="可选：显式指定预处理好的 poi_region.pkl；若不传，则根据 poi_coos 动态按 geohash 精度生成")
    parser.add_argument("--region_precision", type=int, default=6,
                        help="动态构造 Region 时使用的 geohash 精度，例如 5 或 6")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--log_interval", type=int, default=50, help="每隔多少个 batch 打印一次训练/测试日志")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--deviceID", type=int, default=0)
    parser.add_argument("--keep_rate", type=float, default=1.0)
    parser.add_argument("--keep_rate_poi", type=float, default=1.0)
    parser.add_argument("--num_col_layers", type=int, default=2)
    parser.add_argument("--num_reg_layers", type=int, default=2)
    parser.add_argument("--num_cat_layers", type=int, default=1)
    parser.add_argument("--num_trans_layers", type=int, default=4)
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.1)
    parser.add_argument("--mask_rate_cat", type=float, default=0.2)
    parser.add_argument("--lambda_cat", type=float, default=0.05)
    parser.add_argument("--mask_rate_reg", type=float, default=0.0)
    parser.add_argument("--lambda_reg", type=float, default=0.0)
    parser.add_argument("--mask_alpha", type=float, default=2.0)
    parser.add_argument("--save_dir", type=str, default="logs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.deviceID}" if torch.cuda.is_available() else "cpu")
    meta = load_meta(args.meta_path)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_save_dir = os.path.join(args.save_dir, f"seed{args.seed}_{current_time}")
    os.makedirs(current_save_dir, exist_ok=True)
    build_logger(current_save_dir)
    save_json(os.path.join(current_save_dir, "args.json"), vars(args))
    result_txt_path = os.path.join(current_save_dir, "result.txt")

    def write_result_line(text):
        """将关键信息同步写入 result.txt，便于后续人工汇总。"""
        with open(result_txt_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    logging.info("1. Parse Arguments")
    logging.info(args)
    logging.info("device: %s", device)
    logging.info("meta: %s", meta)

    logging.info("2. Load Dataset")
    train_dataset = HDCHLBDataset(os.path.join(args.data_dir, "train_samples.pkl"), args.data_dir, args, device)
    test_dataset = HDCHLBDataset(os.path.join(args.data_dir, "test_samples.pkl"), args.data_dir, args, device)
    logging.info("dynamic num_regions used in dataset: %d", train_dataset.num_regions)

    logging.info("3. Construct DataLoader")
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, padding_value=meta["padding_idx"]),
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, padding_value=meta["padding_idx"]),
    )

    logging.info("4. Load Model")
    model = HDCHLB(
        num_users=meta["num_users"],
        num_pois=meta["num_pois"],
        num_regions=train_dataset.num_regions,
        num_categories=meta["num_categories"],
        padding_idx=meta["padding_idx"],
        args=args,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss().to(device)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=args.lr_scheduler_factor)

    logging.info("5. Start Training")
    ks_list = [1, 5, 10, 20]
    best_test_rec5 = 0.0
    final_results = {"Rec1": 0.0, "Rec5": 0.0, "Rec10": 0.0, "Rec20": 0.0,
                     "NDCG1": 0.0, "NDCG5": 0.0, "NDCG10": 0.0, "NDCG20": 0.0}
    best_rec10 = -1.0
    best_rec10_epoch = -1
    best_rec10_row = None

    write_result_line("# 每轮汇总：best-by-metric 与 best-Rec10-epoch row")

    for epoch in range(args.num_epochs):
        logging.info("================= Epoch %d/%d =================", epoch, args.num_epochs)
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            predictions, aux_loss = model(train_dataset, batch)
            loss_rec = criterion(predictions, batch["label"].to(device))
            loss = loss_rec + aux_loss
            if idx % args.log_interval == 0 or idx == len(train_dataloader) - 1:
                logging.info(
                    "Train. Batch %d/%d loss_rec=%.4f aux_loss=%.4f loss=%.4f",
                    idx,
                    len(train_dataloader),
                    loss_rec.item(),
                    float(aux_loss),
                    float(loss),
                )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        logging.info("Training finishes at this epoch. It takes %.4f min", (time.time() - t0) / 60)
        logging.info("Training loss: %.4f", train_loss / max(len(train_dataloader), 1))

        test_loss, test_metrics = evaluate(model, test_dataset, test_dataloader, criterion, device, ks_list, args.log_interval)
        logging.info("Testing loss: %.4f", test_loss)
        logging.info("Testing results: %s", {k: f"{v:.4f}" for k, v in test_metrics.items()})

        lr_scheduler.step(test_loss)
        if test_metrics["Rec5"] > best_test_rec5:
            best_test_rec5 = test_metrics["Rec5"]
            logging.info("Update test results and save model at epoch%d", epoch)
            torch.save(model.state_dict(), os.path.join(current_save_dir, "H_DCHL_B.pt"))

        for key in final_results:
            final_results[key] = max(final_results[key], test_metrics[key])

        # 第二套汇报方式：按 Rec10 最优所在 epoch 汇报整行结果
        if test_metrics["Rec10"] > best_rec10:
            best_rec10 = test_metrics["Rec10"]
            best_rec10_epoch = epoch
            best_rec10_row = test_metrics.copy()

        metric_text = {k: f"{v:.4f}" for k, v in final_results.items()}
        row_text = {k: f"{v:.4f}" for k, v in best_rec10_row.items()}
        write_result_line(
            f"epoch={epoch}\tbest-by-metric={metric_text}\tbest-Rec10-epoch={best_rec10_epoch}\tbest-Rec10-row={row_text}"
        )
        logging.info("==================================")

    logging.info("6. Final Results")
    logging.info("best-by-metric: %s", {k: f"{v:.4f}" for k, v in final_results.items()})
    logging.info(
        "best-Rec10-epoch row (epoch=%d): %s",
        best_rec10_epoch,
        {k: f"{v:.4f}" for k, v in best_rec10_row.items()},
    )
    write_result_line("# Final Summary")
    write_result_line(f"best-by-metric={ {k: f'{v:.4f}' for k, v in final_results.items()} }")
    write_result_line(f"best-Rec10-epoch={best_rec10_epoch}")
    write_result_line(f"best-Rec10-row={ {k: f'{v:.4f}' for k, v in best_rec10_row.items()} }")


if __name__ == "__main__":
    main()
