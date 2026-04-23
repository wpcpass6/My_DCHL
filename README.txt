python train.py --data_dir datasets/TSMC2014 --meta_path datasets/TSMC2014/meta.pkl --num_epochs 30 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 3

固定随机种子2026， --num_trans_layers 2，3，4
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 2 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 3 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 4 --seed 2026

num_trans_layers=4 最优         num_col_layers2，3,4
(同上)python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 3 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 4 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 4 --seed 2026

num_trans_layers=4  num_col_layers2 最优      num_reg_layers 1,2,3  
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 1 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 3 --num_cat_layers 1 --num_trans_layers 4 --seed 2026

num_trans_layers=4  num_col_layers2 num_reg_layers 2 最优      num_cat_layers 1,2,3
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 2 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 3 --num_trans_layers 4 --seed 2026

num_trans_layers=4  num_col_layers2 num_reg_layers 2 num_cat_layers 1 最优 dropout 0.1/0.3/0.5
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.1 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.5 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 2026

num_trans_layers=4  num_col_layers2 num_reg_layers 2 num_cat_layers 1  dropout 0.3 最优 三seed均值
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 2026
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 2025
python train.py --data_dir datasets/TKY --meta_path datasets/TKY/meta.pkl --num_epochs 15 --batch_size 200 --emb_dim 128 --lr 0.001 --decay 0.0005 --dropout 0.3 --keep_rate 1 --keep_rate_poi 1 --num_col_layers 2 --num_reg_layers 2 --num_cat_layers 1 --num_trans_layers 4 --seed 42


cat_reg_mask:
python train.py --mask_rate_cat 0.2 --lambda_cat 0.05 --mask_rate_reg 0.2 --lambda_reg 0.05 --seed 42 --save_dir logs_cat_reg_mask
python train.py --mask_rate_cat 0.2 --lambda_cat 0.05 --mask_rate_reg 0.2 --lambda_reg 0.05 --seed 2025 --save_dir logs_cat_reg_mask
python train.py --mask_rate_cat 0.2 --lambda_cat 0.05 --mask_rate_reg 0.2 --lambda_reg 0.05 --seed 2026 --save_dir logs_cat_reg_mask


python train.py --mask_rate_cat 0.2 --lambda_cat 0.05 --mask_rate_reg 0.2 --lambda_reg 0.02 --region_precision 5 --seed 2026 --save_dir logs_catreg_p5