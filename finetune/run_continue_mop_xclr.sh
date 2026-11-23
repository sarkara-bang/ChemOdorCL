CUDA_VISIBLE_DEVICES=1 python continual_fcl_xclr_tarreducedim.py \
        --device cuda \
        --batch_size 32 \
        --n_head 12 \
        --n_layer 12 \
        --n_embd 768 \
        --d_dropout 0.1 \
        --dropout 0.1 \
        --lr_start 1e-5 \
        --num_workers 8\
        --max_epochs 140\
        --num_feats 32 \
        --checkpoint_every 100 \
        --data_root ../data/mop_cleanxclr\
        --seed_path '../data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt' \
        --dataset_name mop_cleanxclr\
        --measure_name odor \
        --dims 768 768 768 1 \
        --checkpoints_folder 'fingerprint_injected/mop_cleanxclr'\ 

#--measure_name alpha \ for avoiding errors from original code. It will be ignored in actual training.
#batch_size==128意味着一个训练step会使用128个分子进行训练 1个epoch会进行多个训练step完整的使用整个数据集