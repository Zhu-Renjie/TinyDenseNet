# -*- coding: utf-8 -*-
# author: lm 

python train_tiny.py \
       --train=/home/aistudio/work/data/tiny-imagenet-200/tfrecords/train_64_a0/ \
       --eval=/home/aistudio/work/data/tiny-imagenet-200/tfrecords/val_64/ \
       --height=64 \
       --width=64 \
       --num_labels=200 \
       --epoches=12 \
       --eval_epoches=1 \
       --batch_size=128 \
       --eval_batch_size=100 \
       --weight_decay=0.0002 \
       --lr=0.000001 \
       --max_lr=0.000006 \
       --step_size=1564 \
       --ckpt_iters=781 \
       --show_iters=20 \
       --use_bn=1 \
       --model_dir=models/td2_tiny \
       --log_path=logs/td2_tiny_1029.txt \
       --restore=1 \
       --reset_lr=1  

# FILE END.
