export ALFWORLD_DATA=~/alfworld-storage
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file config_zero2.yaml --main_process_port 29330 ../main_alf.py \
    --env-name "AlfredThorEnv" \
    --alf_config ../alf-config.yaml \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 12000 \
    --num-steps 1024 \
    --grad-accum-steps 256 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.2 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path /your_sft_checkpoint_for_alfworld \
    --use-lora \
    --train-vision all \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4