CONFIG_FILE=$(realpath ../conf/nemotron_nano_v2_vl_override_example.yaml)
HF_MODEL_PATH=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
MEGATRON_MODEL_PATH=/lustre/fs1/portfolios/nvr/projects/nvr_lacr_llm/users/jaehunj/models/megatron/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16
DATASET_MAKER_NAME=make_raven_dataset
WANDB_PROJECT=lpt3-nemo-v2-vl
WANDB_EXP_NAME=debug


SAVE_DIR=$(realpath ./checkpoints)
SCRIPT_DIR=$(realpath ../)


# run script
cd "$SCRIPT_DIR";
torchrun --nproc-per-node=8 ./finetune_nemotron_nano_v2_vl.py \
  --config-file "$CONFIG_FILE" \
  --hf-model-path $HF_MODEL_PATH \
  --pretrained-checkpoint $MEGATRON_MODEL_PATH \
  dataset.maker_name=$DATASET_MAKER_NAME \
  logger.wandb_project=$WANDB_PROJECT \
  logger.wandb_exp_name=$WANDB_EXP_NAME \
  logger.wandb_save_dir="$SAVE_DIR" \
  checkpoint.load="$SAVE_DIR"/$WANDB_EXP_NAME \
  checkpoint.save="$SAVE_DIR"/$WANDB_EXP_NAME




