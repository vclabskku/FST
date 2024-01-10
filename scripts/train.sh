MODEL=$1 # resnet18 resnet34 resnet50
EXP_NAME=$2
GPU=$3
DATASET_DIR=./dataset/z_bare
META_PATH=./dataset/meta.json

mkdir -m 777 ./logs/${EXP_NAME}

python train.py \
    --dataset_dir ${DATASET_DIR} \
    --meta_path ${META_PATH} \
    --model_name ${MODEL} \
    --gpu ${GPU} \
    --exp_name ${EXP_NAME} | tee ./logs/${EXP_NAME}/${EXP_NAME}.log