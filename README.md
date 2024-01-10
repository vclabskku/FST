# FST

## Setting
```
conda create -n fst python==3.8
conda activate fst

cd FST

# cuda 11.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Data Processing
```
ln -s DATASET_DIR ./dataset/z_bare # DATASET_DIR=/mnt/hdd0/leemiso/FST/231213/z_bare
zsh scripts/data_process.sh
```
The output file will be saved in ./dataset/ as `meta.json`.

## Training
```
# class
zsh scripts/train.sh MODEL_NAME EXP_NAME GPU_NUM

# binary
zsh scripts/train_binary.sh MODEL_NAME EXP_NAME GPU_NUM
```
The log file will be saved `logs` as `EXP_NAME.log`.
Checkpoints will be saved in `./ckpts/EXP_NAME` as `EXP_NAME_best.pth`.