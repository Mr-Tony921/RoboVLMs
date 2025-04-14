set -x

!/usr/bin/env bash
set -x
T=`date +%Y%m%d_%H%M%S`

export NCCL_DEBUG=INFO


export PATH=/mnt/afs1/luojiapeng/miniconda3/bin:$PATH
export TORCH_EXTENSIONS_DIR=/mnt/afs1/luojiapeng/.cache/torch_extensions

cd
cp /mnt/afs1/tianhao2/aoss.conf ./

ROOT=/mnt/afs/share_data/tongronglei/work/RoboVLMs
cd $ROOT

source activate
conda activate /mnt/afs/tongronglei/.conda/envs/robovlms

export PYTHONPATH=$ROOT:$PYTHONPATH
export HOME=/mnt/afs/tongronglei
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-10086}

GPUS=4
GPUS_PER_NODE=4
RANK=${RANK:-0} # srun env node rank
WORLD_SIZE=${WORLD_SIZE:-1} # srun env node num
echo "nnodes=${WORLD_SIZE}, node_rank=${RANK}"

# 判断是否为主节点
if [ "$RANK" -ne 0 ]; then
    HYDRA_OUTPUT_SUBDIR=hydra.output_subdir=null
    echo "Hydra output is disabled for node_rank=${RANK}"
else
    HYDRA_OUTPUT_SUBDIR=""
    echo "Hydra output is enabled for node_rank=0"
fi


echo $(which python)

bash scripts/run.sh configs/kosmos_ph_post_train_lab.json