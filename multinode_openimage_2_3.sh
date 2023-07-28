#!/bin/bash
#SBATCH --job-name=Flute_EX_opnimg_mn_hetero_1+1
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=mauao,ngongotaha
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH -c 8


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

export EXPERIMENT_START_TIME=$( date '+%F_%H:%M:%S' )

export MAU_SIZE=2
export NGON_SIZE=3
export WORLD_SIZE=$(($MAU_SIZE + $NGON_SIZE))
echo "WORLD_SIZE="$WORLD_SIZE




master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
export WORKER_TASKS=1

# export NCCL_DEBUG=INFO

# echo "SLURMD_NODENAME="$SLURMD_NODENAME

### init virtual environment if needed
source /nfs-share/aai30/miniconda3/bin/activate
conda activate Flute

### the command to run
srun launch_flute_multi.sh