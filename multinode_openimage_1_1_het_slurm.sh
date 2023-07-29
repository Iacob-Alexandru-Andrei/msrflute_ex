#!/bin/bash
#SBATCH --job-name=Flute_EX_opnimg_mn_hetero_1+1
### e.g. request 4 nodes with 1 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodelist=mauao,ngongotaha
#SBATCH --gres=gpu:a40:2 -c 8 -w mauao --ntasks=2
#SBATCH packjob
#SBATCH --gres=gpu:rtx2080:1 -c 8 -w ngongotaha --ntasks=1


export MASTER_PORT=6379
# Make this the sum of the number of tasks launched in total across nodes
export WORLD_SIZE=3

echo "WORLD_SIZE="$WORLD_SIZE

export MASTER_ADDR="mauao"
echo "MASTER_ADDR="$MASTER_ADDR

# export NCCL_DEBUG=INFO

# echo "SLURMD_NODENAME="$SLURMD_NODENAME

### init virtual environment if needed
srun launch_flute_multi.sh 
