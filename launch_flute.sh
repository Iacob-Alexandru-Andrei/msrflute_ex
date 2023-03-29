#!/bin/bash
source /nfs-share/aai30/miniconda3/bin/activate
conda activate Flute
echo "$@"

monitor_pid=$(python -m launch_monitor "$@")
python -m torch.distributed.run $1  e2e_trainer.py "${@:2}"
echo $monitor_pid
kill $monitor_pid 
exit
exit
exit
