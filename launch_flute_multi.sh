#!/bin/bash
source /nfs-share/aai30/miniconda3/bin/activate
conda activate Flute


monitor_pid=$(python -m launch_monitor -config experiments/openImg/config_json_fast_agg.yaml)
python e2e_trainer.py -config experiments/openImg/config_json_fast_agg.yaml
echo $monitor_pid
kill $monitor_pid 
exit
exit
exit
