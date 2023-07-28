import torch.distributed
import argparse
import yaml
import subprocess
import time
import os
import sys
import psutil
from datetime import datetime


def get(dic, key, else_val):
    val = dic.get(key)
    if val:
        return val
    return else_val


NOT_SLURM = "RANK" in os.environ.keys()
HOMOGENEOUS = (
    "SLURM_NTASKS_PER_NODE" in os.environ.keys()
    and os.environ["SLURM_NTASKS_PER_NODE"] is not None
)


if __name__ == "__main__":
    og_args = sys.argv
    time_stamp = str(datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes")
    parser.add_argument("--nproc_per_node")
    parser.add_argument("-config")
    parser.add_argument("--outputPath", default=None)
    parser.add_argument("--task", default=None, help="Define the task for the run")
    args = parser.parse_args()
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    node_name = os.environ["SLURMD_NODENAME"]
    if MASTER_ADDR != node_name:
        time.sleep(1)
    mau_path = "/local/scratch/fertilizer/Flute/monitor"
    rank = int(os.environ["SLURM_PROCID"])

    if NOT_SLURM:
        gpus_per_node = int(os.environ["LOCAL_RANK"])
    elif HOMOGENEOUS:
        gpus_per_node = int(os.environ["SLURM_NTASKS_PER_NODE"])
    else:
        gpus_per_node = int(os.environ["SLURM_NTASKS"])

    local_rank = int(rank - gpus_per_node * (rank // gpus_per_node))
    if local_rank == 0:
        with open(args.config) as f:
            cfg_dict = yaml.safe_load(f)
            task = (
                args.task
                if args.task is not None
                else get(cfg_dict["setup"], "task", None)
            )
            outputPath = (
                args.outputPath
                if args.outputPath is not None
                else get(cfg_dict["setup"], "outputPath", None)
            )
            # =========== Starting monitoring GPU ==============
            # print("Starting Monitor")
            monitor_log_dir = os.path.join(".", "monitor")
            if not os.path.isdir(monitor_log_dir):
                os.makedirs(monitor_log_dir, exist_ok=True)
            # Now is one monitor per `driver.py` execution
            monitor_filename = os.path.join(
                monitor_log_dir,
                f"{task}_{time_stamp}_{node_name}.csv",
            )
            monitor_shell_false_cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,name,index,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                "--format=csv",
                f"--filename={monitor_filename}",
                "--loop-ms=1000",
            ]
            with open(f"{task}_logging", "a") as fout:
                process = psutil.Popen(
                    monitor_shell_false_cmd, shell=False, stdout=fout, stderr=fout
                )
            print(process.pid)
