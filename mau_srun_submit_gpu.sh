#!/bin/bash

srun -w mauao -u --job-name "FlUTE" -c 44 --gres=gpu:a40:4 launch_flute.sh "$@"

