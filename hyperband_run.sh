#!/bin/bash
#PBS -l walltime=16:0:0
#PBS -l nodes=4:ppn=4,gpus=2

#module load Miniconda3
source activate cloned_DeepPurpose
#source $EBROOTMINICONDA3/bin/activate /kyukon/data/gent/444/vsc44416/miniconda3/envs/DeepPurpose

cd "/kyukon/data/gent/vo/000/gvo00048/vsc44416/DeepPurpose"
python hyperband_run.py