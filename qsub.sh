#!/bin/bash
#------- qsub option -----------
#PBS -A CVLABPJ
#PBS -q gen_S
#PBS -b 1
#PBS -l elapstim_req=12:00:00
#PBS –M vaz.valois@hotmail.com
#PBS –m e #バッチリクエスト終了時に指定したメールアドレスに通知
#PBS -N sub_pvalois #リクエスト名
#------- Program execution -----------

module load cuda/12.3.2
module load openmpi/$NQSV_MPI_VER

cd $PBS_O_WORKDIR
uv run python -c "import torch; print(torch.__version__)"
make run