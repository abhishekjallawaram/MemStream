#!/bin/bash
# Bash script for RQ1 - hyper-parameter tuning

python3 memstream.py --dataset KDD --beta 1 --memlen 256 --RQ1 True
python3 memstream.py --dataset NSL --beta 0.1 --memlen 2048 --RQ1 True
python3 memstream.py --dataset UNSW --beta 0.1 --memlen 2048 --RQ1 True
python3 memstream.py --dataset DOS --beta 0.1 --memlen 2048 --RQ1 True
python3 memstream.py --dataset ionosphere --beta 0.001 --memlen 4 --RQ1 True
python3 memstream.py --dataset cardio --beta 1 --memlen 64 --RQ1 True
python3 memstream.py --dataset statlog --beta 0.01 --memlen 32 --RQ1 True
python3 memstream.py --dataset satimage-2 --beta 10 --memlen 256 --RQ1 True
python3 memstream.py --dataset mammography --beta 0.1 --memlen 128 --RQ1 True
python3 memstream.py --dataset pima --beta 0.001 --memlen 64 --RQ1 True
python3 memstream.py --dataset cover --beta 0.0001 --memlen 2048 --RQ1 True
