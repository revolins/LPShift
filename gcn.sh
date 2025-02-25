#!/bin/bash

device=$1

python main_gnn.py --data_name ogbl-collab_CN_2_1_0 --lr 1e-2 --dropout 0.1 --device $device --epochs 100

# python main_gnn.py --data_name ogbl-collab_CN_4_2_0 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-collab_CN_5_3_0 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-collab_CN_0_1_2 --lr 1e-3 --dropout 0.3 --device $device 

# python main_gnn.py --data_name ogbl-collab_CN_0_2_4 --lr 1e-3 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-collab_CN_0_3_5 --lr 1e-3 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-collab_PA_0_50_100 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_PA_0_100_200 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_PA_0_150_250 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_PA_100_50_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_PA_200_100_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_PA_250_150_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_SP_00_017_026 --dropout 0.3 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_SP_00_026_036 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-collab_SP_026_017_00 --dropout 0.3 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-collab_SP_036_026_00 --dropout 0.3 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-ddi_CN_0_1_2 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-ddi_CN_0_2_4 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-ddi_CN_0_3_5 --lr 1e-2 --dropout 0.1 --device $device

# python main_gnn.py --data_name ogbl-ddi_CN_2_1_0 --lr 1e-2 --dropout 0.1 --device $device

# python main_gnn.py --data_name ogbl-ddi_CN_4_2_0 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-ddi_CN_5_3_0 --lr 1e-2 --dropout 0.1 --device $device

# python main_gnn.py --data_name ogbl-ddi_PA_0_5000_10000 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ddi_PA_0_10000_20000 --dropout 0.1 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-ddi_PA_0_15000_25000 --dropout 0.1 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-ddi_PA_10000_5000_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ddi_PA_20000_10000_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ddi_PA_25000_15000_0 --dropout 0.1 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-ppa_CN_0_1_2 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-ppa_CN_0_2_4 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-ppa_CN_0_3_5 --lr 1e-2 --dropout 0.1 --device $device

# python main_gnn.py --data_name ogbl-ppa_CN_2_1_0 --lr 1e-2 --dropout 0.3 --device $device

# python main_gnn.py --data_name ogbl-ppa_CN_4_2_0 --lr 1e-3 --dropout 0.1 --device $device

# python main_gnn.py --data_name ogbl-ppa_CN_5_3_0 --lr 1e-3 --dropout 0.1 --device $device

# python main_gnn.py --data_name ogbl-ppa_PA_0_5000_10000 --dropout 0.3 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-ppa_PA_0_10000_20000 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ppa_PA_0_15000_25000 --dropout 0.3 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-ppa_PA_10000_5000_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ppa_PA_20000_10000_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ppa_PA_25000_15000_0 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ppa_SP_00_017_026 --dropout 0.3 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ppa_SP_00_026_036 --dropout 0.1 --lr 0.001 --device $device

# python main_gnn.py --data_name ogbl-ppa_SP_026_017_00 --dropout 0.3 --lr 0.01 --device $device

# python main_gnn.py --data_name ogbl-ppa_SP_036_026_00 --dropout 0.1 --lr 0.01 --device $device

