import pandas as pd
import os
import glob

dataset_csv = '/media/data/mu/ML2/data2/our_data_processed/experiments/dataset.csv'
data = pd.read_csv(dataset_csv)

len_data_all = len(data)
len_tb_all = len(data[data['id']==1])
len_non_tb_all = len(data[data['id']==0])
print('len_data_all', len_data_all)
print('len_tb_all', len_tb_all)
print('len_non_tb_all', len_non_tb_all)
print('')

diabetes = data[data['type']=='D']
diabetes_tb = diabetes[diabetes['id']==1]
len_tb_D = len(diabetes_tb)
len_non_tb_D = len(diabetes) - len_tb_D
print('len_diabetes_all', len(diabetes))
print('len_tb_D', len_tb_D)
print('len_non_tb_D', len_non_tb_D)
print('')


HIV = data[data['type']=='H']
HIV_tb = HIV[HIV['id']==1]
len_tb_H = len(HIV_tb)
len_non_tb_H = len(HIV) - len_tb_H
print('len_HIV_all', len(HIV))
print('len_tb_H', len_tb_H)
print('len_non_tb_H', len_non_tb_H)
print('')

TB_only = data[data['type']=='T']
len_tb_only = len(TB_only)
print('len_TB_only', len(TB_only))
print('len_tb_only', len_tb_only)

