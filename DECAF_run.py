from tests.utils import *
from tests.test_decaf import test_run_experiments

# Add files to sys 
import os, sys
import json

sys.path.append(os.getcwd())

df = load_adult()

df_train, df_test = train_test_split(df.iloc[:1000], test_size=200,
                                               stratify=df.iloc[:1000]['label'])

final_ND = {'precision': [],
'recall': [],
'auroc': [],
'dp': [],
'ftu': []}

final_FTU = {'precision': [],
'recall': [],
'auroc': [],
'dp': [],
'ftu': []}

final_DP = {'precision': [],
'recall': [],
'auroc': [],
'dp': [],
'ftu': []}

final_CF = {'precision': [],
'recall': [],
'auroc': [],
'dp': [],
'ftu': []}

def multi_run_all(RUNS=10):

   for i in range(RUNS):
      res = test_run_experiments(df_train, df_test, '')

      final_ND['precision'].append(res['precision'])
      final_ND['recall'].append(res['recall'])
      final_ND['auroc'].append(res['auroc'])
      final_ND['dp'].append(res['dp'])
      final_ND['ftu'].append(res['ftu'])

      print(f'Results for run {i} no debiasing:', final_ND)

      res = test_run_experiments(df_train, df_test, 'ftu')

      final_FTU['precision'].append(res['precision'])
      final_FTU['recall'].append(res['recall'])
      final_FTU['auroc'].append(res['auroc'])
      final_FTU['dp'].append(res['dp'])
      final_FTU['ftu'].append(res['ftu'])

      print(f'Results for run {i} FTU:', final_FTU)

      res = test_run_experiments(df_train, df_test, 'dp')

      final_DP['precision'].append(res['precision'])
      final_DP['recall'].append(res['recall'])
      final_DP['auroc'].append(res['auroc'])
      final_DP['dp'].append(res['dp'])
      final_DP['ftu'].append(res['ftu'])

      print(f'Results for run {i} DP:', final_DP)

      res = test_run_experiments(df_train, df_test, 'cf')

      final_CF['precision'].append(res['precision'])
      final_CF['recall'].append(res['recall'])
      final_CF['auroc'].append(res['auroc'])
      final_CF['dp'].append(res['dp'])
      final_CF['ftu'].append(res['ftu'])

      print(f'Results for run {i} CF:', final_CF)

multi_run_all()
