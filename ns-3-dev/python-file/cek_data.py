import pandas as pd
import numpy as np

df = pd.read_csv('/home/nru/CBNS_NEW2/ns-3-dev/python-file/result_experiment/dataset_complete.csv')

print(df.head())

nilai_lain = df[~df['PDR'].isin([0.0, 100.0])]

print(nilai_lain)
