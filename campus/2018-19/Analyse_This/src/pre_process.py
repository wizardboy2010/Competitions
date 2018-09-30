import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import Counter

def assign_missing(val):
    if val in ['missing', 'na', 'NA', 'Na', 'Nan', 'NaN', 'N/A']:
        return np.NaN
    try:
        return float(val)
    except:
        return str(val)

try:
    path = os.path.dirname(os.path.realpath(__file__))
except:
    path = 'E:/ubuntu_data/Projects/Competitions/campus/2018-19/Analyse_This/src'

os.chdir(path)

data_path = os.path.join(os.path.dirname(path), 'data')

data = pd.read_csv(os.path.join(data_path,'Training_dataset_Original.csv'), low_memory=False)

plt.matshow(data.corr())
plt.show()

c = Counter(data.mvar1)