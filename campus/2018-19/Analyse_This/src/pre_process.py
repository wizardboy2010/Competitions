import pandas as pd
import numpy as np
import os

from collections import Counter

data = pd.read_csv('campus/2018-19/Analyse_This/data/Training_dataset_Original.csv', low_memory=False)

c = Counter(data.mvar1)