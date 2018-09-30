import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from collections import Counter

data = pd.read_csv('campus/2018-19/Analyse_This/data/Training_dataset_Original.csv', low_memory=False)

plt.matshow(data.corr())
plt.show()

c = Counter(data.mvar1)