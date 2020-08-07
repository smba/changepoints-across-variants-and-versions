import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import joypy
import matplotlib.pyplot as plt
from matplotlib import cm

def verteilung(cp, maximum=100):
    if len(cp) == 0:
        return np.zeros(100)
    if cp.shape == (1,):
        count = np.zeros(maximum)
        for a, b in itertools.combinations(np.arange(100), 2):
            if a < cp and b > cp:
                count[a:b] += 1/(a-b)**2
    else:
        count = np.zeros(maximum)
        for c in cp:
            count += verteilung(np.array([c]))
    count = count/np.sum(count)
    count += np.random.normal(0, 0.0007, size=maximum)
    count = np.abs(count)
    return count/np.sum(count)

import pandas as pd
import joypy
import numpy as np

df = pd.DataFrame()
for i in "ABE":
    if i == "A":
        count = verteilung(np.array([15, 70]))
        df[i] = np.random.choice(np.arange(100), size=8000, p=count)
    elif i == "B":
        count = verteilung(np.array([45]))
        df[i] = np.random.choice(np.arange(100), size=8000, p=count) 
    else:
        count = verteilung(np.array([ 90]))
        df[i] = np.random.choice(np.arange(100), size=8000, p=count)
joypy.joyplot(df, overlap=3, linewidth=.5, bins=100, color="lightgrey")
plt.show()
