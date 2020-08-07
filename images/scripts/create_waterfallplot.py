#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import itertools
from scipy.stats import cauchy as cauchy


fig = plt.figure()
ax = fig.gca(projection='3d')

def verteilung(cp, maximum=100):
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
    count += np.random.normal(0, 0.0005, size=maximum)
    return count
        
            

xs = np.arange(0, 100, 0.1)


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)
def fc(arg):
    return "white"#mcolors.to_rgba(arg, alpha=0.0)

xs = np.arange(0, 100)
verts = []
zs = [0.0, 0.5, 1.0, 1.5]
for z in zs:
    r = np.random.randint(5,10)
    ys = verteilung(np.random.choice(np.arange(100), size=r))
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors=[fc('r'), fc('g'), fc('b'), fc('y')], edgecolors=[cc('black'),cc('black'),cc('black'),cc('black')])
poly.set_alpha(0.9)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('time [commit]')
ax.set_xlim3d(0, 100)
ax.set_ylabel('Configuration')
ax.set_ylim3d(0, 2)
ax.set_zlabel('change point probability')
ax.set_zlim3d(0, 0.06)

plt.show()