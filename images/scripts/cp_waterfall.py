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
    return np.abs(count)
        
            

xs = np.arange(0, 100, 0.1)


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)
def fc(arg):
    return "lightgrey"#mcolors.to_rgba(arg, alpha=0.0)

xs = np.arange(0, 100)
verts = []
zs = np.arange(10)
cps = [66, 40, 20, 30, 80]
for z in zs:
    r = np.random.randint(1,3)
    ys = verteilung(np.random.choice(cps, size=r))
    ys[0], ys[-1] = 0, 0
    verts.append(list(zip(xs, ys)))

poly = PolyCollection(verts, facecolors=[fc('r'), fc('g'), fc('b'), fc('y')], edgecolors=[cc('black'),cc('black'),cc('black'),cc('black')])
poly.set_alpha(0.9)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('time [commit]')
ax.set_xlim3d(0, 100)
ax.set_ylabel('Configuration')
ax.set_ylim3d(0, 10)
ax.set_zlabel('change point probability')
ax.set_zlim3d(0, 0.06)
plt.axvline(4)
plt.show()

