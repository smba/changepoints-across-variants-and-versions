#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:05:26 2020

@author: stefan
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

np.random.seed(156)
plt.style.use('ggplot')
sns.set_style("whitegrid")


plt.figure(figsize=(4,2.5))

for x in np.random.choice(np.arange(100), size=3):
    ys = np.random.choice(np.arange(100), size=10)
    plt.scatter([x for y in ys], ys, color="black", marker=".", zorder=4)
    plt.axvline(x, alpha=0.5)
    

plt.xlim(0,100)
plt.ylim(0,100)
plt.xlabel("time")
plt.ylabel("configuration")
plt.draw()
plt.savefig("1.pdf", bbox_inches="tight")
plt.clf()

for y in np.random.choice(np.arange(100), size=3):
    xs = np.random.choice(np.arange(100), size=10)
    plt.scatter(xs, [y for x in xs], color="black", marker=".", zorder=4)
    plt.axhline(y, alpha=0.5)

plt.xlim(0,100)#np.random.normal()
plt.ylim(0,100)
plt.xlabel("time")
plt.ylabel("configuration")
plt.draw()
plt.savefig("2.pdf", bbox_inches="tight")
plt.clf()

ys = np.random.choice(np.arange(100), size=8)
for y in ys: plt.axhline(y, alpha=0.5)
for x in np.random.choice(np.arange(100), size=8):  
    plt.scatter([x for y in ys], ys, color="black", marker=".", zorder=4)
    plt.axvline(x, alpha=0.5)

plt.xlim(0,100)
plt.ylim(0,100)
plt.xlabel("time")
plt.ylabel("configuration")
plt.draw()
plt.savefig("3.pdf", bbox_inches="tight")
plt.clf()

ys = sorted(np.random.choice(np.arange(100), size=5))
xs = sorted(np.random.choice(np.arange(100), size=5))
#for y in ys: plt.axhline(y, alpha=0.5)
for y in ys:  
    plt.scatter(np.random.choice(np.arange(100), size=5), ys, color="black", marker=".", zorder=4)
    plt.axhline(y, alpha=0.5)
ynew = sorted(np.random.choice(np.arange(ys[2]-15, ys[2]+15), size=3))
xnew = [sorted(np.random.choice(np.arange(xs[2]-10, xs[2]+10), size=3)) for i in range(3)]

for i, xs in enumerate(xnew):
    plt.scatter(xs, [ynew[i] for j in range(3)], color="blue", marker=".", zorder=4)

for i, xs in enumerate(xnew):
    plt.plot((xs[0], xs[-1]), (ynew[i], ynew[i]), color="skyblue", alpha=0.6)
    #plt.plot([xs[0], xs[-1]], [ynew[i], ynew[i]], color="skyblue")
    
plt.xlim(0,100)
plt.ylim(0,100)
plt.xlabel("time")
plt.ylabel("configuration")
plt.draw()
plt.savefig("4.pdf", bbox_inches="tight")
plt.clf()
