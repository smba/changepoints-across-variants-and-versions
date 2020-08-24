#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns

np.random.seed(3213)

sns.set_context("paper")
plt.style.use('seaborn-bright')
sns.set_style("whitegrid")

import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns


ys = np.ones(1000)
cps = [450,800]

for cp in cps:
    add = np.random.randint(1,10)
    ys[cp:] += add

ys += np.random.normal(0, 0.1, size=1000)

obs1 = np.arange(0, 1000, 1)#np.random.choice(, size=100)
count = np.zeros(1000)
for a, b in itertools.combinations(obs1, 2):
    if np.abs(ys[a] - ys[b]) > 0.5:
        count[min(a, b):max(a,b)] += 1/((a-b)**2)
count = count / np.sum(count)

obs2 = np.random.choice(np.arange(0, 1000), size=50)
count2 = np.zeros(1000)
for a, b in itertools.combinations(obs2, 2):
    if np.abs(ys[a] - ys[b]) > 0.5:
        count2[min(a, b):max(a,b)] += 1/((a-b)**2)
count2 = count2 / np.sum(count2)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(count, label="all versions")
ax1.plot(count2, label="approximation")
ax1.set_xlabel("time")
ax1.set_ylabel("$p(v)$")
for cp in cps: 
    ax1.axvline(cp, color="black", linestyle="--")
    ax2.axvline(cp, color="black", linestyle="--")
ax2.set_xlabel("time")
ax2.set_ylabel("performance")
ax2.plot(ys)
ax1.legend()
for a, b in itertools.combinations(obs2, 2):
    if np.abs(ys[a] - ys[b]) > 0.5:
        count2[min(a, b):max(a,b)] += 1/((a-b)**2)
count2 = count2 / np.sum(count2)

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot([])
ax1.plot(count, label="all versions")
ax1.plot(count2, label="approximation")
ax1.set_xlabel("time")
ax1.set_ylabel("$p(v)$")
for cp in cps: 
    ax1.axvline(cp, color="black", linestyle="--")
    ax2.axvline(cp, color="black", linestyle="--")
    
sns.rugplot(obs2, ax=ax1)
ax1.scatter([], [], marker="|", linewidth=2, s=100, label="sample measurements")
ax2.set_xlabel("Time")
ax2.set_ylabel("Performance")
ax2.plot(ys)
ax1.legend()

plt.draw()
plt.savefig("cpp_example.eps", bbox_inches="tight")
