# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.linear_model as lm
import sklearn.preprocessing as pre
import itertools

euro = ["#09ff67", "#bf0778", "#ff6709", "#09a1ff"]
europalette = sns.color_palette(euro)
sns.set_palette(europalette)

sns.set_context("paper")
plt.figure(figsize=(4.9, 2))
#plt.style.use('seaborn-bright')

sns.set_style("whitegrid")
labels = ["vanilla", "compression", "encryption", "compr & encr"]

configs = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,1]])

perf = np.ones(shape=(100, 4))

perf[:, 1] += 1
perf[25:, 1] += 0.45
perf[:, 2] += 0.3
perf[:, 3] += 0.7
perf[50:, 3] += 0.25
perf[80:, :] -=0.5

#perf[:25, 2:4] = 1
perf += np.random.normal(0,0.01, size=(100, 4))


# PI models
#poly = pre.PolynomialFeatures(degree=2, interaction_only=True)
#polyconfigs = poly.fit_transform(configs)[:,1:]
#print(polyconfigs)
coefs = []
intercepts = []
for t in range(100):
    lmod = lm.LinearRegression()
    lmod.fit(configs, perf[t, :])
    coefs.append(lmod.coef_)
    intercepts.append(lmod.intercept_)
coefs = np.array(coefs)
cc = coefs[:25,1:3]
coefs[:25,1:3]=0

perf[:25,2:] = np.nan
for i in range(4):
    plt.plot(perf[:, i], label=labels[i], linewidth=2)
for x in [25,50, 80]: plt.axvline(x, color="black", linestyle="--")
plt.xlabel("Time [commit]")
plt.ylabel("Performance [s]")
plt.legend(loc=9, bbox_to_anchor=(1.25, 1.03))

#plt.show()
plt.draw()
plt.savefig("dbms_performance.eps", bbox_inches="tight")
plt.clf()

perf[:25,1:3] = np.nan

plt.plot(intercepts, label="y-intercept", linewidth=2)
for i in range(3):
    plt.plot(coefs[:,i], linewidth=2, label=labels[i+1])
for x in [25,50, 80]: plt.axvline(x, color="black", linestyle="--")

plt.xlabel("time [commit]")
plt.ylabel("performance influence")
plt.legend(loc=9, bbox_to_anchor=(1.25, 1.03))

#plt.show()
plt.draw()
plt.savefig("dbms_influences.eps", bbox_inches="tight")
plt.clf()


# influence changes
obs = np.arange(0, 100, 1)#np.random.choice(, size=100)
count = np.zeros(100)
for a, b in itertools.combinations(obs, 2):
    if np.abs(intercepts[a] - intercepts[b]) > 0.2:
        count[min(a, b):max(a,b)] += 1/((a-b)**2)

count = count / np.sum(count)
plt.plot(count, label="y-intercept", linewidth=2)

coefs[:25, 1] = coefs[26, 1]
coefs[:25, 2] = coefs[26, 2]
for i in range(3):
    obs = np.arange(0, 100, 1)#np.random.choice(, size=100)
    count = np.zeros(100)
    for a, b in itertools.combinations(obs, 2):
        if np.abs(coefs[a,i] - coefs[b, i]) > 0.2:
            count[min(a, b):max(a,b)] += 1/((a-b)**2)
    count /= np.sum(count)
    plt.plot(count, label=labels[i+1], linewidth=2)
    for x in [25,50, 80]: plt.axvline(x, color="black", linestyle="--")

plt.xlabel("time [commit]")
plt.ylabel("feature change-p. probability")
plt.legend(loc=9, bbox_to_anchor=(1.25, 1.03))

#plt.show()
plt.draw()
plt.savefig("dbms_change_point_probability.eps", bbox_inches="tight")
plt.clf()
