#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import os

from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
import matplotlib
import numpy as np
import ot  # https://github.com/rflamary/POT
import matplotlib.pyplot as plt

###############################################################################
# TOY EXAMPLES                                                                #
markerSize = 50
matplotlib.rcParams['font.size'] = 16
fig = plt.figure(1, figsize=(18, 10))
###############################################################################


def plotPoints(ax, P, label, marker, xMin, xMax, yMin, yMax, c):
    linewidth = 2
    if marker == "o":
        linewidth = 0
    ax.scatter(P[:, 0], P[:, 1], label=label, marker=marker, s=markerSize,
               c=c, linewidth=linewidth)
    ax.set_xlim([xMin, xMax])
    ax.set_ylim([yMin, yMax])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', framealpha=0.7)


# First toy example


def makeDomain(outer, inner, step, name, marker):
    # First toy example
    points = []
    for y in np.arange(-outer, outer+1, step):
        xMin = -(outer - abs(y))
        xMax = outer - abs(y)
        for x in np.arange(xMin, xMax+1, step):
            if x <= -inner or x >= inner or y <= -inner or y >= inner:
                points.append([x, y])
    return np.array(points)


T = makeDomain(32, 16, 4, "Target", "+")
S = makeDomain(14, 8, 2, "Source", "o")
ax = fig.add_subplot(2, 3, 1)
plotPoints(ax, T, "Target", "+", -35, 35, -35, 35, "blue")
plotPoints(ax, S, "Source", "o", -35, 35, -35, 35, "orange")

transp = ot.da.EMDTransport()
transp.fit(Xs=S, Xt=T)
indexesNearest = np.argmax(transp.coupling_, axis=1)
ax = fig.add_subplot(2, 3, 2)
plotPoints(ax, np.array([T[i] for i in indexesNearest]),
           "Target Selection OT", "+", -35, 35, -35, 35, "blue")

dist = cdist(S, T, metric='sqeuclidean')
indexesNearest = np.argmin(dist, axis=1)
T2 = np.array([T[i] for i in indexesNearest])
ax = fig.add_subplot(2, 3, 3)
plotPoints(ax, np.array([T[i] for i in indexesNearest]),
           "Target selection 1NN", "+", -35, 35, -35, 35, "blue")


# Second toy example
fig = plt.figure(1, figsize=(18, 5))
X, y = make_moons(150, shuffle=False, noise=0.05)
S = np.array(random.sample(list(X[y == 0]), 35))
T = X[y == 1]
ax = fig.add_subplot(2, 3, 4)
plotPoints(ax, T, "Target", "+", -1.3, 2.3, -0.7, 1.2, "blue")
plotPoints(ax, S, "Source", "o", -1.3, 2.3, -0.7, 1.2, "orange")

transp = ot.da.EMDTransport()
transp.fit(Xs=S, Xt=T)
indexesNearest = np.argmax(transp.coupling_, axis=1)
ax = fig.add_subplot(2, 3, 5)
plotPoints(ax, np.array([T[i] for i in indexesNearest]),
           "Target Selection OT", "+", -1.3, 2.3, -0.7, 1.2, "blue")

dist = cdist(S, T, metric='sqeuclidean')
indexesNearest = np.argmin(dist, axis=1)
ax = fig.add_subplot(2, 3, 6)
plotPoints(ax, np.array([T[i] for i in indexesNearest]),
           "Target selection 1NN", "+", -1.3, 2.3, -0.7, 1.2, "blue")

# Save both toy examples in a png file
plt.subplots_adjust(wspace=0.0, hspace=0.0)
savePath = os.path.join(".", "results", "toySampleSelection.png")
if not os.path.exists("./results"):
    os.makedirs("./results")
fig.savefig("./results/toySampleSelection.png", bbox_inches="tight")
