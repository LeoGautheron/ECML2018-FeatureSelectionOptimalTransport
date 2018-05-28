#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from matplotlib.colors import ListedColormap
import matplotlib
import numpy as np
import ot  # https://github.com/rflamary/POT
import matplotlib.pyplot as plt

###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

nbS = 30
nbT = 30
np.random.seed(41)
matplotlib.rcParams['font.size'] = 16

plotResults = True
markerSize = 100
c11 = "#0088FF"
c12 = "#FF8800"
c13 = "#00FF88"

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################


def generateGaussian(isTarget):
    if isTarget:
        n = nbT
    else:
        n = nbS
    y = np.floor((np.arange(n)*1.0/n*3))+1
    x = np.zeros((n, 2))

    x[y == 1, 1] = 1
    x[y == 2, 1] = 2
    x[y == 3, 1] = 3

    if isTarget:
        x[y == 1, 0] = 2.
        x[y == 2, 0] = 2.
        x[y == 3, 0] = 2.
    x += 0.4*np.random.randn(n, 2)
    return x, y


def drawPoints(ax, X, Y, c1, c2, c3, m, z, label):
    ax.scatter(X[:, 0], X[:, 1], c=Y, label=label, edgecolor='black',
               linewidth='2', marker=m, s=[markerSize] * len(X),
               cmap=ListedColormap([c1, c2, c3]), zorder=z)

    ax.set_xticks([0, 2])
    ax.set_yticks([0, 1, 2, 3])


XS, YS = generateGaussian(isTarget=False)
XT, YT = generateGaussian(isTarget=True)

rows = 2
columns = 3
subplotNumber = 0
fig = plt.figure(1, figsize=(columns * 6, rows * 5))
names = ["OT", "OT2", "OT3"]

for (i, name) in enumerate(names):
    # Compute optimal coupling G with one of the three optimal transport algo
    if i == 0:
        transp = ot.da.OTDA()
        transp.fit(XS, XT)
    elif i == 1:
        transp = ot.da.OTDA_sinkhorn()
        transp.fit(XS, XT, reg=1)
    elif i == 2:
        transp = ot.da.OTDA_lpl1()
        transp.fit(XS, YS, XT, reg=1, eta=1)
    G = transp.G

    # Show coupling matrix
    subplotNumber += 1
    ax = fig.add_subplot(rows, columns, subplotNumber)
    ax.grid(b=False)
    im = ax.imshow(G, cmap="Blues", aspect="equal", interpolation='none')
    fig.colorbar(im, ax=ax, orientation='horizontal', format='%.1e',
                 ticks=[np.min(G), np.max(G)])
    ax.set_title("Coupling with " + name)

    # Show coupling points
    ax = fig.add_subplot(rows, columns, subplotNumber+3)
    ax.grid(b=False)
    nbPerSample = nbT
    cls = np.argsort(-G)[:, :nbPerSample]
    mx = G.max()
    for i in range(XS.shape[0]):
        color = c12
        if YS[i] == 1:
            color = c11
        elif YS[i] == 2:
            color = c12
        else:
            color = c13
        for j in range(nbPerSample):
            alpha = G[i, cls[i, j]] / mx
            alpha /= 3
            ax.plot([XS[i, 0], XT[cls[i, j], 0]], [XS[i, 1], XT[cls[i, j], 1]],
                    alpha=alpha, color=color, zorder=0)
    # Draw the source and target points
    drawPoints(ax, XS, YS, c11, c12, c13, "o", 1, "Source")
    drawPoints(ax, XT, YT, c11, c12, c13, "+", 2, "Target")
    ax.legend(loc='lower left', framealpha=0.7)

# Save figure
plt.subplots_adjust(wspace=0.30, hspace=0.0)
savePath = os.path.join(".", "results", "comparisonOT.png")
if not os.path.exists("results"):
    os.makedirs("results")
fig.savefig(savePath, bbox_inches="tight")
