#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle

from cycler import cycler
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 22
color_c = cycler('color', ['k'])
style_c = cycler('linestyle', ['-', '--', ':', '-.'])
markr_c = cycler('marker', ['', '.', 'o'])
c_cms = color_c * markr_c * style_c
c_csm = color_c * style_c * markr_c

# Load data
filename = "./results/officeCaltech.pkl"
f = open(filename, "rb")
r = pickle.load(f)
f.close()


features = r["features"]
adaptationAlgoNames = r["adaptationAlgoNames"]
orderNames = r["orders"]


def plotAll():
    for fts in features:
        nbrsFeatures = r["resFeatures"][fts]["nbrsFeatures"]
        numberFeatures = nbrsFeatures[-1]
        res = r["resFeatures"][fts]["resTests"]

        for algo in adaptationAlgoNames:
            fig = plt.figure(figsize=(8.5, 7))

            plt.tick_params(axis="both", which="both", bottom="off", top="off",
                            labelbottom="on", left="off", right="off",
                            labelleft="on")

            ax = fig.add_subplot(1, 1, 1)
            ax.grid(True)

            ax.set_xlim([0, numberFeatures])
            ticksX = [int(numberFeatures/4), int(2*numberFeatures/4),
                      int(3*numberFeatures/4), numberFeatures]
            ax.set_xticks(ticksX)
            ax.set_xticklabels(ticksX)
            plt.xlabel('Number of features selected', fontsize=30)

            ax.set_ylim([15, 45])  # accuracy
            ax.set_yticks(np.arange(15, 46, 5))
            ax.set_yticklabels(np.arange(15, 46, 5))
            plt.ylabel('Accuracy', fontsize=30, labelpad=0)

            styles = ["--", "-.", "--"]
            colors = ['gray', 'k', 'silver']

            for i, orderName in enumerate(orderNames):
                acc = [np.mean(
                       [np.mean(
                        [p["acc"]
                         for p in res[da]["resTest"][orderName][algo][d]])
                        for da in res.keys()])
                       for d in range(len(nbrsFeatures))]

                plt.plot(np.array(nbrsFeatures), acc, linestyle=styles[i],
                         color=colors[i], label=orderNames[i], linewidth=6)

            accAll = np.mean(
                     [np.mean(
                      [p["acc"]
                       for p in res[da]["resTest"][orderNames[0]][algo][-1]])
                      for da in res.keys()])
            ax.plot([0, numberFeatures], [accAll, accAll],
                    color='slategray', label="All features", linewidth=4)
            ax.legend(loc='upper center', ncol=2)
            filename = algo + "_" + fts + ".png"
            savePath = os.path.join(".", "results", filename)
            if not os.path.exists("results"):
                os.makedirs("results")
            plt.savefig(savePath, bbox_inches="tight")
            plt.close()


def latexArrayAllPairs():
    d = 8  # to change, index in array nbrsFeatures
    for fts in features:
        nbrsFeatures = r["resFeatures"][fts]["nbrsFeatures"]
        numberFeatures = nbrsFeatures[-1]
        res = r["resFeatures"][fts]["resTests"]

        for algo in adaptationAlgoNames:
            print("\n%", fts, algo)
            print("\\begin{tabular}{c | c c c}")
            print("DA pairs", end="")
            print(" & $\\searrow " + str(nbrsFeatures[d]) + "$", end="")
            print(" & $\\nearrow " + str(nbrsFeatures[d]) + "$", end="")
            print(" & $" + str(numberFeatures) + "$", end="")
            print("\\\\")
            print("\\hline")
            for da in sorted(res.keys()):
                print(da.replace("->", "$\\rightarrow$"), end="")
                meanAccDsc = np.mean(
                 [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                meanStdDsc = np.std(
                 [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                print(" & {0:.1f}".format(meanAccDsc) +
                      "$\\pm${0:.1f}".format(meanStdDsc), end="")

                meanAccAsc = np.mean(
                 [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][d]])
                meanStdAsc = np.std(
                 [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][d]])
                print(" & {0:.1f}".format(meanAccAsc) +
                      "$\\pm${0:.1f}".format(meanStdAsc), end="")

                meanAccAll = np.mean(
                 [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][-1]])
                meanStdAll = np.std(
                 [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][-1]])
                print(" & {0:.1f}".format(meanAccAll) +
                      "$\\pm${0:.1f}".format(meanStdAll), end="")
                print("\\\\")

            print("\\hline")
            print("Mean", end="")
            meanAccDsc = np.mean([np.mean(
             [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                                  for da in res.keys()])
            meanStdDsc = np.mean([np.std(
             [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                                  for da in res.keys()])
            print(" & {0:.1f}".format(meanAccDsc) +
                  "$\\pm${0:.1f}".format(meanStdDsc), end="")

            meanAccAsc = np.mean([np.mean(
             [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][d]])
                                  for da in res.keys()])
            meanStdAsc = np.mean([np.std(
             [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][d]])
                                  for da in res.keys()])
            print(" & {0:.1f}".format(meanAccAsc) +
                  "$\\pm${0:.1f}".format(meanStdAsc), end="")

            meanAccAll = np.mean([np.mean(
             [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][-1]])
                                  for da in res.keys()])
            meanStdAll = np.mean([np.std(
             [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][-1]])
                                  for da in res.keys()])
            print(" & {0:.1f}".format(meanAccAll) +
                  "$\\pm${0:.1f}".format(meanStdAll), end="")
            print("\\\\")
            print("\\end{tabular}")


def latexArrayFeatures():
    for algo in adaptationAlgoNames:
        print("\n%", algo)
        print("\\begin{tabular}{r |", end="")
        for i in range(len(features)):
            print(" c", end="")
        print("}")
        print("\\#features", end="")
        for i in range(len(features)):
            print(" & " + features[i], end="")
        print("\\\\")
        print("\\hline")
        idxs = [0, 2, 8]  # to change, indexes in array nbrsFeatures
        titlesRows = ["d/32$", "d/8$", "d/2$"]
        for idRow, d in enumerate(idxs):
            print("$\\searrow " + titlesRows[idRow], end="")
            for fts in features:
                res = r["resFeatures"][fts]["resTests"]
                meanAccDsc = np.mean([np.mean(
                 [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                                      for da in res.keys()])
                meanStdDsc = np.mean([np.std(
                 [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                                      for da in res.keys()])
                print(" & {0:.1f}".format(meanAccDsc) +
                      "$\\pm${0:.1f}".format(meanStdDsc), end="")
            print("\\\\")
            print("$\\nearrow " + titlesRows[idRow], end="")
            for fts in features:
                res = r["resFeatures"][fts]["resTests"]
                meanAccAsc = np.mean([np.mean(
                 [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][d]])
                                      for da in res.keys()])
                meanStdAsc = np.mean([np.std(
                 [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][d]])
                                      for da in res.keys()])
                print(" & {0:.1f}".format(meanAccAsc) +
                      "$\\pm${0:.1f}".format(meanStdAsc), end="")
            print("\\\\")
            print("\\hline")

        # Last row, all features
        print("d", end="")
        for fts in features:
            res = r["resFeatures"][fts]["resTests"]
            meanAccAll = np.mean([np.mean(
             [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][-1]])
                                  for da in res.keys()])
            meanStdAll = np.mean([np.std(
             [p["acc"] for p in res[da]["resTest"]["Ascending"][algo][-1]])
                                  for da in res.keys()])
            print(" & {0:.1f}".format(meanAccAll) +
                  "$\\pm${0:.1f}".format(meanStdAll), end="")
        print("\\\\")
        print("\\end{tabular}")


def latexArrayComputationTime():
    for fts in features:
        nbrsFeatures = r["resFeatures"][fts]["nbrsFeatures"]
        numberFeatures = nbrsFeatures[-1]
        idxs = [2, 4, 8, 16]  # to change, indexes in array nbrsFeatures
        res = r["resFeatures"][fts]["resTests"]

        print("\n%", fts, nbrsFeatures)
        print("\\begin{tabular}{c |", end="")
        for d in idxs:
            print(" c r|", end="")
        print("}")
        print("Method", end="")
        for i, d in enumerate(idxs):
            if i < len(idxs)-1:
                print(" & \multicolumn{2}{c}{$\\searrow$" +
                      str(nbrsFeatures[d]) + "}", end="")
        print(" & \multicolumn{2}{c}{" + str(numberFeatures) + "}\\\\")
        print("\\hline")
        for algo in adaptationAlgoNames:
            print(algo, end="")
            for d in idxs:
                meanAcc = np.mean([np.mean(
                 [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                                   for da in res.keys()])
                meanStd = np.mean([np.std(
                 [p["acc"] for p in res[da]["resTest"]["Descending"][algo][d]])
                                   for da in res.keys()])
                sumTime = np.sum([np.sum(
                          [p["time"]
                           for p in res[da]["resTest"]["Descending"][algo][d]])
                                  for da in res.keys()])
                print(" & {0:.1f}".format(meanAcc) +
                      "$\\pm${0:.1f}".format(meanStd) +
                      " & {:6.2f}s".format(sumTime), end="")
            print("\\\\")
        print("\\hline")
        print("\\end{tabular}")


plotAll()
# latexArrayAllPairs()
# latexArrayFeatures()
# latexArrayComputationTime()
