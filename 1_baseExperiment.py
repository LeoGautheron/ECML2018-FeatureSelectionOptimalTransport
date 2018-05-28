#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import random
import time

from sklearn.neighbors import KNeighborsClassifier
from scipy.io import loadmat
import numpy as np
import ot  # https://github.com/rflamary/POT
from sklearn import preprocessing

###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

featuresToUse = ["surf", "CaffeNet4096", "GoogleNet1024"]
numberIteration = 1
adaptationAlgoNames = ["NA", "SA", "OT", "OT2", "OT3", "TCA", "CORAL"]
# see function adaptData for available algorithms

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################


def SampleSelection(S, T):
    transp = ot.da.EMDTransport()
    if len(S) < len(T):
        transp.fit(Xs=S, Xt=T)
        indexesNearest = np.argmax(transp.coupling_, axis=1)
        return (S, [T[i] for i in indexesNearest])
    else:
        transp.fit(Xs=T, Xt=S)
        indexesNearest = np.argmax(transp.coupling_, axis=1)
        return ([S[i] for i in indexesNearest], T)


def FeatureRankingForDomainAdaptation(S, T):
    (Su, Tu) = SampleSelection(S, T)
    Su = preprocessing.scale(np.transpose(Su))  # zscore transpose
    Tu = preprocessing.scale(np.transpose(Tu))  # zscore transpose
    transp2 = ot.da.SinkhornTransport(reg_e=1, norm="log")
    transp2.fit(Xs=Su, Xt=Tu)
    # Return the features sorted decreasingly by their coupling between S & T
    F = np.argsort(-np.diag(transp2.coupling_))
    return F, np.diag(transp2.coupling_)


def getAdaptedData(adaptationAlgoName, Sx, Sy, Tx, Ty):
    algoStartTime = time.time()
    if adaptationAlgoName == "NA":
        Sa = Sx
        Ta = Tx
    if adaptationAlgoName == "OT":
        transp = ot.da.EMDTransport()
        transp.fit(Xs=Sx, Xt=Tx)
        Sa = transp.transform(Xs=Sx)
        Ta = Tx
    elif adaptationAlgoName == "OT2":
        transp2 = ot.da.SinkhornTransport(reg_e=2, norm="median")
        transp2.fit(Xs=Sx, Xt=Tx)
        Sa = transp2.transform(Xs=Sx)
        Ta = Tx
    elif adaptationAlgoName == "OT3":
        transp3 = ot.da.SinkhornLpl1Transport(reg_e=2, reg_cl=1, norm="median")
        transp3.fit(Xs=Sx, ys=Sy, Xt=Tx)
        Sa = transp3.transform(Xs=Sx)
        Ta = Tx
    elif adaptationAlgoName == "SA":
        # Subspace Alignment, described in:
        # Unsupervised Visual Domain Adaptation Using Subspace Alignment, 2013,
        # Fernando et al.
        from sklearn.decomposition import PCA
        d = 80
        if Sx.shape[1] < d:
            d = Sx.shape[1]
        pcaS = PCA(d).fit(Sx)
        pcaT = PCA(d).fit(Tx)
        XS = np.transpose(pcaS.components_)[:, :d]
        XT = np.transpose(pcaT.components_)[:, :d]
        Xa = XS.dot(np.transpose(XS)).dot(XT)
        Sa = Sx.dot(Xa)
        Ta = Tx.dot(XT)
    elif adaptationAlgoName == "TCA":
        # Domain adaptation via transfer component analysis. IEEE TNN 2011
        d = 80  # subspace dimension
        if Sx.shape[1] < d:
            d = Sx.shape[1]
        Ns = Sx.shape[0]
        Nt = Tx.shape[0]
        L_ss = (1. / (Ns * Ns)) * np.full((Ns, Ns), 1)
        L_st = (-1. / (Ns * Nt)) * np.full((Ns, Nt), 1)
        L_ts = (-1. / (Nt * Ns)) * np.full((Nt, Ns), 1)
        L_tt = (1. / (Nt * Nt)) * np.full((Nt, Nt), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        X = np.vstack((Sx, Tx))
        K = np.dot(X, X.T)  # linear kernel
        H = (np.identity(Ns+Nt)-1./(Ns+Nt)*np.ones((Ns + Nt, 1)) *
             np.ones((Ns + Nt, 1)).T)
        inv = np.linalg.pinv(np.identity(Ns + Nt) + K.dot(L).dot(K))
        D, W = np.linalg.eigh(inv.dot(K).dot(H).dot(K))
        W = W[:, np.argsort(-D)[:d]]  # eigenvectors of d highest eigenvalues
        Sa = np.dot(K[:Ns, :], W)  # project source
        Ta = np.dot(K[Ns:, :], W)  # project target
    elif adaptationAlgoName == "CORAL":
        # Return of Frustratingly Easy Domain Adaptation. AAAI 2016
        from scipy.linalg import sqrtm
        Cs = np.cov(Sx, rowvar=False) + np.eye(Sx.shape[1])
        Ct = np.cov(Tx, rowvar=False) + np.eye(Tx.shape[1])
        Ds = Sx.dot(np.linalg.inv(np.real(sqrtm(Cs))))  # whitening source
        Ds = Ds.dot(np.real(sqrtm(Ct)))  # re-coloring with target covariance
        Sa = Ds
        Ta = Tx
    algoTime = time.time() - algoStartTime
    return (Sa, Ta, algoTime)


def generateSubset(X, Y, nPerClass):
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        random.shuffle(idxClass)
        idx.extend(idxClass[0:min(nPerClass, len(idxClass))])
    return (X[idx, :], Y[idx])


orderNames = ["Ascending", "Descending", "Random"]
resFeatures = {}
for featureToUse in featuresToUse:
    tests = []
    data = {}

    # Loop loading data
    for sourceDomain in ['amazon', 'caltech10', 'dslr', 'webcam']:
        possible_data = loadmat("./features/" + featureToUse + "/" +
                                sourceDomain + '.mat')
        if featureToUse == "surf":
            # Normalize the surf histograms
            feat = (possible_data['fts'].astype(float) /
                    np.tile(np.sum(possible_data['fts'], 1),
                            (np.shape(possible_data['fts'])[1], 1)).T)
        else:
            feat = possible_data['fts'].astype(float)

        # Z-score everything at beginning
        feat = preprocessing.scale(feat)
        (n, numberFeatures) = feat.shape
        labels = possible_data['labels'].ravel()
        data[sourceDomain] = [feat, labels]
        for targetDomain in ['amazon', 'caltech10', 'dslr', 'webcam']:
            if sourceDomain != targetDomain:
                perClassSource = 20
                if sourceDomain == 'dslr':
                    perClassSource = 8
                tests.append([sourceDomain, targetDomain, perClassSource])

    step = int(numberFeatures / 16)
    nbrsFeatures = [int(numberFeatures/32)]
    nbrsFeatures.extend(range(step, numberFeatures+step, step))
    print(featureToUse, nbrsFeatures)

    resTests = {}
    for (Sname, Tname, numberPerClass) in tests:  # Loop over the 12 DA pairs
        startTime = time.time()
        nameDA = (Sname.upper()[:1] + '->' + Tname.upper()[:1])
        print(nameDA, end=" ")

        fullSx = data[Sname][0]
        fullSy = data[Sname][1]
        Tx = data[Tname][0]
        Ty = data[Tname][1]

        resTest = {orderName:
                   {algo: [[] for j in range(len(nbrsFeatures))]
                    for algo in adaptationAlgoNames}
                   for orderName in orderNames}
        couplingDiags = []
        for iteration in range(numberIteration):
            (Sx, Sy) = generateSubset(fullSx, fullSy, numberPerClass)
            # Apply our proposed method to obtain the ordered list of features
            # by decreasing coupling between source and target domains
            orderDesc, diag = FeatureRankingForDomainAdaptation(Sx, Tx)

            orderAsc = orderDesc[::-1]  # reverse of descending
            orderRandom = orderAsc.copy()  # random
            random.shuffle(orderRandom)
            couplingDiags.append([float(c) for c in diag])

            for order, orderName in zip([orderAsc, orderDesc, orderRandom],
                                        orderNames):
                for (j, nbrFeatures) in enumerate(nbrsFeatures):
                    F = order[:nbrFeatures]
                    for algo in adaptationAlgoNames:
                        Sa, Ta, t = getAdaptedData(algo,
                                                   Sx[:, F], Sy, Tx[:, F], Ty)
                        knn = KNeighborsClassifier(n_neighbors=1)
                        knn.fit(Sa, Sy)
                        acc = 100 * knn.score(Ta, Ty)
                        resTest[orderName][algo][j].append({"acc": acc,
                                                            "time": t})
            currentTime = time.time()
            print(".", end="")
        resTests[nameDA] = {"couplings": couplingDiags,
                            "resTest": resTest}

        currentTime = time.time()
        print(" {:6.2f}".format(currentTime - startTime) + "s")
    resFeatures[featureToUse] = {"nbrsFeatures": nbrsFeatures,
                                 "resTests": resTests}
dic = {"features": featuresToUse,
       "adaptationAlgoNames": adaptationAlgoNames,
       "orders": orderNames,
       "resFeatures": resFeatures}
f = open("./results/res" + str(time.time()) + ".pkl", "wb")
pickle.dump(dic, f)
f.close()
