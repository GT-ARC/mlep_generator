# -*- coding: utf-8 -*-
"""
This program benchmarks algorithms on problem instances. In this case, specifically for machine learning classifiers.
Created on 08.01.2020
@author: christian.geissler@gt-arc.com
"""
#python basic imports
import logging
import os, sys
#3rd party imports (from packages, the environment)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

    
#custom
import util.config as cfg
from util.logging import setupLogging, shutdownLogging
from classifiers import *


challenge = 0.#np.random.rand(1)#[0,1] 0 simplest, 1 hardest.

n_classes = int( 2 + np.random.rand() * challenge * 100 )
n_features = 2


def distributeIntoDiscretePartititions(weights, discretes):
    print(weights)
    print(discretes)
    totalWeights = np.sum(weights)
    fractions = (weights * discretes) / totalWeights
    discreteFractions = np.around(fractions).astype(dtype=int)
    while np.sum(discreteFractions) < discretes:
        discreteFractions[np.argmax( fractions - discreteFractions )] += 1
        print(discreteFractions)
    print(discreteFractions)
    return discreteFractions


[n_informative, n_redundant, n_repeated] = distributeIntoDiscretePartititions(np.random.rand(3), n_features)

n_samples_per_class = int( 100 * (1 / (1 + challenge)) * n_features )
n_samples = n_samples_per_class * n_classes
n_clusters_per_class = int(np.ceil( 0.5 + ( 2**n_informative / n_classes ) * challenge ))

while ( n_classes * n_clusters_per_class > 2** n_informative):
    if ( n_clusters_per_class > 1 ):
        n_clusters_per_class -= 1
    elif( n_classes > 2 ):
        n_classes -= 1
    else:
        n_informative += 1
        n_features += 1
        

class_sep = 2 / (1.0 + challenge)

print("challenge: "+str(challenge))
print("n_classes: "+str(n_classes))
print("n_features: "+str(n_features))
print("n_informative: "+str(n_informative))
print("n_redundant: "+str(n_redundant))
print("n_repeated: "+str(n_repeated))
print("n_samples_per_class: "+str(n_samples_per_class))
print("n_samples: "+str(n_samples))
print("n_clusters_per_class: "+str(n_clusters_per_class))

X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, weights=None, flip_y=0.01, class_sep=class_sep, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)

randomTransformationMatrix = np.random.rand(*np.shape(X.T))
randomTransformationMatrix = randomTransformationMatrix / np.linalg.det(randomTransformationMatrix)
X = X * randomTransformationMatrix

for yi in np.unique(y):
    yMask = (y == yi)
    x0 = X[yMask,0]
    x1 = X[yMask,1]
    plt.scatter(x = x0, y = x1)
plt.title("The first two features of a dataset")
plt.xlabel('x0', fontsize=6)
plt.ylabel('x1', fontsize=6)
plt.show()