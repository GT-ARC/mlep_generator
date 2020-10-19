# -*- coding: utf-8 -*-
"""
This program benchmarks algorithms on problem instances. In this case, specifically for machine learning classifiers.
Created on 08.01.2020
@author: christian.geissler@gt-arc.com
"""
#python basic imports
import logging
import os, sys
from pathlib import Path
import json
#3rd party imports (from packages, the environment)
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.neural_network import MLPClassifier
#custom
import util.config as cfg
from util.logging import setupLogging, shutdownLogging
from classifiers import *

VERSION = '2.0'
def classificationGenerator(classifierFactoryClass, n_samples, n_initialSamples, n_features, n_hidden_features, n_classes, challenge, scaling , repeats, seed):
    '''
    
        repeats - number of times a specific prototype examples should be duplicated. Helps especially for random forests and other algorithms that tend to ignore singled out examples as noise.
    '''
    random = np.random.RandomState(seed = seed)
    n_features = n_features + n_hidden_features

    if scaling == None:
        scaling = 1.0
    else:
        scaling = ( random.rand(n_features)*2 - 1 ) * scaling
    X_proto = random.rand(n_initialSamples, n_features) * scaling
    y_proto = [(i % n_classes) for i in range(n_initialSamples)]
    #y_proto = y_proto * (n_classes)
    #np.floor( y_proto, y_proto )
    #y_proto[y_proto >=n_classes] = n_classes-1
    X_proto = np.repeat(X_proto, repeats = repeats, axis = 0)
    y_proto = np.repeat(y_proto, repeats = repeats, axis = 0)
    
    uniques, counts = np.unique(y_proto, return_counts = True)
    #print("no of prototypical examples: ")
    #print(counts)
    
    #print("selected: "+str(classifierFactoryClass.__name__))
    classifierFactory = classifierFactoryClass()
    classifier = classifierFactory(X = X_proto, y = y_proto, seed = seed)

    X = random.rand(n_samples, n_features) * scaling
    y = classifier(X)
    
    #X = np.concatenate([X,X_proto], axis = 0)
    #y = np.concatenate([y, y_proto], axis = 0)
    #remove the hidden features:
    if (n_hidden_features > 0):
        choosenIndexes = np.arange(n_features)
        np.random.shuffle(choosenIndexes)
        choosenIndexes = choosenIndexes[:-n_hidden_features]
        X = X[:,choosenIndexes]
    return X, y, classifierFactoryClass.__name__
    
def classificationGeneratorWithQualityCheck(seed):
    random = np.random.RandomState(seed = seed)
    classifierFactoryClass = random.choice( OBOE_SET )
    
    n_min_features = 2
    n_max_features = 20
    
    n_max_classes = 5
    n_min_samples = 100 #per class
    n_max_samples = 5000 #per class
    
    n_min_init__samples = 1 #per class
    n_max_init_samples = 100 #per class
    
    repeats = 1
    
    #print("n_classes")
    counter = 0
    while True:
        n_features = np.random.randint(n_min_features,n_max_features+1)
        n_hidden_features = np.random.randint(0,2)
        challenge = np.random.rand()
        n_classes = int(2 + np.random.rand()*(n_max_classes-2))
        n_samples = int( ((np.random.rand()* (n_max_samples - n_min_init__samples)) + n_min_init__samples) * n_classes )
        n_initialSamples = int( n_classes * (n_min_samples+(n_max_init_samples-n_min_samples)*challenge) )
        scaling = (1.0  + random.rand() * 99)
        #Create artificial classification dataset
        #classifierFactoryClass = random.choice( OBOE_SET )
        allClassesRepresented = False
        allClassesRepresented2 = False
        #try:
        X, y, classifierFactoryName = classificationGenerator(classifierFactoryClass, n_samples, n_initialSamples, n_features, n_hidden_features, n_classes, challenge, scaling, repeats = repeats, seed = random.randint(10000000))
        uniques, counts = np.unique(y, return_counts = True)
        #check the quality conditions
        allClassesRepresented = np.all(counts > 0) and (len(uniques) == n_classes)
        #contain at least 1% of each class and at least 10 instances of each class (10-Fold Cross-Validation)
        minRepresentativeTreshold = np.max( [0.01 * n_samples, 10] )
        minNoOfInstance = np.min(counts)
        allClassesRepresented2 = minNoOfInstance>=minRepresentativeTreshold
        #print(counts)
        #stop the while loop if all conditions have been met
        #except Exception as exceptionInstance:
        #    logging.error( exceptionInstance, exc_info = True )
        if (allClassesRepresented and allClassesRepresented2):
            break
            
        if ( allClassesRepresented == False ):
            print("The generated dataset did not contain all the desired classes")
            
        if ( allClassesRepresented2 == False ):
            print("The generated dataset did not contain enough instances for all the desired classes")
            
        print("Generation info: n_samples: "+str(n_samples)+" n_features: "+str(n_features)+" n_hidden_features: "+str(n_hidden_features)+" n_classes: "+str(n_classes)+" challenge: "+str(challenge))
        #print(classifierFactoryClass.__name__)
        #reduce the difficulty of the generation randomly:
        
        #in case we did not find a dataset, we by chance increase the number of duplicates in the prototype examples to counter classifiers that filter out noise too strongly (and therefore converge to a single prediction that does not help to generate datasets that contain instances of all the classes).
        switch = random.randint(0,5)
        if (switch == 0):
            repeats = repeats * 2
        '''
        if (switch == 1) and (n_classes > 2):
            n_classes -= 1
        if (switch == 2) and (n_features > 1):
            n_features -= 1
        if (switch == 3) and (n_hidden_features > 0):
            n_hidden_features -= 1
        if (switch == 4) and n_samples < (1000 * n_classes):
            n_samples = int( np.max( [0.5 * n_samples, 100 * n_classes]) )
            n_initialSamples *= 2
        '''
        counter += 1
        
        if (counter % 100 == 0) and ( counter>0) :
            print("selected classifier factory: "+str(classifierFactoryClass.__name__))
            for yi in np.unique(y):
                yMask = (y == yi)
                x0 = X[yMask,0]
                x1 = X[yMask,1]
                plt.scatter(x = x0, y = x1)
            plt.title("Randomly generated dataset by "+classifierFactoryName)
            plt.xlabel('x0', fontsize=6)
            plt.ylabel('x1', fontsize=6)
            plt.show()
            
    #return the generated dataset
    return {'X':X.tolist(), 'y':y.tolist(), 'n_samples':n_samples, 'n_features':n_features, 'n_hidden_features':n_hidden_features, 'n_classes':n_classes, 'challenge':challenge, 'seed':seed, 'classifierFactoryName':classifierFactoryName, 'generatorVersion':VERSION}
    
I = 10000
for i in range(I):
    print(str(i+1)+'/'+str(I)+' = '+str(int(100*(i+1)/I))+" %")
    
    seed = np.random.randint(10000000)
    result = classificationGeneratorWithQualityCheck(seed = seed)
    
    #store dataset:
    
    basedir = cfg.tmpDir + 'artificial_datasets2/'
    filedir = basedir + str(i) + '/'
    filepath = filedir + 'dataset.json'
    Path(filedir).mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as outfile:
        json.dump(result, outfile)
    
    '''
    X = np.array(result['X'])
    y = np.array(result['y'])
    classifierFactoryName = result['classifierFactoryName']
    for yi in np.unique(y):
        yMask = (y == yi)
        x0 = X[yMask,0]
        x1 = X[yMask,1]
        plt.scatter(x = x0, y = x1)
    plt.title("Randomly generated dataset by "+classifierFactoryName)
    plt.xlabel('x0', fontsize=6)
    plt.ylabel('x1', fontsize=6)
    plt.show()
    '''

    