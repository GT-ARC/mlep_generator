# -*- coding: utf-8 -*-
"""
Created on 30.01.2020
@author: christian.geissler@gt-arc.com
"""

#python basic imports
import math, sys
#3rd party imports (from packages, the environment)
from sklearn.metrics import accuracy_score, r2_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, Perceptron
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
import numpy as np
#custom imports
import util.config as cfg


'''
Sklearn set of algorithms

'''
class GenericSklearnModel():
    def __init__(self, model):
        super(GenericSklearnModel, self).__init__()
        self.model = model
        
    def __call__(self, X):
        return self.model.predict(X) 
        
class SKLearnClassificationFactory():
    '''
    This class's purpose is to bundle the set of Sklear Classification Factories as it's children.
    '''


class DummyMajorityClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( DummyClassifier(strategy='most_frequent', random_state=seed, constant=None).fit(X, y) )
        
class RidgeRegressionClassifierCV(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( RidgeClassifierCV(alphas=(0.1, 0.3, 0.9, 2.7, 81), scoring = scorer, cv = 3).fit(X, y) )
        
'''       
#Generically create Ridge Regression Classifiers equal to RidgeRegressionClassifierCV (to beeing able to run specific ones seperately)

RIDGE_REGRESSION_CLASSIFIERS = []
alphas=(0.1, 0.3, 0.9, 2.7, 81)
for alpha in alphas:
    functions = {'__call__': ( lambda self, X, y, seed: GenericSklearnModel( RidgeClassifier(alpha=alpha, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=seed).fit(X, y)))}
    genericClassObject = type('RidgeRegressionClassifier_alpha'+str(alpha), (SKLearnClassificationFactory,), functions)
    RIDGE_REGRESSION_CLASSIFIERS.append( genericClassObject )
'''

class RidgeRegressionClassifier_alpha01(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( RidgeClassifier(alpha=0.1, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=seed).fit(X, y) )

class RidgeRegressionClassifier_alpha03(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( RidgeClassifier(alpha=0.3, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=seed).fit(X, y) )

class RidgeRegressionClassifier_alpha09(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( RidgeClassifier(alpha=0.9, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=seed).fit(X, y) )

class RidgeRegressionClassifier_alpha27(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( RidgeClassifier(alpha=2.7, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=seed).fit(X, y) )

class RidgeRegressionClassifier_alpha81(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( RidgeClassifier(alpha=81, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=seed).fit(X, y) )

RIDGE_REGRESSION_CLASSIFIERS = [RidgeRegressionClassifier_alpha01, RidgeRegressionClassifier_alpha03, RidgeRegressionClassifier_alpha09, RidgeRegressionClassifier_alpha27, RidgeRegressionClassifier_alpha81]

from sklearn.linear_model import LogisticRegressionCV
class LogisticRegressionClassifierLBFGSCV(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( LogisticRegressionCV(Cs=10, solver='lbfgs', max_iter = 100, multi_class='auto', scoring = scorer, cv = 3, random_state = seed).fit(X, y) )

class LogisticRegressionClassifierSAGACV(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( LogisticRegressionCV(Cs=10, solver='saga', max_iter = 100, multi_class='auto', scoring = scorer, cv = 3, random_state = seed).fit(X, y) )

class LogisticRegressionClassifierLiblinearCV(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        scorer = make_scorer(r2_score)
        return GenericSklearnModel( LogisticRegressionCV(Cs=10, solver='liblinear', max_iter = 100, multi_class='auto', scoring = scorer, cv = 3, random_state = seed).fit(X, y) )
        
class LogisticRegressionClassifier(SKLearnClassificationFactory):
    '''
    Generic Logistic Regression Classifier with LBFGS
    '''
    def __init__(self, cstep, csteps, solver):
        super().__init__()
        #set c between 1e-4 and 1e4, steps: 10, logarithmic
        self.c = math.exp(  ((cstep-1)/(csteps-1)) * 8 - 4)
        self.solver = solver
        
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=self.c, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=self.solver, max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None).fit(X, y) )
        
def LogisticRegressionClassifierLBFGS1(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=1, csteps=10, solver = 'lbfgs')
        
def LogisticRegressionClassifierLBFGS2(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=2, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS3(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=3, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS4(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=4, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS5(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=5, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS6(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=6, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS7(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=7, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS8(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=8, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS9(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=9, csteps=10, solver = 'lbfgs')

def LogisticRegressionClassifierLBFGS10(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=10, csteps=10, solver = 'lbfgs')
        
def LogisticRegressionClassifierSAGA1(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=1, csteps=10, solver = 'saga')
        
def LogisticRegressionClassifierSAGA2(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=2, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA3(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=3, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA4(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=4, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA5(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=5, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA6(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=6, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA7(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=7, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA8(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=8, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA9(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=9, csteps=10, solver = 'saga')

def LogisticRegressionClassifierSAGA10(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=10, csteps=10, solver = 'saga')
        
def LogisticRegressionClassifierLiblinear1(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=1, csteps=10, solver = 'liblinear')
        
def LogisticRegressionClassifierLiblinear2(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=2, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear3(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=3, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear4(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=4, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear5(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=5, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear6(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=6, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear7(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=7, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear8(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=8, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear9(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=9, csteps=10, solver = 'liblinear')

def LogisticRegressionClassifierLiblinear10(LogisticRegressionClassifierLBFGS):
    def __init__(self):
        super().__init__(cstep=10, csteps=10, solver = 'liblinear')

LOGISTIC_REGRESSION_LBFGS_CLASSIFIERS = [getattr(sys.modules[__name__], 'LogisticRegressionClassifierLBFGS'+str(i)) for i in range(1,10)]
LOGISTIC_REGRESSION_SAGA_CLASSIFIERS = [getattr(sys.modules[__name__], 'LogisticRegressionClassifierSAGA'+str(i)) for i in range(1,10)]
LOGISTIC_REGRESSION_LIBLINEAR_CLASSIFIERS = [getattr(sys.modules[__name__], 'LogisticRegressionClassifierLiblinear'+str(i)) for i in range(1,10)]
LOGISTIC_REGRESSION_CLASSIFIERS = LOGISTIC_REGRESSION_LBFGS_CLASSIFIERS + LOGISTIC_REGRESSION_SAGA_CLASSIFIERS + LOGISTIC_REGRESSION_LIBLINEAR_CLASSIFIERS

class LogisticRegressionClassifierOBOE(SKLearnClassificationFactory):
    '''
    Generic Logistic Regression Classifier
    '''
    def __init__(self, c, solver, penalty):
        super().__init__()
        self.c = c
        self.solver = solver
        self.penalty = 'l1' or 'l2'
        
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LogisticRegression(penalty=self.penalty, C=self.c, multi_class = 'auto', random_state=seed, solver=self.solver).fit(X, y) )
        
        
class LogisticRegressionClassifierOBOEC025SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.25, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC05SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.5, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC075SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.75, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC1SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC15SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1.5, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC2SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=2, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC3SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=3, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC4SLiblinearL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=4, solver = 'liblinear', penalty='l1')
        
class LogisticRegressionClassifierOBOEC025SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.25, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC05SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.5, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC075SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.75, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC1SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC15SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1.5, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC2SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=2, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC3SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=3, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC4SSagaL1(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=4, solver = 'saga', penalty='l1')
        
class LogisticRegressionClassifierOBOEC025SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.25, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC05SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.5, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC075SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.75, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC1SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC15SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1.5, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC2SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=2, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC3SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=3, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC4SLiblinearL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=4, solver = 'liblinear', penalty='l2')
        
class LogisticRegressionClassifierOBOEC025SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.25, solver = 'saga', penalty='l2')
        
class LogisticRegressionClassifierOBOEC05SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.5, solver = 'saga', penalty='l2')
        
class LogisticRegressionClassifierOBOEC075SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=0.75, solver = 'saga', penalty='l2')
        
class LogisticRegressionClassifierOBOEC1SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1, solver = 'saga', penalty='l2')
        
class LogisticRegressionClassifierOBOEC15SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=1.5, solver = 'saga', penalty='l2')
        
class LogisticRegressionClassifierOBOEC2SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=2, solver = 'saga', penalty='l2')
        
class LogisticRegressionClassifierOBOEC3SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=3, solver = 'saga', penalty='l2')
        
class LogisticRegressionClassifierOBOEC4SSagaL2(LogisticRegressionClassifierOBOE):
    def __init__(self):
        super().__init__(c=4, solver = 'saga', penalty='l2')
        
LOGISTIC_REGRESSION_CLASSIFIERS_OBOE = [LogisticRegressionClassifierOBOEC025SLiblinearL1, LogisticRegressionClassifierOBOEC05SLiblinearL1, LogisticRegressionClassifierOBOEC075SLiblinearL1, LogisticRegressionClassifierOBOEC1SLiblinearL1, LogisticRegressionClassifierOBOEC15SLiblinearL1, LogisticRegressionClassifierOBOEC2SLiblinearL1, LogisticRegressionClassifierOBOEC3SLiblinearL1, LogisticRegressionClassifierOBOEC4SLiblinearL1, LogisticRegressionClassifierOBOEC025SSagaL1, LogisticRegressionClassifierOBOEC05SSagaL1, LogisticRegressionClassifierOBOEC075SSagaL1, LogisticRegressionClassifierOBOEC1SSagaL1, LogisticRegressionClassifierOBOEC15SSagaL1, LogisticRegressionClassifierOBOEC2SSagaL1, LogisticRegressionClassifierOBOEC3SSagaL1, LogisticRegressionClassifierOBOEC4SSagaL1, LogisticRegressionClassifierOBOEC025SLiblinearL2, LogisticRegressionClassifierOBOEC05SLiblinearL2, LogisticRegressionClassifierOBOEC075SLiblinearL2, LogisticRegressionClassifierOBOEC1SLiblinearL2, LogisticRegressionClassifierOBOEC15SLiblinearL2, LogisticRegressionClassifierOBOEC2SLiblinearL2, LogisticRegressionClassifierOBOEC3SLiblinearL2, LogisticRegressionClassifierOBOEC4SLiblinearL2, LogisticRegressionClassifierOBOEC025SSagaL2, LogisticRegressionClassifierOBOEC05SSagaL2, LogisticRegressionClassifierOBOEC075SSagaL2, LogisticRegressionClassifierOBOEC1SSagaL2, LogisticRegressionClassifierOBOEC15SSagaL2, LogisticRegressionClassifierOBOEC2SSagaL2, LogisticRegressionClassifierOBOEC3SSagaL2, LogisticRegressionClassifierOBOEC4SSagaL2]
        
class KNeighborsKDTClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        parameters = {'n_neighbors':[1, 2, 3, 4, 5, 10]}
        scorer = make_scorer(accuracy_score)
        gridSearch = GridSearchCV(KNeighborsClassifier(algorithm = 'kd_tree'), parameters, iid = False, scoring = scorer, cv = 3)
        return GenericSklearnModel( gridSearch.fit(X, y) )
        
class KNeighborsKDTClassifier1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'kd_tree', n_neighbors = 1).fit(X, y) )
        
class KNeighborsKDTClassifier2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'kd_tree', n_neighbors = 2).fit(X, y) )
        
class KNeighborsKDTClassifier3(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'kd_tree', n_neighbors = 3).fit(X, y) )
        
class KNeighborsKDTClassifier4(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'kd_tree', n_neighbors = 4).fit(X, y) )
        
class KNeighborsKDTClassifier5(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'kd_tree', n_neighbors = 5).fit(X, y) )
        

     
class KNeighborsKDTClassifier10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'kd_tree', n_neighbors = 10).fit(X, y) )


        
KNEIGHBORS_KDT_CLASSIFIERS = [KNeighborsKDTClassifier1, KNeighborsKDTClassifier2, KNeighborsKDTClassifier3, KNeighborsKDTClassifier4, KNeighborsKDTClassifier5, KNeighborsKDTClassifier10]

class KNeighborsClassifierN1P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=1, p=1, ).fit(X,y) )

class KNeighborsClassifierN3P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=3, p=1, ).fit(X,y) )

class KNeighborsClassifierN5P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=5, p=1, ).fit(X,y) )

class KNeighborsClassifierN7P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=7, p=1, ).fit(X,y) )

class KNeighborsClassifierN9P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=9, p=1, ).fit(X,y) )

class KNeighborsClassifierN11P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=11, p=1, ).fit(X,y) )

class KNeighborsClassifierN13P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=13, p=1, ).fit(X,y) )

class KNeighborsClassifierN15P1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=15, p=1, ).fit(X,y) )

class KNeighborsClassifierN1P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=1, p=2, ).fit(X,y) )

class KNeighborsClassifierN3P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=3, p=2, ).fit(X,y) )

class KNeighborsClassifierN5P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=5, p=2, ).fit(X,y) )

class KNeighborsClassifierN7P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=7, p=2, ).fit(X,y) )

class KNeighborsClassifierN9P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=9, p=2, ).fit(X,y) )

class KNeighborsClassifierN11P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=11, p=2, ).fit(X,y) )

class KNeighborsClassifierN13P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=13, p=2, ).fit(X,y) )

class KNeighborsClassifierN15P2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(n_neighbors=15, p=2, ).fit(X,y) )


KNEIGHBORS_KDT_CLASSIFIERS_OBOE = [KNeighborsClassifierN1P1, KNeighborsClassifierN3P1, KNeighborsClassifierN5P1, KNeighborsClassifierN7P1, KNeighborsClassifierN9P1, KNeighborsClassifierN11P1, KNeighborsClassifierN13P1, KNeighborsClassifierN15P1, KNeighborsClassifierN1P2, KNeighborsClassifierN3P2, KNeighborsClassifierN5P2, KNeighborsClassifierN7P2, KNeighborsClassifierN9P2, KNeighborsClassifierN11P2, KNeighborsClassifierN13P2, KNeighborsClassifierN15P2]

class KNeighborsBTClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        parameters = {'n_neighbors':[1, 2, 3, 4, 5, 10]}
        scorer = make_scorer(accuracy_score)
        gridSearch = GridSearchCV(KNeighborsClassifier(algorithm = 'ball_tree'), parameters, iid = False, scoring = scorer, cv = 3)
        return GenericSklearnModel( gridSearch.fit(X, y) )    
        
class KNeighborsBTClassifier1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'ball_tree', n_neighbors = 1).fit(X, y) )
        
class KNeighborsBTClassifier2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'ball_tree', n_neighbors = 2).fit(X, y) )

class KNeighborsBTClassifier3(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'ball_tree', n_neighbors = 3).fit(X, y) )

class KNeighborsBTClassifier4(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'ball_tree', n_neighbors = 4).fit(X, y) )

class KNeighborsBTClassifier5(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'ball_tree', n_neighbors = 5).fit(X, y) )

class KNeighborsBTClassifier10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'ball_tree', n_neighbors = 10).fit(X, y) )

KNEIGHBORS_BT_CLASSIFIERS = [KNeighborsBTClassifier1, KNeighborsBTClassifier2, KNeighborsBTClassifier3, KNeighborsBTClassifier4, KNeighborsBTClassifier5, KNeighborsBTClassifier10]
         
class KNeighborsBFClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        parameters = {'n_neighbors':[1, 2, 3, 4, 5, 10]}
        scorer = make_scorer(accuracy_score)
        gridSearch = GridSearchCV(KNeighborsClassifier(algorithm = 'brute'), parameters, iid = False, scoring = scorer, cv = 3)
        return GenericSklearnModel( gridSearch.fit(X, y) )

class KNeighborsBFClassifier1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'brute', n_neighbors = 1).fit(X, y) )

class KNeighborsBFClassifier2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'brute', n_neighbors = 2).fit(X, y) )

class KNeighborsBFClassifier3(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'brute', n_neighbors = 3).fit(X, y) )

class KNeighborsBFClassifier4(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'brute', n_neighbors = 4).fit(X, y) )

class KNeighborsBFClassifier5(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'brute', n_neighbors = 5).fit(X, y) )

class KNeighborsBFClassifier10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( KNeighborsClassifier(algorithm = 'brute', n_neighbors = 10).fit(X, y) )

KNEIGHBORS_BF_CLASSIFIERS = [KNeighborsBFClassifier1, KNeighborsBFClassifier2, KNeighborsBFClassifier3, KNeighborsBFClassifier4, KNeighborsBFClassifier5, KNeighborsBFClassifier10]
KNEIGHBORS_CLASSIFIERS = KNEIGHBORS_KDT_CLASSIFIERS + KNEIGHBORS_BT_CLASSIFIERS + KNEIGHBORS_BF_CLASSIFIERS

class RadiusNeighborsKDTClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        distances, indices = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X).kneighbors(X)
        d = np.max( distances[:,1] ) #distances of the closest point that is not itself (index 0 is the distance to itself)
        parameters = {'radius':[2*d, 3*d, 4*d, 5*d]}
        scorer = make_scorer(accuracy_score)
        gridSearch = GridSearchCV(RadiusNeighborsClassifier(algorithm = 'kd_tree'), parameters, iid = False, scoring = scorer, cv = 3)
        return GenericSklearnModel( gridSearch.fit(X, y) )  
      
class RadiusNeighborsBTClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        distances, indices = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X).kneighbors(X)
        d = np.max( distances[:,1] ) #distances of the closest point that is not itself (index 0 is the distance to itself)
        parameters = {'radius':[2*d, 3*d, 4*d, 5*d]}
        scorer = make_scorer(accuracy_score)
        gridSearch = GridSearchCV(RadiusNeighborsClassifier(algorithm = 'ball_tree'), parameters, iid = False, scoring = scorer, cv = 3)
        return GenericSklearnModel( gridSearch.fit(X, y) )  
  
class RadiusNeighborsBFClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        distances, indices = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X).kneighbors(X)
        d = np.max( distances[:,1] ) #distances of the closest point that is not itself (index 0 is the distance to itself)
        parameters = {'radius':[2*d, 3*d, 4*d, 5*d]}
        scorer = make_scorer(accuracy_score)
        gridSearch = GridSearchCV(RadiusNeighborsClassifier(algorithm = 'brute'), parameters, iid = False, scoring = scorer, cv = 3)
        return GenericSklearnModel( gridSearch.fit(X, y) )
  
class LinearSupportVectorClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        svc = SVC(kernel="linear", C=0.025, max_iter = 100, gamma = 'scale', random_state = seed)
        return GenericSklearnModel( svc.fit(X,y) )

class LinearSupportVectorClassifierGridSearch(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        parameters = {'C':[0.025, 0.05, 0.1, 0.2, 0.4, 0.8]}
        scorer = make_scorer(accuracy_score)
        gridSearch = GridSearchCV(SVC(kernel="linear", max_iter = 100, gamma = 'scale', random_state = seed), parameters, iid = False, scoring = scorer, cv = 3)
        return GenericSklearnModel( gridSearch.fit(X,y) )        

class LinearSupportVectorClassifier005(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="linear", C=0.05, max_iter = 100, gamma = 'scale', random_state = seed).fit(X,y) )
        
class LinearSupportVectorClassifier01(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="linear", C=0.1, max_iter = 100, gamma = 'scale', random_state = seed).fit(X,y) )
        
class LinearSupportVectorClassifier02(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="linear", C=0.2, max_iter = 100, gamma = 'scale', random_state = seed).fit(X,y) )
        
class LinearSupportVectorClassifier04(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="linear", C=0.4, max_iter = 100, gamma = 'scale', random_state = seed).fit(X,y) )
        
class LinearSupportVectorClassifier08(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="linear", C=0.8, max_iter = 100, gamma = 'scale', random_state = seed).fit(X,y) )

LINEAR_SUPPORT_VECTOR_CLASSIFIERS = [LinearSupportVectorClassifier, LinearSupportVectorClassifier005, LinearSupportVectorClassifier01, LinearSupportVectorClassifier02, LinearSupportVectorClassifier04, LinearSupportVectorClassifier08]
        
class RBFSupportVectorClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        svc = SVC(gamma = 2, C=1, max_iter = 100, random_state = seed)
        return GenericSklearnModel( svc.fit(X,y) )      
        
class GPClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GaussianProcessClassifier(1.0 * RBF(1.0), random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class DTreeClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(max_depth = 5, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class DTreeClassifier2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 2, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class DTreeClassifier4(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 4, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class DTreeClassifier8(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 8, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier16(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 16, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier32(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 32, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier64(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 64, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier128(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 128, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier256(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 256, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier512(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 512, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier1024(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 1024, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 0.01, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier0001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 0.001, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 0.0001, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class DTreeClassifier000001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = DecisionTreeClassifier(min_samples_split = 0.00001, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

DECISION_TREE_CLASSIFIERS = [ DTreeClassifier2, DTreeClassifier4, DTreeClassifier8, DTreeClassifier16, DTreeClassifier32, DTreeClassifier64, DTreeClassifier128, DTreeClassifier256, DTreeClassifier512, DTreeClassifier1024, DTreeClassifier001, DTreeClassifier0001, DTreeClassifier00001, DTreeClassifier000001 ]

class ETreeClassifier2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 2, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ETreeClassifier4(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 4, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ETreeClassifier8(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 8, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier16(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 16, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier32(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 32, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier64(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 64, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier128(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 128, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier256(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 256, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier512(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 512, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier1024(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 1024, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 0.01, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier0001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 0.001, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 0.0001, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ETreeClassifier000001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = ExtraTreeClassifier(min_samples_split = 0.00001, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

EXTRA_TREE_CLASSIFIERS = [ ETreeClassifier2, ETreeClassifier4, ETreeClassifier8, ETreeClassifier16, ETreeClassifier32, ETreeClassifier64, ETreeClassifier128, ETreeClassifier256, ETreeClassifier512, ETreeClassifier1024, ETreeClassifier001, ETreeClassifier0001, ETreeClassifier00001, ETreeClassifier000001 ]
   
class RFClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(max_depth = 5, n_estimators=10, max_features=1, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class MultiLayerPerceptronClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = MLPClassifier(alpha=1, max_iter=1000, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )


class ABClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier50LR1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier100LR1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.0, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier50LR15(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.5, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier100LR15(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=1.5, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier50LR2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=2.0, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier100LR2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=2.0, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ABClassifier50LR25(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=2.5, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier100LR25(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=2.5, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class ABClassifier50LR3(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=3.0, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class ABClassifier100LR3(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=3.0, algorithm='SAMME.R', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

ADABOOST_CLASSIFIERS = [ABClassifier50LR1, ABClassifier100LR1, ABClassifier50LR15, ABClassifier100LR15, ABClassifier50LR2, ABClassifier100LR2, ABClassifier50LR25, ABClassifier100LR25, ABClassifier50LR3, ABClassifier100LR3]


class GBoostingClassifierLR0001MD3MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.001, max_depth = 3, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR0001MD3MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.001, max_depth = 3, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR001MD3MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.01, max_depth = 3, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR001MD3MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.01, max_depth = 3, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR0025MD3MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.025, max_depth = 3, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR0025MD3MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.025, max_depth = 3, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR005MD3MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.05, max_depth = 3, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR005MD3MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.05, max_depth = 3, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR01MD3MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.1, max_depth = 3, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR01MD3MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.1, max_depth = 3, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR025MD3MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.25, max_depth = 3, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR025MD3MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.25, max_depth = 3, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR05MD3MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.5, max_depth = 3, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR05MD3MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.5, max_depth = 3, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR0001MF6MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.001, max_depth = 6, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR0001MF6MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.001, max_depth = 6, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR001MF6MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.01, max_depth = 6, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR001MF6MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.01, max_depth = 6, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR0025MF6MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.025, max_depth = 6, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR0025MF6MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.025, max_depth = 6, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR005MF6MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.05, max_depth = 6, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR005MF6MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.05, max_depth = 6, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR01MF6MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.1, max_depth = 6, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR01MF6MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.1, max_depth = 6, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR025MF6MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.25, max_depth = 6, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR025MF6MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.25, max_depth = 6, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class GBoostingClassifierLR05MF6MFNone(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.5, max_depth = 6, n_estimators = 10, max_features = None, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class GBoostingClassifierLR05MF6MFLog2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GradientBoostingClassifier(learning_rate=0.5, max_depth = 6, n_estimators = 10, max_features = 'log2', random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

GRADIENT_BOOSTING_CLASSIFIERS = [GBoostingClassifierLR0001MD3MFNone,GBoostingClassifierLR0001MD3MFLog2,GBoostingClassifierLR001MD3MFNone,GBoostingClassifierLR001MD3MFLog2,GBoostingClassifierLR0025MD3MFNone,GBoostingClassifierLR0025MD3MFLog2,GBoostingClassifierLR005MD3MFNone,GBoostingClassifierLR005MD3MFLog2,GBoostingClassifierLR01MD3MFNone,GBoostingClassifierLR01MD3MFLog2,GBoostingClassifierLR025MD3MFNone,GBoostingClassifierLR025MD3MFLog2,GBoostingClassifierLR05MD3MFNone,GBoostingClassifierLR05MD3MFLog2,GBoostingClassifierLR0001MF6MFNone,GBoostingClassifierLR0001MF6MFLog2,GBoostingClassifierLR001MF6MFNone,GBoostingClassifierLR001MF6MFLog2,GBoostingClassifierLR0025MF6MFNone,GBoostingClassifierLR0025MF6MFLog2,GBoostingClassifierLR005MF6MFNone,GBoostingClassifierLR005MF6MFLog2,GBoostingClassifierLR01MF6MFNone,GBoostingClassifierLR01MF6MFLog2,GBoostingClassifierLR025MF6MFNone,GBoostingClassifierLR025MF6MFLog2,GBoostingClassifierLR05MF6MFNone,GBoostingClassifierLR05MF6MFLog2]

class GaussianNBClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = GaussianNB()
        return GenericSklearnModel( cls.fit(X,y) )        

class QDAClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = QuadraticDiscriminantAnalysis(random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class MLPClassifier_OBOE_00001_SGD_00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.0001, learning_rate = 'adaptive', solver = 'sgd', alpha = 0.0001, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_0001_SGD_00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.001, learning_rate = 'adaptive', solver = 'sgd', alpha = 0.0001, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_001_SGD_00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.01, learning_rate = 'adaptive', solver = 'sgd', alpha = 0.0001, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_00001_ADAM_00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.0001, learning_rate = 'adaptive', solver = 'adam', alpha = 0.0001, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_0001_ADAM_00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.001, learning_rate = 'adaptive', solver = 'adam', alpha = 0.0001, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_001_ADAM_00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.01, learning_rate = 'adaptive', solver = 'adam', alpha = 0.0001, random_state = seed).fit(X,y) )
        
class MLPClassifier_OBOE_00001_SGD_001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.0001, learning_rate = 'adaptive', solver = 'sgd', alpha = 0.01, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_0001_SGD_001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.001, learning_rate = 'adaptive', solver = 'sgd', alpha = 0.01, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_001_SGD_001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.01, learning_rate = 'adaptive', solver = 'sgd', alpha = 0.01, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_00001_ADAM_001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.0001, learning_rate = 'adaptive', solver = 'adam', alpha = 0.01, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_0001_ADAM_001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.001, learning_rate = 'adaptive', solver = 'adam', alpha = 0.01, random_state = seed).fit(X,y) )

class MLPClassifier_OBOE_001_ADAM_001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( MLPClassifier(learning_rate_init = 0.01, learning_rate = 'adaptive', solver = 'adam', alpha = 0.01, random_state = seed).fit(X,y) )

        
MLP_CLASSIFIERS_OBOE = [MLPClassifier_OBOE_00001_SGD_00001, MLPClassifier_OBOE_0001_SGD_00001, MLPClassifier_OBOE_001_SGD_00001, MLPClassifier_OBOE_00001_ADAM_00001, MLPClassifier_OBOE_0001_ADAM_00001, MLPClassifier_OBOE_001_ADAM_00001, MLPClassifier_OBOE_00001_SGD_001, MLPClassifier_OBOE_0001_SGD_001, MLPClassifier_OBOE_001_SGD_001, MLPClassifier_OBOE_00001_ADAM_001, MLPClassifier_OBOE_0001_ADAM_001, MLPClassifier_OBOE_001_ADAM_001]


class Perceptron_Classifier(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( Perceptron(random_state = seed).fit(X,y) )

class RandomForestClassifier2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 2, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
class RandomForestClassifier4(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 4, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
class RandomForestClassifier8(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 8, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier16(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 16, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier32(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 32, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier64(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 64, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier128(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 128, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier256(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 256, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier512(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 512, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier1024(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 1024, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 0.01, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier0001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 0.001, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier00001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 0.0001, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )

class RandomForestClassifier000001(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        cls = RandomForestClassifier(min_samples_split = 0.00001, n_estimators = 10, random_state = seed)
        return GenericSklearnModel( cls.fit(X,y) )
        
RANDOM_FOREST_CLASSIFIERS_OBOE = [ RandomForestClassifier2, RandomForestClassifier4, RandomForestClassifier8, RandomForestClassifier16, RandomForestClassifier32, RandomForestClassifier64, RandomForestClassifier128, RandomForestClassifier256, RandomForestClassifier512, RandomForestClassifier1024, RandomForestClassifier001, RandomForestClassifier0001, RandomForestClassifier00001, RandomForestClassifier000001 ]
class KernelSVMClassifierRBFC0125C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.125, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC025C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.25, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC05C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.5, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC075C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.75, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC1C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=1, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC2C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=2, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC4C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=4, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC8C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=8, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC16C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=16, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )

class KernelSVMClassifierPOLYC0125C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.125, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC025C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.25, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC05C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.5, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC075C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.75, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC1C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=1, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC2C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=2, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC4C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=4, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC8C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=8, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC16C0(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=16, coef0 = 0, gamma = 'scale', random_state = seed).fit(X,y) )

class KernelSVMClassifierRBFC0125C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.125, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC025C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.25, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC05C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.5, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC075C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=0.75, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC1C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=1, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC2C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=2, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC4C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=4, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC8C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=8, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierRBFC16C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="rbf", C=16, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )

class KernelSVMClassifierPOLYC0125C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.125, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC025C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.25, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC05C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.5, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC075C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=0.75, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC1C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=1, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC2C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=2, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC4C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=4, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC8C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=8, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
class KernelSVMClassifierPOLYC16C10(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( SVC(kernel="poly", C=16, coef0 = 10, gamma = 'scale', random_state = seed).fit(X,y) )
        
KERNEL_SVM_CLASSIFIERS_OBOE = [KernelSVMClassifierRBFC0125C0, KernelSVMClassifierRBFC025C0, KernelSVMClassifierRBFC05C0, KernelSVMClassifierRBFC075C0, KernelSVMClassifierRBFC1C0, KernelSVMClassifierRBFC2C0, KernelSVMClassifierRBFC4C0, KernelSVMClassifierRBFC8C0, KernelSVMClassifierRBFC16C0, KernelSVMClassifierPOLYC0125C0, KernelSVMClassifierPOLYC025C0, KernelSVMClassifierPOLYC05C0, KernelSVMClassifierPOLYC075C0, KernelSVMClassifierPOLYC1C0, KernelSVMClassifierPOLYC2C0, KernelSVMClassifierPOLYC4C0, KernelSVMClassifierPOLYC8C0, KernelSVMClassifierPOLYC16C0, KernelSVMClassifierRBFC0125C10, KernelSVMClassifierRBFC025C10, KernelSVMClassifierRBFC05C10, KernelSVMClassifierRBFC075C10, KernelSVMClassifierRBFC1C10, KernelSVMClassifierRBFC2C10, KernelSVMClassifierRBFC4C10, KernelSVMClassifierRBFC8C10, KernelSVMClassifierRBFC16C10, KernelSVMClassifierPOLYC0125C10, KernelSVMClassifierPOLYC025C10, KernelSVMClassifierPOLYC05C10, KernelSVMClassifierPOLYC075C10, KernelSVMClassifierPOLYC1C10, KernelSVMClassifierPOLYC2C10, KernelSVMClassifierPOLYC4C10, KernelSVMClassifierPOLYC8C10, KernelSVMClassifierPOLYC16C10]

class LinearSVMClassifierC0125(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=0.125, random_state = seed).fit(X,y) )

class LinearSVMClassifierC025(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=0.25, random_state = seed).fit(X,y) )

class LinearSVMClassifierC05(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=0.5, random_state = seed).fit(X,y) )

class LinearSVMClassifierC075(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=0.75, random_state = seed).fit(X,y) )

class LinearSVMClassifierC1(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=1, random_state = seed).fit(X,y) )

class LinearSVMClassifierC2(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=2, random_state = seed).fit(X,y) )

class LinearSVMClassifierC4(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=4, random_state = seed).fit(X,y) )

class LinearSVMClassifierC8(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=8, random_state = seed).fit(X,y) )

class LinearSVMClassifierC16(SKLearnClassificationFactory):
    def __call__(self, X, y, seed):
        return GenericSklearnModel( LinearSVC(C=16, random_state = seed).fit(X,y) )
        
LINEAR_SVM_CLASSIFIERS_OBOE = [LinearSVMClassifierC0125, LinearSVMClassifierC025, LinearSVMClassifierC05, LinearSVMClassifierC075, LinearSVMClassifierC1, LinearSVMClassifierC2, LinearSVMClassifierC4, LinearSVMClassifierC8, LinearSVMClassifierC16 ]
       
#Set equivalent to the one used in   KDD'19 Oboe:Collaborative Filtering for AutoML Model Selection   
OBOE_SET = ADABOOST_CLASSIFIERS + DECISION_TREE_CLASSIFIERS + EXTRA_TREE_CLASSIFIERS + GRADIENT_BOOSTING_CLASSIFIERS + [GaussianNBClassifier] + KNEIGHBORS_KDT_CLASSIFIERS_OBOE + LOGISTIC_REGRESSION_CLASSIFIERS_OBOE + MLP_CLASSIFIERS_OBOE + [Perceptron_Classifier] + RANDOM_FOREST_CLASSIFIERS_OBOE + KERNEL_SVM_CLASSIFIERS_OBOE + LINEAR_SVM_CLASSIFIERS_OBOE