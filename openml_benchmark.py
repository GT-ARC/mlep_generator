# -*- coding: utf-8 -*-
"""
Created on 08.01.2020
@author: christian.geissler@gt-arc.com
"""
#python basic imports
import logging
#3rd party imports (from packages, the environment)
import numpy as np
import openml
import pmlb
from sklearn.model_selection import train_test_split
#custom (local) imports
from concepts import *
from classification import *

 
class GenericSklearnModel(Classifier):
    def __init__(self, model):
        super(GenericSklearnModel, self).__init__()
        self.model = model
        
    def __call__(self, X):
        return self.model.predict(X) 
        
class SKLearnClassificationFactory(ClassifierFactory):
    '''
    This class's purpose is to bundle the set of Sklear Classification Factories as it's children.
    '''


from sklearn.linear_model import LinearRegression
class LinearRegressionClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y):
        return GenericSklearnModel( LinearRegression().fit(X, y) )
        
from sklearn.linear_model import Ridge
class RidgeRegressionClassifier(SKLearnClassificationFactory):
    def __call__(self, X, y):
        return GenericSklearnModel( Ridge(alpha=.5).fit(X, y) )


class OpenMLTask(ClassificationTask):
    def getX():
        return NotImplemented
        
    def getY():        
        return NotImplemented
        
class OpenMLBenchmark(Benchmark):
    def __init__(self):
        super(OpenMLBenchmark, self).__init__()
        
        benchmark_suite_id = 'OpenML-CC18'
        benchmark_suite = openml.study.get_suite(benchmark_suite_id)
        
        self.tasks = list()

        for task_id in benchmark_suite.tasks:
            task = openml.tasks.get_task(task_id)
    
            X, Y = task.get_X_and_y()
            self.tasks.append( type('OpenMLTask_'+str(task_id), (ClassificationTask,), {'getX': lambda self: X, 'getY': lambda self: Y}) )
        
    def getAlgorithms(self):
        return SKLearnClassificationFactory.__subclasses__()
        
    def getTasks(self):
        return self.tasks
        
    def getEvaluations(self):
        return [Classifier3FoldEvaluation()]#ClassifierEvaluation.__subclasses__()
        
    def getSeeds(self):
        return range(100)
        
class PennMLBenchmark(Benchmark):
    def __init__(self):
        super(PennMLBenchmark, self).__init__()
        self.tasks = list()
        print("Download datasets...")
        total = len(pmlb.classification_dataset_names)
        counter = 0
        for task_id in pmlb.classification_dataset_names:
            X, Y = pmlb.fetch_data(task_id, return_X_y=True, local_cache_dir=self.tmpDirectory)
            self.tasks.append( type('PennMLTask_'+str(task_id), (ClassificationTask,), {'getX': lambda self: X, 'getY': lambda self: Y}) )
            counter = counter + 1
            if (counter % 10 == 0):
                print(str( int(100 * counter / total) )+"%")
        
    def getAlgorithms(self):
        return [cf() for cf in SKLearnClassificationFactory.__subclasses__()]
        
    def getTasks(self):
        return self.tasks
        
    def getEvaluations(self):
        return [Classifier3FoldEvaluation()]#ClassifierEvaluation.__subclasses__()
        
    def getSeeds(self):
        return range(100)



    #classInstance = myNewClass()
    
    #task.get_train_test_split_indices()
    #task.get_dataset() #datasets.OpenMLDataset -> datasets.get_dataset(self.dataset_id)
    #dataset = task.get_dataset()
    #X = dataset.get_data()
    
    
benchmark = PennMLBenchmark()

print("Starting Benchmark")
print(" Algorithms:")
for a in benchmark.getAlgorithms():
    print(" "+str(a.__name__))
print(" Tasks:")
for a in benchmark.getTasks():
    print(" "+str(a.__name__))
print(" Evaluations:")
for a in benchmark.getEvaluations():
    print(" "+str(a.__name__))

benchmark.run()