# -*- coding: utf-8 -*-
"""
This program benchmarks algorithms on problem instances. In this case, specifically for machine learning classifiers.
Created on 08.01.2020
@author: christian.geissler@gt-arc.com
"""
#python basic imports
import logging
import os, sys
import argparse
import time
from pathlib import Path
import json
import hashlib, inspect
import concurrent.futures
import multiprocessing
import random
#3rd party imports (from packages, the environment)
import numpy as np
import openml
import pmlb
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from scipy.spatial import distance_matrix
import pandas as pd
import scipy.sparse as sps

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
except ModuleNotFoundError as error:
    pass
    
#custom
import util.config as cfg
from util.logging import setupLogging, shutdownLogging
from classifiers import *


def executeEvaluation(args):
    self = args['self']
    algorithm_id = args['algorithm_id']
    task_id = args['task_id']
    evaluation_id = args['evaluation_id']
    seed = args['seed']
    try:
        result = self.run(algorithm_id, task_id, evaluation_id, seed)
    except Exception as exceptionInstance:
        self.logger.error( exceptionInstance, exc_info = True )
                
class Benchmark():
    def __init__(self):
        self.baseResultDirectory = cfg.resultDir
        self.tmpDirectory = cfg.tmpDir
        self.logger = logging.getLogger(__name__+"."+str(self.__class__.__name__))
        
    def getCodeHash(self):
        try:
            return self.__class__.codehash
        except:
            sha3 = hashlib.sha3_256()
            sha3.update( inspect.getsource(sys.modules[__name__]).encode('utf-8' ) )
            self.__class__.codehash = sha3.hexdigest()
        return self.__class__.codehash
        
    def getInstanceClassCodeHash(self, targetInstance):
        sha3 = hashlib.sha3_256()
        sha3.update( inspect.getsource(targetInstance.__class__).encode('utf-8' ) )
        return sha3.hexdigest()
        
    def getMergedCodeHash(self, instances):
        sha3 = hashlib.sha3_256()
        for i in instances:
            sha3.update( self.getInstanceClassCodeHash(i).encode('utf-8') )
        return sha3.hexdigest()
        
    def getDirAndPath(self, algorithm_Id, task_id, evaluation_id, seed):
        filedir = self.baseResultDirectory + str(self.__class__.__name__) + '/' + str(task_id)  + '/' + str(algorithm_Id) + '/' + str(evaluation_id) + '/'
        filepath = filedir + str(seed) + '.json'
        return filedir, filepath
        
    def loadResult(self, algorithm_id, task_id, evaluation_id, seed):
        '''
        returns the result if it exists
        '''
        filedir, filepath = self.getDirAndPath(algorithm_id, task_id, evaluation_id, seed)
        if os.path.isfile(filepath):
            with open(filepath, 'r') as infile:
                return json.load(infile)
        return None
        
    def run(self, algorithm_id, task_id, evaluation_id, seed, skipcalculation = False, verbose = True):
        '''
        Get the specific evaluation.
        skipcalulcation - If True, do not calculate the evaluation if it does not exist yet but return a None. Default:False
        '''
        filedir, filepath = self.getDirAndPath(algorithm_id, task_id, evaluation_id, seed)
        
        if (skipcalculation and os.path.isfile(filepath)==False):
            return None

        #Initialize the instance
        algorithm = self.algorithmProvider(algorithm_id)
        task = self.taskProvider(task_id)
        evaluate = self.evaluationProvider(evaluation_id)
        
        #hash = self.getCodeHash() #too sensitive.
        hash = self.getMergedCodeHash([algorithm, task, evaluate])

        if (verbose):
            self.logger.info("Current evaluation: "+filedir)
        if ( ((cfg.forceRecalculation == False) or (skipcalculation == True)) and ( os.path.isfile(filepath) ) ):
            with open(filepath, 'r') as infile:
                result = json.load(infile)
                #self.logger.info("Hash: "+str(hash)+" == "+str(result['hash'])+" -> "+str(hash == result['hash']))
                if (hash == result['hash']):
                    if (verbose):
                        self.logger.info("Skip already existing evaluation: "+filedir)
                    return result
                else:
                    if (verbose):
                        self.logger.info("Repeat already existing evaluation because the code changed: "+filedir)

        #Do the actual evaluation
        result = evaluate(algorithm, task, seed)
        
        #Store and return the results
        result['hash'] = hash
        
        Path(filedir).mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as outfile:
            json.dump(result, outfile)
        return result

    def runAll(self, processes = 1):
        self.logger.info("Starting "+str(self.__class__.__name__)+".runAll()")
        
        pool = multiprocessing.Pool(processes = processes)
        
        
        self.logger.info('seeds: '+str(self.allSeeds()))
        self.logger.info('no of seeds: '+str(np.shape(self.allSeeds())))
        self.logger.info('no of alg: '+str(len(self.allAlgorithms())))
        self.logger.info('no of tasks: '+str(len(self.allTasks())))
        self.logger.info('no of eval: '+str(len(self.allEvaluations())))
        total = len(self.allSeeds()) * len(self.allAlgorithms()) * len(self.allTasks()) * len(self.allEvaluations())
        self.logger.info('total: '+str(total))
        
        counter = 0
        nextProgressUpdate = time.time()
        progressUpdateInterval = 10
        
        taskList = []
        for seed in self.allSeeds():
            for algorithm_id in self.allAlgorithms():
                for task_id in self.allTasks():
                    for evaluation_id in self.allEvaluations():
                        arguments = {
                            'self':self,
                            'seed':seed,
                            'algorithm_id':algorithm_id,
                            'task_id':task_id,
                            'evaluation_id':evaluation_id,
                        }
                        taskList.append(arguments)
        

        for result in pool.imap_unordered(executeEvaluation, taskList, chunksize=1 ):    
            counter += 1
            if nextProgressUpdate < time.time():
                self.logger.info("Progress: "+str(counter)+"/"+str(total)+" = "+str(int(100*counter/total))+"%")
                nextProgressUpdate = time.time() + progressUpdateInterval

            
        self.logger.info("finished runAll()")
        
    def scatterPlot(self, title, x, y, xLabel = "", yLabel = ""):
        x = np.array(x)
        y = np.array(y)
        self.logger.info(title)
        self.logger.info(np.shape(x))
        self.logger.info(np.shape(y))
        #outputFile = args.destinationdir + 'diagram_' + diagram['title'] + '.png'
        plt.figure(figsize=(16,9))
        plt.scatter(x, y)
        plt.grid(True, alpha=0.3)
        plt.xlabel( xLabel )
        plt.ylabel( yLabel )
        plt.title( title )
        plt.draw()
        #plt.savefig(outputFile)
        plt.show()
        

    def loadExistingResults(self, noCache = False):
        if (not hasattr(self, '_loadExistingResults') or noCache):
            self.logger.info("collect information...") 
            total = len(self.allSeeds()) * len(self.allAlgorithms()) * len(self.allTasks()) * len(self.allEvaluations())
            counter = 0
            progressUpdateInterval = 10
            nextProgressUpdate = time.time()
            allresults = dict()
             
            for seed in self.allSeeds():
                allresults[seed] = dict()
                for algorithm_id in self.allAlgorithms():
                    algorithm_id = str(algorithm_id)
                    allresults[seed][algorithm_id] = dict()
                    for task_id in self.allTasks():
                        task_id = str(task_id)
                        allresults[seed][algorithm_id][task_id] = dict()
                        for evaluation_id in self.allEvaluations():
                            arguments = {
                                'self':self,
                                'seed':seed,
                                'algorithm_id':algorithm_id,
                                'task_id':task_id,
                                'evaluation_id':evaluation_id,
                            }
                            result = self.loadResult(algorithm_id, task_id, evaluation_id, seed)
                            if not(result is None):
                                counter += 1
                                allresults[seed][algorithm_id][task_id][evaluation_id] = result
                                
                            if nextProgressUpdate < time.time():
                                self.logger.info("Progress: "+str(counter)+"/"+str(total)+" = "+str(int(100*counter/total))+"%")
                                nextProgressUpdate = time.time() + progressUpdateInterval
            self.logger.info("Finished with "+str(counter)+"/"+str(total)+" = "+str(int(100*counter/total))+"% runs.")
            self._loadExistingResults = allresults
        return self._loadExistingResults
        
    def analyzeResults(self, allresults, noCache = False):
        if (not hasattr(self, '_analyzeResults') or noCache):
            self.logger.info("analyze results...")  
            algorithmRuns = dict()
            for algorithm_id in self.allAlgorithms():
                algorithmRuns[str(algorithm_id)] = 0
            
            taskRuns = dict()
            for task_id in self.allTasks():
                taskRuns[str(task_id)] = 0
            
            evaluationRuns = dict()
            for evaluation_id in self.allEvaluations():
                evaluationRuns[str(evaluation_id)] = 0
             
            noOfSeeds = len(self.allSeeds())
            noOfAlgorithms = len(self.allAlgorithms())
            noOfTasks = len(self.allTasks())
            noOfEvaluations = len(self.allEvaluations())
            total = noOfSeeds * noOfAlgorithms * noOfTasks * noOfEvaluations
            counter = 0
            
            for seed, d1 in allresults.items():
                for algorithm_id, d2 in d1.items():
                    for task_id, d3 in d2.items():
                        for evaluation_id, d4 in d3.items():
                            counter += 1
                            algorithmRuns[algorithm_id] += 1
                            taskRuns[task_id] += 1
                            evaluationRuns[evaluation_id] += 1
            
            removeTaskIdsList = list()
            for task_id, noOfRuns in taskRuns.items():
                if ( noOfRuns == 0 ):
                    logging.warning(str(task_id)+" has zero runs. This is unusual and looks like there is an implementation error in the benchmark. We remove this task for now to be able to continue.")
                    removeTaskIdsList.append(task_id)
            for task_id in removeTaskIdsList:
                del taskRuns[task_id]
                
            #figure out, which algorithms are completely evaluated:
            numOfRunsToBeComplete = len(self.allSeeds()) * len(taskRuns.keys()) * len(self.allEvaluations())
            completeAlgorithmIds = list()
            numOfRunsToHaveAtLeastOneSeedComplete = len(taskRuns.keys()) * len(self.allEvaluations())
            atLeastOneSeedCompleteAlgorithmIds = list()
            for algorithm_id, d in algorithmRuns.items():
                #print(str(algorithm_id)+": "+str(algorithmRuns[algorithm_id]) + " >= " + str(numOfRunsToHaveAtLeastOneSeedComplete))
                if algorithmRuns[algorithm_id] == numOfRunsToBeComplete:
                    completeAlgorithmIds.append(algorithm_id)
 
                if algorithmRuns[algorithm_id] >= numOfRunsToHaveAtLeastOneSeedComplete:
                    atLeastOneSeedCompleteAlgorithmIds.append(algorithm_id)
                    
            #create performance matrix:
            noOfCompleteAlgorithms = len( completeAlgorithmIds )
            scores = np.zeros((noOfCompleteAlgorithms, noOfTasks))
            costs = np.zeros((noOfCompleteAlgorithms, noOfTasks))
            errors = np.zeros((noOfCompleteAlgorithms, noOfTasks))
            timeouts = np.zeros((noOfCompleteAlgorithms, noOfTasks))
            numbers = np.zeros((noOfCompleteAlgorithms, noOfTasks))
            allcosts = list()
            allscores = list()
            
            for seed, d1 in allresults.items():
                ai = 0
                for algorithm_id, d2 in d1.items():
                    if (algorithm_id in completeAlgorithmIds):
                        ti = 0
                        for task_id, d3 in d2.items():
                            for evaluation_id, d4 in d3.items():
                                allcosts.append(d4['cost'])
                                allscores.append(d4['score'])
                                scores[ai,ti] += d4['score']
                                costs[ai,ti] += d4['cost']
                                errors[ai,ti] += d4['noOfErrors']
                                timeouts[ai,ti] += d4['noOfTimeouts']
                                numbers[ai,ti] += 1
                            ti += 1
                        ai += 1

            #print("ai"+str(ai))    
            #print("ti"+str(ti))    
            #normingFactor = 1.0 / ( noOfEvaluations * noOfSeeds )
            #scores /= normingFactor
            #costs /= normingFactor
            #scores = np.linalg.norm(scores, axis = 0)
            #costs = np.linalg.norm(costs, axis = 0)
            #scoreDistances = distance_matrix(scores, scores, p=2)
            #costDistances = distance_matrix(costs, costs, p=2)
            
            ai = 0
            scoreDataFrame = pd.DataFrame()
            costDataFrame = pd.DataFrame()
            statisticsDataFrame = pd.DataFrame()

            for algorithm_id in completeAlgorithmIds: 
                scoreDataFrame[str(algorithm_id)] = scores[ai,:]
                costDataFrame[str(algorithm_id)] = costs[ai,:]
                statisticsDataFrame[str(algorithm_id)] = [np.mean(errors[ai,:]),np.mean(timeouts[ai,:])]
                ai += 1
                
                        
            taskScoreDataFrame = pd.DataFrame()
            taskCostDataFrame = pd.DataFrame()
            taskStatisticsDataFrame = pd.DataFrame()
            
            ti = 0
            for task_id in taskRuns.keys(): 
                taskScoreDataFrame[str(task_id)] = scores[:,ti]
                taskCostDataFrame[str(task_id)] = costs[:,ti]
                taskStatisticsDataFrame[str(task_id)] = [np.mean(errors[:,ti]),np.mean(timeouts[:,ti])]
                ti += 1                
            
            params =  {'legend.fontsize': 'x-large',
                  'figure.figsize': (15, 5),
                 'axes.labelsize': 'x-small',
                 'axes.titlesize':'medium',
                 'xtick.labelsize':'xx-small',
                 'ytick.labelsize':'xx-small',
                 }
            pylab.rcParams.update(params)
            
            diagram_display_time = 1
            filedir = self.baseResultDirectory       
            filepath = filedir + str(self.__class__.__name__) + '/'
            figsize = (16,12)
            
            
            #Show the algorithm score correlation matrix
            df = scoreDataFrame
            plt.figure(figsize=figsize)
            plt.matshow(df.corr(), fignum = 1)
            plt.xticks(range(len(df.columns)), df.columns, rotation=70)
            plt.yticks(range(len(df.columns)), df.columns)
            plt.colorbar()
            plt.title("score correlations")
            plt.savefig(filepath + 'scores.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            
            #Show the sorted algorithm score correlation matrix
            data = df.corr()
            data = data.to_numpy()
            
            pca = PCA(n_components = 2).fit(data)
            pcaTransformedData = pca.transform(data)
            sortedIndexes = np.argsort(pcaTransformedData[:,0])
            sortedIndexes2 = sortedIndexes
            #sortedIndexes2 = np.argsort(pcaTransformedData[:,1])
            data = data[sortedIndexes]
            data = data[:,sortedIndexes2]
            print("np.shape(data): "+str(np.shape(data)))
            columns = df.columns.to_numpy()
            plt.figure(figsize=figsize)
            plt.matshow(data, fignum = 1)
            plt.xticks(range(len(columns)), columns[sortedIndexes2], rotation=90)
            plt.yticks(range(len(columns)), columns[sortedIndexes])
            plt.colorbar()
            plt.title("score correlations")
            plt.savefig(filepath + 'scores_pca_sorted.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            
            iso = Isomap(n_neighbors=1, n_components=1, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None).fit(data)
            isoTransformedData = iso.transform(data)
            sortedIndexes = np.argsort(isoTransformedData[:,0])
            sortedIndexes2 = sortedIndexes #np.argsort(isoTransformedData[:,1])
            data = data[sortedIndexes]
            data = data[:,sortedIndexes2]
            print("np.shape(data): "+str(np.shape(data)))
            columns = df.columns.to_numpy()
            plt.figure(figsize=figsize)
            plt.matshow(data, fignum = 1)
            plt.xticks(range(len(columns)), columns[sortedIndexes2], rotation=90)
            plt.yticks(range(len(columns)), columns[sortedIndexes])
            plt.colorbar()
            plt.title("score correlations")
            plt.savefig(filepath + 'scores_iso_sorted.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            

            #Show the algorithm cost correlation matrix
            df = costDataFrame
            plt.figure(figsize=figsize)
            plt.matshow(df.corr(), fignum = 1)
            plt.xticks(range(len(df.columns)), df.columns, rotation=70)
            plt.yticks(range(len(df.columns)), df.columns)
            plt.colorbar()
            plt.title("cost correlations")
            plt.savefig(filepath + 'costs.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            #Show the sorted algorithm score correlation matrix
            data = df.corr()
            data = data.to_numpy()
            
            pca = PCA(n_components = 2).fit(data)
            pcaTransformedData = pca.transform(data)
            sortedIndexes = np.argsort(pcaTransformedData[:,0])
            sortedIndexes2 = sortedIndexes
            #sortedIndexes2 = np.argsort(pcaTransformedData[:,1])
            data = data[sortedIndexes]
            data = data[:,sortedIndexes2]
            print("np.shape(data): "+str(np.shape(data)))
            columns = df.columns.to_numpy()
            plt.figure(figsize=figsize)
            plt.matshow(data, fignum = 1)
            plt.xticks(range(len(columns)), columns[sortedIndexes2], rotation=90)
            plt.yticks(range(len(columns)), columns[sortedIndexes])
            plt.colorbar()
            plt.title("cost correlations")
            plt.savefig(filepath + 'costs_pca_sorted.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            
            iso = Isomap(n_neighbors=1, n_components=1, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None).fit(data)
            isoTransformedData = iso.transform(data)
            sortedIndexes = np.argsort(isoTransformedData[:,0])
            sortedIndexes2 = sortedIndexes #np.argsort(isoTransformedData[:,1])
            data = data[sortedIndexes]
            data = data[:,sortedIndexes2]
            print("np.shape(data): "+str(np.shape(data)))
            columns = df.columns.to_numpy()
            plt.figure(figsize=figsize)
            plt.matshow(data, fignum = 1)
            plt.xticks(range(len(columns)), columns[sortedIndexes2], rotation=90)
            plt.yticks(range(len(columns)), columns[sortedIndexes])
            plt.colorbar()
            plt.title("cost correlations")
            plt.savefig(filepath + 'costs_iso_sorted.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            def positivePCA(n_components, data):
                pca = PCA(n_components = n_components).fit(scoreDataFrame)
                pcaComponents = pca.components_
                normalizedComponents = ( pcaComponents.T * np.sign( np.sum(pcaComponents, axis=1) ) ).T
                pca.components_ = normalizedComponents
                return pca.transform(data)
            '''
            dfScore = positivePCA(n_components = 1, data = scoreDataFrame)
            dfCost = positivePCA(n_components = 1, data = costDataFrame)
            plt.figure(figsize=figsize)
            plt.scatter(x = dfCost, y = dfScore)
            plt.title("PCA of score and cost per algorithm")
            plt.xlabel('cost', fontsize=6)
            plt.ylabel('score', fontsize=6)
            plt.savefig(filepath + 'pca_scores_costs.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            '''

            #Show the algorithm statistics matrix
            df = statisticsDataFrame
            plt.figure(figsize=(3,12))
            plt.matshow(df.T, fignum = 1)
            plt.xticks(range(len(df.index)), ['errors','timeouts'], rotation=70)
            plt.yticks(range(len(df.columns)), df.columns)
            plt.colorbar()
            plt.title("statistics")
            plt.savefig(filepath + 'statistics.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            
            df = PCA(n_components = 2).fit_transform(taskScoreDataFrame)
            print(np.shape(df))
            plt.figure(figsize=figsize)
            plt.scatter(x = df[:,0], y = df[:,1])
            plt.title("PCA of scores per dataset")
            plt.savefig(filepath + 'pca_scores.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            df = PCA(n_components = 2).fit_transform(taskCostDataFrame)
            plt.figure(figsize=figsize)
            plt.scatter(x = df[:,0], y = df[:,1])
            plt.title("PCA of costs per dataset")
            plt.savefig(filepath + 'pca_costs.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            
            df = PCA(n_components = 2).fit_transform(taskCostDataFrame)
            plt.figure(figsize=figsize)
            plt.scatter(x = allcosts, y = allscores )
            plt.xlabel('cost', fontsize=6)
            plt.ylabel('score', fontsize=6)
            plt.title("score vs. cost per evaluation")
            plt.savefig(filepath + 'score_cost_per_evaluation.png')
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()

            '''
            dfScore = PCA(n_components = 1).fit_transform(taskScoreDataFrame)
            dfCost = PCA(n_components = 1).fit_transform(taskCostDataFrame)
            plt.scatter(x = dfScore, y = dfCost)
            plt.title("PCA of score/cost per dataset")
            plt.xlabel('score', fontsize=6)
            plt.ylabel('cost', fontsize=6)
            plt.show(block=False)
            plt.pause(diagram_display_time)
            plt.close()
            '''
            
            self._analyzeResults = {'possibleRuns':total,'availableRuns':counter, 'algorithmRuns':algorithmRuns, 'taskRuns':taskRuns, 'evaluationRuns':evaluationRuns, 'completeAlgorithmRuns':completeAlgorithmIds, 'oneSeedCompleteAlgorithmRuns':atLeastOneSeedCompleteAlgorithmIds, 'noOfSeeds':noOfSeeds, 'noOfAlgorithms':noOfAlgorithms, 'noOfTasks':noOfTasks, 'noOfEvaluations':noOfEvaluations, 'scoreDistances':scoreDataFrame, 'costDistances':costDataFrame }
        return self._analyzeResults
        

    def createASBenchmark(self, allresults, statistics):
        self.logger.info("create algorithm selection benchmark...")  
        algorithmdIds = statistics['oneSeedCompleteAlgorithmRuns']
        noOfAlgorithms = len(algorithmdIds)
        taskIds = self.allTasks()
        noOfTasks = len(taskIds)
        seed = (self.allSeeds())[0]
        
        for evaluation_id in self.allEvaluations():
            scores = np.zeros((noOfAlgorithms, noOfTasks))
            costs = np.zeros((noOfAlgorithms, noOfTasks))
            for a_index in range(noOfAlgorithms):
                for t_index in range(noOfTasks):
                    algorithm_id = str(algorithmdIds[a_index])
                    task_id = str(taskIds[t_index])
                    evaluation_id = str(evaluation_id)
                    #self.logger.info(str(seed) + ", " + str(algorithm_id) + ", " + str(task_id) + ", "+ str(evaluation_id))
                    '''
                    d = allresults[seed]
                    d = d[algorithm_id]
                    d = d[task_id]
                    d = d[evaluation_id]
                    d = d['score']
                    '''
                    scores[a_index, t_index] = allresults[seed][algorithm_id][task_id][evaluation_id]['score']
                    costs[a_index, t_index] = allresults[seed][algorithm_id][task_id][evaluation_id]['cost']
                    
            description = 'Performance scores and costs of applying algorithms to tasks. algorithms[a] and tasks[t] contain the names, scores[a,t] and costs[a,t] contain the values, where a and t are the indices.'
            
            taskIds = [str(task_id) for task_id in taskIds]
            algorithmdIds = [str(algorithm_id) for algorithm_id in algorithmdIds]
            dataset = {'description':description, 'algorithms':algorithmdIds, 'tasks':taskIds, 'scores':scores.tolist(), 'costs':costs.tolist()}
            
            filedir = self.baseResultDirectory       
            filepath = filedir + str(self.__class__.__name__) + '/merged_'+str(evaluation_id)+'.json'
            Path(filedir).mkdir(parents=True, exist_ok=True)
            self.logger.info("save as "+filepath)  
            with open(filepath, 'w') as outfile:
                json.dump(dataset, outfile)
                
    def aggregate(self):
        results = self.loadExistingResults()
        statistics = self.analyzeResults(results)
        self.createASBenchmark(results, statistics)
        
    def showAll(self):
        results = self.loadExistingResults()
        statistics = self.analyzeResults(results)
      
        self.logger.info("===========================")           
        self.logger.info("Benchmark Result Statistics")
        self.logger.info("===========================")  
        self.logger.info("Total Evaluations: "+str(statistics['availableRuns'])+"/"+str(statistics['possibleRuns'])+"("+str(int(100*statistics['availableRuns']/statistics['possibleRuns']))+")")
        self.logger.info("== by algorithm ==")
        for k, v in statistics['algorithmRuns'].items():
            self.logger.info(str(k)+": "+str(v))
        #self.logger.info("== by task ==")
        #for k, v in statistics['taskRuns'].items():
        #    self.logger.info(str(k)+": "+str(v))
        self.logger.info("== by evaluation ==")
        for k, v in statistics['evaluationRuns'].items():
            self.logger.info(str(k)+": "+str(v))
        self.logger.info("")
        self.logger.info("Alg runs that are complete: "+str(statistics['completeAlgorithmRuns']))
        self.logger.info("Alg runs with at least one seed: "+str(statistics['oneSeedCompleteAlgorithmRuns']))
        self.logger.info("")
        self.logger.info("")
        
        evaluation_id = str((self.allEvaluations())[-1])
        
        #get score and cost for one algorithm for one evaluation method for all tasks       
        for seed in self.allSeeds():
            for algorithm_id in self.allAlgorithms():
                algorithm_id = str(algorithm_id)
                self.logger.info("Seed: "+str(seed)+" in results: "+str(seed in results))
                self.logger.info("Algorithm: "+str(algorithm_id)+" in results[seed]: "+str(algorithm_id in results[seed]))
                if seed in results and algorithm_id in results[seed]:
            
                    scores = list()
                    costs = list()
                
                    for task_id in self.allTasks():
                        task_id = str(task_id)
                        if  task_id in results[seed][algorithm_id] and evaluation_id in results[seed][algorithm_id][task_id]:
                            scores.append( results[seed][algorithm_id][task_id][evaluation_id]['score'] )
                            costs.append( results[seed][algorithm_id][task_id][evaluation_id]['cost'] )
                    if len(scores)>0:
                        self.scatterPlot(title = str(algorithm_id + " " +evaluation_id) , x = scores, y = costs, xLabel = 'score', yLabel = 'cost')
                    


    
    def allAlgorithms(self):
        return NotImplementedError
        
    def allTasks(self):
        return NotImplementedError
    
    def allEvaluations(self):
        return NotImplementedError
    
    def allSeeds(self):
        return NotImplementedError
    
    def algorithmProvider(self, id):
        return NotImplementedError
    
    def taskProvider(self, id):
        return NotImplementedError
    
    def evaluationProvider(self, id):
        return NotImplementedError


class ClassificationTask():
    '''

    '''
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def getFeatures(self):
        return self.features
    
    def getLabels(self):
        return self.labels
        
class BasicEvaluator():
    '''
    An evaluation defines and implements an evaluation method and metric.
    '''
    
    def __init__(self):
        self.logger = logging.getLogger(__name__+"."+str(self.__class__.__name__))
        
    def getSplits(self, task, seed):
        return NotImplementedError
        
    def addToDict(self, d, key, value):
        if not(key in d):
            d[key] = list()
        d[key].append(value)
        return d   
        
    def addscores(self, scoredict, yTrue, yPredicted):
        self.addToDict(scoredict, 'accuracy', accuracy_score( yTrue, yPredicted, normalize = True, sample_weight = None ))
        self.addToDict(scoredict, 'balanced_adjusted_accuracy', balanced_accuracy_score( yTrue, yPredicted, sample_weight = None, adjusted = True ))
        return scoredict

    def __call__(self, algorithm, task, seed):
        cutoffTimePerRepetionInSeconds = cfg.cutOffTime
        results = dict()
        trainingTimes = list()
        predictionTimes = list()
        totalTime = time.process_time()
        timeoutOccured = 0
        errorsOccured = 0
        for xTrain, xValid, ytrain, yValid in self.getSplits(task, seed):       
            t0 = time.process_time() #current process time in seconds
            try:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                t0 = time.process_time()
                #print("start executor.submit")
                future = executor.submit(algorithm, xTrain, ytrain, seed)
                
                global t1
                t1 = 0
                def stopTime(future):
                    global t1
                    if ( t1 == 0 ):
                        t1 = time.process_time()
                
                #this done callback makes sure that we also record runtimes that are below the sleep intervall of the following while loop
                future.add_done_callback(stopTime)
                
                while ( future.running() == True ):
                    t_runtime = time.process_time() - t0
                    if ( t_runtime > cutoffTimePerRepetionInSeconds ):
                        raise concurrent.futures.TimeoutError()
                        break
                    time.sleep(1)
                    
                executor.shutdown(wait=False)
                stopTime(future)
                #print(t1)
                #print("submitted. t1-t0 = "+str(t1-t0))
                model = future.result(timeout = cutoffTimePerRepetionInSeconds)
                t2 = time.process_time()
                #print("result. t2-t0 = "+str(t2-t0))
                assert not( model == None ), "Error evaluating "+str(algorithm)
                t1 = time.process_time() #current process time in seconds
                yPredicted = model(xValid)
                t2 = time.process_time() #current process time in seconds
                trainingTime = t1-t0
                predictionTime = t2-t1
                self.addscores(results, yValid, yPredicted)
            except Exception as exceptionInstance:
                #if anything goes wrong, we set the results to "bad" numbers to indicate an non-successfull run.
                executor.shutdown(wait=False)
                future.cancel()
                t1 = time.process_time() #current process time in seconds
                print("Exception occured after time = "+str(t1-t0))
                print("cutoffTime = "+str(cutoffTimePerRepetionInSeconds))
                #since the original model failed, we random guessing as fallback solution:
                yPredicted = yValid.copy()
                np.random.shuffle(yPredicted)#we randomly guess the right label since we have no model. This can be a worse result than just the majority classifier but that is on purpose.
                t2 = time.process_time() #current process time in seconds
                self.addscores(results, yValid, yPredicted)
                #calculate the training time and prediction time. Note: even if the calculation fails, it might be a usefull information how long the training time until failure was.
                trainingTime = t1-t0
                predictionTime = t2-t1
                if (type(exceptionInstance) is concurrent.futures.TimeoutError):
                    self.logger.warning("We hit the cut-off time for the evaluation. Stopping the evaluation and returning worst score.")
                    trainingTime = cutoffTimePerRepetionInSeconds + 1
                    timeoutOccured += 1
                elif (type(exceptionInstance) is concurrent.futures.CancelledError):
                    self.logger.warning("The execution has been cancelled. Returning worst score.")
                    trainingTime = cutoffTimePerRepetionInSeconds + 1
                    timeoutOccured += 1
                else:
                    self.logger.warn("An error occured during the evaluation. Stopping the evaluation and returning worst score.")
                    self.logger.warn( exceptionInstance, exc_info = True )
                    errorsOccured += 1
                        
            
            self.addToDict(results, 'training_cost_in_seconds', trainingTime)
            self.addToDict(results, 'prediction_cost_in_seconds', predictionTime)
            
        result = dict()
        
        for key, value in results.items():
            result['splitwise_' + key] = results[key]
            result['mean_' + key] = np.mean(results[key])
        result['totalTime'] = time.process_time() - totalTime
        
        #most important summary:
        result['cost'] =result['totalTime']
        result['score'] =np.mean(result['mean_accuracy'])
        result['noOfTimeouts'] = timeoutOccured
        result['noOfErrors'] = errorsOccured
        
        return result
        
class SimpleSplitEvaluator(BasicEvaluator):
    def getSplits(self, task, seed):
        repetitions = 1
        random = np.random.RandomState(seed)
        int32Info = np.iinfo(np.int32)
        #Creates subsets
        for i in range(repetitions):
            yield train_test_split( task.getFeatures(), task.getLabels(), train_size = 0.66, random_state = random.randint(0,int32Info.max-1), shuffle = True)
            
class KFold3Evaluator(BasicEvaluator):
    def getSplits(self, task, seed):
        kfoldInstance = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
        X = task.getFeatures()
        y = task.getLabels()
        for train_index, test_index in kfoldInstance.split(X, y):
            yield X[train_index], X[test_index], y[train_index], y[test_index]
            
class KFold10Evaluator(BasicEvaluator):
    def getSplits(self, task, seed):
        kfoldInstance = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
        X = task.getFeatures()
        y = task.getLabels()  
        for train_index, test_index in kfoldInstance.split(X, y):
            yield X[train_index], X[test_index], y[train_index], y[test_index]
            
            
class SKLearnBenchmarkTemplate( Benchmark ):
    def __init__(self):
        super().__init__()
        self.classifiers = [
        DummyMajorityClassifier,
        RidgeRegressionClassifierCV,
        LogisticRegressionClassifierLBFGSCV,
        LogisticRegressionClassifierSAGACV,
        LogisticRegressionClassifierLiblinearCV,
        KNeighborsKDTClassifier,
        KNeighborsBTClassifier,
        KNeighborsBFClassifier,
        RadiusNeighborsKDTClassifier,
        RadiusNeighborsBTClassifier ,
        RadiusNeighborsBFClassifier,
        LinearSupportVectorClassifier,
        LinearSupportVectorClassifierGridSearch,  
        RBFSupportVectorClassifier,
        GPClassifier,
        DTreeClassifier,
        RFClassifier,
        MLPClassifier,
        RFClassifier,
        ABClassifier,
        GaussianNBClassifier ,    
        QDAClassifier
        ]
        
        self.classifierDict = dict()
        for scls in self.classifiers:
            self.classifierDict[scls.__name__] = scls
        
    def allAlgorithms(self):
        return [scls.__name__ for scls in self.classifiers]
        
    def algorithmProvider(self, id):
        return self.classifierDict[id]()
        
class SKLearnBenchmarkTemplateV2( Benchmark ):
    '''
    Difference to V1: We do not use CV grid search classifiers but treat them as individuals to get a larger total number of algorithms, such that the AS can exploit individual runs of algorithms
    to estimate other performances.
    '''
    def __init__(self):
        super().__init__()
        self.classifiers = [
        DummyMajorityClassifier,
        DTreeClassifier,
        RFClassifier,
        MLPClassifier]

        self.classifiers = self.classifiers + RIDGE_REGRESSION_CLASSIFIERS
        self.classifiers = self.classifiers + LOGISTIC_REGRESSION_CLASSIFIERS
        self.classifiers = self.classifiers + KNEIGHBORS_CLASSIFIERS
        self.classifiers = self.classifiers + LINEAR_SUPPORT_VECTOR_CLASSIFIERS
        
        self.classifiers = self.classifiers + [
            RBFSupportVectorClassifier,
            RadiusNeighborsKDTClassifier,
            RadiusNeighborsBTClassifier ,
            RadiusNeighborsBFClassifier,
            GPClassifier,
            RFClassifier,
            ABClassifier,
            GaussianNBClassifier,    
            QDAClassifier]

        self.classifierDict = dict()
        for scls in self.classifiers:
            self.classifierDict[scls.__name__] = scls
        
    def allAlgorithms(self):
        return [scls.__name__ for scls in self.classifiers]
        
    def algorithmProvider(self, id):
        return self.classifierDict[id]()
        
class SKLearnBenchmarkTemplateOboe( Benchmark ):
    '''
    Difference to V2: We use the same classifier set from SKLearn as in the following paper:
    KDD'19 Chengrun Yang et. Al - Oboe: Collaborative Filtering for AutoML Model Selection
    '''
    def __init__(self):
        super().__init__()
        self.classifiers = [DummyMajorityClassifier] + OBOE_SET

        self.classifierDict = dict()
        for scls in self.classifiers:
            self.classifierDict[scls.__name__] = scls
        
    def allAlgorithms(self):
        return [scls.__name__ for scls in self.classifiers]
        
    def algorithmProvider(self, id):
        return self.classifierDict[id]()

        
class PennMLBenchmarkTemplate( Benchmark ):
    def allTasks(self):
        return pmlb.classification_dataset_names#[:2]
        
    def taskProvider(self, id):
        X, Y = pmlb.fetch_data(id, return_X_y=True, local_cache_dir=self.tmpDirectory)
        X = StandardScaler().fit_transform(X)
        if sps.issparse(X):
            X=X.todense()
        return ClassificationTask( features = X, labels = Y)

        
class OpenMLBenchmarkTemplate( Benchmark ):
    def __init__(self):
        super().__init__()
        benchmark_suite_id = 'OpenML-CC18'
        self.benchmark_suite = openml.study.get_suite(benchmark_suite_id)
    
    def allTasks(self):
        return self.benchmark_suite.tasks
        
    def taskProvider(self, id):
        task = openml.tasks.get_task(id)
        X, Y = task.get_X_and_y()
        X = StandardScaler().fit_transform(X)
        
        if sps.issparse(X):
            X=X.todense()
        return ClassificationTask( features = X, labels = Y)
        

        
class ArtificialProblemsTemplate( Benchmark ):
    def __init__(self):
        super().__init__()
        self.noOfTasks = 1000
        self.minNoOfClasses = 2
        self.maxNoOfClasses = 10
        self.minNoOfInstances = 200 #if 100, it is likely that with 10-Fold-Cross Validation we get splits that are missing one class.
        self.maxNoOfInstances = 10000
        self.minNoOfFeatures = 1
        self.maxNoOfFeatures = 100
        
    def allTasks(self):
        return [str(t) for t in range(self.noOfTasks)]
        
    def taskProvider(self, id):
        id = int(id)
        randomState = np.random.RandomState(id)
        n_classes = randomState.randint(self.minNoOfClasses,self.maxNoOfClasses) #will be modified below
        n_clusters_per_class = 1 #will be modified below
        n_samples = int(self.minNoOfInstances + ( randomState.uniform(0,1) )**2 * self.maxNoOfInstances)
        n_features = randomState.randint(self.minNoOfFeatures,self.maxNoOfFeatures)
        n_informative = n_features#randomState.randint(1,n_features)
        n_redundant = 0
        n_repeated = 0
        weights = None #will be set below
        flip_y = 0.01 #will be set below
        class_sep = randomState.uniform(0.5,1)
        hypercube = True
        shift = 0.0
        scale = 1.0
        shuffle = True        
        
        if ( ( n_features-n_informative ) != 0):
            n_redundant = randomState.randint(0,n_features-n_informative)

        maxClustersPerClass = int(n_informative)

        if ( maxClustersPerClass > 1 ):
            n_clusters_per_class = randomState.randint( 1, maxClustersPerClass )
            
        if ( n_clusters_per_class * n_classes > 2**n_informative ):
            n_classes = int( 2**n_informative / n_clusters_per_class )
        
        #weights
        weights = [(0.1 + randomState.rand()*0.9) for i in range(n_classes)]
        #noise
        flip_y = randomState.rand()*0.1 #add label noise in between 0 and 0.1

        #datasetName = "artificial_"+str(i)+"i"+str(n_samples)+"f"+str(n_features)+"c"+str(n_classes)
        #print("Generating: "+str(datasetName))
        
        X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=n_redundant, n_repeated=n_repeated, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class, weights=weights, flip_y=flip_y, class_sep=class_sep, hypercube=hypercube, shift=shift, scale=scale, shuffle=shuffle, random_state=randomState)
        
        X = StandardScaler().fit_transform(X)
        #y = np.expand_dims(y, axis=1)
        '''
        datasetProperties = {
            'id':datasetName,
            'n_samples':n_samples,
            'n_features':n_features,
            'n_informative':n_informative,
            'n_redundant':n_redundant,
            'n_repeated':n_repeated,
            'n_classes':n_classes,
            'n_clusters_per_class':n_clusters_per_class,
            'weights':weights,
            'flip_y':flip_y,
            'hypercube':hypercube,
            'shift':shift,
            'scale':scale,
            'shuffle':shuffle,
            'random_state':random_state
        }
        '''
        return ClassificationTask( features = X, labels = y)
    
    def allEvaluations(self):
        return [KFold10Evaluator.__name__]
        
    def evaluationProvider(self, id):
        return (getattr(sys.modules[__name__], id))()
    
    def allSeeds(self):
        return [a for a in range(1)]
    
class ArtificialProblemsTemplate1( ArtificialProblemsTemplate ):
    def __init__(self):
        super().__init__()
        self.noOfTasks = 1
        
class PregeneratedArtificialProblemsTemplate( Benchmark ):
    
    def __init__(self):
        super().__init__()
        self.basedir = cfg.tmpDir + 'artificial_datasets/'
        self.tasks = None
        self.allTasks()
        
    def getTaskPath(self, id):
        filedir = self.basedir + str(id) + '/'
        filepath = filedir + 'dataset.json'
        return filepath
        
    def allTasks(self):
        if (self.tasks is None):
            counter = 0
            while True:
                filepath = self.getTaskPath(counter)
                if (os.path.isfile(filepath) == False):
                    break
                counter += 1
            self.tasks = np.arange(counter)
        return self.tasks
            
        
    def taskProvider(self, id):
        filepath = self.getTaskPath(id)
        if (os.path.isfile(filepath) == False):
            return None
        
        with open(filepath, 'r') as infile:
            dataset = json.load(infile)
            X = np.array(dataset['X'])
            y = np.array(dataset['y'])
            X = StandardScaler().fit_transform(X)
            return ClassificationTask( features = X, labels = y)
        return None
        
class BasicEvaluationAndSeedTemplate( Benchmark ):
    def __init__(self):
        super().__init__()

    def allEvaluations(self):
        return [KFold10Evaluator.__name__]
        
    def evaluationProvider(self, id):
        return (getattr(sys.modules[__name__], id))()
    
    def allSeeds(self):
        return [a for a in range(1)]
        
class ExhuastiveEvaluationAndSeedTemplate( Benchmark ):
    def __init__(self):
        super().__init__()

    def allEvaluations(self):
        return [SimpleSplitEvaluator.__name__, KFold3Evaluator.__name__, KFold10Evaluator.__name__]
        
    def evaluationProvider(self, id):
        return (getattr(sys.modules[__name__], id))()
    
    def allSeeds(self):
        return [a for a in range(100)]
        
class SKLearnOnPennMLBenchmark( SKLearnBenchmarkTemplate, PennMLBenchmarkTemplate, BasicEvaluationAndSeedTemplate ):
    pass
        
class SKLearnOnOpenMLBenchmark( SKLearnBenchmarkTemplate, OpenMLBenchmarkTemplate, BasicEvaluationAndSeedTemplate ):
    pass
    
class SKLearnOnArtificialMLBenchmark( SKLearnBenchmarkTemplate, ArtificialProblemsTemplate, BasicEvaluationAndSeedTemplate ):
    pass
    
class SKLearnV2OnPennMLBenchmark( SKLearnBenchmarkTemplateV2, PennMLBenchmarkTemplate, BasicEvaluationAndSeedTemplate ):
    pass
        
class SKLearnV2OnOpenMLBenchmark( SKLearnBenchmarkTemplateV2, OpenMLBenchmarkTemplate, BasicEvaluationAndSeedTemplate ):
    pass
    
class SKLearnV2OnArtificialMLBenchmark( SKLearnBenchmarkTemplateV2, ArtificialProblemsTemplate, BasicEvaluationAndSeedTemplate ):
    pass
    
class SKLearnOboeOnPennMLBenchmark( SKLearnBenchmarkTemplateOboe, PennMLBenchmarkTemplate, BasicEvaluationAndSeedTemplate ):
    pass
        
class SKLearnOboeOnOpenMLBenchmark( SKLearnBenchmarkTemplateOboe, OpenMLBenchmarkTemplate, BasicEvaluationAndSeedTemplate ):
    pass
    
class SKLearnOboeOnArtificialMLBenchmark( SKLearnBenchmarkTemplateOboe, ArtificialProblemsTemplate, BasicEvaluationAndSeedTemplate ):
    pass
    
class SKLearnOboeOnPregeneratedArtificialMLBenchmark( SKLearnBenchmarkTemplateOboe, PregeneratedArtificialProblemsTemplate, BasicEvaluationAndSeedTemplate ):
    pass
    
class SKLearnOboeOnPregeneratedArtificialMLBenchmarkV2( SKLearnBenchmarkTemplateOboe, PregeneratedArtificialProblemsTemplate, BasicEvaluationAndSeedTemplate ):
    def __init__(self):
        super().__init__()
        self.basedir = cfg.tmpDir + 'artificial_datasets2/'
        self.tasks = None
        self.allTasks()

class DebugBenchmark( SKLearnBenchmarkTemplateOboe, ArtificialProblemsTemplate1, BasicEvaluationAndSeedTemplate ):
    pass
    
class DebugBenchmark2( SKLearnBenchmarkTemplateOboe, PennMLBenchmarkTemplate, BasicEvaluationAndSeedTemplate ):
    def allTasks(self):
        return ['poker']#pmlb.classification_dataset_names#[:2]

if __name__ == '__main__':
    setupLogging(logfile = cfg.logfile)
    logger = logging.getLogger(__name__)
    
    benchmarks = [SKLearnOboeOnPennMLBenchmark, SKLearnOboeOnOpenMLBenchmark, SKLearnOboeOnArtificialMLBenchmark, SKLearnV2OnPennMLBenchmark, SKLearnV2OnOpenMLBenchmark, SKLearnV2OnArtificialMLBenchmark, SKLearnOnPennMLBenchmark, SKLearnOnOpenMLBenchmark, SKLearnOnArtificialMLBenchmark, SKLearnOboeOnPregeneratedArtificialMLBenchmark, SKLearnOboeOnPregeneratedArtificialMLBenchmarkV2, DebugBenchmark]

    parser = argparse.ArgumentParser(description='run a single optimizer on a single problem instance')
    parser.add_argument('--benchmark','-b',
                    dest='benchmark',
                    action='store',
                    type=type(' '),
                    default='SKLearnOnPennMLBenchmark',
                    help='the benchmark suite that is about to run '+str([cls.__name__ for cls in benchmarks]))
                    
    parser.add_argument('--processes','-pr',
                    dest='processes',
                    action='store',
                    type=type(' '),
                    default='1',
                    help='number of parallel processes that are used for the calculation.')
                    
                
    parser.add_argument('--no-evaluate','-ne',
                dest='evaluate',
                action='store_false',
                help='don \'t run benchmark.runAll() (but use existing results where available)')
    
    parser.set_defaults(evaluate = True)
                    
    parser.add_argument('--show','-s',
                dest='show',
                action='store_true',
                help='run benchmark.showAll()')
                
    parser.set_defaults(show = False)

    parser.add_argument('--aggregate','-a',
                dest='aggregate',
                action='store_true',
                help='aggregates the individual results to one common result file for further processing. Also shows some performance diagrams at the end.')
                
    parser.set_defaults(aggregate = False)
                        
    args = parser.parse_args()
    
    logging.info(args.benchmark)
    benchmarkClass = getattr(sys.modules[__name__], args.benchmark) #flexible, get and create an instance of the provided class.
    #make sure that class is actually really a benchmark class. The run it.
    if (issubclass(benchmarkClass, Benchmark)):
        benchmark = benchmarkClass()
        if args.evaluate == True:
            benchmark.runAll(processes = int(args.processes))
        if args.aggregate == True:
            benchmark.aggregate()
        if args.show == True:
            benchmark.showAll()
    shutdownLogging()