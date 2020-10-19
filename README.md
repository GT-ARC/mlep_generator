# Meta Learning Evaluation Policies #
Author: Christian Geißler <christian.geissler@gt-arc.com>
License: Copyright 2020 by GT-ARC gGmbH
Acknowledgement: This work is supported in part by the German Federal Ministry of Education and Research (BMBF) under the grant number 01IS16046.

# Introduction
This repository contains code to generate artificial machine learning classification datasets and to run machine learning benchmarks on real datasets and the generated ones.

To generate artificial datsets, run any of the run_generate*.py files. run_demo_genearte_artifical_problems.py is intended to give a visual impression on how the generated datasets look like.

Based on the generated datasets or real classificaiton problem collections, a benchmark can be run (run.py) that applies a whole set of different classification learning algorithms on the set and stores their performance and resource consumption (time) under .results. The very same script allows to aggregate all the results from such a benchmark into a single file. Example for executing this for the SKLearn PennMLBenchmark dataset:
python run.py -b SKLearnOnPennMLBenchmark -ne -s, where -b <name> indicates the name of the machine learning benchmark run before. Note: You can run run.py -h to see all the options available. After aggregation, a file is created in the specific benchmark folder .results/<benchmarkname>/merged_KFold10Evaluator.json. This file can be used as input in the experiments.

# Principles
The artificial dataset generation is based on sampling the decision spaces from the set of machine learning classification approaches provided. It thus does not intend to approximate real datasets, but focuses on creating datasets that make it possible to find out the differences between the machine learning approaches. The basic idea behind it is to create a large number of artificial datasets to learn effective evaluation policies for running the machine learning approaches on new dataset in a purely data driven way.