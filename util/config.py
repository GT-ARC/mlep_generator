#python basic imports

#3rd party imports (from packages, the environment)

#custom (local) imports


tmpDir = './.tmp/'
resultDir = './.results/'
logfile = tmpDir + 'debug.log'

forceRecalculation = False
cutOffTime = 600 #cut off time in seconds. We set it to 10 minutes for now. Every training pass that takes longer will be stopped.

