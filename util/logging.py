# -*- coding: utf-8 -*-
"""
Created on 28.03.2019
@author: christian.geissler@gt-arc.com
"""
#system
import logging

#3rd party
import numpy as np

#custom



class LevelDependendFormatter(logging.Formatter):
    '''
    Level specific formatting, such that info does not clog the log with unnessesary information but errors and warnings get more detailed ones.
    '''
    def __init__(self, fmt="%(levelno)s: %(msg)s", critical = None, error = None, warning = None, debug = None, info = None):
        super(LevelDependendFormatter, self).__init__(self, fmt)
        self.dflt_fmt = fmt
        self.crt_fmt  = critical
        self.err_fmt  = error
        self.wrn_fmt  = warning
        self.dbg_fmt  = debug
        self.info_fmt = info

    
    def format(self, record):

        # Set to default format (in case none of the following applies)
        #self._fmt = dflt_fm
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        #the reason why we do the check here and not in init is, that this is more robust: the user could set self.info_fmt to None even after __init__ and it still works.
        if record.levelno == logging.DEBUG:
            self._fmt = self.dflt_fmt if self.dbg_fmt is None else self.dbg_fmt

        elif record.levelno == logging.INFO:
            self._fmt = self.dflt_fmt if self.info_fmt is None else self.info_fmt
            
        elif record.levelno == logging.WARNING:
            self._fmt = self.dflt_fmt if self.wrn_fmt is None else self.wrn_fmt

        elif record.levelno == logging.ERROR:
            self._fmt = self.dflt_fmt if self.err_fmt is None else self.err_fmt
            
        elif record.levelno == logging.CRITICAL:
            self._fmt = self.dflt_fmt if self.crt_fmt is None else self.crt_fmt
        
        self._style = logging.PercentStyle(self._fmt)
        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)
        
        self._fmt = format_orig

        return result

def setupLogging(logfile="debug.log"):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    
    loggingFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    loggingFileHandler = logging.FileHandler(logfile, mode='w')
    loggingFileHandler.setLevel(logging.INFO)
    loggingFileHandler.setFormatter(loggingFormatter)
    logging.getLogger('').addHandler(loggingFileHandler)
    
    #loggingFormatter = logging.Formatter('%(levelname)s - %(pathname)s %(lineno)d - %(message)s')
    #loggingFormatter = logging.Formatter('%(message)s')
    #loggingFormatter = LevelDependendFormatter('%(message)s')
    loggingFormatter = LevelDependendFormatter(fmt = '%(levelname)s - %(pathname)s %(lineno)d - %(message)s', info = '%(message)s')
    loggingStreamHandler = logging.StreamHandler()
    loggingStreamHandler.setLevel(logging.INFO)
    loggingStreamHandler.setFormatter(loggingFormatter)
    logging.getLogger('').addHandler(loggingStreamHandler)
    
    #replace regular print with logger info function, because some sklearn implementation (I'm not looking into a specific direction - RandomizedSearchCV) use print for logging...
    #print = logger.info
    
    logger.info('Start Logging')
    return logger
    
def shutdownLogging():
    logger = logging.getLogger('')
    logger.info("Shutdown logger, bye bye!")
    #create a copy, so that we can savely manipulate the original list:
    copyOfList = list(logger.handlers)
    for handlerInstance in copyOfList:
        logger.removeHandler(handlerInstance)
    logger.handlers = []
    del logger
    logging.shutdown()