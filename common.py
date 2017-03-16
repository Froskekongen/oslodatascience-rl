import logging
import pandas as pd
import time

class LogProgress(object):
    '''
    Class for logging progress of RL agents
    logfile: file name of log file.
    console: output to console.
    level: logging level.
    name: name of logger. (not really relevant as of now...)
    '''
    def __init__(self, logfile, console=False, level=logging.INFO, name=None):
        self.logfile = logfile
        self.level = level
        if name is None: name = __name__
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        formatter = logging.Formatter('%(asctime)s;%(message)s')
        self.fh = logging.FileHandler(logfile)
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)

        if console:
            self.ch = logging.StreamHandler()
            self.ch.setFormatter(formatter)
            self.logger.addHandler(self.ch)

    def info(self, msg):
        '''
        Use logger info method to write to logfile (and console).
        msg: string message
        '''
        self.logger.info(msg)
        # if msg.__class__ is str:
            # self.logger.info(msg)
        # elif hasattr(msg, '__iter__'):
            # self.logger.info(';'.join(msg))
        # else:
            # raise ValueError('msg should be string or iterable')

class LogPong(LogProgress):
    '''
    Class for logging progress in pong game.
    logfile: file name of log file.
    console: output to console.
    **kwargs: arguments passed to LogProgress.
    '''
    def __init__(self, logfile, console=False, **kwargs):
        super().__init__(logfile, console, **kwargs)

    def log(self, episode, rewardSum):
        '''
        Function for writing to log file (and console).
        episole: episode number.
        rewardSum: sum of rewards in episode.
        '''
        msg = '%d;%f' % (episode, rewardSum)
        self.info(msg)

def readLogPong(filename, **kwargs):
    '''
    Get pong log file (LogPong) as a dataframe.
    filename: file name of log file.
    **kwargs: arguments passed to pd.read_csv.
    '''
    df = pd.read_csv(filename, sep=';', names=('time', 'episode', 'rewardSum'), **kwargs)
    df.time = pd.to_datetime(df.time)
    return df


class StreamLog(object):
    '''A way to read a file as it is written.
    fileName: file name
    start: 0 for beginnning of file, 2 for end of file.
    sleepSec: number of seconds to sleep when we reach end of file.
    '''
    def __init__(self, fileName, start=0, sleepSec=2):
        self.fileName = fileName
        self.handle = open(self.fileName, 'r')
        self.start = start
        self.sleepSec=sleepSec

    def close(self):
        '''Close file handle.'''
        self.handle.close()

    def streamContinuously(self):
        '''Stream continuously. 
        Run a for loop over <object>.streamContinuously()'''
        while True:
            line = self.handle.readline()
            if not line:
                time.sleep(self.sleepSec)
                continue
            yield line

    def streamAvailable(self):
        '''Read available file, and break. 
        Continues where the previous read ended.
        Run a for loop over <object>.streamAvailable()'''
        line = 'something'
        while line:
            line = self.handle.readline()
            yield line

    @staticmethod
    def removeNewLineCharacter(line):
        return line[:-1]

    
def main():
    pass

if __name__ == '__main__':
    main()
