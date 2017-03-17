'''
Should be run with:
    bokeh serve --show plotLog.py --args <logfile>
'''

from common import readLogPong, StreamLog
from bokeh.plotting import figure, curdoc
import argparse 
import pandas as pd


class StreamPongLog(StreamLog):
    episodes = []
    rewardSums = []
    timestamps = []

    def readData(self):
        for line in self.streamAvailable():
            if not line: break
            line = self.parse(line)
            self.timestamps.append(line[0])
            self.episodes.append(line[1])
            self.rewardSums.append(line[2])

    def parse(self, line):
        line = self.removeNewLineCharacter(line).split(';')
        line[0] = pd.to_datetime(line[0])
        line[1] = int(line[1])
        line[2] = float(line[2])
        return line

        
parser = argparse.ArgumentParser()
parser.add_argument('logfile', help="name of log files")
args = parser.parse_args()

p = figure(plot_width=1200, plot_height=400)
p.xaxis.axis_label = 'episode'
p.yaxis.axis_label = 'sum rewards'
r1 = p.line([], [], color="firebrick", line_width=2)
ds1 = r1.data_source
# r2 = p.line([], [], color="navy", line_width=2)
# ds2 = r2.data_source

stream = StreamPongLog(args.logfile)
def update():
    stream.readData()
    ds1.data['x'] = stream.episodes
    ds1.data['y'] = stream.rewardSums
    ds1.trigger('data', ds1.data, ds1.data)

curdoc().add_root(p)
curdoc().add_periodic_callback(update, 1000)

