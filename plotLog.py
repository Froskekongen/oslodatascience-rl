'''
Just give it the name of the log file and it will return a bokeh plot.
'''

from common import readLogPong
from bokeh.charts import Line
from bokeh.plotting import output_file, show
import argparse 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logfile', help="name of log files")
    args = parser.parse_args()
    
    df = readLogPong(args.logfile)
    p = Line(df, x='episode', y='rewardSum', plot_width=1200, plot_height=400)
    output_file("line.html")
    show(p)


if __name__ == '__main__':
    main()
