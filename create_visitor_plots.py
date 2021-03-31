from flask import Response, Blueprint

import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import random as rand
import datetime
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd

import matplotlib.dates as mdates


myFmt = mdates.DateFormatter('%H:%M')

plots = Blueprint('plots', __name__, template_folder='templates')

from app import mysql


@plots.route('/plot_visitor_today')
def plot_visitor_today():
    data = get_daily_visitor()
    df = pd.DataFrame(data)
    print(df)
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@plots.route('/plot_visitor_yesterday')
def plot_visitor_yesterday():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@plots.route('/plot_visitor_week')
def plot_visitor_week():
    fig = create_figure_week()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@plots.route('/plot_visitor_overall')
def plot_visitor_overall():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def create_figure():
    fig = Figure(figsize=(11, 4))
    axis = fig.add_subplot(1, 1, 1)

    x = np.array([datetime.datetime(2013, 9, 28, i, 0) for i in range(7, 22)])
    y = np.random.randint(100, size=x.shape)

    # axis.set_major_formatter(myFmt)
    axis.xaxis.set_major_formatter(myFmt)
    xs = range(100)
    ys = [rand.randint(1, 50) for x in xs]
    axis.plot(x, y)
    return fig


def create_figure_week():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    dates = []
    for index in range(7):
        label = (dt.now() - timedelta(days=index)).strftime('%Y-%m-%d')
        # dates.append((dt.now() - timedelta(days=date)).strftime('%Y-%m-%d'))

        x = np.array([datetime.datetime(2013, 9, 28, i, 0) for i in range(7, 22)])
        y = np.random.randint(100, size=x.shape)

        # axis.set_major_formatter(myFmt)
        axis.xaxis.set_major_formatter(myFmt)
        xs = range(100)
        ys = [rand.randint(1, 50) for x in xs]
        axis.plot(x, y, label=label)

    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1,0))

    return fig


def get_daily_visitor():
    cur = mysql.connection.cursor()
    result = cur.execute(
        "SELECT * FROM person_counter WHERE timestamp >= CURDATE() AND timestamp < CURDATE() + INTERVAL 1 DAY ORDER "
        "BY timestamp")
    daily = cur.fetchall()

    rows = []

    for row in cur:
        rows.append(row)
        print(type(row))


    if result > 0:
        cur.close()
        return rows
    else:
        cur.close()
        return 0


"""
fileNames = [file for file in fileNames if '.csv' in file]

### Loop over all files
for file in fileNames:
    ### Read .csv file and append to list
    df = pd.read_csv(PATH + file, index_col=0)

    ### Create line for every file
    plt.plot(df)

### Generate the plot
plt.show()
"""
