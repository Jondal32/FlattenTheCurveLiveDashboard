from flask import Response, Blueprint, send_file, make_response

import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import random as rand
import datetime
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%H:%M')

plots = Blueprint('plots', __name__, template_folder='templates')

from app import mysql


@plots.route('/plot_visitor_today')
def plot_visitor_today():
    try:
        data = get_daily_visitor()
        # print(type(data))
        df = pd.DataFrame(data)
        # print(df.dtypes)
        df_cut = df[["timestamp", "amount"]].to_csv('csv/today.csv', date_format='%Y/%m/%d %H:%M:%S', index=False)
        # date_format = '%H:%M:%S'
        # print(type(df_cut))

        return Response(df_cut, mimetype='text/csv')
        # return send_file("csv/today.csv", mimetype='text/csv')
        # fig = create_figure()
        # output = io.BytesIO()
        # FigureCanvas(fig).print_png(output)
        # return Response(output.getvalue(), mimetype='image/png')
    except ValueError:
        return 0


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


@plots.route('/plot_heatmap', methods=['GET', 'POST'])
def plot_heatmap():
    file_path = createDistanceMessurementHeatmap()

    return send_file(file_path, mimetype='image/png')


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
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0))

    return fig


def get_daily_visitor():
    try:
        cur = mysql.connection.cursor()
        result = cur.execute(
            "SELECT * FROM person_counter WHERE timestamp >= CURDATE() AND timestamp < CURDATE() + INTERVAL 1 DAY ORDER "
            "BY timestamp")
        daily = cur.fetchall()

        rows = []

        for row in cur:
            rows.append(row)

        if result > 0:
            cur.close()
            return rows
        else:
            cur.close()
            return 0
    except ValueError:
        return 0


def createDistanceMessurementHeatmap():
    img = plt.imread("static/img/imgHeatMapRAW.JPG")
    df = pd.read_csv("static/img/testdata.csv")

    fig = Figure(dpi=300)

    ax = fig.add_subplot(1, 1, 1)

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    # print(xmax, ymin)

    # Create the heatmap
    kde = sns.kdeplot(
        x=df['x'],
        y=df['y'],
        shade=True,
        thresh=0.05,
        alpha=.4,
        n_levels=10,
        cmap='magma',
        ax=ax
    )

    ax.imshow(img)

    # plt.ylim(0,ymin)
    # plt.xlim(0,xmax)
    # ax.invert_yaxis()
    ax.set_axis_off()
    file_path = "static/img/imgHeatMapFINAL.png"
    fig.savefig(file_path, bbox_inches='tight', pad_inches=0)
    return file_path


def getVisitorData():
    cur = mysql.connection.cursor()
    """
    1. Abfrage: durchschnittliche Personen im Laden (gesamter Zeitraum)
    2. Abfrage: durchschnittliche Personen im Laden (Heute)
    3. Abfrage: durchschnittliche Personen im Laden (Gestern)
    4. Abfrage: durchschnittliche Personen im Laden (Woche)
    5. Abfrage: überwachter Zeitraum (Heute)
    6. Abfrage: überwachter Zeitraum (Gestern)
    7. Abfrage: höchste Besucherzahl (Heute)
    8. Abfrage: höchste Besucherzahl (Gestern)
    9. Abfrage: höchste Besucherzahl (Woche)
    
    """
    results = []

    queries = ["SELECT AVG(amount) as avg_total from person_counter",
               "SELECT AVG(amount)  as avg_daily FROM person_counter WHERE timestamp = CURDATE()",
               "SELECT AVG(amount) as avg_yesterday FROM person_counter WHERE timestamp <= CURDATE() AND timestamp >= CURDATE() - "
               "INTERVAL 1 DAY ",
               "SELECT AVG(amount) as avg_week FROM person_counter WHERE timestamp <= CURDATE() AND timestamp >= CURDATE() - "
               "INTERVAL 7 DAY ",
               "SELECT TIMESTAMPDIFF(Minute, Min(timestamp), Max(timestamp)) as range_today FROM (SELECT * FROM "
               "person_counter WHERE timestamp = CURDATE()) as dt",
               "SELECT TIMESTAMPDIFF(Minute, Min(timestamp), Max(timestamp)) as range_yesterday FROM(SELECT * FROM person_counter WHERE "
               "timestamp >= CURDATE() - INTERVAL 1 DAY) as dt",
               "SELECT max(amount) as max_persons_today, timestamp from person_counter WHERE timestamp = CURDATE()",
               "SELECT max(amount) as max_persons_yesterday, timestamp from person_counter WHERE timestamp <= CURDATE() AND "
               "timestamp >= CURDATE() - INTERVAL 1 DAY",
               "SELECT max(amount) as max_persons_week, timestamp from person_counter WHERE timestamp <= CURDATE() AND "
               "timestamp >= CURDATE() - INTERVAL 7 DAY "
               ]

    for query in queries:
        cur.execute(query)
        result = cur.fetchall()
        results.append(result)
    print(results)
    print(results[0][0]["avg_total"])
    print(type(results[0][0]["avg_total"]))
    return results
