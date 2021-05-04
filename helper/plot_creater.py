from flask import Response, Blueprint, send_file, make_response

from matplotlib.figure import Figure

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%H:%M')

plots = Blueprint('plots', __name__, template_folder='templates')

from app import mysql


@plots.route('/plot_heatmap', methods=['GET', 'POST'])
def plot_heatmap():
    file_path = createDistanceMessurementHeatmap()
    return send_file(file_path, mimetype='image/png')


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

    queries = ["SELECT ROUND(AVG(amount),1) as avg_total from person_counter",
               "SELECT ROUND(AVG(amount),1) as avg_daily FROM person_counter WHERE DATE(timestamp) = CURDATE()",
               "SELECT ROUND(AVG(amount),1) as avg_yesterday FROM person_counter WHERE DATE(timestamp) = CURDATE() - INTERVAL 1 DAY ",
               "SELECT ROUND(AVG(amount),1) as avg_week FROM person_counter WHERE DATE(timestamp) <= CURDATE() "
               "AND DATE(timestamp) >= CURDATE() - INTERVAL 7 DAY ",
               "SELECT TIMESTAMPDIFF(Minute, Min(timestamp), Max(timestamp)) as range_today FROM (SELECT * FROM "
               "person_counter WHERE DATE(timestamp) = CURDATE()) as dt",
               "SELECT TIMESTAMPDIFF(Minute, Min(timestamp), Max(timestamp)) as range_yesterday FROM (SELECT * FROM person_counter WHERE "
               "DATE(timestamp) = CURDATE() - INTERVAL 1 DAY ) as dt",
               "SELECT max(amount) as max_persons_today, DATE_FORMAT(timestamp, '%e-%m %H:%i') as timestamp_modi from person_counter WHERE DATE(timestamp) = CURDATE()",
               "SELECT max(amount) as max_persons_yesterday, DATE_FORMAT(timestamp, '%e-%m %H:%i') as timestamp_modi from person_counter WHERE DATE(timestamp) = CURDATE() - INTERVAL 1 DAY ",
               "SELECT max(amount) as max_persons_week, DATE_FORMAT(timestamp, '%e-%m %H:%i') as timestamp_modi from person_counter WHERE DATE(timestamp) <= CURDATE() "
               "AND DATE(timestamp) >= CURDATE() - INTERVAL 7 DAY "
               ]

    for query in queries:
        cur.execute(query)
        result = cur.fetchall()
        results.append(result)

    return results
