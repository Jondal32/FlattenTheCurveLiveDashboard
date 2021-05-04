from flask import Response, Blueprint, send_file
import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%H:%M')

piePlots = Blueprint('piePlots', __name__, template_folder='templates')

from app import mysql


class PieChartData:
    """Diese Klasse erstellt alle PiePlots die für Analytics relevant sind"""

    def __init__(self):
        self.dataToday = None
        self.dataYesterday = None
        self.dataWeek = None
        self.dataTotal = None

    def createPieChartDataToday(self):
        try:

            cur = mysql.connection.cursor()
            result = cur.execute(
                "SELECT CAST(SUM(maskFrames) AS CHAR) as maskFrames, CAST(SUM(noMaskFrames) AS CHAR) as noMaskFrames FROM maskproportion WHERE date = "
                "CURDATE()")

            if result > 0:
                try:
                    re = cur.fetchall()
                    data = [int(re[0]["maskFrames"]), int(re[0]["noMaskFrames"])]

                    cur.close()

                    self.dataToday = data
                except TypeError:
                    self.dataToday = None


            else:
                self.dataToday = None


        except ValueError:
            print("Value Error beim speichern vom pie chart!")

    def createPieChartDataYesterday(self):
        try:

            cur = mysql.connection.cursor()
            result = cur.execute(
                "SELECT CAST(SUM(maskFrames) AS CHAR) as maskFrames, CAST(SUM(noMaskFrames) AS CHAR) as noMaskFrames FROM maskproportion WHERE date = "
                "CURDATE() - INTERVAL 1 DAY")

            if result > 0:
                try:
                    re = cur.fetchall()
                    data = [int(re[0]["maskFrames"]), int(re[0]["noMaskFrames"])]

                    cur.close()

                    self.dataYesterday = data
                except TypeError:
                    self.dataYesterday = None

            else:
                self.dataYesterday = None


        except ValueError:
            print("Value Error beim speichern vom pie chart!")

    def createPieChartDataWeek(self):
        try:

            cur = mysql.connection.cursor()
            result = cur.execute(
                "SELECT CAST(SUM(maskFrames) AS CHAR) as maskFrames, CAST(SUM(noMaskFrames) AS CHAR) as noMaskFrames FROM maskproportion WHERE date "
                "<= CURDATE() AND date >= CURDATE() - INTERVAL 7 DAY")

            if result > 0:
                try:
                    re = cur.fetchall()
                    data = [int(re[0]["maskFrames"]), int(re[0]["noMaskFrames"])]

                    cur.close()

                    self.dataWeek = data
                except TypeError:
                    self.dataWeek = None

            else:
                self.dataWeek = None

        except ValueError:
            print("Value Error beim speichern vom pie chart!")

    def createPieChartDataTotal(self):
        try:

            cur = mysql.connection.cursor()
            result = cur.execute(
                "SELECT CAST(SUM(maskFrames) AS CHAR) as maskFrames, CAST(SUM(noMaskFrames) AS CHAR) as noMaskFrames FROM maskproportion")

            if result > 0:
                try:
                    re = cur.fetchall()
                    data = [int(re[0]["maskFrames"]), int(re[0]["noMaskFrames"])]

                    cur.close()

                    self.dataTotal = data
                except TypeError:
                    self.dataTotal = None

            else:
                print("Fail")


        except ValueError:
            print("Value Error beim speichern vom pie chart!")

    def createAllCharts(self):
        self.createPieChartDataToday()
        self.createPieChartDataYesterday()
        self.createPieChartDataWeek()
        self.createPieChartDataTotal()

