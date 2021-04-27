from flask import Response, Blueprint, send_file

import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%H:%M')

backgroundTasks = Blueprint('backgroundTask', __name__, template_folder='templates')


@backgroundTasks.route('/backgroundDistance', methods=["GET", "POST"])
def backgroundDistance():
    return 0


def createDictFromVisitorDataList(results):
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

    avg_total = results[0][0]["avg_total"]
    avg_daily = results[1][0]["avg_daily"]
    avg_yesterday = results[2][0]["avg_yesterday"]
    avg_week = results[3][0]["avg_week"]
    range_today = results[4][0]["range_today"]
    range_yesterday = results[5][0]["range_yesterday"]
    max_persons_today, max_person_today_timestamp = results[6][0]["max_persons_today"], results[6][0]["timestamp_modi"]
    max_persons_yesterday, max_persons_yesterday_timestamp = results[7][0]["max_persons_yesterday"], results[7][0][
        "timestamp_modi"]
    max_persons_week, max_persons_week_timestamp = results[8][0]["max_persons_week"], results[8][0]["timestamp_modi"]



    visitorSettings = {'avg_total_available': avg_total if not None else False,
                       'avg_daily_available': avg_daily if not None else False,
                       'avg_yesterday_available': avg_yesterday if not None else False,
                       'avg_week_available': avg_week if not None else False,
                       'range_today_available': range_today if not None else False,
                       'range_yesterday_available': range_yesterday if not None else False,
                       'max_persons_today_available': max_persons_today if not None else False,
                       'max_persons_yesterday_available': max_persons_yesterday if not None else False,
                       'max_persons_week_available': max_persons_week if not None else False, 'avg_total': avg_total,
                       'avg_daily': avg_daily, 'avg_yesterday': avg_yesterday, 'avg_week': avg_week,
                       'range_today': range_today, 'range_yesterday': range_yesterday,
                       'max_persons_today': max_persons_today, 'max_persons_yesterday': max_persons_yesterday,
                       'max_persons_week': max_persons_week, 'max_persons_today_timestamp': max_person_today_timestamp,
                       'max_persons_yesterday_timestamp': max_persons_yesterday_timestamp,
                       'max_persons_week_timestamp': max_persons_week_timestamp}


    return visitorSettings


def createDictFromMaskDetectionData(PieChartData):

    piePlotsSettings = {'todayAvailable': PieChartData.dataToday if not None else False,
                        'yesterdayAvailable': PieChartData.dataYesterday if not None else False,
                        'weekAvailable': PieChartData.dataWeek if not None else False,
                        'totalAvailable': PieChartData.dataTotal if not None else False,
                        'pieChartDataToday': PieChartData.dataToday,
                        'pieChartDataYesterday': PieChartData.dataYesterday, 'pieChartDataWeek': PieChartData.dataWeek,
                        'pieChartDataTotal': PieChartData.dataTotal}

    return piePlotsSettings
