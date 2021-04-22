from flask import Response, Blueprint, send_file

import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%H:%M')

backgroundTasks = Blueprint('backgroundTask', __name__, template_folder='templates')


@backgroundTasks.route('/backgroundDistance', methods=["GET", "POST"])
def backgroundDistance():
    return 0
