import decimal

from flask import Response, Blueprint, send_file

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
import json

import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%H:%M')

backgroundTasks = Blueprint('backgroundTask', __name__, template_folder='templates')



@backgroundTasks.route('/backgroundDistance', methods=["GET", "POST"])
def backgroundDistance():
    return 0
