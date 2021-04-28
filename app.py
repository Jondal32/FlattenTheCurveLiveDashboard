# flask
from flask import flash, redirect, url_for, session, request
from flask_mysqldb import MySQL
from flask import Flask, render_template

app = Flask(__name__)
# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask_dashboard'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# init MYSQL
mysql = MySQL(app)

from wtforms import Form, StringField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from helper.plot_creater import *
from helper.create_pie_plots import *
from helper.backgroundTasks import *

###
import threading
import time as clock

from tensorflow.keras.models import load_model

import cv2
import json

from time import time

lock = threading.Lock()

app.register_blueprint(plots)
app.register_blueprint(piePlots)
app.register_blueprint(backgroundTasks)

prototxtPath = 'models/face_detector/deploy.prototxt'
weightsPath = 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model('models/face_detection_mobilenetv2')

from core.mask_detection_2 import Stream as mask_stream
from core.person_counter_script_2 import Stream as person_counter_stream

camera_off = False


# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):

        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Sie sind nicht autorisiert, bitte anmelden!', 'danger')
            return redirect(url_for('login'))

    return wrap


def check_rights(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT rights FROM users WHERE username = %s", [session['username']])
        user_rights = cur.fetchall()
        cur.close()

        """berechtige Personen haben in den Datenbank bei rights als bit-value = 1  und nicht berechtige Personen 
        bit-value = 0 """

        if 'logged_in' in session and user_rights[0]['rights'] == b'\x01':
            return f(*args, **kwargs)
        else:
            flash(
                'Kein Zugriff! Bitte kontaktieren Sie Ihren Vorgesetzen oder den Datenbank Administrator um die passendes Rechte zu erhalten',
                'danger')
            return redirect(url_for('index'))

    return wrap


# Index
@app.route('/')
def index():
    return render_template('home.html')


# About
@app.route('/maskdetection')
@is_logged_in
@check_rights
def maskdetection():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off' and maskStream.status() == True:
        maskStream.close()
        flash("Stream beendet", "info")
    elif camera is not None and camera == 'on' and maskStream.status() == False:
        maskStream.open()
        flash("Stream erfolgreich gestartet!", "success")

    setting = dict(
        stream_on=maskStream.status(),
        fps=maskStream.get_fps(),
        w=600
    )

    return render_template('maskdetection.html', setting=setting)


@app.route('/person_counter')
@is_logged_in
@check_rights
def person_counter():
    use_traffic_light = False
    camera = request.args.get("camera")
    predefined_person_count = request.args.get("persons")
    threshold = request.args.get("threshold")
    ## Werte für Ampel
    yellow_mark = request.args.get("yellow")
    red_mark = request.args.get("red")

    if camera is not None and camera == 'off' and personCounterStream.status() == True:
        personCounterStream.close()
        flash("Stream beendet", "info")
    elif camera is not None and camera == 'on' and personCounterStream.status() == False:
        if predefined_person_count is not None:
            personCounterStream.totalIn = int(predefined_person_count)
            personCounterStream.totalOut = 0

        if threshold is not None:
            personCounterStream.minConfidence = float(threshold.replace(',', '.'))

        if yellow_mark is not None:
            personCounterStream.yellow_mark = int(yellow_mark)
            # Sobald die gelbe oder rote Markierung vorliegt wird die Ampel benutzt
            use_traffic_light = True

        if red_mark is not None:
            personCounterStream.red_mark = int(red_mark)


        personCounterStream.open()
        flash("Stream erfolgreich gestartet", "success")

    setting = dict(
        stream_on=personCounterStream.status(),
        use_traffic_light=use_traffic_light,

    )
    return render_template('person_counter.html', setting=setting)


@app.route('/distance_messurement')
@is_logged_in
@check_rights
def distance_messurement():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off' and maskStream.status() == True:
        maskStream.close()
        flash("Stream beendet", "info")
    elif camera is not None and camera == 'on' and maskStream.status() == False:
        maskStream.open()
        flash("Stream erfolgreich gestartet", "success")

    setting = dict(
        stream_on=maskStream.status(),
        w=600
    )
    return render_template('distance_messurement.html', setting=setting)


@app.route('/kontakt')
def kontakt():
    return render_template('kontakt.html')


@app.route('/analytics', methods=['GET', 'POST'])
@is_logged_in
@check_rights
def analytics():
    """Sql-Abfragen zu Besucherzahlen, dessen Daten zu einem dictionary zusammengefasst werden"""
    results = getVisitorData()
    visitorSettings = createDictFromVisitorDataList(results)

    """Sql-Abfragen in der Datenbank zur MaskDetection und anschließende Erstellung eines dictionary"""
    PieChartData.createAllCharts()
    piePlotsSettings = createDictFromMaskDetectionData(PieChartData)

    return render_template('analytics.html', piePlotsSettings=piePlotsSettings,
                           visitorSettings=visitorSettings)


# Register Form Class
class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Benutzername', [validators.Length(min=4, max=25)])
    email = StringField('Email Adresse', [validators.Length(min=6, max=50)])
    password = PasswordField('Passwort', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Die angegebenen Passwörter stimmen nicht überein')
    ])
    confirm = PasswordField('Passwort bestätigen')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        # Create cursor
        cur = mysql.connection.cursor()

        # Execute query
        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)",
                    (name, email, username, password))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('Sie haben sich erfolgreich registriert', 'success')

        return redirect(url_for('login'))
    return render_template('register.html', form=form)


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username

                flash('Erfolgreich eingeloggt', 'success')
                return redirect(url_for('index'))
            else:
                error = 'Benutzername oder Passwort falsch'
                return render_template('login.html', error=error)
            # Close connection
            cur.close()
        else:
            error = 'Benutzername nicht gefunden'
            return render_template('login.html', error=error)

    return render_template('login.html')


# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('Erfolgreich abgemeldet', 'success')
    return redirect(url_for('login'))


@app.route('/datenschutz')
def datenschutz():
    return render_template('datenschutz.html')


@app.route('/mask_detection_video', methods=['GET', 'POST'])
def mask_detection_video():
    return Response(maskStream.generateFrames(faceNet=faceNet, maskNet=maskNet),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/person_counter_video', methods=['GET', 'POST'])
def person_counter_video():
    return Response(personCounterStream.generateFrames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/data', methods=["GET", "POST"])
def data():
    current_in = personCounterStream.get_TotalIn()
    current_out = personCounterStream.get_TotalOut()

    # %Y-%m-%d , time2.strftime('%H:%M:%S')

    data = [time() * 1000, current_in, current_out]

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO person_counter(timestamp, amount) VALUES (%s,%s)",
                [clock.strftime('%Y-%m-%d %H:%M:%S'), current_in - current_out])

    mysql.connection.commit()
    cur.close()

    response = make_response(json.dumps(data))

    response.content_type = 'application/json'

    return response


@app.route('/mask_detection_pie_chart_data', methods=["GET", "POST"])
def mask_detection_pie_chart_data():
    maskFrames = maskStream.maskFrames
    noMaskFrames = maskStream.noMaskFrames
    maskProportion = None
    noMaskProportion = None

    if maskFrames != 0 or noMaskFrames != 0:
        maskProportion = round((maskFrames / (maskFrames + noMaskFrames)), 2)
        noMaskProportion = round((noMaskFrames / (maskFrames + noMaskFrames)), 2)
    else:
        maskProportion = 0
        noMaskProportion = 0

    data = [maskProportion, noMaskProportion]

    response = make_response(json.dumps(data))

    response.content_type = 'application/json'

    return response


@app.route('/backgroundMask', methods=["GET", "POST"])
def backgroundMask():
    infos = dict(
        currentFps=maskStream.get_fps(),
        detected=maskStream.amount_detected,
        withMask=maskStream.withMask,
        withoutMask=maskStream.withoutMask
    )

    return infos


@app.route('/backgroundPersonCounter', methods=["GET", "POST"])
def backgroundPersonCounter():
    infos = dict(
        currentFps=personCounterStream.get_fps(),
        currentAmountOfPersonInside=personCounterStream.totalPersonsInside,
        personsIn=personCounterStream.totalIn,
        personsOut=personCounterStream.totalOut,
        yellow_mark=personCounterStream.yellow_mark,
        red_mark=personCounterStream.red_mark,


    )

    return infos


@app.route('/backgroundMask', methods=["GET", "POST"])
def backgroundDistance():
    infos = dict(
        currentFps=maskStream.get_fps(),
        detected=maskStream.amount_detected,
        withMask=maskStream.withMask,
        withoutMask=maskStream.withoutMask
    )

    return infos


if __name__ == '__main__':
    ##jetson_cam = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    maskStream = mask_stream(camera_src=0)
    personCounterStream = person_counter_stream()
    PieChartData = PieChartData()

    app.secret_key = 'secret123'
    app.run(debug=True)
