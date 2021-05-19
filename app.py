# flask
from typing import Dict, Any

from flask import flash, redirect, url_for, session, request, jsonify
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

prototxtPath = 'models/face_detector/deploy.prototxt'
weightsPath = 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model('models/face_detection_mobilenetv2')


from core.mask_detection import Stream as mask_stream
from core.person_counter import Stream as person_counter_stream
from core.distance_detection import Stream as distance_detection_stream

"""wenn die Jetson spezifischen Files genutzt werden sollen dann die vorherigen 3 Import Statements auskommentieren und diese verwenden"""


# from core.jetson.mask_detection_jetson import Stream as mask_stream
# from core.jetson.person_counter_jetson import Stream as person_counter_stream
# from core.jetson.distance_detection_jetson import Stream as distance_detection_stream


# region Login und Nutzer Verwaltung

# Überprüfung ob der Nutzer eingeloggt ist
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):

        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Sie sind nicht autorisiert, bitte anmelden!', 'danger')
            return redirect(url_for('login'))

    return wrap


# Überprüfung der Nutzerrechte und entsprechend Zugang freigeben oder verwehren
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


# Registierungsformular
class RegisterForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=50)])
    username = StringField('Benutzername', [validators.Length(min=4, max=25)])
    email = StringField('Email Adresse', [validators.Length(min=6, max=50)])
    password = PasswordField('Passwort', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Die angegebenen Passwörter stimmen nicht überein')
    ])
    confirm = PasswordField('Passwort bestätigen')


# User  Registierung
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        email = form.email.data
        username = form.username.data
        password = sha256_crypt.encrypt(str(form.password.data))

        cur = mysql.connection.cursor()

        cur.execute("INSERT INTO users(name, email, username, password) VALUES(%s, %s, %s, %s)",
                    (name, email, username, password))

        mysql.connection.commit()

        cur.close()

        flash('Sie haben sich erfolgreich registriert', 'success')

        return redirect(url_for('login'))
    return render_template('register.html', form=form)


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Daten von Formular erhalten
        username = request.form['username']
        password_candidate = request.form['password']

        cur = mysql.connection.cursor()

        # Nutzer anhand der Nutzernamen finden
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Passwörter vergleichen
            if sha256_crypt.verify(password_candidate, password):
                session['logged_in'] = True
                session['username'] = username

                flash('Erfolgreich eingeloggt', 'success')
                return redirect(url_for('index'))
            else:
                error = 'Benutzername oder Passwort falsch'
                return render_template('login.html', error=error)
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


# endregion

# region Dashboard Seiten
# Home Seite
@app.route('/')
def index():
    return render_template('home.html')


@app.route('/maskdetection')
@is_logged_in
@check_rights
def maskdetection():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off' and MaskStream.status() == True:
        MaskStream.close()
        flash("Stream beendet", "info")
    elif camera is not None and camera == 'on' and MaskStream.status() == False:
        MaskStream.open()
        flash("Stream erfolgreich gestartet!", "success")

    setting = dict(
        stream_on=MaskStream.status(),
        fps=MaskStream.get_fps(),
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

    if camera is not None and camera == 'off' and PersonCounterStream.status() == True:
        PersonCounterStream.close()
        flash("Stream beendet", "info")
    elif camera is not None and camera == 'on' and PersonCounterStream.status() == False:
        if predefined_person_count is not None:
            PersonCounterStream.totalIn = int(predefined_person_count)
            PersonCounterStream.totalOut = 0

        if threshold:
            PersonCounterStream.minConfidence = float(threshold.replace(',', '.'))

        if yellow_mark is not None:
            PersonCounterStream.yellow_mark = int(yellow_mark)
            # Sobald die gelbe oder rote Markierung vorliegt wird die Ampel benutzt
            use_traffic_light = True

        if red_mark is not None:
            PersonCounterStream.red_mark = int(red_mark)

        PersonCounterStream.open()
        flash("Stream erfolgreich gestartet", "success")

    setting = dict(
        stream_on=PersonCounterStream.status(),
        use_traffic_light=use_traffic_light,

    )
    return render_template('person_counter.html', setting=setting)


@app.route('/distance_messurement')
@is_logged_in
@check_rights
def distance_messurement():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off' and DistanceDetectionStream.status() == True:
        DistanceDetectionStream.close()
        flash("Stream beendet", "info")
    elif camera is not None and camera == 'on' and DistanceDetectionStream.status() == False:
        DistanceDetectionStream.open()
        flash("Stream erfolgreich gestartet", "success")

    setting = dict(
        stream_on=DistanceDetectionStream.status(),
    )
    return render_template('distance_messurement.html', setting=setting)


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


@app.route('/datenschutz')
def datenschutz():
    return render_template('datenschutz.html')


@app.route('/kontakt')
def kontakt():
    return render_template('kontakt.html')

# endregion

# region VideoStreams
@app.route('/mask_detection_video', methods=['GET', 'POST'])
def mask_detection_video():
    return Response(MaskStream.generateFrames(faceNet=faceNet, maskNet=maskNet),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/person_counter_video', methods=['GET', 'POST'])
def person_counter_video():
    return Response(PersonCounterStream.generateFrames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/distance_detection_video', methods=['GET', 'POST'])
def distance_detection_video():
    return Response(DistanceDetectionStream.generateFrames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# endregion

## Daten für das Linien Diagram von PersonCounter + Überführung in die Datenbank
@app.route('/person_counter_data', methods=["GET", "POST"])
def person_counter_data():
    current_in = PersonCounterStream.totalIn
    current_out = PersonCounterStream.totalOut


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
    maskFrames = MaskStream.maskFrames
    noMaskFrames = MaskStream.noMaskFrames
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
        currentFps=MaskStream.get_fps(),
        detected=MaskStream.amount_detected,
        withMask=MaskStream.withMask,
        withoutMask=MaskStream.withoutMask
    )

    return infos


@app.route('/backgroundAbstandskontrolle', methods=["GET", "POST"])
def backgroundAbstandskontrolle():
    distance_info = dict(currentFps=DistanceDetectionStream.get_fps(), detected=DistanceDetectionStream.amount_detected,
                         violations=DistanceDetectionStream.violations)

    return jsonify(distance_info)


@app.route('/backgroundPersonCounter', methods=["GET", "POST"])
def backgroundPersonCounter():
    person_counter_info = dict(
        currentFps=PersonCounterStream.get_fps(),
        currentAmountOfPersonInside=PersonCounterStream.totalPersonsInside,
        personsIn=PersonCounterStream.totalIn,
        personsOut=PersonCounterStream.totalOut,
        yellow_mark=PersonCounterStream.yellow_mark,
        red_mark=PersonCounterStream.red_mark,

    )

    return person_counter_info


if __name__ == '__main__':
    person_counter_video = r"C:\Users\manue\PycharmProjects\einfaches_dashboard_feb_2021\videos\1.mp4"
    distance_detection_video = r"C:\Users\manue\PycharmProjects\einfaches_dashboard_feb_2021\videos\pedestrians.mp4"

    MaskStream = mask_stream(camera_src=0)
    PersonCounterStream = person_counter_stream(
        camera_src=person_counter_video)
    DistanceDetectionStream = distance_detection_stream(
        camera_src=distance_detection_video)

    PieChartData = PieChartData()
    app.secret_key = 'secret123'
    app.run()#debug=True)
