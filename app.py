# flask
from flask import flash, redirect, url_for, session, request, send_file

from flask_mysqldb import MySQL
from flask import Flask, render_template, make_response, Response

# core
from core.stream import Stream
from core.jetson_cam import camera
from core.object_counter import Counter
from core.detector import Detector

from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from core.mask_detection_scripts import generateFrames
from person_counter_scripts import generateFrames_PersonCounter
from create_visitor_plots import *

from core.person_counter_script import genFrames
from core.person_counter_script import get_TotalIn, get_TotalOut

from core.mask_detection_2 import Stream as mask_stream

###
import threading

from tensorflow.keras.models import load_model

import time as time2
import cv2
import json
from random import random

from time import time

outputFrame = None
lock = threading.Lock()

###


app = Flask(__name__)

app.register_blueprint(plots)

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flask_dashboard'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
# init MYSQL
mysql = MySQL(app)

prototxtPath = 'models/face_detector/deploy.prototxt'
weightsPath = 'models/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# faceNet.setPreferableTarget(cv2.dnn. DNN_TARGET_CUDA) #or cv2.dnn.DNN_TARGET_CUDA_FP16

maskNet = load_model('models/face_detection_mobilenetv2')

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
        rights = cur.execute("SELECT rights FROM users WHERE username = %s", [session['username']])

        cur.close()

        if 'logged_in' in session and rights == 1:
            return f(*args, **kwargs)
        else:
            flash('Kein Zugriff! Bitte kontaktieren Sie ihren Vorgesetzen oder den Datenbank Administrator', 'danger')
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
        flash("Camera turn off!", "info")
    elif camera is not None and camera == 'on' and maskStream.status() == False:
        maskStream.open()
        flash("Camera turn on!", "success")

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
    return render_template('person_counter.html')


@app.route('/distance_messurement')
@is_logged_in
@check_rights
def distance_messurement():
    camera = request.args.get("camera")

    if camera is not None and camera == 'off' and maskStream.status() == True:
        maskStream.close()
        flash("Camera turn off!", "info")
    elif camera is not None and camera == 'on' and maskStream.status() == False:
        maskStream.open()
        flash("Camera turn on!", "success")

    setting = dict(
        stream_on=maskStream.status(),
        w=600
    )
    return render_template('distance_messurement.html', setting=setting)


# Articles
@app.route('/articles')
def articles():
    # Create cursor
    cur = mysql.connection.cursor()

    # Get articles
    result = cur.execute("SELECT * FROM articles")

    articles = cur.fetchall()

    if result > 0:
        return render_template('articles.html', articles=articles)
    else:
        msg = 'No Articles Found'
        return render_template('articles.html', msg=msg)
    # Close connection
    cur.close()


@app.route('/kontakt')
def kontakt():
    return render_template('kontakt.html')


@app.route('/analytics', methods=['GET', 'POST'])
@is_logged_in
@check_rights
def analytics():
    plot_visitor_today()
    return render_template('analytics.html')


# Single Article
@app.route('/article/<string:id>/')
def article(id):
    # Create cursor
    cur = mysql.connection.cursor()

    # Get article
    result = cur.execute("SELECT * FROM articles WHERE id = %s", [id])

    article = cur.fetchone()

    return render_template('article.html', article=article)


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


# Dashboard
@app.route('/dashboard')
@is_logged_in
def dashboard():
    # Create cursor
    cur = mysql.connection.cursor()

    # Get articles
    # result = cur.execute("SELECT * FROM articles")
    # Show articles only from the user logged in 
    result = cur.execute("SELECT * FROM articles WHERE author = %s", [session['username']])

    articles = cur.fetchall()

    if result > 0:
        return render_template('dashboard.html', articles=articles)
    else:
        msg = 'No Articles Found'
        return render_template('dashboard.html', msg=msg)
    # Close connection
    cur.close()


# Article Form Class
class ArticleForm(Form):
    title = StringField('Title', [validators.Length(min=1, max=200)])
    body = TextAreaField('Body', [validators.Length(min=30)])


# Add Article
@app.route('/add_article', methods=['GET', 'POST'])
@is_logged_in
def add_article():
    form = ArticleForm(request.form)
    if request.method == 'POST' and form.validate():
        title = form.title.data
        body = form.body.data

        # Create Cursor
        cur = mysql.connection.cursor()

        # Execute
        cur.execute("INSERT INTO articles(title, body, author) VALUES(%s, %s, %s)", (title, body, session['username']))

        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('Article Created', 'success')

        return redirect(url_for('dashboard'))

    return render_template('add_article.html', form=form)


# Edit Article
@app.route('/edit_article/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_article(id):
    # Create cursor
    cur = mysql.connection.cursor()

    # Get article by id
    result = cur.execute("SELECT * FROM articles WHERE id = %s", [id])

    article = cur.fetchone()
    cur.close()
    # Get form
    form = ArticleForm(request.form)

    # Populate article form fields
    form.title.data = article['title']
    form.body.data = article['body']

    if request.method == 'POST' and form.validate():
        title = request.form['title']
        body = request.form['body']

        # Create Cursor
        cur = mysql.connection.cursor()
        app.logger.info(title)
        # Execute
        cur.execute("UPDATE articles SET title=%s, body=%s WHERE id=%s", (title, body, id))
        # Commit to DB
        mysql.connection.commit()

        # Close connection
        cur.close()

        flash('Article Updated', 'success')

        return redirect(url_for('dashboard'))

    return render_template('edit_article.html', form=form)


# Delete Article
@app.route('/delete_article/<string:id>', methods=['POST'])
@is_logged_in
def delete_article(id):
    # Create cursor
    cur = mysql.connection.cursor()

    # Execute
    cur.execute("DELETE FROM articles WHERE id = %s", [id])

    # Commit to DB
    mysql.connection.commit()

    # Close connection
    cur.close()

    flash('Article Deleted', 'success')

    return redirect(url_for('dashboard'))


@app.route('/video', methods=['GET', 'POST'])
def video():
    return Response(maskStream.generateFrames(faceNet=faceNet, maskNet=maskNet),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(generateFrames(faceNet, maskNet, camera_off), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/person_counter_video', methods=['GET', 'POST'])
def person_counter_video():
    # return Response(stream.gen_frames(),
    #                mimetype='multipart/x-mixed-replace; boundary=frame')
    global camera_off
    return Response(genFrames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/data', methods=["GET", "POST"])
def data():


    current_in = get_TotalIn()
    current_out = get_TotalOut()

    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO person_counter(timestamp, amount) VALUES (%s,%s)",
                [time2.strftime('%Y-%m-%d %H:%M:%S'), current_in - current_out])

    mysql.connection.commit()
    cur.close()

    data = [time2.strftime('%Y-%m-%d %H:%M:%S'), current_in, current_out]

    response = make_response(json.dumps(data))

    response.content_type = 'application/json'

    return response


@app.route('/mask_detection_data', methods=["GET", "POST"])
def mask_detection_data():
    current_in = get_TotalIn()
    current_out = get_TotalOut()

    response = make_response(json.dumps(current_out))

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
    maskStream = mask_stream(camera_src=0)

    app.secret_key = 'secret123'
    app.run(debug=True)

    """global detector
    w, h = 640, 480



    # load classess data
    classesFile = "models/coco.json"
    with open(classesFile) as json_labels:
        classes = json.load(json_labels)

    # initialize counter object
    lines = []
    lines.append([int(w * 0.50), 0, int(w * 0.50), h])  # LINE 0, x0, y0, x1, y1
    #lines.append([int(w * 0.80), 0, int(w * 0.80), h])  # LINE 1, x0, y0, x1, y1
    counter = Counter(classes, mode='line', lines=lines, threshDist=30)  # mode='line', 'area', 'multiline'

    # initialize model
    detector = Detector(counter, socketio, classes)
    detector.generate_color_maps()
    detector.load_model(model="models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
                        config="models/ssd_mobilenet_v2_coco_2018_03_29/ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

    # initialize stream object
    stream = Stream(camera(0, w, h), detector, counter, socketio, classes)

    # initialize background task
    socketio.start_background_task(target=detector.main)


    # run flask-socketio
    socketio.run(app, debug=True)
    stream.close()"""
