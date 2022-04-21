from flask import Flask, request, render_template
import numpy as np


app = Flask(__name__)


menu = []


def get_prediction(person):
    prediction = 'Test'
    return str(prediction)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/elastic/')
def elastic():
    return render_template('elastic.html', target1='target')


@app.route('/predict_form/', methods=['post', 'get'])
def login():
    message = ''
    if request.method == 'POST':
        person = request.form.get('username')

        person_parameters = person.split(" ")
        person = [float(param) for param in person_parameters]
        person = np.array([person])

        message = get_prediction(person)

    return render_template('login.html', message=message)


if __name__ == '__main__':
    app.run()
