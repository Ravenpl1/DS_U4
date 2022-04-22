import pandas as pd
from flask import Flask, request, render_template


app = Flask(__name__)


def prediction_elastic(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return 1


def prediction_strength(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return 2


def prediction_filler_matrix(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return 3


def data_preparation(post_data) -> dict:
    data = dict(post_data)
    error = [elem for elem in data.values() if not is_digit(elem)]
    print(error)
    if not error:
        for elem in data:
            data[elem] = float(data.get(elem))
        return data
    return {}


def is_digit(string):
    if string.isdigit():
       return True
    else:
        try:
            float(string)
            return True
        except ValueError:
            return False


@app.route('/')
def index():
    return render_template('index.html', target='target')


@app.route('/elastic/', methods=['post', 'get'])
def elastic():
    if request.method == 'POST':
        data = data_preparation(request.form)
        if data:
            predict = prediction_elastic(data)
            return render_template('message.html',
                                   table_name='Predict',
                                   el1='Модуль упругости при растяжении, ГПа',
                                   el2=str(predict)
                                   )
        else:
            return render_template('error.html')
    return render_template('elastic.html', target1='target')


@app.route('/strength/', methods=['post', 'get'])
def strength():
    if request.method == 'POST':
        data = data_preparation(request.form)
        if data:
            predict = prediction_strength(data)
            return render_template('message.html',
                                   table_name='Predict',
                                   el1='Прочность при растяжении, МПа',
                                   el2=str(predict)
                                   )
        else:
            return render_template('error.html')
    return render_template('strength.html', target2='target')


@app.route('/filler-matrix/', methods=['post', 'get'])
def filler_matrix():
    if request.method == 'POST':
        data = data_preparation(request.form)
        if data:
            predict = prediction_filler_matrix(data)
            return render_template('message.html',
                                   table_name='Predict',
                                   el1='Соотношение матрица-наполнитель',
                                   el2=str(predict)
                                   )
        else:
            return render_template('error.html')
    return render_template('filler-matrix.html', target3='target')


if __name__ == '__main__':
    app.run()
