import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, request, render_template

rft_elastic = pickle.load(open('../model/rft_elastic_v01.pkl', 'rb'))
rft_strength = pickle.load(open('../model/rft_strength_v01.pkl', 'rb'))
nrn_mfiller = tf.keras.models.load_model('../model/nrn_mfiller_v01')

std_scaler = pickle.load(open('../data/processed/EDA_out_v01.pkl', 'rb'))
#inversed = std_scaler.inverse_transform(np_std_scaler)
#test = df_std_scaler['Плотность, кг/м3'].iloc[1] * std_scaler.scale_[1] + std_scaler.mean_[1]

app = Flask(__name__)


def input_standartization(data: dict):
    for elem in list(data.keys()):
        if elem == 'Соотношение матрица-наполнитель':
            data['Соотношение матрица-наполнитель'] = (data['Соотношение матрица-наполнитель'] - std_scaler.mean_[0]) / std_scaler.scale_[0]
        elif elem == 'Плотность, кг/м3':
            data['Плотность, кг/м3'] = (data['Плотность, кг/м3'] - std_scaler.mean_[1]) / std_scaler.scale_[1]
        elif elem == 'Модуль упругости, ГПа':
            data['Модуль упругости, ГПа'] = (data['Модуль упругости, ГПа'] - std_scaler.mean_[2]) / std_scaler.scale_[2]
        elif elem == 'Количество отвердителя, м.%':
            data['Количество отвердителя, м.%'] = (data['Количество отвердителя, м.%'] - std_scaler.mean_[3]) / std_scaler.scale_[3]
        elif elem == 'Содержание эпоксидных групп,%_2':
            data['Содержание эпоксидных групп,%_2'] = (data['Содержание эпоксидных групп,%_2'] - std_scaler.mean_[4]) / std_scaler.scale_[4]
        elif elem == 'Температура вспышки, С_2':
            data['Температура вспышки, С_2'] = (data['Температура вспышки, С_2'] - std_scaler.mean_[5]) / std_scaler.scale_[5]
        elif elem == 'Поверхностная плотность, г/м2':
            data['Поверхностная плотность, г/м2'] = (data['Поверхностная плотность, г/м2'] - std_scaler.mean_[6]) / std_scaler.scale_[6]
        elif elem == 'Модуль упругости при растяжении, ГПа':
            data['Модуль упругости при растяжении, ГПа'] = (data['Модуль упругости при растяжении, ГПа'] - std_scaler.mean_[7]) / std_scaler.scale_[7]
        elif elem == 'Прочность при растяжении, МПа':
            data['Прочность при растяжении, МПа'] = (data['Прочность при растяжении, МПа'] - std_scaler.mean_[8]) / std_scaler.scale_[8]
        elif elem == 'Потребление смолы, г/м2':
            data['Потребление смолы, г/м2'] = (data['Потребление смолы, г/м2'] - std_scaler.mean_[9]) / std_scaler.scale_[9]
        elif elem == 'Угол нашивки, град':
            data['Угол нашивки, град'] = (data['Угол нашивки, град'] - std_scaler.mean_[10]) / std_scaler.scale_[10]
        elif elem == 'Шаг нашивки':
            data['Шаг нашивки'] = (data['Шаг нашивки'] - std_scaler.mean_[11]) / std_scaler.scale_[11]
        elif elem == 'Плотность нашивки':
            data['Плотность нашивки'] = (data['Плотность нашивки'] - std_scaler.mean_[12]) / std_scaler.scale_[12]


def prediction_elastic(data: dict) -> pd.DataFrame:
    input_standartization(data)
    x_elastic = pd.DataFrame([data])
    predict_elastic = float(rft_elastic.predict(x_elastic)) * std_scaler.scale_[7] + std_scaler.mean_[7]
    return predict_elastic


def prediction_strength(data: dict) -> pd.DataFrame:
    input_standartization(data)
    x_strength = pd.DataFrame([data])
    predict_strength = float(rft_strength.predict(x_strength)) * std_scaler.scale_[8] + std_scaler.mean_[8]
    return predict_strength


def prediction_filler_matrix(data: dict) -> pd.DataFrame:
    input_standartization(data)
    x_mfiller = pd.DataFrame([data])
    predict_mfiller = float(nrn_mfiller.predict(x_mfiller)) * std_scaler.scale_[0] + std_scaler.mean_[0]
    return predict_mfiller


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
