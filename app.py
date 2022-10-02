from recommender import anime_recommendations

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

# print(anime_recommendations('One Punch Man'))

app = Flask(__name__,static_folder='', static_url_path='')

similarity_data=pd.read_pickle("./dummy.pkl")

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    data = request.get_json(force=True)

    try:
        prediction = anime_recommendations(data,similarity_data)
        print(prediction)
    except NameError:
        print("hai")

    result = prediction

    return jsonify({'value': result})

if __name__ == "__main__":
    app.run(debug=True)