from flask import Flask,render_template,request
import pickle
import numpy as np
from inference import Inference
import pandas as pd


sym = pd.read_csv('./data/raw_data/symptom_severity.csv')
sym_list = list(sym['Symptom'].values)
print("Total Known Symptom:- ",len(sym_list))


model_path = "bert-base-uncased"
local_model_dict = "./model/model_bert-base-uncased_fold-1"
max_length = 64
num_classes = 41
sample_text = [""]
inf = Inference(model_path,local_model_dict,num_classes,max_length)

res1 = inf.get_results(sample_text)
print(res1)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html",sym_list=sym_list)


@app.route('/predict',methods=['post'])
def predict():
    pass


if __name__ == '__main__':
    app.run(debug=True)