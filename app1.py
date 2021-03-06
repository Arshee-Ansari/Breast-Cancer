import numpy as np
import pandas as pd
from flask import Flask,request, render_template
import pickle 

app = Flask(__name__)
model = pickle.load(open('Cancer_B', 'rb'))

@app.route('/')
def home():
 return render_template('home1.html')

@app.route('/log1')
def log1():
   return render_template('log1.html')


@app.route('/Abstract')
def Abstract():
   return render_template('Abstract-pg1.html')

@app.route('/Output')
def Output():
   return render_template('output-pg.html')

@app.route('/predict',methods=['POST'])
def predict():
   input_features = [float(x) for x in request.form.values()]
   features_value = [np.array(input_features)]

   features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry','worst fractal dimension']


   df = pd.DataFrame(features_value, columns=features_name)
   output = model.predict(df)
        
   if output == 0:
      res_val = " breast cancer (Malignant)"
   else:
      res_val = "no breast cancer(Benign)"
        

   return render_template('output-pg.html', prediction_text='Patient has {}'.format(res_val)) 

if __name__ == "__main__":
   app.run()