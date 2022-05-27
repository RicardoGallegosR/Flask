from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('Flask/model.pkl','rb'))
# input_query = np.array([['8.8','100','80']])
# input_query = np.array([[8.8,100,80]])

# model.predict(input_query)[0]

app = Flask(__name__)


@app.route('/')
def index():
    return "<h1>Hello world</h1>"

@app.route('/predict',methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')
    
    input_query = np.array([[cgpa,iq,profile_score]])
    result = model.predict(input_query)[0]
    
    return jsonify({'placement':str(result)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)