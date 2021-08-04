from flask import Flask, request
import pickle
import numpy as np
import sys

app = Flask(__name__)
app.config["DEBUG"] = True

# file path to the saved model
model_filepath = 'model.sav'
try:
    model = pickle.load(open(model_filepath, 'rb'))
except:
    sys.exit('Unable to load the model')


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get parameters for temperature
        temperature = float(request.args.get('temp'))
        # Get parameters for humidity
        humidity = float(request.args.get('hmd'))

        # Predict Apparent temperature
        # Same order as the x_train dataframe
        features = [np.array([temperature, humidity])]
        prediction = model.predict(features)
        output = round(prediction[0][0], 2)

        return {'apparent_temp': output}
    except Exception as e:
        print(e)
        return 'Calculation Error', 500


app.run()