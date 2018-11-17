import pickle
import json
import numpy
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model
import os

def init():
    global models
    models = []

    model_path = Model.get_model_path('outputs')

    # loop over it to select all models
    alphas = numpy.arange(0.0, 1.0, 0.05)
    for alpha in alphas:
        model_file_name = 'ridge_{0:.2f}.pkl'.format(alpha)
        current_model_path = os.path.join(model_path, model_file_name)
        model = joblib.load(current_model_path)
        tup = (model_file_name, model)
        models.append(tup)    

# note you can pass in multiple rows for scoring
def run(raw_data):
    # randomly select a model
    import random
    random_model_index = random.randint(0, len(models) - 1)

    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result = models[random_model_index][1].predict(data)
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result.tolist(), "model_name": models[random_model_index][0]})
