import json
import joblib
from azureml.core.model import Model


# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path("name_lstm")
    model = joblib.load(model_path)


# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    input = json.loads(raw_data)
    if input is list:
        max_id = max(input)
        expanded_list = [model.make_recommendation() for _ in range(max_id)]
        predictions = [expanded_list[i] for i in input]
    else:
        predictions = [model.make_recommendation() for _ in range(input)]

    # Return the predictions as JSON
    return json.dumps(predictions)
