import mlflow
import mlflow.azureml
from types import SimpleNamespace
from azureml.core import Run
from argparse import ArgumentParser
from name_recommender.datasets import allowed_names, name_sentiments
from name_recommender.model import (
    Dec2VecModel,
    LstmModel,
    SimpleRnnModel,
    AutoencoderModel,
)
import joblib
import os

# parser = ArgumentParser()
# parser.add_argument("--allowed-names", type=str)
# parser.add_argument("--name-sentiments", type=str)
# args = parser.parse_args()

with mlflow.start_run():
    run = Run.get_context()
    ws = run.experiment.workspace
    allowed_names_ds = allowed_names.get_dataset(ws)
    name_sentiments_ds = name_sentiments.get_dataset(ws)

    allowed_names_df = allowed_names_ds.to_pandas_dataframe()
    name_sentiments = name_sentiments_ds.to_pandas_dataframe()

    allowed_names = [n.lower() for n in allowed_names_df.iloc[:, 0].tolist()]
    name_sentiments = name_sentiments.set_index("name")["score"]
    model = LstmModel(
        SimpleNamespace(**{"balance_classes": "no"}), allowed_names, name_sentiments
    )
    model.train()
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/name_lstm.pkl"
    joblib.dump(value=model, filename=model_path)
    run.upload_file(name=model_path, path_or_stream=model_path)

    # Register the model
    run.register_model(
        model_path=model_path,
        model_name="name_lstm",
    )

    run.complete()
