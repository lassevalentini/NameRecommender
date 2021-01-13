import mlflow
import mlflow.azureml
from types import SimpleNamespace
from azureml.core import Run, Dataset
from argparse import ArgumentParser
from name_recommender.datasets import allowed_names, name_sentiments
from name_recommender.model import (
    Dec2VecModel,
    LstmModel,
    SimpleRnnModel,
    AutoencoderModel,
)

# parser = ArgumentParser()
# parser.add_argument("--allowed-names", type=str)
# parser.add_argument("--name-sentiments", type=str)
# args = parser.parse_args()

with mlflow.start_run():
    run = Run.get_context()
    ws = run.experiment.workspace
    allowed_names_ds = allowed_names.get_dataset()
    name_sentiments_ds = name_sentiments.get_dataset()

    allowed_names = allowed_names_ds.to_pandas_dataframe()
    name_sentiments = name_sentiments_ds.to_pandas_dataframe()

    name_sentiments = name_sentiments.set_index("name")["score"]
    model = LstmModel(
        SimpleNamespace(**{"balance_classes": "no"}), allowed_names, name_sentiments
    )
    model.train()
    print(model.make_recommendation())
