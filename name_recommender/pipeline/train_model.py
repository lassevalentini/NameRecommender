import mlflow
import mlflow.azureml
from types import SimpleNamespace
from azureml.core import Run, Dataset
from argparse import ArgumentParser
from ..model import Dec2VecModel, LstmModel, SimpleRnnModel, AutoencoderModel

parser = ArgumentParser()
parser.add_argument("--allowed-names", type=str)
parser.add_argument("--names-sentiments", type=str)
args = parser.parse_args()

with mlflow.start_run():
    run = Run.get_context()
    ws = run.experiment.workspace
    allowed_names_ds = Dataset.get_by_id(ws, id=args.allowed_names)
    name_sentiments_ds = Dataset.get_by_id(ws, id=args.name_sentiments)

    allowed_names = allowed_names_ds.to_pandas_dataframe()
    name_sentiments = name_sentiments_ds.to_pandas_dataframe()

    name_sentiments = name_sentiments.set_index("name")["score"]
    model = LstmModel(SimpleNamespace(**{}), allowed_names, name_sentiments)
    model.train()
    print(model.make_recommendation())
