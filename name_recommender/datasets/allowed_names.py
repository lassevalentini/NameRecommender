from azureml.core import Workspace, Datastore, Dataset

_dataset_name = "allowed_names"
_datastore_name = "lvjdatalake"


def get_dataset(ws: Workspace) -> Dataset:
    if _dataset_name not in ws.datasets:
        datastore = Datastore.get(ws, _datastore_name)

        datastore_paths = [(datastore, f"{_dataset_name}/*.csv")]

        dataset = Dataset.Tabular.from_delimited_files(path=datastore_paths, header=False)
        dataset.register(
            workspace=ws, name=_dataset_name, description="Legally allowed names"
        )
    else:
        dataset = ws.datasets[_dataset_name]
    return dataset
