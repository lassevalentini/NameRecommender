import argparse
from pathlib import Path

from azureml.core import Workspace, Datastore, Environment
from azureml.core.compute import ComputeTarget, AmlCompute, AksCompute

# setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="")
parser.add_argument("--subscription-id", type=str, default=None)
parser.add_argument("--workspace-name", type=str, default="default")
parser.add_argument("--resource-group", type=str, default="azureml")
parser.add_argument("--location", type=str, default="westeu")
parser.add_argument("--datalake-name", type=str, default="lvjdatalake")
parser.add_argument("--datalake-rg-name", type=str, default="datalake")
parser.add_argument("--create-aks", type=bool, default=False)
args = parser.parse_args()

amlcomputes = {
    "cpu-cluster": {
        "vm_size": "STANDARD_DS3_V2",
        "min_nodes": 0,
        "max_nodes": 10,
        "idle_seconds_before_scaledown": 1200,
    },
    "gpu-cluster": {
        "vm_size": "STANDARD_NC6",
        "min_nodes": 0,
        "max_nodes": 4,
        "idle_seconds_before_scaledown": 1200,
    },
}

akscomputes = {
    # "aks-cpu-deploy": {
    #     "vm_size": "STANDARD_DS3_V2",
    #     "agent_count": 3,
    # },
    # "aks-gpu-deploy": {
    #     "vm_size": "STANDARD_NC6S_V3",
    #     "agent_count": 3,
    # },
}


try:
    ws = Workspace.from_config(args.config)
except Exception:
    ws = Workspace.create(
        args.workspace_name,
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        location=args.location,
        create_resource_group=True,
        exist_ok=True,
        show_output=True,
    )
    ws.write_config()

# create aml compute targets
for ct_name in amlcomputes:
    if ct_name not in ws.compute_targets:
        print(f"Creating AML compute {ct_name}")
        compute_config = AmlCompute.provisioning_configuration(**amlcomputes[ct_name])
        ct = ComputeTarget.create(ws, ct_name, compute_config)
        ct.wait_for_completion(show_output=True)

# create aks compute targets
if args.create_aks:
    for ct_name in akscomputes:
        if ct_name not in ws.compute_targets:
            print(f"Creating AKS compute {ct_name}")
            compute_config = AksCompute.provisioning_configuration(**akscomputes[ct_name])
            ct = ComputeTarget.create(ws, ct_name, compute_config)
            ct.wait_for_completion(show_output=True)

if args.datalake_name not in ws.datastores:
    print(f"Creating datastore {args.datalake_name}")
    adlsgen2_datastore = Datastore.register_azure_data_lake_gen2(
        workspace=ws,
        datastore_name=args.datalake_name,
        account_name=args.datalake_name,
        subscription_id=ws.subscription_id,
        resource_group=args.datalake_rg_name,
        filesystem="datalake",
        grant_workspace_access=True,
    )

for env_file in Path("./environments/").glob("*.yml"):
    print(f"Creating env {env_file.name}")
    env = Environment.from_conda_specification(name=env_file.name, file_path=env_file)
    env.register(ws)
