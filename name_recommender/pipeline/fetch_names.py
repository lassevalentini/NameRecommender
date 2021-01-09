from parsel import Selector
import requests
from argparse import ArgumentParser


def run():
    parser = ArgumentParser()
    parser.add_argument(
        "base_url",
        help="Base URL to find file on",
        default="https://familieretshuset.dk/navne/navne/godkendte-fornavne",
    )

    args = parser.parse_args()

    resp = requests.get(args.base_url)
    doc = Selector(text=resp.text)


if __name__ == "__main__":
    run()
