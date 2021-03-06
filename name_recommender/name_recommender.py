from argparse import ArgumentParser
from model import Dec2VecModel, LstmModel, SimpleRnnModel, AutoencoderModel
import re


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def ask_name(name):
    score = input("{0}? (0-5)".format(name))
    try:
        return int(score)
    except ValueError:
        return ask_name(name)


def add_to_negative(name, score):
    negative_names_file.write("\n{0} {1}".format(score, name))
    negative_names_file.flush()
    name_scores[name] = score


def add_to_positive(name, score):
    positive_names_file.write("\n{0} {1}".format(score, name))
    positive_names_file.flush()
    name_scores[name] = score


def clean_name(name):
    return name.lower().replace("\n", "")


def read_name_file(name_file_name, name_scores):
    name_file = open(name_file_name, "a+", encoding="utf-8")
    name_file.seek(0)
    for line in name_file:
        score, name = line.split(" ", 1)
        name = clean_name(name)
        if name not in name_scores and name in all_names:
            name_scores[name] = int(score)
    return name_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="Train a new dov2vec model (only used by Dec2VecModel)",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--model-path",
        help="Path to the model pickle (only used by Dec2VecModel)",
        default="model.pickle",
    )
    parser.add_argument(
        "-p",
        "--positive-names",
        help="File with positive ratings (score > 2)",
        default="positive-names.txt",
    )
    parser.add_argument(
        "-n",
        "--negative-names",
        help="File with positive ratings (score <= 2)",
        default="negative-names.txt",
    )
    parser.add_argument(
        "-b",
        "--balance-classes",
        help="Should positive and negative names be balanced for training (possible values: upsample, downsample, none)",
        default="upsample",
    )
    parser.add_argument("-s", "--sample-factor", help="", type=int, default=10)
    parser.add_argument(
        "--match-regex",
        help="Names must match regex to be considered",
        default="^[a-væøå]{3,10}$",
    )
    parser.add_argument(
        "--ignore-regex",
        help="Names must not match regex to be considered",
        default="([wxyzq]|aa|ee|uu|ii|oo)",
    )
    parser.add_argument(
        "--ngrams", help="Size of n-grams used for Dec2VecModel", default=2
    )
    parser.add_argument(
        "names", help="Text file with all names, each name on a separate line"
    )

    args = parser.parse_args()

    match_regex = re.compile(args.match_regex, re.IGNORECASE)
    ignore_regex = re.compile(args.ignore_regex, re.IGNORECASE)

    all_names = set()
    with open(args.names, "r", encoding="utf-8") as names:
        for name in names:
            if (not match_regex.search(name)) or ignore_regex.search(name):
                pass
            else:
                name = clean_name(name)
                all_names.add(name)

        print("{0} names after filtering".format(len(all_names)))

    name_scores = {}
    positive_names_file = read_name_file(args.positive_names, name_scores)
    negative_names_file = read_name_file(args.negative_names, name_scores)

    print("Using LstmModel - for other models see name_recommender.py")
    model = LstmModel(args, all_names, name_scores)
    ### Alternative models ###
    # model = SimpleRnnModel(args, all_names, name_scores)
    # model = Dec2VecModel(args, all_names, name_scores)
    # model = AutoencoderModel(args, all_names, name_scores)
    model.train()

    try:
        suggestions_made = 0
        while True:
            suggestions_made += 1
            suggestion = model.make_recommendation()

            if not suggestion:
                model.retrain()
                continue

            score = ask_name(suggestion)
            if score > 2:
                add_to_positive(suggestion, score)
            else:
                add_to_negative(suggestion, score)

            if suggestions_made % 10 == 0:
                model.retrain()

    except KeyboardInterrupt:
        pass
