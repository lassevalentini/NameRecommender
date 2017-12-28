from argparse import ArgumentParser
from model import Dec2VecModel, LstmModel, SimpleRnnModel
import re

parser = ArgumentParser()
parser.add_argument("-t", "--train", action="store_true", help="Train a new prod2vec model", default=False)
parser.add_argument("-m", "--model-path", help="Path to the model pickle", default='model.pickle')
parser.add_argument("-p", "--positive-names", help="", default='positive-names.txt')
parser.add_argument("-n", "--negative-names", help="", default='negative-names.txt')
parser.add_argument("--match-regex", help="", default='^[a-væøå]{3,10}$')
parser.add_argument("--ignore-regex", help="", default='([wxyzq]|aa|ee|uu|ii|oo)')
parser.add_argument("--ngrams", help="", default=2)
parser.add_argument("names", help="Text file with all names, each name on a separate line")

args = parser.parse_args()


def ask_name(name):
	decision = input("{0}? (y/n)".format(name))
	return decision == "y"


def add_to_negative(name):
	negative_names_file.write("\n"+name)
	negative_names_file.flush()
	negative_names.add(name)


def add_to_positive(name):
	positive_names_file.write("\n"+name)
	positive_names_file.flush()
	positive_names.add(name)


def clean_name(name):
	return name.lower().replace('\n', '')


match_regex = re.compile(args.match_regex, re.IGNORECASE)
ignore_regex = re.compile(args.ignore_regex, re.IGNORECASE)

all_names = set()
with open(args.names, "r") as names:
	for name in names:
		if (not match_regex.search(name)) or ignore_regex.search(name):
		# if match_regex.search(name):
			# print("Skipping {0}".format(name[:-1]))
			# print(match_regex.search(name), ignore_regex.search(name))
			# print()
			pass
		else:
			name = clean_name(name)
			all_names.add(name)

	print("{0} names after filtering".format(len(all_names)))

# positive_names = TaggedLineDocument(args.positive_names)
positive_names = set()
positive_names_file = open(args.positive_names, "a+")
positive_names_file.seek(0)
for name in positive_names_file:
	positive_names.add(clean_name(name))

negative_names = set()
negative_names_file = open(args.negative_names, "a+")
negative_names_file.seek(0)
for name in negative_names_file:
	negative_names.add(clean_name(name))


model = LstmModel(args, all_names, positive_names, negative_names)
model.train()

try:
	suggestions_made = 0
	while True:
		suggestions_made += 1
		suggestion = model.make_recommendation()
		decision = ask_name(suggestion)
		if decision:
			add_to_positive(suggestion)
		else:
			add_to_negative(suggestion)
		if suggestions_made % 10 == 0:
			model.retrain()
except KeyboardInterrupt:
	pass
