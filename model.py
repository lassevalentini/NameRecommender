from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from argparse import ArgumentParser
import json
import pickle
import os
import re

parser = ArgumentParser()
parser.add_argument("-t", "--train", action="store_true", help="Train a new prod2vec model", default=False)
parser.add_argument("-m", "--model-path", help="Path to the model pickle", default='model.pickle')
parser.add_argument("-p", "--positive-names", help="", default='positive-names.txt')
parser.add_argument("-n", "--negative-names", help="", default='negative-names.txt')
parser.add_argument("--match-regex", help="", default='^[a-væøå]+$')
parser.add_argument("--ignore-regex", help="", default='([wxyzq]|aa|ee|uu)')
parser.add_argument("names", help="Text file with all names, each name on a separate line")

args = parser.parse_args()

match_regex = re.compile(args.match_regex, re.IGNORECASE)
ignore_regex = re.compile(args.ignore_regex, re.IGNORECASE)

if args.train or not os.path.exists(args.model_path):
	# Train model
	# documents = TaggedLineDocument(args.names)
	documents = []
	with open(args.names, "r") as names:
		i = 0
		for name in names:
			if (not match_regex.search(name)) or ignore_regex.search(name):
			# if match_regex.search(name):
				# print("Skipping {0}".format(name[:-1]))
				# print(match_regex.search(name), ignore_regex.search(name))
				# print()
				pass
			else:
				documents.append(TaggedDocument(list(name.lower().replace('\n', '')), [i]))
				i += 1

	print("{0} names after filtering".format(len(documents)))
	model = Doc2Vec(size=50, window=2, min_count=5, workers=4, iter=20)
	model.build_vocab(documents)
	model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
	with open(args.model_path, 'wb') as model_pickle:
		pickle.dump(model, model_pickle)
else:
	with open(args.model_path, 'rb') as model_pickle:
		model = pickle.load(model_pickle)

# positive_names = TaggedLineDocument(args.positive_names)
positive_names = []
with open(args.positive_names, "r") as names:
	for i, name in enumerate(names):
		positive_names.append(list(name.lower().replace('\n', '')))

negative_names = []
with open(args.negative_names, "r") as names:
	for i, name in enumerate(names):
		negative_names.append(model.infer_vector(list(name.lower().replace('\n', ''))))

for name in positive_names:
	inferred_vector = model.infer_vector(name)
	sims = model.docvecs.most_similar([inferred_vector], topn=5)
	print("{0} top {1}:".format("".join(name), len(sims)))
	for sim in [("".join(documents[i].words),s) for i,s in sims]:
		print("\t", sim)

sims = model.docvecs.most_similar(positive=[model.infer_vector(name) for name in positive_names], negative=negative_names, topn=10)
print("Top 10 of all:")
for sim in [("".join(documents[i].words),s) for i,s in sims]:
	print("\t", sim)