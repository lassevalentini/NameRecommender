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
parser.add_argument("-l", "--like-names", help="", default='like-names.txt')
parser.add_argument("--ignore-regex", help="", default='[xzq]')
parser.add_argument("names", help="Text file with all names, each name on a separate line")

args = parser.parse_args()

ignore_regex = re.compile(args.ignore_regex)

if args.train or not os.path.exists(args.model_path):
	# Train model
	# documents = TaggedLineDocument(args.names)
	documents = []
	with open(args.names, "r") as names:
		for i, name in enumerate((name if not ignore_regex.match(name) for name in names)):
			print("Adding {0}".format(name))
			documents.append(TaggedDocument(list(name.replace('\n', '')), [i]))
	print("{0} names after filtering".format(len(documents)))
	model = Doc2Vec(size=10, window=2, min_count=5, workers=4, iter=100)
	model.build_vocab(documents)
	model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
	with open(args.model_path, 'wb') as model_pickle:
		pickle.dump(model, model_pickle)
else:
	with open(args.model_path, 'rb') as model_pickle:
		model = pickle.load(model_pickle)

# like_names = TaggedLineDocument(args.like_names)
like_names = []
with open(args.like_names, "r") as names:
	for i, name in enumerate(names):
		like_names.append(list(name.replace('\n', '')))

for name in like_names:
	inferred_vector = model.infer_vector(name)
	sims = model.docvecs.most_similar([inferred_vector], topn=5)
	print("{0} top {1}:".format("".join(name), len(sims)))
	for sim in [("".join(documents[i].words),s) for i,s in sims]:
		print("\t", sim)

sims = model.docvecs.most_similar(positive=[model.infer_vector(name) for name in like_names], topn=10)
print("Top 10 of all:")
for sim in [("".join(documents[i].words),s) for i,s in sims]:
	print("\t", sim)