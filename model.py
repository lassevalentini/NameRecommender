from gensim.models import Doc2Vec
from gensim.models.phrases import Phrases
from gensim.models.doc2vec import TaggedDocument
import pickle
import os
from nltk import ngrams
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking, Dropout
from keras.layers import LSTM, SimpleRNN
from keras.optimizers import RMSprop
import numpy as np


class NameRecommendModel:
	def __init__(self, opts, all_names, positive_names, negative_names):
		self.opts = opts
		self.all_names = all_names
		self.negative_names = negative_names
		self.positive_names = positive_names
		print(len(all_names), len(positive_names), len(negative_names))

	def possible_names(self):
		return [name for name in self.all_names if not name in self.negative_names and not name in self.positive_names]

	def train(self):
		pass

	def retrain(self):
		pass

	def make_recommendation(self):
		pass


class Dec2VecModel(NameRecommendModel):
	def make_bigrams(self, name):
		r = ["".join(list(gram)) for gram in ngrams(name, self.opts.ngrams)]
		return r
		# return bigram_transformer[name]


	def train(self):
		# bigram_transformer = Phrases((list(name) for name in self.all_names), threshold=1)
		if self.opts.train or not os.path.exists(self.opts.model_path):
			# Train model
			documents = [TaggedDocument(self.self.make_bigrams(list(name)), [i]) for i,name in enumerate(self.all_names)]
			self.model = Doc2Vec(size=25, window=2, min_count=5, workers=4, iter=20)
			self.model.build_vocab(documents)
			self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.iter)
			with open(self.opts.model_path, 'wb') as model_pickle:
				pickle.dump(self.model, model_pickle)
		else:
			with open(self.opts.model_path, 'rb') as model_pickle:
				self.model = pickle.load(model_pickle)


	def vectorize(self, name):
		return self.model.infer_vector(self.make_bigrams(name))


	def make_recommendation(self):
		possible_suggestions = []

		for name in self.positive_names:
			print(name, self.make_bigrams(name))
			inferred_vector = self.vectorize(name)
			sims = self.model.docvecs.most_similar([inferred_vector], topn=100)
			print("{0} top {1}:".format("".join(name), len(sims)))
			n_printed = 0
			for sim in [(self.all_names[i],s) for i,s in sims]:
				if n_printed >= 5:
					break
				if not sim[0] in self.negative_names and not sim[0] in self.positive_names:
					print("\t", sim)
					n_printed+=1
					possible_suggestions.append(sim)

		negative_name_vectors=[self.vectorize(name) for name in self.negative_names]
		positive_name_vectors=[self.vectorize(name) for name in self.positive_names]
		
		sims = self.model.docvecs.most_similar(positive=positive_name_vectors, negative=negative_name_vectors, topn=10)

		print("Top 10 of all:")
		for sim in [(self.all_names[i],s) for i,s in sims]:
			print("\t", sim)
			possible_suggestions.append(sim)

		sum_similarity = sum((sim[1] for sim in possible_suggestions))
		choice = random.uniform(0, sum_similarity)
		comul = 0
		for name,sim in possible_suggestions:
			comul+=sim
			if comul > choice:
				return name
		return possible_suggestions[-1][0]



class KerasSequentialModel(NameRecommendModel):
	def __init__(self, *args):
		NameRecommendModel.__init__(self, *args)
		self.name_scores = None

	def name_to_features(self, name):
		x = np.zeros((self.maxlen, len(self.chars_map)), dtype=np.bool)
		for i,c in enumerate(name):
			char_index = self.chars_map[c]
			x[i, char_index] = 1
		return x

	def retrain(self):
		self.train();
		self.name_scores = None


	def build_model(self):
		pass

	def fit(self, x, y):
		pass

	def train(self):
		chars = set()
		self.maxlen = 1
		for name in self.all_names:
			self.maxlen = max(len(name), self.maxlen)
			for char in name:
				chars.add(char)
		self.chars_map = {c: i for i,c in enumerate(chars)}

		self.build_model()

		train_names = [name for name in self.positive_names if name in self.all_names] + [name for name in self.negative_names if name in self.all_names]

		x = np.zeros((len(train_names), self.maxlen, len(self.chars_map)), dtype=np.bool)
		y = np.zeros(len(train_names), dtype=np.bool)

		for i,name in enumerate(train_names):
			if name in self.positive_names:
				y[i] = 1
			elif name in self.negative_names:
				y[i] = 0
			else:
				y[i] = 0.5
				print("Name not in positive or negative: {0}".format(name))

			# for j,c in enumerate(name):
			# 	char_index = self.chars_map[c]
			# 	x[i, j, char_index] = 1
			x[i] = self.name_to_features(name)

		self.fit(x, y)


	def make_recommendation(self):
		if self.name_scores is None:
			self.name_scores = []
			possible_names = self.possible_names()
			x_pred = np.zeros((len(possible_names), self.maxlen, len(self.chars_map)))
			for i, name in enumerate(possible_names):
				x_pred[i] = self.name_to_features(name)

			scores = self.model.predict(x_pred, verbose=0)

			for i, (name, score) in enumerate(zip(possible_names, scores)):
				self.name_scores.append((name, score[0]))
				self.name_scores = sorted(self.name_scores, key=lambda x: x[1])

		selected_name = self.name_scores.pop(-1)
		print(selected_name)
		return selected_name[0]



class LstmModel(KerasSequentialModel):
	def build_model(self):
		self.batch_size = 5

		self.model = Sequential()
		self.model.add(Masking(
			mask_value=0, 
			input_shape=(self.maxlen, len(self.chars_map))))
		self.model.add(LSTM(64))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1, activation='sigmoid'))

		optimizer = RMSprop(lr=0.003)
		self.model.compile(loss='binary_crossentropy', optimizer=optimizer)


	def fit(self, x, y):
		self.model.fit(x, y, batch_size=self.batch_size, epochs=120)



class SimpleRnnModel(KerasSequentialModel):
	def build_model(self):
		self.model = Sequential()
		self.model.add(Masking(mask_value=0, input_shape=(self.maxlen, len(self.chars_map))))
		self.model.add(SimpleRNN(64))
		# self.model.add(Dropout(0.2))
		self.model.add(Dense(1, activation='sigmoid'))

		optimizer = RMSprop()
		self.model.compile(loss='binary_crossentropy', optimizer=optimizer)


	def fit(self, x, y):
		self.model.fit(x, y, batch_size=5, epochs=80)

