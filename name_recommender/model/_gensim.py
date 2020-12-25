from gensim.models import Doc2Vec
from gensim.models.phrases import Phrases
from gensim.models.doc2vec import TaggedDocument
import pickle
import os
from nltk import ngrams
import random
from ._base import NameRecommendModel


class Dec2VecModel(NameRecommendModel):
    def __init__(self, *args):
        NameRecommendModel.__init__(self, *args)
        self.all_names_list = list(self.all_names)

    def make_bigrams(self, name):
        r = ["".join(list(gram)) for gram in ngrams(name, self.opts.ngrams)]
        return r

    def train(self):
        if self.opts.train or not os.path.exists(self.opts.model_path):
            # Train model
            documents = [
                TaggedDocument(self.make_bigrams(list(name)), [i])
                for i, name in enumerate(self.all_names_list)
            ]
            self.model = Doc2Vec(size=5, window=6, min_count=2, workers=4, iter=10)
            self.model.build_vocab(documents)
            self.model.train(
                documents,
                total_examples=self.model.corpus_count,
                epochs=self.model.iter,
            )
            with open(self.opts.model_path, "wb") as model_pickle:
                pickle.dump(self.model, model_pickle)
        else:
            with open(self.opts.model_path, "rb") as model_pickle:
                self.model = pickle.load(model_pickle)

    def vectorize(self, name):
        return self.model.infer_vector(self.make_bigrams(name))

    def make_recommendation(self):
        possible_suggestions = []

        for name in self.positive_names:
            inferred_vector = self.vectorize(name)
            sims = self.model.docvecs.most_similar([inferred_vector], topn=100)
            print("{0} top {1}:".format("".join(name), len(sims)))
            print(sims[0])
            print(self.model.docvecs.index_to_doctag(sims[0][0]))
            n_printed = 0
            for sim in [(self.all_names_list[i], s) for i, s in sims]:
                if n_printed >= 5:
                    break
                if (
                    not sim[0] in self.negative_names
                    and not sim[0] in self.positive_names
                ):
                    print("\t", sim)
                    n_printed += 1
                    possible_suggestions.append(sim)

        negative_name_vectors = [self.vectorize(name) for name in self.negative_names]
        positive_name_vectors = [self.vectorize(name) for name in self.positive_names]

        sims = self.model.docvecs.most_similar(
            positive=positive_name_vectors, negative=negative_name_vectors, topn=10
        )

        print("Top 10 of all:")
        for sim in [(self.all_names_list[i], s) for i, s in sims]:
            print("\t", sim)
            possible_suggestions.append(sim)

        sum_similarity = sum((sim[1] for sim in possible_suggestions))
        choice = random.uniform(0, sum_similarity)
        comul = 0
        for name, sim in possible_suggestions:
            comul += sim
            if comul > choice:
                return name
        return possible_suggestions[-1][0]
