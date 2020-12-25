from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Masking, Dropout
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from random import shuffle
from ._base import NameRecommendModel


class KerasSequentialModel(NameRecommendModel):
    def __init__(self, *args):
        NameRecommendModel.__init__(self, *args)
        self.estimated_name_scores = None

    def name_to_features(self, name):
        x = np.zeros((self.maxlen, len(self.chars_map)), dtype=np.uint8)
        for i, c in enumerate(name):
            char_index = self.chars_map[c]
            x[i, char_index] = 1
        return x

    def retrain(self):
        self.train()
        self.estimated_name_scores = None
        self.recommendations_made = 0

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
        self.chars_map = {c: i for i, c in enumerate(chars)}

        self.build_model()

        x = np.zeros(
            (len(self.training_set), self.maxlen, len(self.chars_map)), dtype=np.uint8
        )
        y = np.zeros(len(self.training_set), dtype=np.uint8)

        for i, (name, score) in enumerate(self.training_set):
            y[i] = self.scale_score(score)

            x[i] = self.name_to_features(name)

        self.fit(x, y)
        self.recommendations_made = 0

    def make_recommendation(self):
        if self.estimated_name_scores is None:
            self.estimated_name_scores = []
            possible_names = self.possible_names()
            x_pred = np.zeros((len(possible_names), self.maxlen, len(self.chars_map)))
            for i, name in enumerate(possible_names):
                x_pred[i] = self.name_to_features(name)

            scores = self.model.predict(x_pred, verbose=0)

            for i, (name, score) in enumerate(zip(possible_names, scores)):
                self.estimated_name_scores.append((name, score[0]))
                self.estimated_name_scores = sorted(
                    self.estimated_name_scores, key=lambda x: x[1]
                )

        selected_name = self.estimated_name_scores.pop(-1)
        if selected_name[1] > 0.1 or self.recommendations_made == 0:
            self.recommendations_made += 1
            return selected_name[0]
        return None


class LstmModel(KerasSequentialModel):
    def build_model(self):
        self.batch_size = 5

        self.model = Sequential()
        self.model.add(
            Masking(mask_value=0, input_shape=(self.maxlen, len(self.chars_map)))
        )
        self.model.add(LSTM(20))
        # self.model.add(Dropout(0.3))
        self.model.add(Dense(1, activation="sigmoid"))

        optimizer = RMSprop(lr=0.0001)
        self.model.compile(loss="mean_squared_error", optimizer=optimizer)

    def fit(self, x, y):
        self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=120,
            shuffle=True,
            callbacks=[
                EarlyStopping(
                    monitor="loss", min_delta=0.001, patience=5, verbose=1, mode="min"
                )
            ],
        )


class SimpleRnnModel(KerasSequentialModel):
    def build_model(self):
        self.batch_size = 5

        self.model = Sequential()
        self.model.add(
            Masking(mask_value=0, input_shape=(self.maxlen, len(self.chars_map)))
        )
        self.model.add(SimpleRNN(64, return_sequences=True))
        self.model.add(Dropout(0.4))
        self.model.add(SimpleRNN(32, return_sequences=True))
        self.model.add(SimpleRNN(10))
        # self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation="sigmoid"))

        optimizer = RMSprop(lr=0.0001)
        self.model.compile(loss="mean_squared_error", optimizer=optimizer)

    def fit(self, x, y):
        self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=120,
            shuffle=True,
            callbacks=[
                EarlyStopping(
                    monitor="loss", min_delta=0.002, patience=5, verbose=1, mode="min"
                )
            ],
        )


class GRUModel(KerasSequentialModel):
    def build_model(self):
        self.batch_size = 5

        self.model = Sequential()
        self.model.add(
            Masking(mask_value=0, input_shape=(self.maxlen, len(self.chars_map)))
        )
        self.model.add(GRU(32))
        # self.model.add(GRU(32, dropout=0.2, return_sequences=True))
        # self.model.add(GRU(10))
        # self.model.add(Dropout(0.4))
        self.model.add(Dense(1, activation="sigmoid"))

        optimizer = RMSprop(lr=0.0001)
        self.model.compile(loss="mean_squared_error", optimizer=optimizer)

    def fit(self, x, y):
        self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=120,
            shuffle=True,
            callbacks=[
                EarlyStopping(
                    monitor="loss", min_delta=0.002, patience=5, verbose=1, mode="min"
                )
            ],
        )


class AutoencoderModel(KerasSequentialModel):
    def build_model(self):
        self.batch_size = 50
        self.encoding_dim = 20

        # this is our input placeholder
        input_layer = Input((self.maxlen * len(self.chars_map),))
        # "encoded" is the encoded representation of the input
        encoded = Dense(self.encoding_dim, activation="relu")(input_layer)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(self.maxlen * len(self.chars_map), activation="sigmoid")(
            encoded
        )

        # this model maps an input to its reconstruction
        self.autoencoder = Model(input_layer, decoded)

        # Encoder part: This model maps an input to its encoded representation
        self.model = Model(input_layer, encoded)

        # create a placeholder for an encoded input
        encoded_input = Input(shape=(self.encoding_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        # optimizer = SGD(lr=0.0001)
        self.autoencoder.compile(optimizer="adadelta", loss="binary_crossentropy")

    def fit(self, x, y):
        x = np.zeros(
            (len(self.all_names), self.maxlen * len(self.chars_map)), dtype=np.uint8
        )

        for i, name in enumerate(self.all_names):
            x[i] = self.name_to_features(name).flatten()

        self.autoencoder.fit(
            x, x, batch_size=self.batch_size, epochs=15, shuffle=True
        )  # , callbacks=[EarlyStopping(monitor='loss', min_delta=0.002, patience=5, verbose=1, mode='min')])

        name_encodings = self.model.predict(x)
        with open(
            "name_encodings.csv", "w", encoding="utf-8"
        ) as name_encodings_out_file:
            for i, name in enumerate(self.all_names):
                name_encodings_out_file.write(name)
                name_encodings_out_file.write(",")
                name_encodings_out_file.write(
                    ",".join([str(v) for v in name_encodings[i]])
                )
                name_encodings_out_file.write("\n")

    def make_recommendation(self):
        if self.estimated_name_scores is None:
            self.estimated_name_scores = []
            self.name_distance_mapping = self.possible_names()
            x_pred = np.zeros(
                (len(self.name_distance_mapping), self.maxlen * len(self.chars_map))
            )
            for i, name in enumerate(self.name_distance_mapping):
                x_pred[i] = self.name_to_features(name).flatten()

            possible_names_encodings = self.model.predict(x_pred)
            x_positive = np.zeros(
                (len(self.positive_names), self.maxlen * len(self.chars_map))
            )
            for i, name in enumerate(self.positive_names):
                x_positive[i] = self.name_to_features(name).flatten()

            positive_names_encodings = self.model.predict(x_positive)
            sim_matrix = pairwise_distances(
                positive_names_encodings, possible_names_encodings, "manhattan"
            )

            # TODO: remove print and this line
            positive_names = list(self.positive_names)

            for i in range(sim_matrix.shape[0]):
                dists = [(j, sim_matrix[i, j]) for j in range(sim_matrix.shape[1])]
                print(
                    positive_names[i],
                    [
                        (self.name_distance_mapping[d[0]], d[1])
                        for d in sorted(dists, key=lambda x: x[1])[:3]
                    ],
                )
                best = max(dists, key=lambda x: x[1])
                # dists = sorted(dists, key=lambda x: x[1])[:5]
                self.estimated_name_scores.append(best)
            self.estimated_name_scores = shuffle(
                sorted(self.estimated_name_scores, key=lambda x: x[1])
            )

        if len(self.estimated_name_scores) > 0:
            selection = self.estimated_name_scores.pop()
            name = self.name_distance_mapping[selection[0]]
            print(name, selection[1])
            return name

        return None
