import numpy as np


class NameRecommendModel:
    def __init__(self, opts, all_names, name_scores):
        self.opts = opts
        self.all_names = all_names
        self.name_scores = name_scores

        # TODO: This should be easily refactored so upsample and downsample shares logic
        if opts.balance_classes == "upsample":
            self.negative_names = [
                (name, score) for name, score in name_scores.items() if score <= 2
            ]
            self.positive_names = [
                (name, score) for name, score in name_scores.items() if score > 2
            ]

            if len(self.negative_names) > len(self.positive_names):
                upsample_names = self.positive_names
                self.training_set = self.negative_names
            else:
                upsample_names = self.negative_names
                self.training_set = self.positive_names

            # numpy confuses an array of tuples with a multidim array
            # using an index choice instead
            choices = np.array(upsample_names)
            chosen_indexes = np.random.choice(len(choices), len(self.training_set))
            for name, score in choices[chosen_indexes]:
                self.training_set.append((name, int(score)))

            self.negative_names = set((name for name, score in self.negative_names))
            self.positive_names = set((name for name, score in self.positive_names))

        elif opts.balance_classes == "downsample":
            self.negative_names = [
                (name, score) for name, score in name_scores.items() if score <= 2
            ]
            self.positive_names = [
                (name, score) for name, score in name_scores.items() if score > 2
            ]

            if len(self.negative_names) > len(self.positive_names) * opts.sample_factor:
                downsample_names = self.negative_names
                self.training_set = list(self.positive_names)
            else:
                downsample_names = self.positive_names
                self.training_set = list(self.negative_names)

            # numpy confuses an array of tuples with a multidim array
            # using an index choice instead
            choices = np.array(downsample_names)
            chosen_indexes = np.random.choice(
                len(choices), len(self.training_set) * opts.sample_factor
            )
            for name, score in choices[chosen_indexes]:
                self.training_set.append((name, int(score)))

            self.negative_names = set((name for name, score in self.negative_names))
            self.positive_names = set((name for name, score in self.positive_names))

        else:
            self.training_set = list(name_scores.items())

            self.negative_names = set(
                (name for name, score in name_scores.items() if score <= 2)
            )
            self.positive_names = set(
                (name for name, score in name_scores.items() if score > 2)
            )

        self.max_score = max(
            self.name_scores.items(), key=lambda name_score: name_score[1]
        )[1]

    def possible_names(self):
        return [name for name in self.all_names if not name in self.name_scores]

    def scale_score(self, score):
        return score / self.max_score

    def train(self):
        pass

    def retrain(self):
        pass

    def make_recommendation(self):
        pass
