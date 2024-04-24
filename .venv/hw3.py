
import sys
import math
# sys.argv = [
#     __file__,
#     'train',
#     'E:\\ai\\train_big.dat',
#     'E:\\ai\\finalized_model.sav',
#     'dt'
# ]

sys.argv = [
    __file__,
    'predict',
    'E:\\ai\\finalized_model.sav',
    'E:\\ai\\testfile_2.dat',

]
import pandas as pd
import numpy as np
from scipy.stats import entropy
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

class Decision_Tree:
    def __init__(self):
        self.depth = 10

    def read(self, file_path,option):
        english_attributes = [['is', 'were', 'on', 'with', 'but', 'by', 'who', 'why', 'will', 'would', 'you', 'could'],
                              ['has', 'have', 'can', 'do', 'for', 'we', 'what', 'which'],
                              ['the', 'from', 'him', 'his', 'if', 'my', 'not'],
                              ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'],
                              ['and', 'of', 'or', 'our', 'she', 'that', 'this', 'to', 'us']]

        german_attributes = [['ich', 'sie','du','er','es','ich','sie','wir','auf','aus','durch','für','gegen','hinter','nach','neben', 'unter', 'vor', 'zu'], ['aber','damit','ob','weil','wenn','und', 'oder'], ['warum','wer','wie','wo','woher','wohin','ä', 'ö', 'ü'], ['der', 'das','dein','mein']]

        data = []
        if(option=="train"):
            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    language, text = line.strip().split('|',1)

                    attributes = []

                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))

                    attributes.append(any(len(word) >= 13 for word in text.split()))

                    data.append(attributes + [language])

            self.data = pd.DataFrame(data, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Class'])

            self.weights = np.ones(len(self.data)) / len(self.data)
        else:
            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    text=line
                    attributes = []

                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))

                    attributes.append(any(len(word) >= 13 for word in text.split()))

                    data.append(attributes)

            self.data = pd.DataFrame(data,columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])

    def entropy(self, series):
        return entropy(series.value_counts(normalize=True), base=2)

    def conditional_entropy(self, X, y):
        y_entropy = self.entropy(y)
        values, counts = np.unique(X, return_counts=True)
        weighted_entropy = sum(
            [(counts[i] / np.sum(counts)) * self.entropy(y[X == value]) for i, value in enumerate(values)])

        return y_entropy - weighted_entropy

    def informationgain(self, X, y):
        info_gain = [self.conditional_entropy(X[col], y) for col in X.columns]
        return X.columns[np.argmax(info_gain)]

    def build_tree(self, X, y, depth):
        if depth == 0 or len(X.columns) == 0:
            return y.value_counts().idxmax()

        root_attribute = self.informationgain(X, y)
        tree = {root_attribute: {}}

        for value in [True, False]:
            X_subset = X[X[root_attribute] == value]
            y_subset = y[X[root_attribute] == value]

            if len(X_subset) > 0:
                tree[root_attribute][value] = self.build_tree(X_subset.drop(root_attribute, axis=1), y_subset,
                                                              depth - 1)
            else:
                tree[root_attribute][value] = y.value_counts().idxmax()

        return tree

    def prediction(self, tree, sample):
        if not isinstance(tree, dict):
            return tree

        attribute = list(tree.keys())[0]
        value = sample[attribute]
        return self.prediction(tree[attribute][value], sample)

    def run(self):
        self.tree = self.build_tree(self.data.iloc[:, :-1], self.data.iloc[:, -1], self.depth)


        # filename = sys.argv[3]
        # pickle.dump(self.tree, open(filename, 'wb'))

    def load_model(self, filename):
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model

    def save_model(self):
        filename = sys.argv[3]
        pickle.dump(self.tree, open(filename, 'wb'))


#Adaboost code starts here








class AdaBoost:
    def __init__(self):
        self.data = None
        self.weights = None
        self.hypotheses = []
        self.hypothesis_weights = []

    def read(self, file_path, option):
        english_attributes = [['is', 'were', 'on', 'with', 'but', 'by', 'who', 'why', 'will', 'would', 'you', 'could'],
                              ['has', 'have', 'can', 'do', 'for', 'we', 'what', 'which'],
                              ['the', 'from', 'him', 'his', 'if', 'my', 'not'],
                              ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'],
                              ['and', 'of', 'or', 'our', 'she', 'that', 'this', 'to', 'us']]

        german_attributes = [
            ['ich', 'sie', 'du', 'er', 'es', 'ich', 'sie', 'wir', 'auf', 'aus', 'durch', 'für', 'gegen', 'hinter',
             'nach', 'neben', 'unter', 'vor', 'zu'], ['aber', 'damit', 'ob', 'weil', 'wenn', 'und', 'oder'],
            ['warum', 'wer', 'wie', 'wo', 'woher', 'wohin', 'ä', 'ö', 'ü'], ['der', 'das', 'dein', 'mein']]

        data = []
        if (option == "train"):
            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    language, text = line.strip().split('|',1)

                    attributes = []

                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(all(attribute not in text.split() for attribute in attribute_group))

                    attributes.append(any(len(word) <= 13 for word in text.split()))

                    data.append([language] + attributes)
                self.data = pd.DataFrame(data, columns=['Class','A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])
                self.X = self.data.iloc[:, 1:]
                self.y = np.where(self.data.iloc[:, 0] == 'en', 1, 0)
                self.weights = np.ones(len(self.data)) / len(self.data)



        else:


            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    text = line
                    attributes = []

                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(all(attribute not in text.split() for attribute in attribute_group))

                    attributes.append(any(len(word) <= 13 for word in text.split()))

                    data.append(attributes)

            self.data = pd.DataFrame(data, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])



    def train(self, K):
        N = len(self.data)
        epsilon = 1e-7

        for k in range(K):
            h = self.train_weak_learner()
            current_best=h

            stump = lambda x: 1 if x.iloc[current_best] else 0
            convert= lambda y: 1 if y=='en' else 0



            error = sum(self.weights[i] for i in range(len(self.X)) if stump(self.X.iloc[i]) != self.y[i])
            error = max(epsilon, min(1 - epsilon, error))


            if error > 0.5:
                break
            for i in range(N):

                if stump(self.X.iloc[i]) != convert(self.data.iloc[i,0]):
                    self.weights[i] *= error / (1 - error)





            self.weights /= sum(self.weights)

            z = 0.5 * math.log((1 - error) / error)

            self.hypotheses.append(h)
            self.hypothesis_weights.append(z)


    def weighted_entropy(self, series, weights):
        series = pd.Series(series)
        unique_values = series.unique()
        value_counts = series.value_counts(normalize=True)
        adjusted_weights = [np.sum(weights[series == value]) for value in unique_values]
        weighted_counts = value_counts * adjusted_weights
        return entropy(weighted_counts, base=2)

    def weighted_conditional_entropy(self, X, y):
        y_entropy = self.weighted_entropy(y, self.weights)
        values, counts = np.unique(X, return_counts=True)
        weighted_entropy = sum(
            [(counts[i] / np.sum(counts)) * self.weighted_entropy(y[X == value], self.weights[X == value]) for i, value
             in enumerate(values)])
        return y_entropy - weighted_entropy

    def weighted_informationgain(self, X, y):
        info_gain = [self.weighted_conditional_entropy(X[col], y) for col in X.columns]
        return np.argmax(info_gain)

    def train_weak_learner(self):

        best_attribute = self.weighted_informationgain(self.X, self.y)

        return best_attribute



    def predict(self,X):
        enwei=0
        dlwei=0
        for i in self.hypotheses:
            if(X.iloc[i]==True):
                enwei=enwei+self.hypothesis_weights[i]
            else:
                dlwei=dlwei+self.hypothesis_weights[i]
        if(enwei>dlwei):
            print("en")
        else:
            print("nl")
    def save_model(self):
        with open(sys.argv[3], 'wb') as f:
            pickle.dump(self, f)



if(sys.argv[1]=="train" and sys.argv[4]=="dt"):
    a = Decision_Tree()
    a.read(sys.argv[2],"train")
    a.run()
    a.save_model()


elif(sys.argv[1]=="train" and sys.argv[4]=="ada"):
    model = AdaBoost()


    # Read the data
    model.read(sys.argv[2],"train")

    model.train(50)

    model.save_model()



    # with open(sys.argv[3], 'wb') as f:
    #     pickle.dump(model, f)

elif(sys.argv[1]=="predict"):
    with open(sys.argv[2], 'rb') as file:
        loaded_model = pickle.load(file)
        if(isinstance(loaded_model,AdaBoost)):
            b = AdaBoost()
            with open(sys.argv[2], 'rb') as f:
                model = pickle.load(f)
                b.read((sys.argv[3]),"predict")
                predictions = b.data.iloc[:, :].apply(lambda x: model.predict(x), axis=1)


        else:
            b = Decision_Tree()
            loaded_model = b.load_model(sys.argv[2])
            b.read(sys.argv[3],"predict")
            predictions = b.data.iloc[:, :].apply(lambda x: b.prediction(loaded_model, x), axis=1)
            for each_predictions in predictions:
                print(each_predictions)