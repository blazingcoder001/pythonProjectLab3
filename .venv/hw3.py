
import sys
# sys.argv = [
#     __file__,
#     'train',
#     'E:\\ai\\train.dat.txt',
#     'E:\\ai\\finalized_model_ada.sav',
#     'ada'
# ]
sys.argv = [
    __file__,
    'predict',
    'E:\\ai\\finalized_model_ada.sav',
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

class Solution:
    def __init__(self):
        self.depth = 10  # Set the depth to 10

    def read(self, file_path,option):
        # Define the attributes to check for each language
        english_attributes = [['is', 'were', 'on', 'with', 'but', 'by', 'who', 'why', 'will', 'would', 'you', 'could'],
                              ['has', 'have', 'can', 'do', 'for', 'we', 'what', 'which'],
                              ['the', 'from', 'him', 'his', 'if', 'my', 'not'],
                              ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'],
                              ['and', 'of', 'or', 'our', 'she', 'that', 'this', 'to', 'us']]

        german_attributes = [['ich', 'sie','du','er','es','ich','sie','wir','auf','aus','durch','für','gegen','hinter','nach','neben', 'unter', 'vor', 'zu'], ['aber','damit','ob','weil','wenn','und', 'oder'], ['warum','wer','wie','wo','woher','wohin','ä', 'ö', 'ü'], ['der', 'das','dein','mein','my']]

        # Initialize the data list
        data = []
        if(option=="train"):
            # Open the file and read each line
            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    # Split the line into language and text
                    language, text = line.strip().split('|')

                    # Initialize the attributes list
                    attributes = []

                    # Check the attributes for the corresponding language
                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))

                    # Check if the text contains a word with length greater than or equal to 13
                    attributes.append(any(len(word) >= 13 for word in text.split()))

                    # Append the language and attributes to the data list
                    data.append(attributes + [language])

            # Convert the data list to a DataFrame
            self.data = pd.DataFrame(data, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Class'])
            print(self.data)

            # Initialize the weights
            self.weights = np.ones(len(self.data)) / len(self.data)
        else:
            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    text=line
                    attributes = []

                    # Check the attributes for the corresponding language
                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))

                    # Check if the text contains a word with length greater than or equal to 13
                    attributes.append(any(len(word) >= 13 for word in text.split()))

                    # Append the language and attributes to the data list
                    data.append(attributes)

            # Convert the data list to a DataFrame
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
        # self.read()
        self.tree = self.build_tree(self.data.iloc[:, :-1], self.data.iloc[:, -1], self.depth)
        print(self.tree)


        # Save the model to disk
        filename = 'finalized_model.sav'
        pickle.dump(self.tree, open(filename, 'wb'))

    def load_model(self, filename):
        # Load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model


#Adaboost code starts here

#
# class DecisionStump:
#     def __init__(self):
#         self.polarity = 1
#         self.feature_index = None
#         self.threshold = None
#         self.alpha = None
#
# class AdaBoost:
#     def __init__(self, S=50):
#         self.S = S
#         self.data = None
#         self.weights = None
#
#     def read(self, file_path,option):
#         # Define the attributes to check for each language
#         english_attributes = [['is', 'were', 'on', 'with', 'but', 'by', 'who', 'why', 'will', 'would', 'you', 'could'],
#                               ['has', 'have', 'can', 'do', 'for', 'we', 'what', 'which'],
#                               ['the', 'from', 'him', 'his', 'if', 'my', 'not'],
#                               ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'],
#                               ['and', 'of', 'or', 'our', 'she', 'that', 'this', 'to', 'us']]
#
#         german_attributes = [
#             ['ich', 'sie', 'du', 'er', 'es', 'ich', 'sie', 'wir', 'auf', 'aus', 'durch', 'für', 'gegen', 'hinter',
#              'nach', 'neben', 'unter', 'vor', 'zu'], ['aber', 'damit', 'ob', 'weil', 'wenn', 'und', 'oder'],
#             ['warum', 'wer', 'wie', 'wo', 'woher', 'wohin', 'ä', 'ö', 'ü'], ['der', 'das', 'dein', 'mein', 'my']]
#
#         # Initialize the data list
#         data = []
#         if (option == "train"):
#             # Open the file and read each line
#             with open(file_path, 'r', encoding='utf8') as file:
#
#                 for line in file:
#                     # Split the line into language and text
#                     language, text = line.strip().split('|')
#
#                     # Initialize the attributes list
#                     attributes = []
#
#                     # Check the attributes for the corresponding language
#                     for attribute_group in english_attributes:
#                         attributes.append(any(attribute in text.split() for attribute in attribute_group))
#                     for attribute_group in german_attributes:
#                         attributes.append(any(attribute in text.split() for attribute in attribute_group))
#
#                     # Check if the text contains a word with length greater than or equal to 13
#                     attributes.append(any(len(word) >= 13 for word in text.split()))
#
#                     # Append the language and attributes to the data list
#                     data.append([language]+attributes)
#
#             # Convert the data list to a DataFrame
#             self.data = pd.DataFrame(data,
#                                      columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Class'])
#             # print(self.data)
#
#             # Initialize the weights
#             self.weights = np.ones(len(self.data)) / len(self.data)
#         else:
#             with open(file_path, 'r', encoding='utf8') as file:
#
#                 for line in file:
#                     text = line
#                     attributes = []
#
#                     # Check the attributes for the corresponding language
#                     for attribute_group in english_attributes:
#                         attributes.append(any(attribute in text.split() for attribute in attribute_group))
#                     for attribute_group in german_attributes:
#                         attributes.append(any(attribute in text.split() for attribute in attribute_group))
#
#                     # Check if the text contains a word with length greater than or equal to 13
#                     attributes.append(any(len(word) >= 13 for word in text.split()))
#
#                     # Append the language and attributes to the data list
#                     data.append(attributes)
#
#             # Convert the data list to a DataFrame
#             self.data = pd.DataFrame(data, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])
#
#     def fit(self, X, y):
#         m, n = X.shape
#         W = np.full(m, 1/m)
#         self.clfs = []
#
#         for _ in range(self.S):
#             clf = DecisionStump()
#             min_error = float('inf')
#
#             for feature in range(n):
#                 feature_values = np.sort(np.unique(X[:, feature]))
#                 thresholds = (feature_values[:-1] + feature_values[1:]) / 2
#
#                 for threshold in thresholds:
#                     for polarity in [1, -1]:
#                         y_hat = np.ones(len(y))
#                         y_hat[polarity * X[:, feature] < polarity * threshold] = -1
#                         error = W[(y_hat != y)].sum()
#
#                         if error < min_error:
#                             clf.polarity = polarity
#                             clf.threshold = threshold
#                             clf.feature_index = feature
#                             min_error = error
#
#             eps = 1e-10
#             clf.alpha = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))
#
#             predictions = np.ones(m)
#             negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
#             predictions[negative_idx] = -1
#
#             W *= np.exp(-clf.alpha * y * predictions)
#             W /= np.sum(W)
#
#             self.clfs.append(clf)
#
#     def predict(self, X):
#         m, _ = X.shape
#         y_hat = np.zeros(m)
#
#         for clf in self.clfs:
#             predictions = np.ones(m)
#             negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
#             predictions[negative_idx] = -1
#             y_hat += clf.alpha * predictions
#
#         return np.sign(y_hat)
#
#
# if(sys.argv[1]=="train" and sys.argv[4]=="dt"):
#     a = Solution()
#     a.read(sys.argv[2],"train")
#     a.run()
#
#
# elif(sys.argv[1]=="train" and sys.argv[4]=="ada"):
#     model = AdaBoost(S=50)
#
#     # Read the data
#     model.read(sys.argv[2],"train")
#
#     # Separate features and target
#     X = model.data.iloc[:, 1:].values.astype(bool)
#     y = np.where(model.data.iloc[:, 0] == 'A', 1, -1)
#
#     # Fit the model
#     model.fit(X, y)
#
#     # Save the model
#     with open(sys.argv[3], 'wb') as f:
#         pickle.dump(model, f)
#
# elif(sys.argv[1]=="predict"):
#     with open(sys.argv[2], 'rb') as file:
#         loaded_model = pickle.load(file)
#         if(isinstance(loaded_model,AdaBoost)):
#         #     new_instance = np.array([True, False, True, False, True, False, True, True, False, True]).reshape(1, -1)
#         #     print(f"\\nPrediction for {new_instance}: {Counter(model.predict(new_instance))}")
#             b = AdaBoost(S=50)
#             with open(sys.argv[2], 'rb') as f:
#                 model = pickle.load(f)
#                 b.read((sys.argv[3]),"predict")
#                 # print(b.data)
#                 predictions = b.data.iloc[:, :].apply(lambda x: model.predict(np.array(x).reshape(1,-1)), axis=1)
#                 for each_predictions in predictions:
#                     print(each_predictions)
#         #         # print(f"\\nPrediction for {new_instance}: {Counter(model.predict(new_instance))}")
#
#         else:
#             b = Solution()
#             loaded_model = b.load_model(sys.argv[2])
#             b.read(sys.argv[3],"predict")
#             # print(b.prediction(loaded_model, sample))
#             predictions = b.data.iloc[:, :].apply(lambda x: b.prediction(loaded_model, x), axis=1)
#             for each_predictions in predictions:
#                 print(each_predictions)








class AdaBoost:
    def __init__(self):
        self.data = None
        self.weights = None
        self.hypotheses = []
        self.hypothesis_weights = []

    def read(self, file_path, option):
        # Define the attributes to check for each language
        english_attributes = [['is', 'were', 'on', 'with', 'but', 'by', 'who', 'why', 'will', 'would', 'you', 'could'],
                              ['has', 'have', 'can', 'do', 'for', 'we', 'what', 'which'],
                              ['the', 'from', 'him', 'his', 'if', 'my', 'not'],
                              ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'],
                              ['and', 'of', 'or', 'our', 'she', 'that', 'this', 'to', 'us']]

        german_attributes = [
            ['ich', 'sie', 'du', 'er', 'es', 'ich', 'sie', 'wir', 'auf', 'aus', 'durch', 'für', 'gegen', 'hinter',
             'nach', 'neben', 'unter', 'vor', 'zu'], ['aber', 'damit', 'ob', 'weil', 'wenn', 'und', 'oder'],
            ['warum', 'wer', 'wie', 'wo', 'woher', 'wohin', 'ä', 'ö', 'ü'], ['der', 'das', 'dein', 'mein', 'my']]

        # Initialize the data list
        data = []
        if (option == "train"):
            # Open the file and read each line
            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    # Split the line into language and text
                    language, text = line.strip().split('|')

                    # Initialize the attributes list
                    attributes = []

                    # Check the attributes for the corresponding language
                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(all(attribute not in text.split() for attribute in attribute_group))

                    # Check if the text contains a word with length greater than or equal to 13
                    attributes.append(any(len(word) >= 13 for word in text.split()))

                    # Append the language and attributes to the data list
                    data.append([language] + attributes)

            # Convert the data list to a DataFrame
            self.data = pd.DataFrame(data,
                                     columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Class'])
            # print(self.data)

            # Initialize the weights
            self.weights = np.ones(len(self.data)) / len(self.data)
        else:
            with open(file_path, 'r', encoding='utf8') as file:

                for line in file:
                    text = line
                    attributes = []

                    # Check the attributes for the corresponding language
                    for attribute_group in english_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))
                    for attribute_group in german_attributes:
                        attributes.append(any(attribute in text.split() for attribute in attribute_group))

                    # Check if the text contains a word with length greater than or equal to 13
                    attributes.append(any(len(word) >= 13 for word in text.split()))

                    # Append the language and attributes to the data list
                    data.append(attributes)

            # Convert the data list to a DataFrame
            self.data = pd.DataFrame(data, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])


    def train(self, K):
        N = len(self.data)
        epsilon = 1e-7  # Small positive number to avoid division by zero

        for k in range(K):
            h = self.train_weak_learner()
            current_best=h[0]
            error=h[1]


            stump = lambda x: 'EN' if x[current_best] else 'DE'

            if error>0.5:
                break

            error = max(epsilon, min(1 - epsilon, error))

            for i in range(N):
                if stump(self.data.iloc[i]) != self.data.iloc[i][-1]:
                    self.weights[i] *= error / (1 - error)



            # Normalize the weights
            self.weights /= sum(self.weights)

            # Compute the weight of the hypothesis
            z = 0.5 * math.log((1 - error) / error)

            # Store the hypothesis and its weight
            self.hypotheses.append(h)
            self.hypothesis_weights.append(z)


    def train_weak_learner(self):
        best_stump = None
        best_error = float('inf')


        # For each attribute in the data
        for attribute in self.data.columns[:-1]:#There is a small mistake here, the attribute does not mark which attribute it is.
            # Create a stump that classifies examples based on the attribute
            stump = lambda x: 'EN' if x[attribute] else 'DE'

            # Compute the weighted error of the stump
            error = sum(
                self.weights[i] for i in range(len(self.data)) if stump(self.data.iloc[i]) != self.data.iloc[i,-1])

            # If this stump has the lowest error so far, store it as the best stump
            # if error < best_error:
            #     best_stump = stump
            #     best_error = error
            if error<best_error:
                best_stump=attribute
                best_error=error

        return (best_stump,best_error)

    def classify(self, x):
        # The final classifier is a weighted combination of the hypotheses
        return sum(self.hypothesis_weights[k] * self.hypothesesk for k in range(len(self.hypotheses)))

    def predict(self,X):
        enwei=0
        dlwei=0
        print(self.hypotheses)
        for i in self.hypotheses:
            if(X[int(i[1:])-1]==True):
                enwei=enwei+self.hypothesis_weights
            else:
                dlwei=dlwei+self.hypothesis_weights
        if(enwei>dlwei):
            print("en")
        else:
            print("nl")



# # Create an AdaBoost instance
# adaboost = AdaBoost()
#
# # Read the training data
# adaboost.read('training_data.txt', 'train')
#
# # Train the AdaBoost model with 50 stumps
# adaboost.train(50)

# Classify a new example
# print(adaboost.classify(new_example))

if(sys.argv[1]=="train" and sys.argv[4]=="dt"):
    a = Solution()
    a.read(sys.argv[2],"train")
    a.run()


elif(sys.argv[1]=="train" and sys.argv[4]=="ada"):
    model = AdaBoost()


    # Read the data
    model.read(sys.argv[2],"train")

    model.train(50)

    # Separate features and target
    X = model.data.iloc[:, 1:].values.astype(bool)
    y = np.where(model.data.iloc[:, 0] == 'A', 1, -1)


    # Save the model
    with open(sys.argv[3], 'wb') as f:
        pickle.dump(model, f)

elif(sys.argv[1]=="predict"):
    with open(sys.argv[2], 'rb') as file:
        loaded_model = pickle.load(file)
        if(isinstance(loaded_model,AdaBoost)):
        #     new_instance = np.array([True, False, True, False, True, False, True, True, False, True]).reshape(1, -1)
        #     print(f"\\nPrediction for {new_instance}: {Counter(model.predict(new_instance))}")
            b = AdaBoost()
            with open(sys.argv[2], 'rb') as f:
                model = pickle.load(f)
                b.read((sys.argv[3]),"predict")
                # print(b.data)
                predictions = b.data.iloc[:, :].apply(lambda x: model.predict(x))
                # for each_predictions in predictions:
                #     print(each_predictions)
        #         # print(f"\\nPrediction for {new_instance}: {Counter(model.predict(new_instance))}")

        else:
            b = Solution()
            loaded_model = b.load_model(sys.argv[2])
            b.read(sys.argv[3],"predict")
            # print(b.prediction(loaded_model, sample))
            predictions = b.data.iloc[:, :].apply(lambda x: b.prediction(loaded_model, x), axis=1)
            for each_predictions in predictions:
                print(each_predictions)