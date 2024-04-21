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

    def read(self):
        self.data = pd.read_csv('C:\\Users\\jrpji\\Downloads\\dtree-data.dat.txt', sep=' ', header=None)
        self.data.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                             'Class']  # Adjusted for 10 attributes
        self.weights = np.ones(len(self.data)) / len(self.data)  # Initialize weights

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
        self.read()
        self.tree = self.build_tree(self.data.iloc[:, :-1], self.data.iloc[:, -1], self.depth)
        predictions = self.data.iloc[:, :-1].apply(lambda x: self.prediction(self.tree, x), axis=1)
        print(predictions)

        # Save the model to disk
        filename = 'finalized_model.sav'
        pickle.dump(self.tree, open(filename, 'wb'))

    def load_model(self, filename):
        # Load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model


a = Solution()
a.run()

# Load the model and make a prediction
filename = 'finalized_model.sav'
b = Solution()
loaded_model = b.load_model(filename)
sample = pd.Series(
    {'A1': True, 'A2': False, 'A3': True, 'A4': False, 'A5': True, 'A6': False, 'A7': True, 'A8': False, 'A9': True,
     'A10': False})  # Adjusted for 10 attributes
print(b.prediction(loaded_model, sample))



#Adaboost code starts here


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class AdaBoost:
    def __init__(self, S=50):
        self.S = S

    def fit(self, X, y):
        m, n = X.shape
        W = np.full(m, 1/m)
        self.clfs = []

        for _ in range(self.S):
            clf = DecisionStump()
            min_error = float('inf')

            for feature in range(n):
                feature_values = np.sort(np.unique(X[:, feature]))
                thresholds = (feature_values[:-1] + feature_values[1:]) / 2

                for threshold in thresholds:
                    for polarity in [1, -1]:
                        y_hat = np.ones(len(y))
                        y_hat[polarity * X[:, feature] < polarity * threshold] = -1
                        error = W[(y_hat != y)].sum()

                        if error < min_error:
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_index = feature
                            min_error = error

            eps = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + eps) / (min_error + eps))

            predictions = np.ones(m)
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1

            W *= np.exp(-clf.alpha * y * predictions)
            W /= np.sum(W)

            self.clfs.append(clf)

    def predict(self, X):
        m, _ = X.shape
        y_hat = np.zeros(m)

        for clf in self.clfs:
            predictions = np.ones(m)
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_hat += clf.alpha * predictions

        return np.sign(y_hat)

        def read(self, file_path):
            # Define the attributes to check for each language
            english_attributes = [['is', 'was', 'were'], ['has', 'have'], ['a', 'the'],
                                  ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'], ['and']]
            german_attributes = [['ich', 'sie'], ['und', 'oder'], ['ä', 'ö', 'ü'], ['der', 'die', 'das']]

            # Initialize the data list
            data = []

            # Open the file and read each line
            with open(file_path, 'r') as file:
                for line in file:
                    # Split the line into language and text
                    language, text = line.strip().split('|')

                    # Initialize the attributes list
                    attributes = []

                    # Check the attributes for the corresponding language
                    if language == 'en':
                        for attribute_group in english_attributes:
                            attributes.append(int(any(attribute in text.split() for attribute in attribute_group)))
                    elif language == 'nl':
                        for attribute_group in german_attributes:
                            attributes.append(int(any(attribute in text.split() for attribute in attribute_group)))

                    # Check if the text contains a word with length greater than or equal to 13
                    attributes.append(int(any(len(word) >= 13 for word in text.split())))

                    # Append the language and attributes to the data list
                    data.append([language] + attributes)

            # Convert the data list to a DataFrame
            self.data = pd.DataFrame(data,
                                     columns=['Class', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])

            # Initialize the weights
            self.weights = np.ones(len(self.data)) / len(self.data)

    # Define the AdaBoost model with 50 decision stumps
    model = AdaBoost(S=50)

    # Read the data
    model.read('C:\\Users\\jrpji\\Downloads\\dtree-data.dat.txt')

    # Print the DataFrame
    print(model.data)

# Define the AdaBoost model with 50 decision stumps
model = AdaBoost(S=50)

# Read the data
model.read()

# Separate features and target
X = model.data.iloc[:, :-1].values.astype(int)
y = np.where(model.data.iloc[:, -1] == 'A', 1, -1)

# Fit the model
model.fit(X, y)

# Predict the class of a new instance
new_instance = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1]).reshape(1, -1)
print(f"\\nPrediction for {new_instance}: {Counter(model.predict(new_instance))}")
