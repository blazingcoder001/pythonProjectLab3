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



# Define the data
data = np.array([
    [0, 0, 1, 0, 0, 'B'],
    [1, 0, 1, 0, 1, 'B'],
    [1, 1, 1, 0, 0, 'A'],
    [0, 1, 0, 0, 1, 'B'],
    [1, 1, 1, 0, 1, 'B'],
    [0, 1, 1, 0, 0, 'B'],
    [1, 1, 0, 0, 1, 'B'],
    [1, 1, 1, 0, 1, 'B'],
    [1, 0, 1, 0, 1, 'B'],
    [1, 0, 1, 0, 1, 'B']
])

# Separate features and target
X = data[:, :-1].astype(int)
y = data[:, -1]

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Define the AdaBoost model with 50 decision stumps
model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50)

# Fit the model
model.fit(X, y)

# Print the feature importances
print("Feature importances:")
for i, importance in enumerate(model.feature_importances_):
    print(f"Attribute {i+1}: {importance}")

# Predict the class of a new instance
new_instance = np.array([1, 0, 1, 0, 1]).reshape(1, -1)
print(f"\\nPrediction for {new_instance}: {le.inverse_transform(model.predict(new_instance))}")
