
import pandas as pd
import numpy as np
from scipy.stats import entropy
from math import log, exp

class Solution:
    def read(self):
        self.data = pd.read_csv('C:\\Users\\jrpji\\Downloads\\dtree-data.dat.txt', sep=' ', header=None)
        self.data.columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Class']
        self.weights = np.ones(len(self.data)) / len(self.data)  # Initialize weights

    def entropy(self, series):
        return entropy(series.value_counts(normalize=True), base=2)

    def conditional_entropy(self, X, y):
        y_entropy = self.entropy(y)
        values, counts = np.unique(X, return_counts=True)
        weighted_entropy = sum([(counts[i]/np.sum(counts))*self.entropy(y[X == value]) for i, value in enumerate(values)])

        return y_entropy - weighted_entropy

    def informationgain(self):
        self.X = self.data.iloc[:, :-1]  # attributes
        self.y = self.data.iloc[:, -1]  # class

        info_gain = [self.conditional_entropy(self.X[col], self.y) for col in self.X.columns]
        self.root_attribute = self.X.columns[np.argmax(info_gain)]

        # Calculate next_attribute for each value of root_attribute
        self.next_attribute = {}
        for value in [True, False]:
            X_subset = self.X[self.X[self.root_attribute] == value]
            y_subset = self.y[self.X[self.root_attribute] == value]

            if len(X_subset) > 0:
                info_gain_subset = [self.conditional_entropy(X_subset[col], y_subset) for col in X_subset.columns]
                self.next_attribute[value] = X_subset.columns[np.argmax(info_gain_subset)]

    def prediction(self):
        misclassified = 0
        for value_root in [True, False]:
            for value_next in [True, False]:
                y_leaf = self.y[(self.X[self.root_attribute] == value_root) & (
                            self.X[self.next_attribute[value_root]] == value_next)]
                indices = y_leaf.index
                if len(y_leaf) > 0:
                    prediction = y_leaf.value_counts().idxmax()
                    print(
                        f'When {self.root_attribute} is {value_root} and {self.next_attribute[value_root]} is {value_next}, predict {prediction}.')
                    misclassified += sum((self.y.loc[indices] != prediction) * self.weights[indices])

        error_rate = misclassified / sum(self.weights)
        alpha = 0.5 * log((1 - error_rate) / error_rate)

        # Update weights
        for i in range(len(self.data)):
            if self.y[i] == prediction:
                self.weights[i] *= exp(-alpha)
            else:
                self.weights[i] *= exp(alpha)

        # Normalize weights
        self.weights /= sum(self.weights)

        print(f'Initial weights: {np.ones(len(self.data)) / len(self.data)}')
        print(f'Misclassified items total weight: {misclassified}')
        print(f'Error rate: {error_rate}')
        print(f'Hypothesis weight: {alpha}')
        print(f'New weights (correctly classified): {self.weights[self.y == prediction]}')
        print(f'New weights (incorrectly classified): {self.weights[self.y != prediction]}')

a = Solution()
a.read()
a.informationgain()
a.prediction()
