'''
Done by: Jishnuraj Prakasan
References:-
Artificial Intelligence A Modern Approach 4th Edition Peter Norvig Stuart Russell Pearson,
https://youtu.be/_L39rN6gz7Y?si=oqyyWO_71s_iG3tA
(Decision and Classification Trees, Clearly Explained!!!
https://youtu.be/LsK-xG1cLYA?si=lQquoooOwSTJLqkD
(AdaBoost, Clearly Explained)
StatQuest with Josh Starmer)

'''


import sys
import math
import re
import pandas as pd
import numpy as np
from scipy.stats import entropy
import pickle
import numpy as np


'''
Class Decision_Tree contains the functions to read, do all calculations, predict results, save and load models.
Depth defines the maximum depth of the decision tree to be built.

'''


class Decision_Tree:


    '''
    depth defines the max-depth of the decision tree.
    '''


    def __init__(self):
        self.depth = 10


    '''
    This function reads the training data and converts it into boolean attrivutes based on the attributes specified in the code.
    @:param flie_path is the path of the file to read.
    @:param option specifies whether is file should be read under train command or predict command.
    @:returns a panda data frame containing the data in attribute form.
    '''


    def read(self, file_path,option):
        english_attributes = [['is', 'were', 'on', 'with', 'but', 'by', 'who', 'why', 'will', 'would', 'you', 'could'],
                              ['has', 'have', 'can', 'do', 'for', 'we', 'what', 'which'],
                              ['the', 'from', 'him', 'his', 'if', 'my', 'not'],
                              ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'],
                              ['and', 'of', 'or', 'our', 'she', 'that', 'this', 'to', 'us']]

        german_attributes = [['ich', 'sie','du','de','er','es','ich','sie','wir','auf','aus','durch','für','gegen','hinter','nach','neben', 'unter', 'vor', 'zu','de','aber','damit','ob','weil','wenn','und', 'oder','warum','wer','wie','wo','woher','wohin','ä', 'ö', 'ü','der', 'das','dein','mein']]


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
                    attributes.append(
                        all(len(re.findall(r'([aeiou])\1', word, re.IGNORECASE)) == 1 for word in text.split()))
                    attributes.append(all('z' not in word for word in text.split()))

                    attributes.append(all(
                        not word.endswith(('eid', 'se', 'en', 'ens', 'eid', 'dt', 't', 'et')) for word in text.split()))

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

                    attributes.append(
                        all(len(re.findall(r'([aeiou])\1', word, re.IGNORECASE)) == 1 for word in text.split()))
                    attributes.append(all('z' not in word for word in text.split()))
                    attributes.append(all(
                        not word.endswith(('eid', 'se', 'en', 'ens', 'eid', 'dt', 't', 'et')) for word in text.split()))

                    data.append(attributes)

            self.data = pd.DataFrame(data,columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])

    '''
    This function finds the entropy of the series passed.
    @:param series is the series for which entropy is to be find out.
    @:return the entropy of the series passed.
    '''

    def entropy(self, series):
        return entropy(series.value_counts(normalize=True), base=2)
    '''
    This function is used to calculate condtional entropy of X given y.
    @:param X is the attributes of each class. 
    @:param y contains the list of classes en and nl corresponding to each example.
    @:return the conditional entropy of X given y. 
    '''

    def conditional_entropy(self, X, y):
        y_entropy = self.entropy(y)
        values, counts = np.unique(X, return_counts=True)
        weighted_entropy = sum(
            [(counts[i] / np.sum(counts)) * self.entropy(y[X == value]) for i, value in enumerate(values)])

        return y_entropy - weighted_entropy


    '''
    This function is used to find the information gain of each attributes and return the attribute with maximum information gain.
    @:param X contains the dataframe excluding the class names.
    @:param y contains the class names corresponding to each examples.
    @:return the attribute with maximum information gain.
    '''


    def informationgain(self, X, y):
        info_gain = [self.conditional_entropy(X[col], y) for col in X.columns]
        return X.columns[np.argmax(info_gain)]


    '''
    This function is used to build the decision tree.
    @:param X contains the dataframe excluding the class names.
    @:param y contains the class names corresponding to each examples.
    @:param depth is the maximum depth of the decision tree needed.
    @:return the decision tree in nested dictionary form.
    '''


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


    '''
    This method helps to predict the input given to it as 'en' or 'nl'.
    @:param tree takes the decision tree to be used for prediction.
    @:param sample contains the dataframe of the input converted to similar attributes used in train command.
    @:return a list of predicted values.
    '''


    def prediction(self, tree, sample):
        if not isinstance(tree, dict):
            return tree

        attribute = list(tree.keys())[0]
        value = sample[attribute]
        return self.prediction(tree[attribute][value], sample)


    '''
    This function helps to call the build_tree function.
    @:param none
    @:return none
    '''


    def run(self):
        self.tree = self.build_tree(self.data.iloc[:, :-1], self.data.iloc[:, -1], self.depth)


    '''
    This function loads the decision tree from the filename provided.
    @:param filename is the filename containing the decision tree.
    @:return the loaded decision tree to be used for prediction.
    '''


    def load_model(self, filename):
        loaded_model = pickle.load(open(filename, 'rb'))
        return loaded_model

    '''
    This function saves the decision tree created during the decision process.
    @:param none
    @:return none
    '''
    def save_model(self):
        filename = sys.argv[3]
        pickle.dump(self.tree, open(filename, 'wb'))


'''
This class contains all the methods for training and predicting using AdaBoost.
'''



class AdaBoost:


    '''
    @:var data is used to store the data frame read from the file.
    @:var weights contain the weights of each example in the dataset.
    @:var hypostheses contains all the hypotheses used in adaboost in the order of usage.
    @:var hypothesis_weights contains the weights of all the hypotheses used in adaboost.
    @:return none
    '''


    def __init__(self):
        self.data = None
        self.weights = None
        self.hypotheses = []
        self.hypothesis_weights = []


    '''
        This function reads the training data and converts it into boolean attributes based on the attributes specified in the code.
        This read function is little bit different from the read in Decision_Tree as the order of attributes are different in the file to be written.
        @:param flie_path is the path of the file to read.
        @:param option specifies whether is file should be read under train command or predict command.
        @:returns a panda data frame containing the data in attribute form.
        '''


    def read(self, file_path, option):
        english_attributes = [['is', 'were', 'on', 'with', 'but', 'by', 'who', 'why', 'will', 'would', 'you', 'could'],
                              ['has', 'have', 'can', 'do', 'for', 'we', 'what', 'which'],
                              ['the', 'from', 'him', 'his', 'if', 'my', 'not'],
                              ['she', 'he', 'they', 'those', 'him', 'her', 'them', 'it'],
                              ['and', 'of', 'or', 'our', 'she', 'that', 'this', 'to', 'us']]

        german_attributes = [['ich', 'sie','du','de','er','es','ich','sie','wir','auf','aus','durch','für','gegen','hinter','nach','neben', 'unter', 'vor', 'zu','de','aber','damit','ob','weil','wenn','und', 'oder','warum','wer','wie','wo','woher','wohin','ä', 'ö', 'ü','der', 'das','dein','mein']]


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
                    attributes.append(
                        all(len(re.findall(r'([aeiou])\1', word, re.IGNORECASE)) == 1 for word in text.split()))
                    attributes.append(all('z' not in word for word in text.split()))
                    attributes.append(all(
                        not word.endswith(('eid', 'se', 'en', 'ens', 'eid', 'dt', 't', 'et')) for word in text.split()))

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

                    # checks if the same vowel repeats.
                    attributes.append(
                        all(len(re.findall(r'([aeiou])\1', word, re.IGNORECASE)) == 1 for word in text.split()))
                    attributes.append(all('z' not in word for word in text.split()))
                    attributes.append(all(
                        not word.endswith(('eid', 'se', 'en', 'ens', 'eid', 'dt', 't', 'et')) for word in text.split()))

                    data.append(attributes)

            self.data = pd.DataFrame(data, columns=['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'])


    '''
    This function is used to train the AdaBoost with K stumps.
    @:param K is the number of stumps to be used.
    @:return none
    '''


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


    '''
        This function finds the entropy of the series passed considering the weight of each element in the series.
        @:param series is the series for which entropy is to be find out.
        @:param weights is the weight of each example in the series.
        @:return the weighted entropy of the series passed.
    '''


    def weighted_entropy(self, series, weights):
        series = pd.Series(series)
        unique_values = series.unique()
        value_counts = series.value_counts(normalize=True)
        adjusted_weights = [np.sum(weights[series == value]) for value in unique_values]
        weighted_counts = value_counts * adjusted_weights
        return entropy(weighted_counts, base=2)


    '''
        This function is used to calculate condtional entropy of X given y. also considering the weigths of each examples
        @:param X is the attributes of each class. 
        @:param y contains the list of classes en and nl corresponding to each example.
        @:return the weighted conditional entropy of X given y. 
        '''


    def weighted_conditional_entropy(self, X, y):
        y_entropy = self.weighted_entropy(y, self.weights)
        values, counts = np.unique(X, return_counts=True)
        weighted_entropy = sum(
            [(counts[i] / np.sum(counts)) * self.weighted_entropy(y[X == value], self.weights[X == value]) for i, value
             in enumerate(values)])
        return y_entropy - weighted_entropy


    '''
        This function is used to find the information gain of each attributes and return the attribute with maximum information gain.
        Although the weights of each examples are not directly used in this function, it will be used in the subsequent functions called
        from this function.
        @:param X contains the dataframe excluding the class names.
        @:param y contains the class names corresponding to each examples.
        @:return the attribute index with maximum information gain.
        '''


    def weighted_informationgain(self, X, y):
        info_gain = [self.weighted_conditional_entropy(X[col], y) for col in X.columns]
        return np.argmax(info_gain)


    '''
    This function is used to find the best attribute to be used as a stub.
    @:param none
    @:return the stub to be used for each iteration.
    '''


    def train_weak_learner(self):

        best_attribute = self.weighted_informationgain(self.X, self.y)

        return best_attribute


    '''
    This function is used to predict and print whether the input given to it is 'en' or 'nl'
    @:param X is the dataframe created when reading the input for which prediction is to be made.
    @:return none
    '''


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

    '''
    This function is used to save the adaboost model created during train command.
    @:param none
    @:return none
    '''


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


    model.read(sys.argv[2],"train")

    model.train(50)

    model.save_model()


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