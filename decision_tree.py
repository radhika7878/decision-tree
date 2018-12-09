from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}

    def learn(self, X, y):
        # DONE: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        self.tree = self.buildTree(X, y)
        
    def buildTree(self, X, y):
        #Recursively builds the tree till IG is zero
        (split_attribute, split_value) = self.findMaxIGSplit(X, y)
        if split_attribute is None:
            counts = np.bincount(y)
            class_y = np.argmax(counts)
            node = {"left": None, "right": None, "split_value": None, "split_attribute": None, "class_y": class_y}
            return node
        else:
            (X_left, X_right, y_left, y_right) = partition_classes(X, y, split_attribute, split_value)
            node = {"left": self.buildTree(X_left, y_left), "right": self.buildTree(X_right, y_right), "split_value": split_value, "split_attribute": split_attribute, "class_y": None}
            return node          

    def findMaxIGSplit(self, X, y):
        #If IG is zero returns none for split value and attribute
        #For categorical variables uses just the value of the 0th row to split
        #For continuous variables uses the mean 
        #Not splitting the node further if info gain is less than 0.2
        maxInfoGain = 0.2
        split_value = None
        split_attribute = None
        if(len(np.unique(y))==1):
            return (split_attribute, split_value)
        numerical_cols = set([0,10,11,12,13,15,16,17,18,19,20]) # indices of numeric attributes (columns)
        arr = np.array(X)
        for i in range(len(X[0])):
            if i in numerical_cols:
                mean = np.mean((arr[:,i]).astype(np.float))
            else:
                mean = X[0][i]
            (X_left_t, X_right_t, y_left_t, y_right_t) = partition_classes(X, y, i, mean)
            temp_ig = information_gain(y, [y_left_t, y_right_t])
            if(temp_ig>maxInfoGain):
                maxInfoGain = temp_ig
                split_attribute = i
                split_value = mean
        return (split_attribute, split_value)
        


    def classify(self, record):
        # DONE: classify the record using self.tree and return the predicted label
        curr = self.tree
        numerical_cols = set([0,10,11,12,13,15,16,17,18,19,20]) # indices of numeric attributes (columns)
        while(curr["split_attribute"] is not None):
            split_attribute = curr["split_attribute"]
            split_value = curr["split_value"]
            if((split_attribute in numerical_cols and record[split_attribute] <= float(split_value)) or (record[split_attribute] == split_value)):
                curr = curr["left"]
            else:
                curr = curr["right"]
        return curr["class_y"]
