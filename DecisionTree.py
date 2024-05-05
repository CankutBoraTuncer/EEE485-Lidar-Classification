import numpy as np
from TreeBranch import TreeBranch

class DecisionTree():
    def __init__(self, data, tree_depth):
        self.data         = data
        self.max_depth    = tree_depth
        self.min_data_len = 10 

    def build_tree(self, parent_data, depth = 0):
        """
            Given a splitted group, calculates the entropy impurity
            Parameters:
                split: The group to be calculated for impurity
        """
        label = np.argmax(np.bincount(parent_data[:, -1].astype(int)))

        if depth == self.max_depth or len(parent_data) < self.min_data_len:
            return TreeBranch(leaf_value=label)
        
        best_information_gain, best_feature_idx, best_feature_threshold, right_split, left_split = self.split_best(parent_data)

        if best_information_gain != 0:
            right_branch = self.build_tree(right_split, depth + 1)
            left_branch  = self.build_tree(left_split, depth + 1)
            return TreeBranch(best_feature_idx, best_feature_threshold, left_branch, right_branch, best_information_gain)
        else:
            return TreeBranch(leaf_value=label)
        
    def calc_entropy_impurity(self, split):
        """
            Given a splitted group, calculates the entropy impurity
            Parameters:
                split: The group to be calculated for impurity
        """
        entropy = 0
        labels = np.unique(split[:,-1])
        for label in labels:
            # The probablity of each label occuring in the split
            p = np.count_nonzero(split[:,-1] == label) / split.shape[1] 
            entropy += -1 * p * np.log2(p)
        return entropy
    
    def calc_gini_impurtiy(self, split):
        """
            Given a splitted group, calculates the gini impurity
            Parameters:
                split: The group to be calculated for impurity
        """  
        entropy = 1
        labels = np.unique(split[:,-1])
        for label in labels:
            # The probablity of each label occuring in the split
            p = np.count_nonzero(split[:,-1] == label) / split.shape[1] 
            entropy -= p*p
        return entropy
    
    def calc_information_gain(self, parent_data, right_split, left_split):
        """
            Calculate the information gained
            Parameters:
                parent_data: The data received by the branch
                left_split : The data splitted by the branch (left)
                right_split: The data splitted by the branch (right)
        """   
        parent_entropy = self.calc_entropy_impurity(parent_data)
        right_entropy  = self.calc_entropy_impurity(right_split)
        left_entropy   = self.calc_entropy_impurity(left_split)

        parent_count   = len(parent_data)
        right_count    = len(right_split)
        left_count     = len(left_split)

        right_weight   = right_count / parent_count
        left_weight    = left_count  / parent_count

        return parent_entropy - left_weight * left_entropy - right_weight * right_entropy

    def construct(self):
        self.root_branch = self.build_tree(self.data, 0)
        print("The tree is constructed")

    def predict(self, data, branch):
        """
            Calculate the information gained
            Parameters:
                parent_data: The data received by the branch
                left_split : The data splitted by the branch (left)
                right_split: The data splitted by the branch (right)
        """  
        if branch.leaf_value != -1 :
            return branch.leaf_value
        
        feature_idx = branch.split_feature_idx
        if data[feature_idx] >= branch.split_threshold :
            return self.predict(data, branch.branch_right)
        else:
            return self.predict(data, branch.branch_left)

    def split(self, data, feature_idx, feature_threshold):
        """
            Splits the given data according the feature (feature_idx) and the threshold
            Parameters:
                feature_idx: The index of the feature
                feature_threshold: The numerical threshold
        """
        right_split = []
        left_split = []
        for row in data:
            if(row[feature_idx] >= feature_threshold):
                right_split.append(row)
            else:
                left_split.append(row)
        right_split = np.array(right_split)
        left_split = np.array(left_split)

        return right_split, left_split
    
    def split_best(self, parent_data):
        """
            Splits the given data by finding the best feature and threshold
            Parameters:
                parent_data: The data received by the branch
        """
        best_information_gain   = 1e5
        best_feature_idx        = 0
        best_feature_threshold  = 0
        best_right_split        = 0
        best_left_split         = 0

        count = parent_data.shape[1] - 1
        for feature_idx in range(0, count):
            feature_threshold = np.mean(parent_data[:, feature_idx], axis=0)
            right_split, left_split = self.split(parent_data, feature_idx, feature_threshold)
            information_gain = self.calc_information_gain(parent_data, right_split, left_split)
            if(information_gain < best_information_gain):
                best_information_gain   = information_gain
                best_feature_idx        = feature_idx
                best_feature_threshold  = feature_threshold
                best_right_split        = right_split
                best_left_split         = left_split

        return best_information_gain, best_feature_idx, best_feature_threshold, best_right_split, best_left_split