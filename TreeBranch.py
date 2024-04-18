class TreeBranch():
    def __init__(self, split_feature_idx=None, split_threshold=None, branch_left=None, branch_right=None, information_gain=None, leaf_value=-1):
        self.split_feature_idx  = split_feature_idx
        self.split_threshold    = split_threshold
        self.branch_left        = branch_left
        self.branch_right       = branch_right
        self.information_gain   = information_gain
        self.leaf_value         = leaf_value