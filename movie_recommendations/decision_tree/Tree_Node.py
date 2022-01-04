class Tree_Node:
    def __init__(self, feature=None, question=None, left=None, right=None, information_gain=None, value=None):
        self.feature = feature
        self.question = question
        self.left = left
        self.right = right
        self.information_gain = information_gain
        self.value = value
