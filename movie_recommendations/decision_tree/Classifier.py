import pandas as pd
from Tree_Node import Tree_Node
from anytree import AnyNode

features = ["vote_average", "budget", "popularity"]


def gini(data):
    ratings = data.unique()
    total = 0
    for rating in ratings:
        rating_count = len(data[data == rating])
        total += (rating_count / len(data)) ** 2
    return 1 - total


def calculate_information_gain(parent, left, right):
    left_weight = len(left) / len(parent)
    right_weight = len(right) / len(parent)
    return gini(parent) - (left_weight * gini(left)) + (right_weight * gini(right))


def get_best_split(dataset):
    best_split = {}
    max_information_gain = -float("inf")

    for feature in features:
        feature_values = dataset[feature]
        questions = feature_values.unique()

        for question in questions:
            left, right = dataset[dataset[feature] < question], dataset[dataset[feature] >= question]

            if len(left) > 0 and len(right) > 0:
                y, left_y, right_y = dataset["rating"], left["rating"], right["rating"]
                information_gain = calculate_information_gain(y, left_y, right_y)

                if information_gain > max_information_gain:
                    best_split["feature"] = feature
                    best_split["question"] = question
                    best_split["left"] = left
                    best_split["right"] = right
                    best_split["information_gain"] = information_gain
                    max_information_gain = information_gain
    return best_split


class Classifier:
    def __init__(self, split_stop=2, depth_stop=None):
        self.root = None
        self.split_stop = split_stop
        self.depth_stop = depth_stop

    def generate_tree(self, dataset, depth=0):
        data = dataset.shape[0]

        if data >= self.split_stop and depth <= self.depth_stop:
            best_split = get_best_split(dataset)

            if best_split["information_gain"] > 0:
                left = self.generate_tree(best_split["left"], depth + 1)
                right = self.generate_tree(best_split["right"], depth + 1)
                return Tree_Node(best_split["feature"], best_split["question"], left, right,
                                 best_split["information_gain"])
        leaf = round(dataset["rating"].mean())
        return Tree_Node(value=leaf)

    def fit(self, x, y):
        dataset = pd.concat(objs=[x, y], axis=1)
        self.root = self.generate_tree(dataset)

    def predict(self, x):
        predictions = x.apply(lambda pred: int(self.single_prediction(pred, self.root)), axis=1)
        return predictions.to_numpy()

    def single_prediction(self, x, root):
        if root.value is not None:
            return root.value

        feature = x[root.feature]
        if feature < root.question:
            return self.single_prediction(x, root.left)
        else:
            return self.single_prediction(x, root.right)

    def generate_tree_png(self, node=None, parent_question=None, depth=0, left=False, right=False):
        if node is None:
            node = self.root

        if node.value is not None:
            sign = "<" if left else ">="
            return AnyNode(name=f'Rating: {node.value}, {sign} {parent_question}, depth: {depth}', )
        else:
            condition = f'{node.feature} <= {node.question}'
            return AnyNode(
                name=f'{condition}, depth: {depth}',
                children=[
                    self.generate_tree_png(node.left, parent_question=node.question, depth=depth + 1, left=True),
                    self.generate_tree_png(node.right, parent_question=node.question, depth=depth + 1, right=True)]
            )
