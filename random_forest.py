import numpy as np

class Tree:

    def __init__(self, label, split_attribute, split_value, left, right):

        self.label = label
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.left = left
        self.right = right

    def predict_one(self, example):

        if self.label is not None:

            return self.label

        elif example[self.split_attribute] < self.split_value:

            return self.left.predict_one(example)

        else:

            return self.right.predict_one(example)

    def predict(self, examples):

        predictions = []

        for example in examples:

            predictions.append(self.predict_one(example))

        return np.asarray(predictions).reshape(len(predictions), 1)

    def print_tree(self, depth):

        if self.label is not None:

            print(("\t" * depth) + str(self.label))

        else:

            print(("\t" * depth) + "Split " + str(self.split_attribute) + " on " + str(self.split_value))
            self.left.print_tree(depth + 1)
            self.right.print_tree(depth + 1)


class RegressionTree:

    def gini(self, probabilities):

        return (1 - np.square(np.asarray(probabilities)).sum())

    def gini_remainder(self, examples, labels, split_index, split_val):

        left_examples = []
        left_labels = []
        right_examples = []
        right_labels = []

        for i in range(len(examples)):

            if examples[i][split_index] < split_val:
                left_examples.append(examples[i])
                left_labels.append(labels[i])

            else:
                right_examples.append(examples[i])
                right_labels.append(labels[i])

        num_examples = float(len(examples))
        p_left = float(left_labels.count(1))
        n_left = len(left_examples) - p_left
        p_right = float(right_labels.count(1))
        n_right = len(right_examples) - p_right

        #print("num_examples:" + str(num_examples))
        #print("p_left" + str(p_left))
        #print("n_left" + str(n_left))
        #print("p_right" + str(p_right))
        #print("n_right" + str(n_right))

        if p_left + n_left == 0 or p_right + n_right == 0: return 10000

        return (((p_left + n_left) / num_examples)*self.gini([(p_left / (p_left + n_left)), (n_left / (p_left + n_left))])
              + ((p_right + n_right) / num_examples)*self.gini([(p_right / (p_right + n_right)), (n_right / (p_right + n_right))]))

    def grow_tree(self, examples, labels, depth):

        #TODO
        if (len(set(labels)) == 0 or
           depth > self.max_depth or
           len(examples) < self.min_examples):

            return Tree((float(sum(labels)) / len(labels)) , None, None, None, None)

        else:

            #test_examples = [[0], [0], [1], [0], [1], [1], [1], [0], [0], [0]]
            #test_labels = [1, 0, 0, 1, 1, 1, 0, 0, 1, 1]

            #TODO determine index of best attribute
            min_split_index = 0
            min_split_val = 0
            min_gini_remainder = np.inf

            #create a random subset of the features
            indexes = np.random.choice(len(examples[0]), size = int(len(examples[0])/3), replace = False)
            for column_index in indexes:
                for row_index in range(len(examples)):

                    gr = self.gini_remainder(examples, labels, column_index, examples[row_index][column_index])

                    if gr < min_gini_remainder:

                        min_split_index = column_index
                        min_split_val = examples[row_index][column_index]
                        min_gini_remainder = gr

            best = min_split_index
            best_split = min_split_val

            left_examples = []
            left_labels = []
            right_examples = []
            right_labels = []

            for i in range(len(examples)):

                if examples[i][best] < best_split:
                    left_examples.append(examples[i])
                    left_labels.append(labels[i])

                else:
                    right_examples.append(examples[i])
                    right_labels.append(labels[i])

            if len(left_examples) > 0 and len(right_examples) > 0:

                left_subtree = self.grow_tree(left_examples, left_labels, depth + 1)
                right_subtree = self.grow_tree(right_examples, right_labels, depth + 1)

                return Tree(None, best, best_split, left_subtree, right_subtree)

            else:

                return Tree((float(sum(labels)) / len(labels)), None, None, None, None)


    def __init__(self, max_depth, min_examples):

        self.max_depth = max_depth
        self.min_examples = min_examples
        self.tree = None

    def fit(self, examples, labels):

        self.tree = self.grow_tree(examples, labels, 0)

    def predict(self, examples):

        return self.tree.predict(examples)

    def print_tree(self):

        self.tree.print_tree(0)

#Random Forest Regression
class RandomForest:

    def __init__(self, num_trees, max_depth, min_examples):
        self.num_trees = num_trees
        self.max_depth = max_depth - 1
        self.min_examples = min_examples
        self.trees = []


    def fit(self, examples, labels):

        for tree in range(self.num_trees):

            #sample a random set of examples from the dataset
            indexes = np.random.choice(len(examples), size=len(examples), replace=True)
            bagged_examples = [examples[i] for i in indexes]
            bagged_labels = [labels[i] for i in indexes]

            #fit the current tree to the random sample
            new_tree = RegressionTree(self.max_depth, self.min_examples)
            new_tree.fit(bagged_examples, bagged_labels)

            #add the fitted tree to the forest's trees
            self.trees.append(new_tree)


    def predict(self, examples):

        prediction_sums = [0] * len(examples)

        for tree in self.trees:
            predictions = tree.predict(examples)

            for index, pred in enumerate(predictions):
                prediction_sums[index] += pred

        predictions = [float(pred)/self.num_trees for pred in prediction_sums]
        return np.asarray(predictions).reshape(len(predictions), 1)

