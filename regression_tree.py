import numpy as np

#Class implementing a Tree
class Tree:
	
	#Tree contructor
    def __init__(self, label, split_attribute, split_value, left, right):

        self.label = label
        self.split_attribute = split_attribute
        self.split_value = split_value
        self.left = left
        self.right = right

	#predict one value
	def predict_one(self, example):

		if self.label is not None:

			return self.label

		elif example[self.split_attribute] < self.split_value:

			return self.left.predict_one(example)

		else:

			return self.right.predict_one(example)
		
	#predict using the Tree
    def predict(self, examples):

        predictions = []
		
		#predict each example
        for example in examples:

            predictions.append(self.predict_one(example))
		
		#return the predictions
        return np.asarray(predictions).reshape(len(predictions), 1)

	#print the tree
    def print_tree(self, depth):
        if self.label is not None:
            print(("\t" * depth) + str(self.label))

        else:
            print(("\t" * depth) + "Split " + str(self.split_attribute) + " on " + str(self.split_value))
            self.left.print_tree(depth + 1)
            self.right.print_tree(depth + 1)
			
#Class implementing a Regression Tree
class RegressionTree:
	
	#function to evaluate the gini score
    def gini(self, probabilities):

        return (1 - np.square(np.asarray(probabilities)).sum())

	#function to evaluate the gini remainder
    def gini_remainder(self, examples, labels, split_index, split_val):

        left_examples = []
        left_labels = []
        right_examples = []
        right_labels = []

		#determine the left and right splits of the data
        for i in range(len(examples)):

            if examples[i][split_index] < split_val:
                left_examples.append(examples[i])
                left_labels.append(labels[i])

            else:
                right_examples.append(examples[i])
                right_labels.append(labels[i])

		#calculate the number of positive and negative labels in each split
        num_examples = float(len(examples))
        p_left = float(left_labels.count(1))
        n_left = len(left_examples) - p_left
        p_right = float(right_labels.count(1))
        n_right = len(right_examples) - p_right

		# return a large gini remainder if one left/right split is only one class
        if p_left + n_left == 0 or p_right + n_right == 0: return 10000
		
		#return the gini remainder
        return (((p_left + n_left) / num_examples)*self.gini([(p_left / (p_left + n_left)), (n_left / (p_left + n_left))])
              + ((p_right + n_right) / num_examples)*self.gini([(p_right / (p_right + n_right)), (n_right / (p_right + n_right))]))

	#function to grow the regression tree
    def grow_tree(self, examples, labels, depth):

        #return a leaf if stopping criteria are met
        if (len(set(labels)) == 0 or
           depth > self.max_depth or
           len(examples) < self.min_examples):

            return Tree((float(sum(labels)) / len(labels)) , None, None, None, None)

        else:

            #determine index of best attribute to split on
            min_split_index = 0
            min_split_val = 0
            min_gini_remainder = np.inf

			#find the lowest gini remainder
            for column_index in range(len(examples[0])):
                for row_index in range(len(examples)):

                    gr = self.gini_remainder(examples, labels, column_index, examples[row_index][column_index])

                    if gr < min_gini_remainder:

                        min_split_index = column_index
                        min_split_val = examples[row_index][column_index]
                        min_gini_remainder = gr

			#set the bet split index and the best split value
            best = min_split_index
            best_split = min_split_val

			
            left_examples = []
            left_labels = []
            right_examples = []
            right_labels = []
			
			#split the data into left and right splits on the split value
            for i in range(len(examples)):
				
				#left split
                if examples[i][best] < best_split:
                    left_examples.append(examples[i])
                    left_labels.append(labels[i])
				
				#right split
                else:
                    right_examples.append(examples[i])
                    right_labels.append(labels[i])
			
			#grow the left and right subtrees if necessary
            if len(left_examples) > 0 and len(right_examples) > 0:

                left_subtree = self.grow_tree(left_examples, left_labels, depth + 1)
                right_subtree = self.grow_tree(right_examples, right_labels, depth + 1)
				
				#return this tree
                return Tree(None, best, best_split, left_subtree, right_subtree)

            else:
				#return a leaf node
                return Tree((float(sum(labels)) / len(labels)), None, None, None, None)

	#RegressionTree constructor
    def __init__(self, max_depth, min_examples):

        self.max_depth = max_depth - 1
        self.min_examples = min_examples
        self.tree = None
	
	#fit the regression tree to the training data
    def fit(self, examples, labels):

        self.tree = self.grow_tree(examples, labels, 0)
	
	#predict using the regression tree
    def predict(self, example):

        return self.tree.predict(example)
	
	#print the regression tree
    def print_tree(self):

        self.tree.print_tree(0)
