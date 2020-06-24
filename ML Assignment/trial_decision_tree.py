import numpy as np
import ete3

Data_File = open('Admissions.txt', 'r')  # open the file contents
Data_String = Data_File.read()  # read the contents into a string
full_data = Data_String.splitlines()  # storing each datapoint into a list
Data_File.close()
feature_names = ['GRE test score',
                 'TOEFL test score',
                 'university rating',
                 'rating of strength of letter of purpose',
                 'rating of strength of letter of recommendation',
                 'undergraduate GPA',
                 'is experienced']
target_names = ['not admitted',
                'admitted']

full_data2 = []  # initiate a matrix to store the data in a an elegant way
y_full = []
"""" below  is for creating discrete class values"""
for data_point in full_data:
    data_point = data_point.split(',')
    del data_point[0]

    group = data_point[len(data_point) - 1]  # get the probability of admission

    bias = 0.65
    if float(group) > bias:  # ascertain whether the probability is above threshold
        group = 1  # group one represents accepted class
    else:
        group = 0  # group zero represents rejected class

    data_point.pop(len(data_point) - 1)  # remove the probability value
    # data_point.append(group)  # replace it discrete value
    y_full.append(group)
    full_data2.append(data_point)


class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

    def debug(self, feature_names, class_names, show_details):
        """Print an ASCII visualization of the tree."""
        lines, _, _, _ = self._debug_aux(
            feature_names, class_names, show_details, root=True
        )
        for line in lines:
            print(line)

    def _debug_aux(self, feature_names, class_names, show_details, root=False):
        # See https://stackoverflow.com/a/54074933/1143396 for similar code.
        is_leaf = not self.right
        if is_leaf:
            lines = [class_names[self.predicted_class]]
        else:
            lines = [
                "{} < {:.2f}".format(feature_names[self.feature_index], self.threshold)
            ]
        if show_details:
            lines += [
                "gini = {:.2f}".format(self.gini),
                "samples = {}".format(self.num_samples),
                str(self.num_samples_per_class),
            ]
        width = max(len(line) for line in lines)
        height = len(lines)
        if is_leaf:
            lines = ["║ {:^{width}} ║".format(line, width=width) for line in lines]
            lines.insert(0, "╔" + "═" * (width + 2) + "╗")
            lines.append("╚" + "═" * (width + 2) + "╝")
        else:
            lines = ["│ {:^{width}} │".format(line, width=width) for line in lines]
            lines.insert(0, "┌" + "─" * (width + 2) + "┐")
            lines.append("└" + "─" * (width + 2) + "┘")
            lines[-2] = "┤" + lines[-2][1:-1] + "├"
        width += 4  # for padding

        if is_leaf:
            middle = width // 2
            lines[0] = lines[0][:middle] + "╧" + lines[0][middle + 1:]
            return lines, width, height, middle

        # If not a leaf, must have two children.
        left, n, p, x = self.left._debug_aux(feature_names, class_names, show_details)
        right, m, q, y = self.right._debug_aux(feature_names, class_names, show_details)
        top_lines = [n * " " + line + m * " " for line in lines[:-2]]
        # fmt: off
        middle_line = x * " " + "┌" + (n - x - 1) * "─" + lines[-2] + y * "─" + "┐" + (m - y - 1) * " "
        bottom_line = x * " " + "│" + (n - x - 1) * " " + lines[-1] + y * " " + "│" + (m - y - 1) * " "
        # fmt: on
        if p < q:
            left += [n * " "] * (q - p)
        elif q < p:
            right += [m * " "] * (p - q)
        zipped_lines = zip(left, right)
        lines = (
                top_lines
                + [middle_line, bottom_line]
                + [a + width * " " + b for a, b in zipped_lines]
        )
        middle = n + width // 2
        if not root:
            lines[0] = lines[0][:middle] + "┴" + lines[0][middle + 1:]
        return lines, n + m + width, max(p, q) + 2 + len(top_lines), middle


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def debug(self, feature_names, class_names, show_details=True):
        """Print ASCII visualization of decision tree."""
        self.tree_.debug(feature_names, class_names, show_details)

    def fit(self, X, y, _tree):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y, _tree)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        """Find the best split for a node.

                "Best" means that the average impurity of the two children, weighted by their
                population, is the smallest possible. Additionally it must be less than the
                impurity of the current node.

                To find the best split, we loop through all the features, and consider all the
                midpoints between adjacent training samples as possible thresholds. We compute
                the Gini impurity of the split generated by that particular feature/threshold
                pair, and return the pair with smallest impurity.

                Returns:
                    best_idx: Index of the feature for best split, or None if no split is found.
                    best_thr: Threshold to use for the split, or None if no split is found.
                """
        m = y.size
        if m <= 1:
            return None, None
        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)

        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))  # Sort data along selected feature.
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                # The Gini impurity of a split is the weighted average of the Gini
                # impurity of the children.
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    # The above condition is to make sure we don't try to split two
                    # points with identical values for that feature, as it is impossible
                    # (both have to end up on the same side of a split).
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (float(thresholds[i]) + float(thresholds[i - 1])) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, _tree, depth=0):
        """Build a decision tree by recursively finding the best split."""
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                # indices_left = int(thr)
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr

                _tree = _tree.add_child(name=feature_names[idx])
                _tree.add_feature('threshold', thr)
                _tree.add_feature('predicted class', node.predicted_class)
                _tree_left = _tree.add_child(name='less than {} ->'.format(thr))
                _tree_right = _tree.add_child(name='less greater than {} ->'.format(thr))

                node.left = self._grow_tree(X_left, y_left, _tree_left, depth + 1)

                node.right = self._grow_tree(X_right, y_right, _tree_right, depth + 1)
        return node

    def _predict(self, inputs):
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


"""below we split data into training data and test data"""
train_data = full_data2[:int(0.7 * len(full_data2))]
train_data = np.array(train_data, dtype=np.float64)
y_train = y_full[:int(0.7 * len(y_full))]
test_data = np.array(full_data2[:int(0.7 * len(full_data2))], dtype=np.float64)
y_train = np.array(y_train)

"""create decision to learn the pattern of data, then make predictions based on test data"""
clf = DecisionTreeClassifier(max_depth=3)
_tree = ete3.Tree()
clf.fit(train_data, y_train, _tree)
y_predicted = clf.predict(test_data)
y_true = y_full[:int(0.7 * len(y_full))]
num_correct = 0
confusion_mat = [[0, 0], [0, 0]]  # [ [TP,FN], [FP,TN]
confusion_mat = np.array(confusion_mat)
for i in range(len(y_train)):
    print(' true y is {0} and predicted y is {1}'.format(y_true[i], y_predicted[i]))
    if y_true[i] == y_predicted[i] == 0:  # ascertain if is True Positive
        confusion_mat[0][0] += 1
    elif y_true[i] == y_predicted[i] == 1:  # ascertain if is true Negative
        confusion_mat[1][1] += 1
    elif y_true[i] == 1 and y_predicted == 0:  # ascertain if is False Negative
        confusion_mat[0][1] += 1
    else:  # ascertain if is False Positive
        confusion_mat[1][0] += 1

""""code below is to report accuracy the algorithm"""
classific_acc = (confusion_mat[0][0] + confusion_mat[1][1]) / (np.sum(confusion_mat))
classific_err = 1 - classific_acc
print('classification accuracy is {}%'.format(classific_acc * 100))
print('classification error is {}%'.format(classific_err * 100))
print('TP is {} \t FN is {}'.format(confusion_mat[0][0], confusion_mat[0][1]))
print('FP is {} \t TN is {}'.format(confusion_mat[1][0], confusion_mat[1][1]))

# print(_tree.get_ascii(show_internal=True,attributes=['name','predicted class']))
clf.debug(list(feature_names), list(target_names), False) # print the tree
