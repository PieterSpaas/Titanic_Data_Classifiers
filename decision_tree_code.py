import numpy as np


class DecisionTree:
    def __init__(self, depth=1, max_depth=3):
        self.depth = depth
        self.max_depth = max_depth
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
    

    # Compute Gini index:
    def _gini_ind(self, y):
        classes, count = np.unique(y, return_counts=True)
        prob = count / len(y)
        return 1 - np.sum(prob ** 2)


    # Split a data set according to a threshold for a numerical feature:
    def _split_data_set(self, X, y, feature, threshold):
        left_X = X[X[feature] <= threshold]
        right_X = X[X[feature] > threshold]
        left_y = y[X[feature] <= threshold]
        right_y = y[X[feature] > threshold]
        return left_X, left_y, right_X, right_y


    # Find the best split according to Gini index.
    # For a first easy case, we consider only binary and continuous features (with the latter including integer valued features for now).
    # We split binary with artificial threshold 0.5, and for continuous features we check 100 equally spaced thresholds between the min and max (ignoring NaNs).
    def _find_best_split(self, X, y):
        best_gini = self._gini_ind(y)
        best_split = None

        for feature in list(X):
            if np.array_equal(X[feature].unique(), np.array([0,1])):
                left_X, left_y, right_X, right_y = self._split_data_set(X, y, feature, 0.5)

                gini = (len(left_y) / len(y)) * self._gini_ind(left_y) + (len(right_y) / len(y)) * self._gini_ind(right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature, 0.5)
            else:
                thresholds = np.linspace(np.nanmin(X[feature]), np.nanmax(X[feature]), 100)
                for threshold in thresholds:
                    left_X, left_y, right_X, right_y = self._split_data_set(X, y, feature, threshold)

                    if len(left_y) == 0 or len(right_y) == 0:
                        continue

                    gini = (len(left_y) / len(y)) * self._gini_ind(left_y) + (len(right_y) / len(y)) * self._gini_ind(right_y)
                    if gini < best_gini:
                        best_gini = gini
                        best_split = (feature, threshold)

        return best_split


    def fit(self, X, y):
        if len(np.unique(y)) == 1:
            self.value = np.unique(y)[0]
            return
        
        if self.depth > self.max_depth:
            self.value = y.mode()[0]
            return
        
        best_split = self._find_best_split(X, y)

        if best_split is None:
            self.value = y.mode()[0]
            return
        
        self.feature, self.threshold = best_split
        
        left_X, left_y, right_X, right_y = self._split_data_set(X, y, self.feature, self.threshold)
        self.left = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)
        self.right = DecisionTree(depth=self.depth + 1, max_depth=self.max_depth)

        self.left.fit(left_X, left_y)
        self.right.fit(right_X, right_y)
    

    def predict(self, X):
        if self.value is not None:
            return self.value

        if X[self.feature] <= self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)