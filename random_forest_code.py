from sklearn.tree import DecisionTreeClassifier
import numpy as np

class RandomForest:
    def __init__(self, n_trees=100, max_depth=3, n_features='sqrt', n_samples=0.25):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_samples = n_samples
        self.trees = None

    
    def fit(self, X, y):
        self.trees = []

        for i in range(self.n_trees):
            new_tree = DecisionTreeClassifier(max_depth=self.max_depth, max_features=self.n_features)

            # Generate random sample
            N = int(self.n_samples * X.shape[0])
            X_sample = X.sample(N, replace=True, axis=0)
            y_sample = y.iloc[X_sample.index]

            new_tree.fit(X_sample, y_sample)

            self.trees.append(new_tree)
    

    def predict(self, X):
        tree_predictions = []

        for tree in self.trees:
            tree_predictions.append(tree.predict(X))

        tree_predictions = np.asarray(tree_predictions)

        predictions = []

        for i in range(len(tree_predictions[0])):
            counts = np.bincount(tree_predictions[:, i])
            predictions.append(np.argmax(counts))

        return predictions