import numpy as np


# We do something a bit ad hoc, and allow for both categorical features (cat) and numerical ones (num). 
# For the categorical ones, we use the discrete metric, assigning distance 1 to different values.
class KNearestNeighbor:
    def __init__(self, n_nb = 3):
        self.n_nb = n_nb
        self.X = None
        self.y = None
        self.cat = None
        self.num = None

    
    def fit(self, X, y, cat = None, num = None):
        self.X = X
        self.y = y
        self.cat = cat
        if num is None:
            self.num = list(X)
        else:
            self.num = num

    
    def _dist2(self, x1, x2):
        return np.sum((x1 - x2) ** 2)
    

    def _find_neighbors(self, df, x):
        x_num = x[self.num]
        nearest_neighbors = {}

        for index, row in df.iterrows():
            row_num = row[self.num]

            addsum = 0
            for feat in self.cat:
                if x[feat] != row[feat]:
                    addsum += 1
            
            dist = np.sqrt(addsum + self._dist2(x_num, row_num))

            if len(nearest_neighbors) < self.n_nb:
                nearest_neighbors.update({index: dist})
            elif dist < max(nearest_neighbors.values()):
                m = max(nearest_neighbors.values())
                for key, value in nearest_neighbors.items():
                    if value == m:
                        del nearest_neighbors[key]
                        break
                nearest_neighbors.update({index: dist})
        
        return nearest_neighbors.keys()
    

    def predict(self, row):
        neighbors = self._find_neighbors(self.X, row)

        predictions = []

        for neighbor_index in neighbors:
            predictions.append(self.y.iloc[neighbor_index])
        
        return max(set(predictions), key=predictions.count)