import numpy as np

class KNearestNeighbor(object):
    ''' a KNN classifier using L2 distance '''
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=1):
        distances = self.compute_distances(X)
        return self.predict_labels(distances, k=k)

    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        distances = np.zeros((num_test, num_train))

        distances = np.sqrt(np.sum(X**2, axis=1).reshape(num_test, 1) + np.sum(self.X_train**2, axis=1) - 2*X @ self.X_train.T)
        return distances

    def predict_labels(self, distances, k=1):
        num_test = distances.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(distances[i])[:k]]

            num_classes = np.max(closest_y)+1
            classes_vote_cnt = np.zeros(num_classes)

            for class_vote in closest_y:
                classes_vote_cnt[class_vote]+=1

            y_pred[i] = np.argmax(classes_vote_cnt)     # Note that argmax will resolve ties
            
        return y_pred
