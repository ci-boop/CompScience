import random

class KNearestNeighbors:
    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors

    def fit(self, training_data, training_labels):
        """Store the training data and their corresponding labels."""
        self.training_data = training_data
        self.training_labels = training_labels

    def predict(self, test_data):
        """Predict the class labels for the test data."""
        return [self._predict_single(test_point) for test_point in test_data]

    def _predict_single(self, test_point):
        """Predict the class label for a single test point."""
        distances = []
        
        # Calculate the distance from the test point to each training point
        for i, train_point in enumerate(self.training_data):
            distance = self._euclidean_distance(test_point, train_point)
            distances.append((distance, self.training_labels[i]))
        
        # Sort the distances and select the labels of the nearest neighbors
        distances.sort(key=lambda x: x[0])
        nearest_labels = [label for _, label in distances[:self.num_neighbors]]
        
        # Return the most common label among the nearest neighbors
        return self._most_common_label(nearest_labels)

    def _euclidean_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return sum((x1 - x2) ** 2 for x1, x2 in zip(point1, point2)) ** 0.5

    def _most_common_label(self, labels):
        """Return the most common label from a list of labels."""
        label_count = {}
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        return max(label_count, key=label_count.get)

# Example usage
if __name__ == "__main__":
    # Sample training data
    training_data = [
        [1, 2], [1, 4], [1, 0],
        [4, 2], [4, 4], [4, 0]
    ]
    training_labels = ['A', 'A', 'A', 'B', 'B', 'B']

    # Sample test data
    test_data = [
        [1, 3], [4, 1]
    ]

    # Create and fit the KNN model
    knn = KNearestNeighbors(num_neighbors=3)
    knn.fit(training_data, training_labels)

    # Predict cluster assignments for the test data
    predictions = knn.predict(test_data)
    print("Predicted Labels:", predictions)
