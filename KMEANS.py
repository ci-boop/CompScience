import random

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data):
        # Randomly initialize centroids
        self.centroids = random.sample(data, self.n_clusters)

        for _ in range(self.max_iter):
            # Step 1: Assign clusters
            clusters = self._assign_clusters(data)

            # Step 2: Calculate new centroids
            new_centroids = self._calculate_centroids(data, clusters)

            # Check for convergence
            if self._has_converged(new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, data):
        return self._assign_clusters(data)

    def _assign_clusters(self, data):
        clusters = []
        for point in data:
            distances = [self._euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_centroid = distances.index(min(distances))
            clusters.append(closest_centroid)
        return clusters

    def _calculate_centroids(self, data, clusters):
        new_centroids = [[] for _ in range(self.n_clusters)]
        for i, point in enumerate(data):
            new_centroids[clusters[i]].append(point)

        # Calculate the mean of each cluster
        return [self._mean(cluster) for cluster in new_centroids]

    def _mean(self, cluster):
        if not cluster:
            return [0] * len(cluster[0])  # Return a zero vector if the cluster is empty
        return [sum(dim) / len(cluster) for dim in zip(*cluster)]

    def _euclidean_distance(self, point1, point2):
        return sum((x1 - x2) ** 2 for x1, x2 in zip(point1, point2)) ** 0.5

    def _has_converged(self, new_centroids):
        return all(self._euclidean_distance(c1, c2) < self.tol for c1, c2 in zip(self.centroids, new_centroids))

# Example usage
if __name__ == "__main__":
    # Sample data
    data = [[1, 2], [1, 4], [1, 0],
            [4, 2], [4, 4], [4, 0]]

    # Create and fit the KMeans model
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    # Predict cluster assignments for the training data
    predictions = kmeans.predict(data)
    print("Cluster Labels:", predictions)
    print("Centroids:", kmeans.centroids)