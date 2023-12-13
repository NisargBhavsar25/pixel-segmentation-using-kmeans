import numpy as np
import random

class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=500):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        if len(data) < self.k:
            raise ValueError("Number of data points must be greater than k")

        # Initialize centroids with distinct data points
        self.centroids = {i: data[random.randint(0, len(data) - 1)] for i in range(self.k)}

        for _ in range(self.max_iter):
            self.classifications = {i: [] for i in range(self.k)}

            # Convert centroids to a NumPy array for vectorized operations
            centroids_array = np.array(list(self.centroids.values()))

            # Vectorized distance calculation
            distances = np.linalg.norm(data[:, np.newaxis] - centroids_array, axis=2)
            classifications = np.argmin(distances, axis=1)

            for index, classification in enumerate(classifications):
                self.classifications[classification].append(data[index])

            prev_centroids = dict(self.centroids)

            # Recalculate centroids
            for classification in self.classifications:
                if len(self.classifications[classification]) > 0:
                    self.centroids[classification] = np.mean(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                # Optimization check with a condition to avoid division by zero
                if np.linalg.norm(current_centroid - original_centroid) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        centroids_array = np.array(list(self.centroids.values()))
        distances = np.linalg.norm(data - centroids_array, axis=1)
        classification = np.argmin(distances)
        return self.centroids[classification]

# Example usage
# kmeans = KMeans(k=3, tol=0.01)
# kmeans.fit(data)
# prediction = kmeans.predict(new_data_point)
