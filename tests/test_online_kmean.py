import unittest

from src.online_kmean import *


class OnlineKMeanTestCase(unittest.TestCase):
    def test_simple(self):
        num_features = 5
        num_clusters = 10

        model = OnlineKMeans(num_features, num_clusters)

        # Generate 10 centroids and keep feeding them in
        sample_centroid = np.arange(0, num_clusters).astype(np.float)
        sample_centroid = np.tile(sample_centroid[:, np.newaxis], (1, num_features))

        # Feed the a random number of points into the model
        num_points = 134
        for i in range(num_points):
            sample = sample_centroid[i % num_clusters]
            model.fit(sample)

        self.assertTrue(np.allclose(model.centroid, sample_centroid))

    def test_simple_2(self):
        """ A slight shift of the points """
        num_features = 5
        num_clusters = 10

        model = OnlineKMeans(num_features, num_clusters)

        # Generate 10 centroids and keep feeding them in
        sample_centroid = np.arange(0, num_clusters).astype(np.float)
        sample_centroid = np.tile(sample_centroid[:, np.newaxis], (1, num_features))

        # Feed the a random number of points into the model
        num_points = 20
        for i in range(num_points):
            sample = sample_centroid[i % num_clusters]

            if i >= num_clusters:
                sample = sample + 0.2

            model.fit(sample)

        self.assertTrue(np.allclose(model.centroid, sample_centroid + 0.1))


if __name__ == '__main__':
    unittest.main()
