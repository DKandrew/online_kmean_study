import os.path as osp
import numpy as np

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups_vectorized

from src.data.mnist import load_data_and_label


class DataLoader:
    """ Data loader for various data sets """

    def __init__(self, data_dir: str, dataset: str = 'mnist', split: str = 'trainval', num_features: int = 100):
        """

        :param data_dir: the root directory of the data
        :param dataset: one of 'mnist', 'fashion_mnist', 'news_groups'
        :param split: one of 'train', 'val', 'test', 'trainval'
        :param num_features: the dimension of the data. If the data is smaller than num_features, then all the
        dimension will be preserved. Otherwise, we will do PCA dimension reduction.
        """
        if dataset not in ['mnist', 'fashion_mnist', 'news_groups']:
            raise ValueError(f"Dataset {dataset} is not supported!")
        if split not in ['train', 'val', 'test', 'trainval']:
            raise ValueError(f"split must be in ['train', 'val', 'test', 'trainval']. {split} is not supported.")

        self.data_dir = data_dir
        self.name = dataset
        self.split = split

        if dataset in ['mnist', 'fashion_mnist']:
            train_image = osp.join(data_dir, "train-images-idx3-ubyte.gz")
            train_label = osp.join(data_dir, "train-labels-idx1-ubyte.gz")
            num_train_samples = 60000

            test_image = osp.join(data_dir, "t10k-images-idx3-ubyte.gz")
            test_label = osp.join(data_dir, "t10k-labels-idx1-ubyte.gz")
            num_test_samples = 10000

            collector = []
            if split in ['train', 'trainval']:
                collector.append(load_data_and_label(train_image, train_label, num_train_samples))
            if split in ['val', 'test', 'trainval']:
                collector.append(load_data_and_label(test_image, test_label, num_test_samples))
            self.data = np.concatenate([elem["data"] for elem in collector])
            self.labels = np.concatenate([elem["labels"] for elem in collector])
            self.num_clusters = 10
        elif dataset == 'news_groups':
            if split == 'trainval':
                split = 'all'
            news_groups_dataset = fetch_20newsgroups_vectorized(subset=split, data_home=data_dir)
            self.data = news_groups_dataset.data
            self.labels = news_groups_dataset.target
            self.labels_name = news_groups_dataset.target_names
            self.num_clusters = 20

        # Reduce data dimension
        if self.data.shape[1] > num_features:
            if dataset == 'news_groups':
                # News groups have very sparse matrix, so we cannot use PCA.
                self.data = TruncatedSVD(n_components=num_features).fit_transform(self.data)
            else:
                self.data = PCA(n_components=num_features).fit_transform(self.data)

        # Collect the number of features in the data
        self.num_features = self.data.shape[1]

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels
