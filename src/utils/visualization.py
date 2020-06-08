import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def show_clusters_in_2D_pca(data: np.ndarray, clusters: np.ndarray, num_clusters: int = 10,
                            title: str = "", block: bool = True):
    """
    Show the cluster in 2D image, use PCA for dimension reduction
    :param data: The shape should be (num_samples, num_features)
    :param clusters: The shape should be (num_samples,)
    :param num_clusters: The number of cluster classes
    :param title: The title of the image
    :param block: If True, it will show the plot immediately
    :return:
    """
    try:
        fig, ax = _show_clusters_in_2D_whatever(data, clusters, PCA, num_clusters, title, block)
    except:
        raise

    return fig, ax


def show_clusters_in_2D_tsne(data: np.ndarray, clusters: np.ndarray, num_clusters: int = 10,
                             title: str = "", block: bool = True):
    """
    Show the cluster in 2D image, use tSNE for dimension reduction

    It is highly recommended to use another dimensionality reduction method (e.g. PCA for dense data or TruncatedSVD
    for sparse data) to reduce the number of dimensions to a reasonable amount before you call this function.

    :param data: The shape should be (num_samples, num_features)
    :param clusters: The shape should be (num_samples,)
    :param num_clusters: The number of cluster classes
    :param title: The title of the image
    :param block: If True, it will show the plot immediately
    :return:
    """
    try:
        fig, ax = _show_clusters_in_2D_whatever(data, clusters, TSNE, num_clusters, title, block, verbose=True)
    except:
        raise

    return fig, ax


def _show_clusters_in_2D_whatever(data: np.ndarray, clusters: np.ndarray, dim_reduce_fn, num_clusters: int,
                                  title: str, block: bool, **kwargs):
    """
    Show the cluster in 2D image, use whatever dim_reduce_fn provided for dimension reduction
    :param data: The shape should be (num_samples, num_features)
    :param clusters: The shape should be (num_samples,)
    :param dim_reduce_fn: The dimension reduction function
    :param num_clusters: The number of cluster classes
    :param title: The title of the image
    :param block: If True, it will show the plot immediately
    :param kwargs: arguments for the dim_reduce_fn
    :return:
    """
    assert len(data.shape) == 2

    num_samples, num_features = data.shape
    if num_features < 2:
        print("num_features should be at least 2!")
        raise ValueError
    if num_features > 2:
        data = dim_reduce_fn(n_components=2, **kwargs).fit_transform(data)

    fig, ax = plt.subplots()
    colors = cm.rainbow(np.linspace(0, 1, num_clusters))
    ax.scatter(data[:, 0], data[:, 1], color=colors[clusters.astype(np.int)], s=0.3)

    ax.set_title(title)

    if block:
        plt.show()

    return fig, ax


def show_relative_ratio_with_base(base_list, ratio_list, title: str = "", block: bool = True):
    """ Show the relative ratio between the minimum number of clusters in online k-means and the batch k-means. The
    number of clusters used in the batch k-means is stored in the base_list.
    """
    fig, ax = plt.subplots()
    ax.plot(base_list, ratio_list)

    ax.set_xlabel("Number of Clusters in Batch K-means")
    ax.set_ylabel("Relative Ratio")
    ax.set_title(title)

    if block:
        plt.show()

    return fig, ax


def _show_relative_ratio_with_base_all():
    """ This is an ad hoc visualization function to show all the relative ratio curve of 3 datasets in one image """
    fig, ax = plt.subplots()
    base_list = np.linspace(0, 100, num=11).astype(np.int)
    base_list[0] = 1

    # MNIST
    ratio_list = [1.0, 1.16, 1.1949999999999998,
                  1.1766666666666665, 1.165, 1.1740000000000002, 1.175,
                  1.1585714285714286, 1.19, 1.18, 1.179]
    ax.plot(base_list, ratio_list, label="MNIST")

    # Fashion MNIST
    ratio_list = [1.0, 1.1199999999999999, 1.175,
                  1.1533333333333333, 1.2, 1.15,
                  1.1866666666666668, 1.1857142857142857, 1.23, 1.2044444444444444, 1.202]
    ax.plot(base_list, ratio_list, label="Fashion MNIST")

    # 20 news groups
    ratio_list = [1.0, 1.4300000000000002, 1.58, 1.7633333333333332,
                  1.8275, 1.838, 1.8599999999999999,
                  1.9428571428571428, 2.05375, 2.148888888888889, 2.135]
    ax.plot(base_list, ratio_list, label="Newsgroups")

    ax.set_xlabel("Number of Clusters in Batch K-means")
    ax.set_ylabel("Relative Ratio")
    ax.set_title("Relative Ratio Curve")
    ax.legend()

    plt.show()


def _show_learning_rate():
    """ An ad hoc function to show the learning rate vs minimum overparametrization curve of 3 datasets """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6.4 * 2, 4.8))

    # Visualize c_prime
    c_prime_list = np.linspace(1, 100, num=11)
    x_label = f"c'"
    y_label = "Minimum Clusters Size"
    title = ""

    ax = axes[0]
    x_list = c_prime_list

    # MNIST
    y_list = [161, 16, 14, 15, 20, 21, 24, 27, 30, 30, 35]
    ax.plot(x_list, y_list, label="MNIST")

    # Fashion MNIST
    y_list = [63, 12, 12, 15, 18, 19, 22, 25, 26, 28, 30]
    ax.plot(x_list, y_list, label="Fashion MNIST")

    # 20 news groups
    y_list = [1297, 724, 221, 80, 52, 51, 54, 54, 52, 60, 60]
    ax.plot(x_list, y_list, label="Newsgroups")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')

    # Visualize t0
    t0_list = np.linspace(1, 100, num=11)
    x_label = f"t0"
    y_label = "Minimum Clusters Size"
    title = ""

    ax = axes[1]
    x_list = t0_list

    # MNIST
    y_list = [16, 16, 16, 16, 16, 17, 16, 16, 16, 16, 16]
    ax.plot(x_list, y_list, label="MNIST")

    # Fashion MNIST
    y_list = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    ax.plot(x_list, y_list, label="Fashion MNIST")

    # 20 news groups
    y_list = [765, 765, 767, 772, 772, 773, 789, 789, 793, 796, 799]
    ax.plot(x_list, y_list, label="Newsgroups")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')

    plt.show()


def show_center_and_data(centers, data, title: str = "", block: bool = True):
    # Visualize the results on PCA-reduced data
    reduced_data = PCA(n_components=2).fit_transform(data)
    centers = PCA(n_components=2).fit_transform(centers)

    plt.figure(1)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='b', s=3)
    # Plot the centroids as a white X
    plt.scatter(centers[:, 0], centers[:, 1],
                marker='x', s=169, linewidths=3,
                color='r', zorder=10)
    plt.title(title)
    plt.xticks(())
    plt.yticks(())

    if block:
        plt.show()


if __name__ == '__main__':
    _show_learning_rate()
