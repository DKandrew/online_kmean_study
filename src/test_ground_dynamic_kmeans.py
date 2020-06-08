import numpy as np
import os
import os.path as osp

from sklearn.cluster import KMeans

from src.data.dataloader import DataLoader
from src.utils.logging import setup_logger
from src.online_kmean import DynamicOnlineKmeans


def run_dynamic_kmeans(cfg, num_clusters=None, num_run=10, logger=None):
    # Set up logger
    if logger is None:
        task_name = "run_dynamic_kmeans"
        output_dir = osp.join(cfg.output_dir, task_name)
        os.makedirs(output_dir, exist_ok=True)
        logger = setup_logger(task_name + f".{cfg.data.name}", save_dir=output_dir, prefix=cfg.data.name,
                              timestamp=True)

        logger.info(f"Running with configuration: \n{cfg}")

    # Load the data
    num_features = 100
    data_loader = DataLoader(data_dir=cfg.data.path, dataset=cfg.data.name, split=cfg.data.split,
                             num_features=num_features)
    data = data_loader.get_data()

    if num_clusters is None:
        num_clusters = data_loader.num_clusters

    # Run the batch kmeans
    ref_model = KMeans(n_clusters=num_clusters)
    ref_model.fit(data)
    target_cost = ref_model.inertia_
    logger.info(f"Kmean++ cost: {target_cost}")

    centroids_list = []
    cost_list = []

    for i in range(num_run):
        # Shuffle the data
        num_samples = data.shape[0]
        shuffle_idx = np.random.permutation(num_samples)
        shuffled_data = data[shuffle_idx]

        # Run the model
        model = DynamicOnlineKmeans(num_features=num_features, num_clusters=num_clusters, )
        model.fit(shuffled_data)
        curr_cost = model.calculate_cost(data)

        centroids_list.append(model.num_centroids)
        cost_list.append(curr_cost)
        logger.info(f"Requested: {num_clusters}, Used: {model.num_centroids}, Cost: {curr_cost}")

    avg_clusters = np.mean(centroids_list)
    avg_cost = np.mean(cost_list)

    logger.info(f"Average number of clusters used: {avg_clusters}")
    logger.info(f"Average cost: {avg_cost} | Diff {target_cost - avg_cost}")

    return avg_clusters, avg_cost


def run_dyanmic_kmean_with_change_of_k(cfg):
    task_name = "run_dyanmic_kmean_with_change_of_k"
    output_dir = osp.join(cfg.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(task_name + f".{cfg.data.name}", save_dir=output_dir, prefix=cfg.data.name,
                          timestamp=True)

    logger.info(f"Running with configuration: \n{cfg}")

    cluster_list = np.linspace(0, 100, num=11).astype(np.int)
    cluster_list[0] = 1

    num_run = 5
    avg_clusters_list = []
    for cluster in cluster_list:
        avg_clusters, avg_cost = run_dynamic_kmeans(cfg, num_clusters=cluster, num_run=num_run, logger=logger)
        avg_clusters_list.append(avg_clusters)

    logger.info(f"cluster list:{cluster_list}")
    logger.info(f"avg clusters:{avg_clusters_list}")


def main():
    import src.config.mnist
    import src.config.fashion_mnist
    import src.config.news_groups

    packages = [src.config.mnist, src.config.fashion_mnist, src.config.news_groups]
    cfg_list = [p.get_cfg_defaults() for p in packages]

    for cfg in cfg_list:
        # run_dynamic_kmeans(cfg)
        run_dyanmic_kmean_with_change_of_k(cfg)


if __name__ == '__main__':
    main()
