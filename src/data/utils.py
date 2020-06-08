import numpy as np

from tqdm import tqdm

from src.data.dataloader import DataLoader


def calculate_aspect_ratio(cfg):
    """ Calculate the aspect ratio of each data set"""
    # Load the data
    num_features = 100
    data_loader = DataLoader(data_dir=cfg.data.path, dataset=cfg.data.name, split=cfg.data.split,
                             num_features=num_features)

    data = data_loader.get_data()
    num_samples, num_features = data.shape

    max_value = -np.inf
    min_value = np.inf

    for i in tqdm(range(num_samples)):
        if (i + 1) == num_samples:
            continue

        temp = data - data[i]
        temp = np.linalg.norm(temp[i + 1:], axis=1)

        max_candidate = np.max(temp)
        min_candidate = np.min(temp)

        if max_candidate > max_value:
            max_value = max_candidate

        if min_candidate != 0 and min_candidate < min_value:  # Ignore 0 value
            min_value = min_candidate

    aspect_ratio = max_value / min_value
    print(f"max value:{max_value:.4f}, min value:{min_value:.4f}, aspect ratio:{aspect_ratio:.4f}")


if __name__ == '__main__':
    import src.config.mnist
    import src.config.fashion_mnist
    import src.config.news_groups

    packages = [src.config.mnist, src.config.fashion_mnist, src.config.news_groups]
    cfg_list = [p.get_cfg_defaults() for p in packages]

    for cfg in cfg_list:
        calculate_aspect_ratio(cfg)
