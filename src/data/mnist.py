"""
Modified from https://www.cntk.ai/pythondocs/CNTK_103A_MNIST_DataLoader.html
"""

# Functions to load MNIST images and unpack into train and test set.
# - loadData reads a image and formats it into a 28x28 long array
# - loadLabels reads the corresponding label data, one for each image
# - load packs the downloaded image and label data into a combined format to be read later by
#   the CNTK text reader

import gzip
import matplotlib.pyplot as plt
import numpy as np
import struct
import os.path as osp


def load_data(gz_file, num_images):
    """
    load_data reads a image and formats it into a 28x28 long array
    """
    with gzip.open(gz_file) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x3080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))[0]
        if n != num_images:
            raise Exception('Invalid file: expected {0} entries.'.format(num_images))
        num_row = struct.unpack('>I', gz.read(4))[0]
        num_col = struct.unpack('>I', gz.read(4))[0]
        if num_row != 28 or num_col != 28:
            raise Exception('Invalid file: expected 28 rows/cols per image.')
        # Read data.
        res = np.frombuffer(gz.read(num_images * num_row * num_col), dtype=np.uint8)
    return res.reshape((num_images, num_row * num_col))


def load_labels(gz_file, num_images):
    """
    load_labels reads the corresponding label data, one for each image
    """
    with gzip.open(gz_file) as gz:
        n = struct.unpack('I', gz.read(4))
        # Read magic number.
        if n[0] != 0x1080000:
            raise Exception('Invalid file: unexpected magic number.')
        # Read number of entries.
        n = struct.unpack('>I', gz.read(4))
        if n[0] != num_images:
            raise Exception('Invalid file: expected {0} rows.'.format(num_images))
        # Read labels.
        res = np.frombuffer(gz.read(num_images), dtype=np.uint8)
    return res.reshape((num_images, 1))


def load_data_and_label(image_file, label_file, num_images):
    data = load_data(image_file, num_images)
    labels = load_labels(label_file, num_images)

    out_dict = {
        "data": data,
        "labels": labels,
    }

    return out_dict


if __name__ == '__main__':
    mnist_root = "[path to]/data/mnist"
    train_image = osp.join(mnist_root, "train-images-idx3-ubyte.gz")
    train_label = osp.join(mnist_root, "train-labels-idx1-ubyte.gz")
    num_train_samples = 60000

    train_dataset = load_data_and_label(train_image, train_label, num_train_samples)

    print(train_dataset.shape)

    sample_number = 5001
    plt.imshow(train_dataset[sample_number, :-1].reshape(28, 28), cmap="gray_r")
    plt.axis('off')
    print("Image Label: ", train_dataset[sample_number, -1])

    plt.show()
