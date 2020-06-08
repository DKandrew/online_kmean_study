# Since fashion_mnist is a direct drop in of MNIST dataset, there is nothing we need to do here.
from src.data.mnist import *

if __name__ == '__main__':
    mnist_root = "[path to]/data/fashion_mnist"
    train_image = osp.join(mnist_root, "train-images-idx3-ubyte.gz")
    train_label = osp.join(mnist_root, "train-labels-idx1-ubyte.gz")
    num_train_samples = 60000

    out_dict = load_data_and_label(train_image, train_label, num_train_samples)

    data = out_dict["data"]
    labels = out_dict["labels"]
    print(data.shape)

    sample_number = 4001
    plt.imshow(data[sample_number, :].reshape(28, 28), cmap="gray_r")
    plt.axis('off')
    print("Image Label: ", labels[sample_number])

    plt.show()
