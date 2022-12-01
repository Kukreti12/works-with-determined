import pandas as pd
import numpy as np

def load_training_data(path):
    """
    Load the training dataset
    """
    df = pd.read_csv(path)
    
    return x_train, y_train


def load_validation_data():
    """Loads the Fashion-MNIST dataset.

    Returns:
        Tuple of Numpy arrays: `(x_test, y_test)`.

    License:
        The copyright for Fashion-MNIST is held by Zalando SE.
        Fashion-MNIST is licensed under the [MIT license](
        https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

    """
    download_directory = tempfile.mkdtemp()
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = [
        "t10k-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
    ]

    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=base + fname, cache_subdir=download_directory))

    with gzip.open(paths[0], "rb") as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], "rb") as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return x_test, y_test
