import urllib.request
import os.path
import gzip
import pickle
import os
import numpy as np

url_base = "http://yann.lecun.com/exdb/mnist/"
key_files = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_label": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_label": "t10k-labels-idx1-ubyte.gz",
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def download_file(fn):
    fp = dataset_dir + "/" + fn

    if os.path.exists(fn):
        return

    print("Downloading " + fn + " ... ")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:47.0) Gecko/20100101 Firefox/47.0"
    }
    request = urllib.request.Request(url_base + fn, headers=headers)
    response = urllib.request.urlopen(request).read()
    with open(fp, mode="wb") as f:
        f.write(response)
    print("Done")


def download_mnist():
    for v in key_files.values():
        download_file(v)


def load_label(fn):
    print("Converting " + fn + " to NumPy Array ...")
    fp = dataset_dir + "/" + fn
    with gzip.open(fp, "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def load_img(fn):
    print("Converting " + fn + " to NumPy Array ...")
    fp = dataset_dir + "/" + fn
    with gzip.open(fp, "rb") as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def convert_numpy():
    dataset = {}
    dataset['train_img'] =  load_img(key_files['train_img'])
    dataset['train_label'] = load_label(key_files['train_label'])
    dataset['test_img'] = load_img(key_files['test_img'])
    dataset['test_label'] = load_label(key_files['test_label'])
    return dataset


def init_mnist():
    download_mnist()
    dataset = convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, "wb") as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """MNISTデータセットの読み込み

    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    one_hot_label :
        one_hot_labelがTrueの場合、ラベルはone-hot配列として返す
        one-hot配列とは、たとえば[0,0,1,0,0,0,0,0,0,0]のような配列
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練ラベル), (テスト画像, テストラベル)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, "rb") as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset["train_label"] = change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = change_one_hot_label(dataset["test_label"])

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_img"], dataset["train_label"]), (
        dataset["test_img"],
        dataset["test_label"],
    )


if __name__ == "__main__":
    init_mnist()
