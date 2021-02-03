import gzip
import hashlib
import os
import tempfile

import numpy as np
import requests


def fetch(url):
    fp = os.path.join(
        tempfile.gettempdir(), hashlib.md5(url.encode("utf-8")).hexdigest()
    )
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching %s" % url)
        dat = requests.get(url).content
        with open(fp + ".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp + ".tmp", fp)
    return dat


def fetch_mnist():
    import gzip

    parse = lambda dat: np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
    X_train = (
        parse(fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"))[
            0x10:
        ]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_train = parse(
        fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")
    )[8:]
    X_test = (
        parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"))[
            0x10:
        ]
        .reshape((-1, 28 * 28))
        .astype(np.float32)
    )
    Y_test = parse(fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"))[
        8:
    ]
    return X_train, Y_train, X_test, Y_test
