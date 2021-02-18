import typer
import numpy as np
from utils.data import fetch_mnist
from pathlib import Path
from tinygrad.tensor import Tensor
import tinygrad.optim as optim
from tqdm import trange
from typing import List


def sparse_categorical_crossentropy(out: Tensor, Y: np.ndarray) -> Tensor:
    num_classes = out.shape[-1]
    YY = Y.flatten()
    y = np.zeros((YY.shape[0], num_classes), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),YY] = -1.0*num_classes
    y = y.reshape(list(Y.shape)+[num_classes])
    y = Tensor(y)
    return out.mul(y).mean()


class Dense:
    def __init__(self, in_size: int, out_size: int):
        self.weight = Tensor.randn(in_size, out_size)
        self.bias = Tensor.zeros(1, out_size)

    def __call__(self, x: Tensor) -> Tensor:
        """
        >>> x = Tensor.zeros(2, 10)
        >>> layer = Dense(10, 5)
        >>> y = layer(x)
        >>> y.shape
        (2, 5)
        """
        return x.dot(self.weight) + self.bias

    def parameters(self) -> List[Tensor]:
        return [self.weight, self.bias]


class FCNN:
    def __init__(self, input_size: int, num_classes: int):
        self.layer1 = Dense(input_size, 64)
        self.layer2 = Dense(64, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        y = x
        y = self.layer1(y).relu6()
        return self.layer2(y).logsoftmax()

    def parameters(self) -> List[Tensor]:
        return self.layer1.parameters() + self.layer2.parameters()


def train(outdir: str = None, epochs: int = 1, batch_size: int = 32):
    X_train, Y_train, X_test, Y_test = fetch_mnist()
    model = FCNN(28 * 28, 10)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        msg = "loss={loss}"
        status = trange(len(X_train) // batch_size)

        Tensor.training = True
        for batch_idx in status:
            samp = np.random.choice(len(X_train), size=batch_size)
            x = Tensor(X_train[samp]) / 255
            out = model(x)
            loss = sparse_categorical_crossentropy(out, Y_train[samp])
            status.set_description(msg.format(loss=float(loss.data)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        Tensor.training = False
        correct = []
        for batch_idx in trange(len(X_test) // batch_size, desc="Testing"):
            n = batch_idx * batch_size
            x = Tensor(X_test[n:n + batch_size])
            out = model(x).data.argmax(axis=1)
            correct.append(out == Y_test[n:n + batch_size])

        acc = np.concatenate(correct).mean()
        print("Accuracy:", acc)

    if not outdir:
        return

    weights = {
        "w1": model.layer1.weight.data,
        "b1": model.layer1.bias.data,
        "w2": model.layer2.weight.data,
        "b2": model.layer2.bias.data
    }
    Path(outdir).mkdir(exist_ok=True)
    for name, val in weights.items():
        np.save(Path(outdir) / f"{name}.npy", val.astype(np.float32))

    order = np.argsort(Y_test[:10])
    x = X_test[order].reshape((10, -1)).astype(np.float32) / 255
    print(Y_test[order])
    np.save(Path(outdir) / "samples.npy", x)



if __name__ == "__main__":
    typer.run(train)
