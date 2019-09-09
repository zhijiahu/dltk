
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import click


def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    """
    :param X: features
    :param W: weight matrix
    """
    preds = sigmoid_activation(X.dot(W))

    # step func to threshold the outputs
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds

def next_batch(X, y, batch_size):
    """
    Loop over dataset X in mini-batches,
    yielding a tuple of current batched data and labels

    :param X: Training dataset of feature vectors/raw image pixel intensties
    :param y: Class labels associiated with each of the training data points
    :param batch_size: The size of each mini batch that will be returned
    """

    for i in np.arange(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])


@click.option('-e', '--epochs',
              type=float,
              default=100,
              help='# of epochs')

@click.option('-a', '--alpha',
              type=float,
              default=0.01,
              help='learning rate')

@click.option('-b', '--batch-size',
              type=int,
              default=32,
              help='size of SGD mini-batches')

@click.command()
def main(epochs, alpha, batch_size):
    # generate 2-class clasification problem
    (X, y) = make_blobs(n_samples=1000,
                        n_features=2,
                        centers=2,
                        cluster_std=1.5,
                        random_state=1)
    y = y.reshape((y.shape[0], 1))

    # bias trick
    X = np.c_[X, np.ones((X.shape[0]))]

    (trainX, testX, trainY, testY) = train_test_split(X, y,
                                                       test_size=0.5,
                                                       random_state=42)

    print('[INFO] training...')

    W = np.random.randn(X.shape[1], 1)
    losses = []

    for epoch in np.arange(0, epochs):

        epoch_loss = []
        for (batch_X, batch_Y) in next_batch(X, y, batch_size):

            preds = sigmoid_activation(batch_X.dot(W))

            error = preds - batch_Y
            epoch_loss.append(np.sum(error ** 2))

            gradient = batch_X.T.dot(error)

            W += -alpha * gradient

        loss = np.average(epoch_loss)
        losses.append(loss)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print('[INFO] epoch={}, loss={:.7f}'.format(int(epoch + 1), loss))

    print('[INFO] evaluating')
    preds = predict(testX, W)
    print(classification_report(testY, preds))

    plt.style.use('ggplot')
    plt.figure()
    plt.title('Data')
    plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY, s=30)

    plt.style.use('ggplot')
    plt.figure()
    plt.title('Traning loss')
    plt.plot(np.arange(0, epochs), losses)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.show()


if __name__== '__main__':
    main()

