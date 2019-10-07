
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import click

def sigmoid_activation(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    # compute the derivative of the sigmoid function ASSUMING
    # that `x` has already been passed through the `sigmoid`
    # function
    return x * (1 - x)

def predict(X, W):
    """
    :param X: features
    :param W: weight matrix
    """
    preds = sigmoid_activation(X.dot(W))

    # step func to threshold the outputs
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    return preds

@click.option('-e', '--epochs',
              type=float,
              default=100,
              help='# of epochs')

@click.option('-a', '--alpha',
              type=float,
              default=0.01,
              help='learning rate')

@click.command()
def main(epochs, alpha):
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
        preds = sigmoid_activation(trainX.dot(W))

        error = preds - trainY
        loss = np.sum(error ** 2)
        losses.append(loss)

        d = error * sigmoid_deriv(preds)
        gradient = trainX.T.dot(d)


        W += -alpha * gradient

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

