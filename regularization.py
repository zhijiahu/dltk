
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader
from imutils import paths
import click


@click.option('-d', '--dataset',
              required=True,
              help='path to input dataset')

@click.command()
def main(dataset):
    print('[INFO] loading images')
    image_paths = list(paths.list_images(dataset))


    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])
    (data, labels) = sdl.load(image_paths, verbose_level=500)
    data = data.reshape((data.shape[0], 3072))

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

    for r in (None, 'l1', 'l2'):
        # train a SGD classifier using a softmax loss function and the
        # specified regularization function for 10 epochs
        print('[INFO] training model with `{}` penalty'.format(r))
        model = SGDClassifier(loss='log',
                              penalty=r,
                              max_iter=10,
                              learning_rate='constant',
                              eta0=0.01,
                              random_state=42)
        model.fit(trainX, trainY)

        acc = model.score(testX, testY)
        print('[INFO] `{}` penalty accuracy: {:.2f}%'.format(r, acc * 100))


if __name__== '__main__':
    main()
