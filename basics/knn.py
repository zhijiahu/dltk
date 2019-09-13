
import click

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths

from preprocessing import SimplePreprocessor
from datasets import SimpleDatasetLoader

@click.command()

@click.option('-d', '--dataset',
              required=True,
              prompt='path to input dataset')

@click.option('-k', '--neighbours',
              type=click.INT,
              default=1,
              prompt='# of nearest neighbours for classifications')

@click.option('-j', '--jobs',
              type=click.INT,
              default=-1,
              prompt='# of jobs for K-NN distance (-1 uses all available cores')

def main(dataset, neighbours, jobs):
    image_paths = list(paths.list_images(dataset))

    sp = SimplePreprocessor(32, 32)
    sdl = SimpleDatasetLoader(preprocessors=[sp])

    (data, labels) = sdl.load(image_paths, verbose_level=500)
    data = data.reshape((data.shape[0], 3072))
    print('features matrix: {:.1f}MB'.format(data.nbytes / (1024*1000.0)))

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    (train_x, test_x, train_y, test_y) = train_test_split(data,
                                                          labels,
                                                          test_size=0.25,
                                                          random_state=42)

    model = KNeighborsClassifier(n_neighbors=neighbours, n_jobs=jobs)
    model.fit(train_x, train_y)
    print(classification_report(test_y, model.predict(test_x), target_names=le.classes_))


if __name__ == "__main__":
    main()
