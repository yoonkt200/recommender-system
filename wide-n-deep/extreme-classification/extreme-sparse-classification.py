import argparse

from scipy.io import loadmat
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam


# dataset
dataset = "bibtex"
data_dir = 'data/{}'.format(dataset)

def load_input():
    data = list(loadmat(data_dir + '/input.mat')['data'][0][0])
    return data[:4]


class Deep:
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.model = self.classifier()

    def classifier(self):
        model = Sequential()
        model.add(Dense(5000, activation='relu', input_dim=self.input_dim))
        model.add(Dense(2500, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.output_dim, activation='softmax'))

        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def fit(self, x, y):
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

    def print_performance(self, x, y):
        performance_test = self.model.evaluate(x, y, batch_size=self.batch_size)
        print('Test Loss and Accuracy ->', performance_test)


class Wide:
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.model = self.classifier()

    def classifier(self):
        model = Sequential()
        model.add(Dense(self.output_dim, activation='softmax', input_dim=self.input_dim))

        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def fit(self, x, y):
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

    def print_performance(self, x, y):
        performance_test = self.model.evaluate(x, y, batch_size=self.batch_size)
        print('Test Loss and Accuracy ->', performance_test)


class WideAndDeep:
    def __init__(self, args):
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.model = self.classifier()

    def classifier(self):
        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999)

        # wide part
        wide = Input(shape=(self.input_dim,))

        # deep part
        deep_input = Input(shape=(self.input_dim,))
        deep = Dense(5000, activation='relu')(deep_input)
        deep = Dense(2500, activation='relu')(deep)
        deep = Dense(500, activation='relu')(deep)
        deep = Dense(50, activation='relu')(deep)

        # concatenate : wide and deep
        wide_n_deep = concatenate([wide, deep])
        wide_n_deep = Dense(self.output_dim, activation='softmax')(wide_n_deep)
        model = Model(inputs=[wide, deep_input], outputs=wide_n_deep)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model

    def fit(self, wide_x, deep_x, y):
        self.model.fit([wide_x, deep_x], y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2)

    def print_performance(self, wide_x, deep_x, y):
        performance_test = self.model.evaluate([wide_x, deep_x], y, batch_size=self.batch_size)
        print('Test Loss and Accuracy ->', performance_test)


def main(model_param):
    # prepare dataset
    x_train, y_train, x_test, y_test = load_input()
    x_train = x_train.toarray()
    y_train = y_train.toarray()
    x_test = x_test.toarray()
    y_test = y_test.toarray()

    # extreme test
    extreme_y_train = np.concatenate((y_train, y_train), axis=1)
    extreme_y_train = np.concatenate((extreme_y_train, extreme_y_train), axis=1)
    extreme_y_train = np.concatenate((extreme_y_train, extreme_y_train), axis=1)
    extreme_y_train = np.concatenate((extreme_y_train, extreme_y_train), axis=1)
    extreme_y_train = np.concatenate((extreme_y_train, extreme_y_train), axis=1)
    extreme_y_train = np.concatenate((extreme_y_train, extreme_y_train), axis=1)
    extreme_y_train = np.concatenate((extreme_y_train, extreme_y_train), axis=1)
    extreme_y_train = np.concatenate((extreme_y_train, extreme_y_train), axis=1)
    extreme_y_test = np.concatenate((y_test, y_test), axis=1)
    extreme_y_test = np.concatenate((extreme_y_test, extreme_y_test), axis=1)
    extreme_y_test = np.concatenate((extreme_y_test, extreme_y_test), axis=1)
    extreme_y_test = np.concatenate((extreme_y_test, extreme_y_test), axis=1)
    extreme_y_test = np.concatenate((extreme_y_test, extreme_y_test), axis=1)
    extreme_y_test = np.concatenate((extreme_y_test, extreme_y_test), axis=1)
    extreme_y_test = np.concatenate((extreme_y_test, extreme_y_test), axis=1)
    extreme_y_test = np.concatenate((extreme_y_test, extreme_y_test), axis=1)

    y_train = extreme_y_train
    y_test = extreme_y_test

    # prepare hyper parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for networks')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs for the networks')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--input_dim', type=int, default=x_train.shape[1],
                        help='Input dimension for the generator.')
    parser.add_argument('--output_dim', type=int, default=y_train.shape[1],
                        help='Output dimension for the generator.')
    args = parser.parse_args()

    if model_param == "deep":
        deep = Deep(args)
        deep.fit(x_train, y_train)
        deep.print_performance(x_test, y_test)
    elif model_param == 'wide':
        wide = Wide(args)
        wide.fit(x_train, y_train)
        wide.print_performance(x_test, y_test)
    else:
        wide_n_deep = WideAndDeep(args)
        wide_n_deep.fit(x_train, x_train, y_train)
        wide_n_deep.print_performance(x_test, x_test, y_test)

        # prediction for individual and y_column rank
        x_predict_test = x_test[np.newaxis, 0, :]
        y_predict_test = y_test[0]
        result = wide_n_deep.model.predict([x_predict_test, x_predict_test])
        print('result predicted:', result)
        print('result real:', y_predict_test)

        # select top 10 y's column index in result(softmax prediction)
        top_10_y_column = result[0].argsort()[-10:][::-1].tolist()
        print('result top 10:', top_10_y_column)


if __name__ == '__main__':
    main('wide')

    # X feature : (n, 1836) dim
    # Y_feautre : (n, 40000) dim
    # learning speed good.