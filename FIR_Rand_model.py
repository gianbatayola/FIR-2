import os
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import numpy as np
from sklearn.inspection import permutation_importance

# hides messages about CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_model(inp_int_: int, l1_w_: int,
                l2_w_: int) -> tf.python.keras.engine.functional.Functional:
    """
    Builds a keras model.

    :param inp_int_: integer of number of features.
    :param l1_w_: integer of weight 1.
    :param l2_w_: integer of weight 2.

    :return: keras model to be used.
    """
    # uses (inp * l1 + l1) + (l1 * l2 + l2) + (out * l2 + out) formula
    inp = tf.keras.Input(shape=(inp_int_,))
    l1 = tf.keras.layers.Dense(l1_w_, kernel_initializer='uniform', activation='relu')(inp)
    l2 = tf.keras.layers.Dense(l2_w_, kernel_initializer='uniform', activation='relu')(l1)
    out = tf.keras.layers.Dense(1, kernel_initializer='uniform', activation='sigmoid')(l2)
    model_p = tf.keras.Model(inp, out)
    model_p.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_p


class Model:
    """
    This class represents the keras model that will undergo testing and feature importance ranking.
    """

    def __init__(self, inp_int: int = 15, l1_w: int = 16, l2_w: int = 16, n_test_samples: int = 190,
                 epoch: int = 100,
                 batch: int = 32,
                 verbosity: int = 0):
        """
        Constructs the keras model and uses a wrapper so it can be used for sk-learn functions.

        :param inp_int: integer of number of features.
        :param l1_w: integer of weight 1.
        :param l2_w: integer of weight 2.
        :param n_test_samples: the number of test samples.
        :param epoch: the number of epochs to execute for the input data.
        :param batch: specifies the batch size.
        :param verbosity: 0 or 1 indicating if training progress should be shown.
        """
        weights = (inp_int * l1_w + l1_w) + (l1_w * l2_w + l2_w) + (l1_w + 1)
        if weights < 2 * n_test_samples:
            self.epoch = epoch
            self.batch = batch
            self.verbosity = verbosity
            # wrapper function allows keras models to use sk-learn functions
            self.model = tf.keras.wrappers.scikit_learn.RandomForestModel(build_fn=build_model,
                                                                          inp_int_=inp_int,
                                                                          l1_w_=l1_w,
                                                                          l2_w_=l2_w,
                                                                          epochs=self.epoch,
                                                                          batch_size=self.batch,
                                                                          verbose=self.verbosity)
        else:
            # throws error if exceeds
            raise ValueError(
                "Weight Formula > 2 * Testing Data: Reduce by {}".format(
                    weights - (2 * n_test_samples)))

    def evaluate(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                 y_test: np.ndarray) -> float:
        """
        Evaluates the accuracy of a trained keras model on the testing data.

        :param x_train: Numpy array of training features.
        :param y_train: Numpy array of training labels.
        :param x_test: Numpy array of testing features.
        :param y_test: Numpy array of testing labels.
        :return: Accuracy of testing data.
        """

        # standard train + test
        self.model.fit(x_train, y_train, verbose=self.verbosity)
        # print('Accuracy on test data:', self.model.score(x_test, y_test, verbose=self.verbosity))
        return self.model.score(x_test, y_test, verbose=self.verbosity)

    def rank(self, x_test: np.ndarray, y_test: np.ndarray, feats: np.ndarray, show: int) -> list:
        """
        Performs feature ranking importance on the keras model and prints out important features.

        :param x_test: Numpy array of testing features.
        :param y_test: Numpy array of testing labels.
        :param feats: Numpy array consisting of feature names.
        :param show: integer to show top N features.
        :return: list of features
        """
        # this function does all of the work for this
        r = permutation_importance(self.model, x_test, y_test)
        # print('Most Important Features:')

        imp = r.importances_mean
        ids = np.argsort(imp)
        print(feats[ids[-show:-1]])  # top 5 features
