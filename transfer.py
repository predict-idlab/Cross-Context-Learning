import math
import random
import os

import numpy as np
import tensorflow as tf
from scipy.spatial import distance
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape, Dropout, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model

from make_dataset import FeatureExtractor
from train_model import train_test_all_last_split


# call back for computing AUC at end epoch/training
class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.results = []
        self.results_batch = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        mae = np.mean(np.mean(np.abs(self.X_val - y_pred), axis=1), axis=1)
        try:
            score = roc_auc_score(self.y_val, mae)
        except:
            score = np.NAN
        self.results_batch.append(score)

    def on_train_end(self, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        mae = np.mean(np.mean(np.abs(self.X_val - y_pred), axis=1), axis=1)
        try:
            score = roc_auc_score(self.y_val, mae)
        except:
            score = np.NAN
        self.results.append(score)

    def reset(self):
        self.results = []
        self.results_batch = []


# generic method to define AE
def lstm_ae(x_train, y_train, x_val, y_val, ival, params):
    # define model
    input_shape = (x_train.shape[1], x_train.shape[2])
    visible = Input(shape=input_shape)
    o = visible

    # encoder
    o = LSTM(params['first_neurons'], return_sequences=False)(o)
    o = Dropout(params['dropout'])(o)
    encoder = o

    # duplicator
    o = RepeatVector(input_shape[0])(o)

    # decoder
    o = LSTM(params['first_neurons'], return_sequences=True)(o)
    o = Dropout(params['dropout'])(o)
    o = TimeDistributed(Dense(input_shape[1]))(o)

    model = Model(inputs=visible, outputs=o)
    model_encoder = Model(inputs=visible, outputs=encoder)
    model.compile(optimizer=params['optimizer'], loss='mse')
    model.summary()
    callbacks = [ival, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                 tf.keras.callbacks.ModelCheckpoint(
                     'best_weights', monitor='val_loss', save_best_only=True, save_weights_only=True, )]
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        callbacks=callbacks)
    model.load_weights('best_weights')

    return history, model


def get_indices(x_train_or, x_train_dest, incremental_method='max_diff'):
    # choose X_train_limited intelligently
    # when choosing the last data, no good results
    # when choosing randomly, after 40 epochs good results
    # Lets choose the data that's most different
    size = int(x_train_dest.shape[0] * 0.4)
    if incremental_method == 'max_diff':
        diff_distances = []
        print("Respective sizes:", x_train_or.shape[0], "(origin)", "and", x_train_dest.shape[0], "(destination)")
        for i in range(min(x_train_or.shape[0], x_train_dest.shape[0])):
            diff_distances.append(distance.euclidean(x_train_dest[i].reshape(-1), x_train_or[i].reshape(-1)))
        limited_indices = np.array(diff_distances).argsort()[:size][::-1]
    if incremental_method == 'random':
        # randomly choose the indexes
        limited_indices = random.sample(list(range(len(x_train_dest))), size)
    if incremental_method == 'last':
        limited_indices = list(range(len(x_train_dest)))[:size]
    return limited_indices


def incremental_learn(model, x_train, y_train, x_val, y_val, ival, params):
    print("Updating model with incremental learning...")
    model.summary()
    callbacks = [ival, tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
                 tf.keras.callbacks.ModelCheckpoint(
                     'best_weights', monitor='val_loss', save_best_only=True, save_weights_only=True, )]
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        batch_size=params['batch_size'],
                        epochs=100,
                        verbose=0,
                        callbacks=callbacks)
    model.load_weights('best_weights')
    return history, model


def train_on_context(node, locale, model=None, window=5, features=16, incremental=False, limited=False, indices=None,
                     split_rate=0.4, override=False):
    """
    Train model on specific node context.
    :param features: nr of features in the input dimension
    :param window: window size
    :param node: node context to be trained on
    :param raw_data_location: location of the data to train the model
    :param locale: locale associated with node data
    :param model: model to be trained
    :param incremental: not supported
    :param limited: select only part of the training data
    :param indices: select specific data indices
    :param split_rate: how to split data into train and test sets
    :param override: override standard abort protocol
    :return:
    """
    config = {"first_neurons": 32,
              "optimizer": "adam",
              "dropout": 0.1,
              "batch_size": 32,
              "epochs": 500}

    raw_data_location = os.path.join('temp', 'data', 'raw', 'PreprocessedData_2019')

    # Extract the time series features and preprocess the data
    fa = FeatureExtractor(data_dir=raw_data_location, locale=locale, nodes=[node])

    # Window the time series
    df_combined, node_indexes = fa.create_combined_dataset(window=window, nodes=[node])

    # split_rate = 0.4
    abnormal_threshold = 3  # used threshold to determine the anomalies {'normal': 0, 'warning': 1, 'minor': 2, 'major': 3, 'critical': 4}

    current_node_range = {node_id: node_indexes[node_id] for node_id in [node]}

    # split train/test set
    train_specific, test_specific = train_test_all_last_split(df_combined, current_node_range, test_size=split_rate,
                                                              ignore_nodes=[], window_correction=window)

    df_test_specific = test_specific.reset_index(drop=True)

    # determine normal/abnormal data
    df_train_specific_normal = train_specific[train_specific.anom_score < abnormal_threshold].drop(
        columns=['anom_score'])
    df_train_specific_abnormal = train_specific[train_specific.anom_score >= abnormal_threshold].drop(
        columns=['anom_score'])
    df_test_specific_normal = df_test_specific[df_test_specific.anom_score < abnormal_threshold].drop(
        columns=['anom_score'])
    df_test_specific_abnormal = df_test_specific[df_test_specific.anom_score >= abnormal_threshold].drop(
        columns=['anom_score'])

    print("Abnormal train and test found:", df_train_specific_abnormal.shape[0], df_test_specific_abnormal.shape[0])

    # generate train/test data
    X_train_or = df_train_specific_normal.values.reshape(-1, window, features)
    # perform train-validation split
    X_train = X_train_or[:int(X_train_or.shape[0] * 0.8)]
    X_test = X_train_or[int(X_train_or.shape[0] * 0.8):]
    X_train_all = train_specific.drop(columns=['anom_score']).values.reshape(-1, window, features)
    X_test_all = df_test_specific.drop(columns=['anom_score']).values.reshape(-1, window, features)

    print("Original dimensions <train, val, test>: <", X_train.shape[0], X_test.shape[0], X_test_all.shape[0], ">")
    print("All train data:", X_train_all.shape[0])

    anoms = np.zeros(X_test_all.shape[0])
    anoms[df_test_specific_abnormal.index] = 1
    ival = IntervalEvaluation(validation_data=(X_test_all, anoms), interval=1)

    # select only a part of training data...
    if limited:
        if indices is not None:
            print("Selecting", len(indices), "indices from", X_train.shape[0], "training samples...")
            X_train = X_train[indices]
        else:
            # ... so, until 40% (first 40%, chronologically)
            X_test = np.vstack((X_train[int(X_train.shape[0] * 0.4):], X_test))  # larger validation set
            X_train = X_train[:int(X_train.shape[0] * 0.4)]

            print("Limited dimensions <train, val, test>: <", X_train.shape[0], X_test.shape[0], X_test_all.shape[0],
                  ">")

    if X_train.shape[0] == 0 or (df_test_specific_abnormal.shape[0] == 0 and not override):
        if X_train.shape[0] == 0:
            print("No training data found... aborting.")
        if df_test_specific_abnormal.shape[0] == 0:
            print("No test anomalies found... aborting.")
        return X_train_or, model, anoms, ival, np.nan, np.nan, np.nan, np.nan

    if model is None:
        history, model = lstm_ae(X_train, X_train, X_test, X_test, ival, config)
    else:
        if incremental:
            history, model = incremental_learn(model, X_train, X_train, X_test, X_test, ival, config)

    final_loss = model.evaluate(X_test_all, X_test_all)
    y_pred = model.predict(X_test_all, verbose=0)
    # calculate the MAE
    mae = np.mean(np.mean(np.abs(X_test_all - y_pred), axis=1), axis=1)
    # calculate the AUC for ROC curve (or precision recall curve)
    try:
        auc_score = roc_auc_score(anoms, mae)
    except:
        print('Unable to calculate ROC, only 1 label')
        auc_score = np.NAN
    auc_avg_pr = average_precision_score(anoms, mae)
    print('AUC ROC: ', auc_score)
    print('AUC Precision/Recall: ', auc_avg_pr)
    print('reconstruction loss test set:', final_loss)
    # prints the AUCs throught the training
    print('AUC after each epoch:', ival.results_batch)

    return X_train_or, model, anoms, ival, mae, auc_score, auc_avg_pr, final_loss
