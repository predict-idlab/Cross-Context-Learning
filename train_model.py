import numpy as np


def train_test_all_last_split(x_train, node_ranges, test_size=0.2, ignore_nodes=[], window_correction=0):
    """
    Splits the whole data set in test and train sets. This is achieved by taking the first x% of each time series as training set.
    All train and test sets are combined.
    :param x_train: The whole dataset
    :param node_ranges: For each node the indexes they contain within the whole dataset
    :param test_size: % of the test set size
    :param ignore_nodes: nodes that should not be included in the train/test sets
    :param window_correction: number of samples that need to be removed from the train set in order to avoid data leakage through the window
    :return: train and test sets for the whole dataset.
    """
    train_indexes = [[]]
    test_indexes = [[]]
    for node, range in node_ranges.items():
        if not node in ignore_nodes:
            normal_index = list(set(x_train.index) & set(list(range)))
            train_indexes.append(normal_index[:int((1 - test_size) * len(normal_index)) - window_correction])
            test_indexes.append(normal_index[int((1 - test_size) * len(normal_index)):])

    return x_train.loc[np.concatenate(train_indexes)], x_train.loc[np.concatenate(test_indexes)]
