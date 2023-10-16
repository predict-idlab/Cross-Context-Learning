import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FeatureExtractor():
    """
    Class to facilitate data generation
    """
    def __init__(self, data_dir, locale, nodes=[]):
        self.data_dir = data_dir
        self.locale = locale
        self.node_info = df = pd.read_csv(self.make_info_dir(), names=['node_name', 'node_id', 'cmts'], sep=';')
        if nodes == []:
            nodes = self.node_info['node_id'].unique()
        try:
            self.nodes = self.read_csv(nodes)
            self.alarms = self.read_csv(nodes, alarm=True)
        except:
            print('Some nodes not present')

    def make_ts_dir(self, node, alarm=False):
        if alarm:
            return os.path.join(self.data_dir, self.locale, 'TimeSeriesAlarmData_' + str(node) + '.csv')
        else:
            return os.path.join(self.data_dir, self.locale, 'TimeSeriesData_' + str(node) + '.csv')

    def read_csv(self, nodes, alarm=False):
        info = {}
        for node in nodes:
            try:
                info[node] = pd.read_csv(self.make_ts_dir(node, alarm=alarm))
            except:
                print('Node %s not found' % (str(node)))

        return info

    def make_info_dir(self, info_dir="Node.csv"):
        return os.path.join(self.data_dir, self.locale, info_dir)

    def create_combined_dataset(self, window, nodes=[]):
        """
        Create a dataset for the given nodes, with a given window size
        :param window: window size
        :param nodes: which nodes to include in the dataset
        :return:
        """
        if len(nodes) == 0:
            nodes = self.nodes.keys()
        # save the original time
        self.node_times = {node: self.nodes[node]['time'].values for node in nodes}
        # combine all the time series to fit the scaler
        df_all = pd.concat([self.nodes[node].fillna(0).drop(columns=['time']) for node in nodes], ignore_index=True,
                           sort=False)
        # scale the data
        self.scaler = MinMaxScaler()
        self.scaler.fit(df_all)
        # window the series indendepently
        # save indexes
        total_index = 0
        self.node_indexes = {}
        all_windowed = []
        all_alarms = []
        alarm_substitutions = {'normal': 0, 'warning': 1, 'minor': 2, 'major': 3, 'critical': 4}

        for node in nodes:
            # first scale
            df_temp = self.nodes[node].fillna(0).drop(columns=['time'])
            df_scaled = pd.DataFrame(self.scaler.transform(df_temp), columns=df_temp.columns)
            # then window
            df_node_windowed = window_df(df_scaled, window).fillna(0)
            self.node_indexes[node] = range(total_index, total_index + len(df_node_windowed))
            total_index += len(self.nodes[node])
            all_windowed.append(df_node_windowed)
            # alarm data
            df_alarms = self.alarms[node].drop(columns=['time'])[df_temp.columns]
            for k, v in alarm_substitutions.items():
                df_alarms = df_alarms.replace(k, v)
            # window alarm data
            df_alarms_windowed = window_df(df_alarms, window).fillna(0)
            all_alarms.append(df_alarms_windowed)

        # combine the time series data
        windowed_df = pd.concat(all_windowed, ignore_index=True, sort=False)
        df_alarm_windowed = pd.concat(all_alarms, ignore_index=True, sort=False)

        # compute the indexes that describe the normal and abnormal behaviour
        self.normal_indexes = extract_alarm_indexes(df_alarm_windowed, columns=df_alarm_windowed.columns[1:],
                                                    comparison_function=np.equal, comparison_value=0,
                                                    truth_testing=np.all)
        self.warning_indexes = extract_alarm_indexes(df_alarm_windowed, columns=df_alarm_windowed.columns[1:],
                                                     comparison_function=np.equal, comparison_value=1,
                                                     truth_testing=np.any)
        self.minor_indexes = extract_alarm_indexes(df_alarm_windowed, columns=df_alarm_windowed.columns[1:],
                                                   comparison_function=np.equal, comparison_value=2,
                                                   truth_testing=np.any)
        self.major_indexes = extract_alarm_indexes(df_alarm_windowed, columns=df_alarm_windowed.columns[1:],
                                                   comparison_function=np.equal, comparison_value=3,
                                                   truth_testing=np.any)
        self.critical_indexes = extract_alarm_indexes(df_alarm_windowed, columns=df_alarm_windowed.columns[1:],
                                                      comparison_function=np.equal, comparison_value=4,
                                                      truth_testing=np.any)

        anom_scores = np.zeros(len(windowed_df))
        anom_scores[self.warning_indexes] = 1
        anom_scores[self.minor_indexes] = 2
        anom_scores[self.major_indexes] = 3
        anom_scores[self.critical_indexes] = 4
        windowed_df['anom_score'] = anom_scores

        return windowed_df, self.node_indexes

    def clean_up(self):
        """
        Removes the original data frames from memory
        :return: None
        """
        self.nodes = []
        self.alarms = []


def window_df(df_signal, window_size):
    """
    Window the dataframe
    :param df_signal: dataframe
    :param window_size: window size
    :return:
    """
    return pd.concat([df_signal.shift(-shift) for shift in range(0, window_size)], axis=1)[:-window_size + 1]


def extract_alarm_indexes(data_frame, columns, comparison_function=np.equal, comparison_value=0,
                          truth_testing=np.all):
    """
    Extracts the indexes that contain a certain alarm index.
    :param data_frame: Dataframe containing the alarm indexes (note that this dataframe is windowed, and thus contains multiple of the same columns).
    :param columns: the columns that need to be checked.
    :param comparison_function: function to compare the values (e.g. np.greater,np.equal,...)
    :param comparison_value:  value to compare the comparison_functions against.
    :param truth_testing: truth value testing function (e.g. np.any or np.all)
    :return: list of indexes
    """
    return truth_testing(
        np.array(
            [truth_testing(comparison_function(data_frame[column_name], comparison_value), axis=1) for column_name in
             columns]), axis=0)
